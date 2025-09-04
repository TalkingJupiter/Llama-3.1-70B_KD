import argparse, os, time
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup

from kd.models import load_student
from kd.kd_rb import response_kd_loss
from kd.kd_fb import feature_kd_loss, LinearProjector
from kd.kd_relb import relation_kd_loss
from kd.datasets import RBTopKIterableDataset, FBDataset, RelBDataset, collate_rb, collate_pad

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kd.mode', dest="kd_mode", choices=['rb', 'fb', 'relb'], required=True)
    ap.add_argument('--student', type=str, required=True)
    ap.add_argument('--data', type=str, required=True, help="Parquet path glob")
    ap.add_argument('--seq_len', type=int, default=8192)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--save', type=str, default=1)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--bash_size', type=int, default=2)
    ap.add_argument('--warmup_steps', type=int, default=100)
    ap.add_argument('max_steps', type=int, default=1000)

    ### Response Based KD Arguments ###
    ap.add_argument('--rb.topk', type=int, default=16)
    ap.add_argument('--rb.temperature', type=float, default=2.0)

    ### Feature Based KD Arguments ###
    ap.add_argument('--fb.teacher_layer', type=int, default=22)
    ap.add_argument('--fb.student_layer', type=int, default=12)
    ap.add_argument('--fb.token_subset_ratio', type=float, default=0.25)

    ### Relation Based KD Arguments ###
    ap.add_argument('--relb.lambda_dist', type=float, default=1.0)
    ap.add_argument('--relb.lambda_angle', type=float, default=0.5)

    ### LoRA Setting Arguments
    ap.add_argument('--lora.r', dest='lora_r', type=int, default=16)
    ap.add_argument('--lora.alpha', dest='lora_alpha', type=int, default=32)
    return ap.parse_args()

def main():
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device

    model, tok = load_student(args.student, lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    model = accelerator.prepare(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.max_steps)

    if args.kd_mode == 'rb':
        dataset = RBTopKIterableDataset(args.data)
        collate = collate_rb
    elif args.kd_mode == 'fb':
        dataset = FBDataset(args.data, teacher_layer=args.fb_teacher_layer)
        collate = collate_pad
    else:
        dataset = RelBDataset(args.data)
        collate = collate_pad
    
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate)
    loader = accelerator.prepare(loader)

    model.train()
    projector = None
    step = 0
    t0 = time.time()
    total_tokens = 0 

    for epoch in range(args.epochs):
        for batch in loader:
            if step >= args.max_steps:
                break
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attn_mask'].to(device)

            if args.kd_mode == 'rb':
                out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
                s_logits = out.logits[:, :-1, :]
                min_len = min(s_logits.size(1), batch['topk_ids']).size(1)
                kd = response_kd_loss(
                    s_logits[:, :min_len, :],
                    batch['topk_ids'][:, :min_len, :].to(device),
                    batch['topk_logprobs'][:, :min_len, :].to(device),
                    T=args.rb_temperature
                )
                loss = kd
                token_this = (attn_mask.sum() - input_ids.size(0)).item()

            elif args.kd_mode == 'fb':
                out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False, output_hidden_states=True)
                s_hid = out.hidden_states[args.fb_student_layer]
                t_feats = batch['teacher_feats'].to(device)
                if projector is None:
                    projector = LinearProjector(s_hid.size(-1), t_feats.size(-1)).to(device)
                    projector = accelerator.prepare(projector)
                s_proj = projector(s_hid)
                loss = feature_kd_loss(s_proj, t_feats, token_mask=attn_mask)
                token_this = attn_mask.sum().item()

            else:   ### Relation Based ###
                out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False, output_hidden=True)
                last = out.hidden_states[-1]
                mask = attn_mask.unsqueeze(-1)
                pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                t_emb = batch['teacher_embed'].to(device)
                loss = relation_kd_loss(pooled, t_emb, lambada_dist=args.relb_lambda_dist, lambada_angle=args.relb_lambda_angle)
                token_this = attn_mask.sum().item()

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            total_tokens += token_this
            step += 1
            if accelerator.is_main_process and step % 10 == 0:
                dt = time.time() - t0
                tps = total_tokens / max(dt, 1e-6)
                print(f"[step {step}] loss={loss.item():.4f} tokens={int(total_tokens)} tok/s={tps:.1f}")

        if step >= args.max_steps:
            break

    if accelerator.is_main_process:
        os.makedirs(args.fave, exist_ok=True)
        model.save_pretrained(args.save)
        tok.save_pretrained(args.save)
        print(f"Saved to {args.save}")






if __name__ == '__main__':
    main()