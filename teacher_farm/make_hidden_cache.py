import argparse, os, json, math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pyarrow as pa, pyarrow.parquet as pq
from tqdm import tqdm

def batched(iterable, n):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--input_jsonl', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--layers', type=int, nargs='+', required=True, help='Teacher layer indicates to save, eg. 22 30')
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--max_length', type=int, default=8192)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype= torch.bfloat16, device_map='auto')
    model.eval()

    texts = [json.loads(l)['text'] for l in open(args.input_jsonl) if l.strip()]
    shard_size = 32
    rows, shard_idx = [], 0

    with torch.no_grad():
        for batch in tqdm(batched(texts, args.batch_size), total=math.ceil(len(texts)/args.batch_size)):
            enc = tok(batch, padding=True, truncation=True, max_length=args.max_length, return_tensors='pt')
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True, use_cache=False)
            hidden = out.hidden_states ##TUPLE of [B, T, d]

            input_ids = enc['input_ids'].cpu()
            attn_mask = enc['attention_mask'].cpu()

            for b in range(input_ids.size(0)):
                L = int(attn_mask[b].sum().item())
                row = {
                    'input_ids': input_ids[b, :L].tolist(),
                    'attn_mask': attn_mask[b, :L].tolist(),
                }
                for li in args.layers:
                    ht = hidden[li][b, :L, :].cpu().tolist()
                    row[f'hidden_L{li}'] = ht
                rows.append(row)
            
    if rows:
        table = pa.Table.from_pylist(rows)
        out_path = os.path.join(args.out_dir, f'fb_hints_{shard_idx:06d}.parquet')
        pq.write_table(table, out_path, compression='zstd')
        print('Wrote', out_path)

if __name__ == '__main__':
    main()
