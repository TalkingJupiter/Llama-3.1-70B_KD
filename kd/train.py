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
    ap.add_argument('--lr', type=flaot, default=1e-4)
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


if __name__ == '__main__':
    main()