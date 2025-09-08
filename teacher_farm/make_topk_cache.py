import argparse, os, json, math
import torch
import torch.nn.functional as F
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
    ap.add_argument('--input_jsonl', required=True, help='Each line {\"text\": ...}')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--batch_size', type=int, default=1, help='1 recommended for very long sequences')
    ap.add_argument('--max_length', type=int, default=8192)
    ap.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)