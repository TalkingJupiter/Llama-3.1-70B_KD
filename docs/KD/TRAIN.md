## Llama-3.1-70B_KD-main/kd/train.py

**Classification:** Code (Python)  
**Size (bytes):** 8858  
**Purpose (Official Statement):** Training entrypoint or runner.

**Declared Imports/Modules:** accelerate, argparse, json, kd.datasets, kd.kd_fb, kd.kd_rb, kd.kd_relb, kd.models, torch, torch.utils.data, transformers

**Command-Line Interface (argparse discovered):**
- `'--kd.mode', dest="kd_mode", choices=['rb', 'fb', 'relb'], required=True`
- `'--student', type=str, required=True`
- `'--data', type=str, required=True, help="Parquet path glob"`
- `'--seq_len', type=int, default=8192`
- `'--lr', type=float, default=1e-4`
- `'--save', type=str, default=1`
- `'--epochs', type=int, default=1`
- `'--bash_size', type=int, default=2`
- `'--warmup_steps', type=int, default=100`
- `'max_steps', type=int, default=1000`
- `'--rb.topk', type=int, default=16`
- `'--rb.temperature', type=float, default=2.0`
- `'--fb.teacher_layer', type=int, default=22`
- `'--fb.student_layer', type=int, default=12`
- `'--fb.token_subset_ratio', type=float, default=0.25`
- `'--relb.lambda_dist', type=float, default=1.0`
- `'--relb.lambda_angle', type=float, default=0.5`
- `'--lora.r', dest='lora_r', type=int, default=16`
- `'--lora.alpha', dest='lora_alpha', type=int, default=32`
- `'--save-dir', type=str, required=True, help='Root directory to run + checkpoints'`
- `'--save_every', type=int, default=0, help='Steps between checkpoints (0=off`
- `'--resume', type=str, default='auto', choices=['auto', 'none', 'path'], help='Resume Policy'`
- `'--resume_path', type=str, default='', help='Directory of a specific checkpoint when --resume=path'`

### Inputs
- See usage within code and related runbooks; typical inputs include dataset manifests, cache directories, model references, and SLURM environment variables.

### Outputs
- Logs to `logs/` (stdout/err) and task-specific outputs (e.g., shards JSONL, cache directories, checkpoints, results CSV/JSON).

### Upstream Dependencies
- Python environment (pinned), CUDA/NVML where applicable.
- For data-related scripts: access to Hugging Face datasets (online or local cache).
- For training/eval: model weights (teacher/student), tokenizers, and cache artifacts.

### Downstream Dependencies
- Subsequent pipeline stages rely on the artifacts produced by this file (e.g., training consumes caches/manifests; evaluation consumes checkpoints; parsing consumes evaluation JSONs).

### Standard Operating Procedure (SOP)
1. Verify environment and data access.
2. Execute with validated parameters (see CLI above if present).
3. Confirm expected outputs exist and are non-empty.
4. Register run metadata (dataset revisions, commit SHA, seed) in audit log.

### Error Handling & Recovery
- Fail-fast on missing inputs. Re-run with same parameters after addressing preconditions. For idempotent steps (e.g., cache building), allow safe re-execution.

### Logging & Observability
- Emit structured status lines to stdout/err; prefer JSON logs where possible. Training/eval should produce periodic progress and final summaries. Telemetry via `monitor.py` where applicable.

### Validation & Quality Assurance
- Cross-check counts (samples, steps), schema (expected keys/columns), and spot-validate a random sample of outputs.
- For KD: confirm loss values are finite and decreasing over warmup; for eval: verify task coverage and metric ranges.

### Security & Compliance Notes
- Do not embed secrets (tokens) in code or outputs. Limit permissions on output directories. Ensure dataset licenses permit use. Retain provenance for reproducibility.

### Code Excerpt (for reference)
```
import argparse, os, time
import torch
import json, signal, pathlib
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
    ap.add_argument()
```