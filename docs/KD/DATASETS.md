## Llama-3.1-70B_KD-main/kd/datasets.py

**Classification:** Code (Python)  
**Size (bytes):** 5037  
**Purpose (Official Statement):** Dataset utilities and loaders.

**Declared Imports/Modules:** glob, pyarrow.parquet, torch, torch.utils.data, typing

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
from typing import Dict, List, Optional
import glob
import torch
from torch.utils.data import IterableDataset
import pyarrow.parquet as pq

def _iter_parquet_rows(path_glob: str, columns: Optional[List[str]] = None):
    files = sorted(glob.glob(path_glob))
    assert files, f"No Parquet files found for {path_glob}"
    for f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            batch = pf.read_row_group(rg, columns=columns).to_pydict()
            n = len(next(iter(batch.values()))) if batch else 0
            for i in range(n):
                yield {k: batch[k][i] for k in batch.keys()}

class RBTopKIterableDataset(IterableDataset):
    def __init__(self, path_glob: str):
        super().__init__()
        self.path_glob = path_glob
    def __iter__(self):
        for row in _iter_parquet_rows(self.path_glob, columns=["input_ids", "attn_mask", "topk_ids", "topk_logprobs"]):
            yield {k: torch.tensor(row[k]) for k in row}

class FBDataset(IterableDataset):
    def __init__(self, path_glob: str, teacher_layer: int):
        super().__init__()
        self.path_glob = path_glob
        self.col = f"hidden_L{teacher_layer}"
    def
```
