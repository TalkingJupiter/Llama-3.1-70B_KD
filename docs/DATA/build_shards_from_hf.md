## Llama-3.1-70B_KD-main/data/build_shards_from_hf.py

**Classification:** Code (Python)  
**Size (bytes):** 5644  
**Purpose (Official Statement):** Dataset sharding/manifest builder.

**Shebang:** `#!/usr/bin/env python3`

**Declared Imports/Modules:** argparse, datasets

**Command-Line Interface (argparse discovered):**
- `"--dataset", action="append", required=True,                     help="HF dataset name (repeatable`
- `"--split", default="train", help="Split to load (default: train`
- `"--weights", type=str, default="",                     help="Comma-separated weights matching --dataset count, e.g. '0.8,0.2'"`
- `"--max_samples", type=int, default=None,                     help="Cap total samples across all datasets"`
- `"--seed", type=int, default=42`
- `"--out", default="data/shards.jsonl"`
- `"--streaming", action="store_true", default=True`
- `"--no-streaming", dest="streaming", action="store_false"`
- `"--data_dir", default=None,                     help="If HF offline, point to local data dir/cache"`

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
#!/usr/bin/env python3
import argparse, json, random, sys, os
from datasets import load_dataset, Dataset, concatenate_datasets

# --- Heuristics to extract text from various dataset schemas ---
def record_to_text(rec):
    # 1) Raw "text"
    if "text" in rec and isinstance(rec["text"], str) and rec["text"].strip():
        return rec["text"].strip()

    # 2) Instruction-style pairs
    for a,b in [("instruction","output"), ("prompt","response"), ("input","output")]:
        if a in rec and b in rec and isinstance(rec[a], str) and isinstance(rec[b], str):
            s = rec[a].strip()
            r = rec[b].strip()
            if s or r:
                return f"### Instruction:\n{s}\n\n### Response:\n{r}"

    # 3) Chat "messages": list of {role, content}
    if "messages" in rec and isinstance(rec["messages"], (list, tuple)):
        msgs = []
        for m in rec["messages"]:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                # Sometimes content is list of segments (OpenAI-style tools); flatten strings
                content = "\n".join([seg.get("text","") if isinstance(seg, dict) else 
```
