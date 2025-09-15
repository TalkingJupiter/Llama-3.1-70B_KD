## Llama-3.1-70B_KD-main/eval/harness_runner.sh

**Classification:** Code (Shell)  
**Size (bytes):** 391  
**Purpose (Official Statement):** Evaluation runners or parsers.

**Shebang:** `#!/usr/bin/env bash`

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
#!/usr/bin/env bash
set -euo pipefail

MODEL=$1
if [[ -z "$MODEL" ]]; then
    echo "Usage: bash eval/harness_runner.sh <model_dir>"
    exit 1
fi

echo "[INFO] Running HF LM Eval Harness for $MODEL"

lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL" \
    --tasks mmlu,hellaswag,bbh,arc_challenge \
    --batch_size 4 \
    --output_path "results/harness_${MODEL//\//_}.json"

```