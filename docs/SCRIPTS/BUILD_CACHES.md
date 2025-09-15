## Llama-3.1-70B_KD-main/scripts/build_caches.sh

**Classification:** Code (Shell)  
**Size (bytes):** 1401  
**Purpose (Official Statement):** Teacher cache / artifact builder.

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
#SBATCH --job-name=kd_build_caches
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=h100
#SBATCH --time=48:00:00
#SBATCH --exclusive
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
source scripts/_env_single_node.sh

IN=${IN:-data/shards.jsonl}
TEACHER=${TEACHER:-meta-llama/Llama-3.1-70B-Instruct}

echo "[INFO] Teacher: $TEACHER"
echo "[INFO] Input:   $IN"
echo "[INFO] GPUs visible: ${CUDA_VISIBLE_DEVICES:-unset}"

# Fail fast if input is missing/empty
if [[ ! -s "$IN" ]]; then
  echo "[ERROR] Input jsonl '$IN' not found or empty." >&2
  exit 2
fi

# Make sure output dirs exist
mkdir -p data/topk_k16 data/fb_hints_L22 data/relb_embeds

# ---- RB top-k caches
python teacher_farm/make_topk_cache.py \
  --model "$TEACHER" \
  --input_jsonl "$IN" \
  --out_dir data/topk_k16/ \
  --k 16 \
  --dtype float16

# ---- FB hidden-state caches (e.g., teacher layer 22)
python teacher_farm/make_hidden_cache.py \
  --model "$TEACHER" \
  --input_jsonl "$IN" \
  --out_dir data/fb_hints_L22/ \
  --layers 22 \
  --batch_size 1 \
  --max_length 2048 \
  --dtype bfloat16 \
  --flush_ev
```