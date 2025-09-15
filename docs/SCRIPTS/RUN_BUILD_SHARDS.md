## Llama-3.1-70B_KD-main/scripts/run_build_shards.sh

**Classification:** Code (Shell)  
**Size (bytes):** 1684  
**Purpose (Official Statement):** Dataset sharding/manifest builder.

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
#SBATCH --job-name=kd_build_shards
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=h100
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
source scripts/_env_single_node.sh   # creates/activates env & installs reqs

# Inputs (override with --export)
HF_DATASETS="${HF_DATASETS:-teknium/OpenHermes-2.5}"
SPLIT="${SPLIT:-train}"
WEIGHTS="${WEIGHTS:-}"               # e.g. "0.8,0.2" if multiple datasets
MAX_SAMPLES="${MAX_SAMPLES:-0}"      # 0 = no cap
OUT="${OUT:-data/shards.jsonl}"
STREAMING="${STREAMING:-1}"          # 1 = streaming, 0 = non-streaming
DATA_DIR="${DATA_DIR:-}"             # for offline local cache: set to your path

# Log
echo "[INFO] Datasets: $HF_DATASETS"
echo "[INFO] Split:    $SPLIT"
echo "[INFO] Weights:  ${WEIGHTS:-<uniform>}"
echo "[INFO] Max:      ${MAX_SAMPLES}"
echo "[INFO] Out:      $OUT"
echo "[INFO] Streaming:${STREAMING}"
echo "[INFO] Data dir: ${DATA_DIR:-<none>}"

# Build args
DATASET_ARGS=()
IFS=',' read -ra DS_ARR <<< "$HF_DATASETS"
for d in "${DS_ARR[@]}"; do
  DATASET_ARGS+=(--dataset "$d")
done

[[ "$STREAMING" == "1" ]] && STREAM_FLAG="--streaming" || STR
```