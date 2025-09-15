## Llama-3.1-70B_KD-main/scripts/run_feature_based_single_node.sh

**Classification:** Code (Shell)  
**Size (bytes):** 1193  
**Purpose (Official Statement):** Shell/SLURM orchestration script.

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
#SBATCH --job-name=kd_feature_based_single_node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=h100
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
source scripts/_env_single_node.sh

echo "[INFO] Feature-Based KD | node=1 | gpus=$GPUS_PER_NODE | procs=$NUM_PROCESSES"

mkdir -p logs/telemetry/$SLURM_JOB_ID
python monitor.py --output logs/telemetry/$SLURM_JOB_ID/${HOSTNAME}.jsonl --interval 1 &
MON_PID=$!

RUN_DIR="serialization_dir/$(date +%Y%m%d_%H%M)_FB_1n"
mkdir -p "$RUN_DIR"

accelerate launch \
  --num_machines 1 \
  --num_processes ${NUM_PROCESSES} \
  --deepspeed configs/ds_zero3.json \
  kd/train.py \
    --kd.mode fb \
    --student Qwen2.5-1.5B-Instruct \
    --data "data/fb_hints_L22/*.parquet" \
    --fb.teacher_layer 22 \
    --fb.student_layer 12 \
    --fb.token_subset_ratio 0.25 \
    --lora.r 16 \
    --lora.alpha 32 \
    --lr 1e-4 \
    --batch_size 2 \
    --max_steps 2000 \
    --save_dir "$RUN_DIR" \
    --save_every 200 \
    --resume auto

kill $MON_PID || true
echo "[INFO] FB KD complete"

```