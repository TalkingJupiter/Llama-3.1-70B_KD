## Llama-3.1-70B_KD-main/scripts/_env_single_node.sh

**Classification:** Code (Shell)  
**Size (bytes):** 1288  
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
set -euo pipefail

# -------------------------------
# Conda environment setup
# -------------------------------
ENV_NAME="kd"
REQ_FILE="requirements.txt"

# Ensure conda is initialized
source ~/.bashrc

if ! conda env list | grep -q "^$ENV_NAME "; then
  echo "[INFO] Conda env '$ENV_NAME' not found. Creating..."
  conda create -y -n $ENV_NAME python=3.10
  conda activate $ENV_NAME
  if [[ -f "$REQ_FILE" ]]; then
    echo "[INFO] Installing requirements..."
    pip install -r $REQ_FILE
  else
    echo "[WARN] $REQ_FILE not found, skipping requirements install"
  fi
else
  echo "[INFO] Conda env '$ENV_NAME' exists. Activating..."
  conda activate $ENV_NAME
fi

# -------------------------------
# Node/GPU topology
# -------------------------------
GPUS_PER_NODE=${GPUS_PER_NODE:-4}      # each REPACSS node has 4x H100
PROCS_PER_GPU=${PROCS_PER_GPU:-1}      # set >1 if you want multiple processes per GPU
NUM_PROCESSES=$(( GPUS_PER_NODE * PROCS_PER_GPU ))

# -------------------------------
# NCCL tuning for NVLink on H100
# -------------------------------
export NCCL_P2P_LEVEL=NVL
export NCCL_MIN_NCHANNELS=8
export NCCL_DEBUG=WARN

# -------------------------------
#
```
