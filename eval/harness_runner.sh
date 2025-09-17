#!/usr/bin/env bash
#SBATCH --job-name=kd_eval_harness
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=eval/logs/%x_%j.out
#SBATCH --error=eval/logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

BASE=${1:?Usage: sbatch kd_eval_harness.slurm <base_model_id> <adapter_dir>}
ADAPTER=${2:?Usage: sbatch kd_eval_harness.slurm <base_model_id> <adapter_dir>}

mkdir -p logs results
source ~/.bashrc || true
conda activate kd || true
[[ -f scripts/_env_single_node.sh ]] && source scripts/_env_single_node.sh

SAFE_BASE="${BASE//\//_}"
RUN_NAME="${SAFE_BASE}__$(basename "$ADAPTER")"

echo "[INFO] Base: $BASE"
echo "[INFO] Adapter: $ADAPTER"

lm_eval \
  --model hf \
  --model_args "pretrained=${BASE},peft=${ADAPTER}" \
  --tasks mmlu,hellaswag,bbh,arc_challenge \
  --batch_size 4 \
  --output_path "results/harness_${RUN_NAME}.json"

echo "[INFO] Done -> results/harness_${RUN_NAME}.json"
