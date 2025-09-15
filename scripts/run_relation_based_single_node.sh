#!/usr/bin/env bash
#SBATCH --job-name=kd_relation_based_single_node
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

echo "[INFO] Relation-Based KD | node=1 | gpus=$GPUS_PER_NODE | procs=$NUM_PROCESSES"

mkdir -p logs/telemetry/$SLURM_JOB_ID
python monitor.py --output logs/telemetry/$SLURM_JOB_ID/${HOSTNAME}.jsonl --interval 1 &
MON_PID=$!

RUN_DIR="serialization_dir/$(date +%Y%m%d_%H%M)_RelB_1n"
mkdir -p "$RUN_DIR"

accelerate launch \
  --num_machines 1 \
  --num_processes ${NUM_PROCESSES} \
  --deepspeed_config_file configs/ds_zero3.json \
  kd/train.py \
    --kd.mode relb \
    --student Qwen2.5-1.5B-Instruct \
    --data "data/relb_embeds/*.parquet" \
    --relb.lambda_dist 1.0 \
    --relb.lambda_angle 0.5 \
    --lora.r 16 \
    --lora.alpha 32 \
    --lr 1e-4 \
    --batch_size 4 \
    --max_steps 2000 \
    --save_dir "$RUN_DIR" \
    --save_every 200 \
    --resume auto

kill $MON_PID || true
echo "[INFO] RelB KD complete"
