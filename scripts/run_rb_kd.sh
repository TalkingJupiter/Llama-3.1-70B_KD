## INFO: SLURM SYSTEM INFO
#!/usr/bin/env
#SBATCH -J kd_rb
#SBATCH -N 8
#SBATCH --gpus-per-node=4
#SBATCH -p h100
#SBATCH -t 24:00:00
#SBATCH --exclusive
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#=========================#

set -euo pipefail

## TODO: Change the source to conda detecting enviroment
source ~/.bashrc
conda activate kd || true
#========================#

echo "[INFO] Starting Response-Based Knowledge Distillation | student=Llama-3.1-8B | teacher=Llama-3.1-70B-Instruct "

# Info: Start Telemetry on every node
srun --nodes=${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 --exclusive \
    bash -lc 'mkdir -p logs/telemetry/$SLURM_JOB_ID; \
              python monitor.py --output logs/telemetry/$SLURM_JOB_ID/${HOSTNAME}.jsonl --interval 5' \
    > logs/telemetry/${SLURM_JOB_ID}_monitor.out 2>&1 &
MONITOR_PID=$!

NUM_MACHINES=${SLURM_JOB_NUM_NODES}
NUM_PROCESSES=$((4 * NUM_MACHINES))

accelerate launch \
    --num_machines $(NUM_MACHINES) \
    --num_processes ${NUM_PROCESSES} \
    --deepspeed configs/ds_zero3.json \
    kd/train.py \
        --kd.mode rb \
        --student meta-llama/Llama-3.1-8B \
        --data data/topk_k16/*.parquet \
        --rb.topk 16 --rb.temperature 2.0
        --lora.r 16 --lora.alpha 32 --lr 1e-4 \
        --batch_size 2 --max_steps 2000
        --save serialization_dir/$(date +%Y%m%d_%H%M)_RB_k16_T2

# Info: Stop Telemetry
srun --nodes=${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 --exclusive batch -lc 'pkill -f "python monitor.py" || true'
wait $MONITOR_PID || true
