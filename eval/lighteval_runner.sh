#!/usr/bin/env bash

set -euo pipefail

MODEL=$1
if [[ -z "$MODEL" ]]; then
    echo "Usage: sbatch eval/lighteval_runner.sh <model_dir>"
    exit 1
fi

echo "[INFO] Running LightEval benchmarks for $MODEL"

lighteval --model "$MODEL" --tasks mmlu,gsm8k,arc_challange,truthfulqa --batch_size 4 --save_results "results/lighteval_${MODEL//\//_}.json"
