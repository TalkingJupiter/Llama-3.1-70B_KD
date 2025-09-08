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
