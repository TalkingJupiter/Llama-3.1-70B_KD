#!/usr/bin/env bash
#SBATCH --job-name=kd_eval_harness_submitter
#SBATCH --partition=zen4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=eval/logs/%x_%j.out
#SBATCH --error=eval/logs/%x_%j.err

sbatch eval/harness_runner.sh\
  Qwen/Qwen2.5-1.5B-Instruct \
  serialization_dir/*FB*

sbatch eval/harness_runner.sh\
  Qwen/Qwen2.5-1.5B-Instruct \
  serialization_dir/*RelB*

sbatch eval/harness_runner.sh\
  meta-llama/Llama-3.1-8B \
  serialization_dir/*RB*