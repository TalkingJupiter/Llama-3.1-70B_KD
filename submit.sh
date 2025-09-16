#!/usr/bin/env bash
#SBATCH --job-name=pipeline_launcher
#SBATCH --partition=zen4              # CPU partition to run the launcher itself
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# This job only SUBMITS other jobs using sbatch (very light). Those jobs do the real work.

set -Eeuo pipefail

# -------------- config --------------
# You can override these by: sbatch --export=ALL,ENV_JOB=...,PARTITION_CPU=...,PARTITION_GPU=...
LOGDIR="logs"
ENV_JOB="${ENV_JOB:-scripts/_env_single_node.sh}"
SHARDS_JOB="${SHARDS_JOB:-scripts/run_build_shards.sh}"
CACHES_JOB="${CACHES_JOB:-scripts/build_caches.sh}"
KD_JOB="${KD_JOB:-scripts/submit_all_kd_single_node.sh}"

# Optional: default partitions if you want to pass them to children here (only needed if your child scripts lack #SBATCH)
PARTITION_CPU="${PARTITION_CPU:-zen4}"
PARTITION_GPU="${PARTITION_GPU:-h100}"

mkdir -p "$LOGDIR"
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# -------------- helpers -------------
die() { echo "[FATAL] $*" >&2; exit 1; }

need_file() {
  [[ -f "$1" ]] || die "[ERROR] Missing required file: $1"
}

submit() {
  # usage: submit <script> [additional sbatch flags...]
  local script="$1"; shift || true
  local out
  out=$(sbatch "$script" "$@" 2>&1) || die "sbatch failed for: $script $* :: $out"
  local jid
  jid=$(awk '/Submitted batch job/ {print $4}' <<<"$out")
  [[ -n "${jid:-}" ]] || die "Could not parse job id from sbatch output: $out"
  echo "$jid"
}

submit_cleanup() {
  # usage: submit_cleanup <after-dep-expr> <jobids-to-cancel...>
  local dep="$1"; shift
  [[ $# -gt 0 ]] || return 0
  local targets=("$@")
  sbatch --job-name="pipeline_cleanup" \
         --partition="${PARTITION_CPU}" \
         --time=00:05:00 \
         --output="${LOGDIR}/%x_%j.out" \
         --error="${LOGDIR}/%x_%j.err" \
         --dependency="$dep" \
         --wrap "$(printf 'scancel %s || true\n' "${targets[@]}")" >/dev/null
}

# -------------- presence checks --------------
need_file "$ENV_JOB"
need_file "$SHARDS_JOB"
need_file "$CACHES_JOB"
need_file "$KD_JOB"

echo "[INFO] Submitting pipeline…"

# -------------- submit chain --------------
jid_env=$(submit "$ENV_JOB")
echo "[SUBMIT] env             -> $jid_env"

jid_shards=$(submit "$SHARDS_JOB" --dependency="afterok:${jid_env}")
echo "[SUBMIT] build_shards    -> $jid_shards (afterok:$jid_env)"

jid_caches=$(submit "$CACHES_JOB" --dependency="afterok:${jid_shards}")
echo "[SUBMIT] build_caches    -> $jid_caches (afterok:$jid_shards)"

jid_kd=$(submit "$KD_JOB" --dependency="afterok:${jid_caches}")
echo "[SUBMIT] kd_pipeline     -> $jid_kd (afterok:$jid_caches)"

# -------------- auto-cleanup on failure --------------
submit_cleanup "afternotok:${jid_env}"   "$jid_shards" "$jid_caches" "$jid_kd"
submit_cleanup "afternotok:${jid_shards}"              "$jid_caches" "$jid_kd"
submit_cleanup "afternotok:${jid_caches}"                           "$jid_kd"

echo "[INFO] All jobs submitted:"
echo "  env:        $jid_env"
echo "  shards:     $jid_shards"
echo "  caches:     $jid_caches"
echo "  kd:         $jid_kd"
