#!/bin/bash
#SBATCH --job-name=gpt-oss-120b-fetch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/gpt_oss_120b_fetch_%j.out
#SBATCH --error=logs/gpt_oss_120b_fetch_%j.err

set -euo pipefail

for variable in PYTHON RUNTIME RUNTIME_MANIFEST_SHA256 OUTPUT_ROOT REPORT; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
[[ -x "$PYTHON" ]]
[[ -d "$RUNTIME" && ! -L "$RUNTIME" ]]
[[ "$OUTPUT_ROOT" == /lustre/fs1/home/sa305415/shohin/artifacts/external/* ]]
[[ "$REPORT" == /lustre/fs1/home/sa305415/shohin/artifacts/external/* ]]
[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]]
[[ ! -e "$REPORT" && ! -L "$REPORT" ]]

read -r filesystem used_kib quota_kib hard_kib _ < <(lfs quota -u sa305415 /lustre/fs1 | tail -1)
[[ "$filesystem" == /lustre/fs1 ]]
[[ "$used_kib" =~ ^[0-9]+$ && "$hard_kib" =~ ^[0-9]+$ ]]
headroom_kib=$((hard_kib - used_kib))
# The 65.28-GB projection must leave at least 96 GiB of durable headroom.
minimum_headroom_kib=164500000
if (( headroom_kib < minimum_headroom_kib )); then
  printf 'GPT-OSS acquisition requires %s KiB headroom; observed %s KiB\n' \
    "$minimum_headroom_kib" "$headroom_kib" >&2
  exit 2
fi

source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_runtime
mkdir -p logs "$(dirname "$OUTPUT_ROOT")"

export PATH=/apps/slurm/current/bin:/usr/bin:/bin
export PYTHONPATH="$RUNTIME/pipeline"
export HF_HOME="$SLURM_TMPDIR/hf"
export XDG_CACHE_HOME="$SLURM_TMPDIR/cache"
export TMPDIR="$SLURM_TMPDIR"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

"$PYTHON" -P -s -B "$RUNTIME/pipeline/fetch_gpt_oss_120b.py" \
  --output-root "$OUTPUT_ROOT" \
  --report "$REPORT" \
  --workers 8

[[ -d "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]]
[[ -f "$OUTPUT_ROOT/SHA256SUMS" && ! -L "$OUTPUT_ROOT/SHA256SUMS" ]]
[[ -f "$REPORT" && ! -L "$REPORT" ]]
(cd "$OUTPUT_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
chmod -R a-w "$OUTPUT_ROOT" "$REPORT"
