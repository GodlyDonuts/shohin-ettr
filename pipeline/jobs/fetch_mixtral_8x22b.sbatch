#!/bin/bash
#SBATCH --job-name=mixtral-8x22b-fetch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/mixtral_8x22b_fetch_%j.out
#SBATCH --error=logs/mixtral_8x22b_fetch_%j.err

set -euo pipefail

: "${PYTHON:?PYTHON is required}"
: "${RUNTIME:?RUNTIME is required}"
: "${OUTPUT:?OUTPUT is required}"
: "${REPORT:?REPORT is required}"

[[ -x "$PYTHON" ]]
[[ -d "$RUNTIME" && ! -L "$RUNTIME" ]]
[[ "$OUTPUT" == /lustre/fs1/home/sa305415/shohin/artifacts/external/* ]]
[[ "$REPORT" == /lustre/fs1/home/sa305415/shohin/artifacts/external/* ]]
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]]
[[ ! -e "$REPORT" && ! -L "$REPORT" ]]
[[ "$(dirname "$OUTPUT")" == "$(dirname "$REPORT")" ]]

read -r filesystem used_kib quota_kib hard_kib _ < <(lfs quota -u sa305415 /lustre/fs1 | tail -1)
[[ "$filesystem" == /lustre/fs1 ]]
[[ "$used_kib" =~ ^[0-9]+$ && "$hard_kib" =~ ^[0-9]+$ ]]
headroom_kib=$((hard_kib - used_kib))
minimum_headroom_kib=300000000
if (( headroom_kib < minimum_headroom_kib )); then
  printf 'Mixtral acquisition requires %s KiB headroom; observed %s KiB\n' \
    "$minimum_headroom_kib" "$headroom_kib" >&2
  exit 2
fi

source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
[[ -d "$SLURM_TMPDIR" && ! -L "$SLURM_TMPDIR" ]]
mkdir -p logs

export PATH=/apps/slurm/current/bin:/usr/bin:/bin
export PYTHONPATH="$RUNTIME/pipeline"
export HF_HOME="$SLURM_TMPDIR/hf"
export XDG_CACHE_HOME="$SLURM_TMPDIR/cache"
export TMPDIR="$SLURM_TMPDIR"
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON" -P -s -B "$RUNTIME/pipeline/fetch_mixtral_8x22b.py" \
  --output "$OUTPUT" \
  --report "$REPORT" \
  --workers 8

[[ -d "$OUTPUT" && ! -L "$OUTPUT" ]]
[[ -f "$OUTPUT/SHA256SUMS" && ! -L "$OUTPUT/SHA256SUMS" ]]
[[ -f "$REPORT" && ! -L "$REPORT" ]]
(cd "$OUTPUT" && sha256sum -c SHA256SUMS >/dev/null)
chmod -R a-w "$OUTPUT" "$REPORT"
