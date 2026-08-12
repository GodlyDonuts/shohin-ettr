#!/usr/bin/env bash
#SBATCH --job-name=pcf16-sandbox-qualification
#SBATCH --partition=normal
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --no-requeue

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${PYTHON:?PYTHON is required}"
: "${OUTPUT:?OUTPUT is required}"

scratch="/tmp/pcf16-sandbox-qualification-${SLURM_JOB_ID}"
case "$scratch" in
  /tmp/pcf16-sandbox-qualification-[0-9]*) ;;
  *) printf 'pcf16: unsafe scratch path\n' >&2; exit 2 ;;
esac
[[ ! -e "$scratch" && ! -L "$scratch" ]]
mkdir -m 700 "$scratch"
cleanup() {
  chmod -R u+w "$scratch" 2>/dev/null || true
  rm -rf --one-file-system -- "$scratch"
}
trap cleanup EXIT

export SLURM_TMPDIR="$scratch"
export TMPDIR="$scratch"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"

[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]]
"$PYTHON" -P -s -S -B "$RUNTIME/train/pcf1_code_sandbox.py" qualify \
  --output "$OUTPUT"
chmod a-w "$OUTPUT"

