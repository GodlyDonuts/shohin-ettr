#!/bin/bash
#SBATCH --job-name=cf-replay
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/capability_floor_replay_%j.out
#SBATCH --error=logs/capability_floor_replay_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${COHORT_ROOT:?COHORT_ROOT is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python3.13}

test -x "$PYTHON"
test -d "$RUNTIME"
test -f "$RUNTIME/SHA256SUMS"
test -d "$COHORT_ROOT"
test -f "$COHORT_ROOT/cohort-index.jsonl"
test -f "$COHORT_ROOT/SHA256SUMS"
test ! -e "$OUTPUT_ROOT"

sha256sum -c "$COHORT_ROOT/SHA256SUMS" >/dev/null
INDEX_SHA256=$(sha256sum "$COHORT_ROOT/cohort-index.jsonl" | awk '{print $1}')

mkdir -p logs "$OUTPUT_ROOT"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"

"$PYTHON" "$RUNTIME/train/materialize_capability_floor_replay.py" \
  --index "$COHORT_ROOT/cohort-index.jsonl" \
  --index-sha256 "$INDEX_SHA256" \
  --candidate protected-shohin-125m-step300k \
  --candidate qwen3.5-0.8b-text-backbone \
  --candidate smollm3-3b \
  --output "$OUTPUT_ROOT/replay-publication.json"

sha256sum "$OUTPUT_ROOT/replay-publication.json" > "$OUTPUT_ROOT/SHA256SUMS"
sha256sum -c "$OUTPUT_ROOT/SHA256SUMS" >/dev/null
chmod -R a-w "$OUTPUT_ROOT"
