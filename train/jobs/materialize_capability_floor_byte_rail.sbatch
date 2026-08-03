#!/bin/bash
#SBATCH --job-name=cf-byte-data
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/capability_floor_byte_data_%j.out
#SBATCH --error=logs/capability_floor_byte_data_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${RELEASE_ROOT:?RELEASE_ROOT is required}"
: "${DATA_ROOT:?DATA_ROOT is required}"
: "${RELEASE_SHA256:?RELEASE_SHA256 is required}"
: "${TOKENIZER:?TOKENIZER is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python3.13}
TRAIN_CORES=${TRAIN_CORES:-128}
DEVELOPMENT_CORES=${DEVELOPMENT_CORES:-128}
SELECTION_SEED=${SELECTION_SEED:-11}

test -x "$PYTHON"
test -d "$RUNTIME"
test -f "$RUNTIME/SHA256SUMS"
test -f "$RELEASE_ROOT/release.json"
test -d "$DATA_ROOT"
test -f "$TOKENIZER"
test ! -e "$OUTPUT_ROOT"

mkdir -p logs "$OUTPUT_ROOT"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"

"$PYTHON" "$RUNTIME/train/capability_floor_byte_rail.py" \
  --mode materialize \
  --release-root "$RELEASE_ROOT" \
  --data-root "$DATA_ROOT" \
  --release-sha256 "$RELEASE_SHA256" \
  --tokenizer "$TOKENIZER" \
  --feature-bundle "$OUTPUT_ROOT/features.pt" \
  --train-cores "$TRAIN_CORES" \
  --development-cores "$DEVELOPMENT_CORES" \
  --selection-seed "$SELECTION_SEED"

sha256sum "$OUTPUT_ROOT/features.pt" "$OUTPUT_ROOT/features.pt.receipt.json" \
  > "$OUTPUT_ROOT/SHA256SUMS"
sha256sum -c "$OUTPUT_ROOT/SHA256SUMS" >/dev/null
chmod -R a-w "$OUTPUT_ROOT"
