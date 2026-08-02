#!/bin/bash
#SBATCH --job-name=cf-corpus
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/capability_floor_corpus_%j.out
#SBATCH --error=logs/capability_floor_corpus_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${RELEASE_ROOT:?RELEASE_ROOT is required}"
: "${DATA_ROOT:?DATA_ROOT is required}"
: "${RELEASE_SHA256:?RELEASE_SHA256 is required}"
: "${SHOHIN_TOKENIZER:?SHOHIN_TOKENIZER is required}"
: "${TOKENIZER_CONFIG:?TOKENIZER_CONFIG is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python3.13}
MAXIMUM_CORES=${MAXIMUM_CORES:-}

test -x "$PYTHON"
test -d "$RUNTIME"
test -f "$RUNTIME/SHA256SUMS"
test -f "$RELEASE_ROOT/release.json"
test -f "$SHOHIN_TOKENIZER"
test -f "$TOKENIZER_CONFIG"
test ! -e "$OUTPUT_ROOT"

mkdir -p logs "$OUTPUT_ROOT"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"

arguments=(
  "$RUNTIME/train/capability_floor_corpus.py"
  --release-root "$RELEASE_ROOT"
  --data-root "$DATA_ROOT"
  --release-sha256 "$RELEASE_SHA256"
  --shohin-tokenizer "$SHOHIN_TOKENIZER"
  --tokenizer-config "$TOKENIZER_CONFIG"
  --index-output "$OUTPUT_ROOT/cohort-index.jsonl"
  --receipt-output "$OUTPUT_ROOT/receipt.json"
)
if [ -n "$MAXIMUM_CORES" ]; then
  arguments+=(--maximum-cores-per-split "$MAXIMUM_CORES")
fi

"$PYTHON" "${arguments[@]}"
sha256sum "$OUTPUT_ROOT/cohort-index.jsonl" "$OUTPUT_ROOT/receipt.json" \
  > "$OUTPUT_ROOT/SHA256SUMS"
sha256sum -c "$OUTPUT_ROOT/SHA256SUMS" >/dev/null
chmod -R a-w "$OUTPUT_ROOT"
