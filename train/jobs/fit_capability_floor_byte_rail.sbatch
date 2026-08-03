#!/bin/bash
#SBATCH --job-name=cf-byte-fit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/capability_floor_byte_fit_%j.out
#SBATCH --error=logs/capability_floor_byte_fit_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${BYTE_ROOT:?BYTE_ROOT is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python3.13}
ARCHITECTURE_SEED=${ARCHITECTURE_SEED:-31}
PROBE_BATCH=${PROBE_BATCH:-16}
UPDATES=${UPDATES:-2000}
LEARNING_RATE=${LEARNING_RATE:-0.0003}

test -x "$PYTHON"
test -d "$RUNTIME"
test -f "$RUNTIME/SHA256SUMS"
test -f "$BYTE_ROOT/features.pt"
test -f "$BYTE_ROOT/SHA256SUMS"
test ! -e "$OUTPUT_ROOT"

(cd "$BYTE_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
FEATURE_SHA256=$(sha256sum "$BYTE_ROOT/features.pt" | awk '{print $1}')

mkdir -p logs "$OUTPUT_ROOT"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"

"$PYTHON" "$RUNTIME/train/capability_floor_byte_rail.py" \
  --mode fit \
  --feature-bundle "$BYTE_ROOT/features.pt" \
  --feature-bundle-sha256 "$FEATURE_SHA256" \
  --output "$OUTPUT_ROOT/report.json" \
  --architecture-seed "$ARCHITECTURE_SEED" \
  --probe-batch "$PROBE_BATCH" \
  --updates "$UPDATES" \
  --learning-rate "$LEARNING_RATE"

sha256sum "$OUTPUT_ROOT/report.json" "$OUTPUT_ROOT/report.pt" \
  > "$OUTPUT_ROOT/SHA256SUMS"
sha256sum -c "$OUTPUT_ROOT/SHA256SUMS" >/dev/null
chmod -R a-w "$OUTPUT_ROOT"
