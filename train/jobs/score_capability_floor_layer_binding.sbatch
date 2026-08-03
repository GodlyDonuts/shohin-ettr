#!/bin/bash
#SBATCH --job-name=cf-tap-bind
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/capability_floor_tap_binding_%j.out
#SBATCH --error=logs/capability_floor_tap_binding_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${TAP_ROOT:?TAP_ROOT is required}"
: "${TAP:?TAP is required}"
: "${PROBE_ROOT:?PROBE_ROOT is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python3.13}
PROBE_BATCH=${PROBE_BATCH:-16}

test -x "$PYTHON"
test -d "$RUNTIME"
test -f "$RUNTIME/SHA256SUMS"
test -f "$TAP_ROOT/features.pt"
test -f "$TAP_ROOT/SHA256SUMS"
test -f "$PROBE_ROOT/report.pt"
test -f "$PROBE_ROOT/SHA256SUMS"
test ! -e "$OUTPUT_ROOT"

case "$TAP" in
  embedding|block-00|block-04|block-09|block-14|block-19|block-24|block-29) ;;
  *) echo "unsupported frozen tap: $TAP" >&2; exit 2 ;;
esac

(cd "$TAP_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
(cd "$PROBE_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
FEATURE_SHA256=$(sha256sum "$TAP_ROOT/features.pt" | awk '{print $1}')
PROBE_SHA256=$(sha256sum "$PROBE_ROOT/report.pt" | awk '{print $1}')

mkdir -p logs "$OUTPUT_ROOT"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"

"$PYTHON" "$RUNTIME/train/capability_floor_layer_taps.py" \
  --mode binding \
  --feature-bundle "$TAP_ROOT/features.pt" \
  --feature-bundle-sha256 "$FEATURE_SHA256" \
  --tap "$TAP" \
  --probe-model "$PROBE_ROOT/report.pt" \
  --probe-model-sha256 "$PROBE_SHA256" \
  --probe-batch "$PROBE_BATCH" \
  --output "$OUTPUT_ROOT/report.json"

sha256sum "$OUTPUT_ROOT/report.json" > "$OUTPUT_ROOT/SHA256SUMS"
sha256sum -c "$OUTPUT_ROOT/SHA256SUMS" >/dev/null
chmod -R a-w "$OUTPUT_ROOT"
