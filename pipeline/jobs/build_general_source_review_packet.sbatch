#!/bin/bash
#SBATCH -J shohin-source-review
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 1-00:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/source_review_%j.out
#SBATCH --open-mode=append
# Build a private exact-document review packet only after a v3 candidate
# corpus has completed. This job does not admit data or label rows.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=${PYTHON:-$BASE/miniforge3/bin/python}
SHARD_DIR=${SHARD_DIR:?set SHARD_DIR to the completed v3 candidate}
SELECTION_CODE=${SELECTION_CODE:-$BASE/pipeline/tokenize_shards.py}
PRIVATE_OUT=${PRIVATE_OUT:?set PRIVATE_OUT outside publishable artifacts}
RECEIPT_OUT=${RECEIPT_OUT:?set RECEIPT_OUT to a fresh text-free receipt}
ROWS=${ROWS:-1000}
MAX_REVIEW_CHARS=${MAX_REVIEW_CHARS:-12000}

for path in "$SHARD_DIR" "$SELECTION_CODE" "$PRIVATE_OUT" "$RECEIPT_OUT"; do
  [[ "$path" == /* ]] || {
    echo "[source-review] paths must be absolute: $path" >&2
    exit 2
  }
done
[[ -d "$SHARD_DIR" && ! -L "$SHARD_DIR" ]] || {
  echo "[source-review] candidate directory is unavailable" >&2
  exit 3
}
[[ -f "$SHARD_DIR/manifest.json" && ! -L "$SHARD_DIR/manifest.json" ]] || {
  echo "[source-review] candidate manifest is unavailable" >&2
  exit 3
}
[[ -f "$SELECTION_CODE" && ! -L "$SELECTION_CODE" ]] || {
  echo "[source-review] selection code is unavailable" >&2
  exit 3
}
[[ ! -e "$PRIVATE_OUT" && ! -L "$PRIVATE_OUT" ]] || {
  echo "[source-review] refusing existing private packet" >&2
  exit 4
}
[[ ! -e "$RECEIPT_OUT" && ! -L "$RECEIPT_OUT" ]] || {
  echo "[source-review] refusing existing receipt" >&2
  exit 4
}

mkdir -p "$(dirname "$PRIVATE_OUT")" "$(dirname "$RECEIPT_OUT")"
chmod 0700 "$(dirname "$PRIVATE_OUT")"
cd "$BASE"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8

"$PY" -u pipeline/build_general_source_review_packet.py \
  --shard-dir "$SHARD_DIR" \
  --selection-code "$SELECTION_CODE" \
  --rows "$ROWS" \
  --max-review-chars "$MAX_REVIEW_CHARS" \
  --private-out "$PRIVATE_OUT" \
  --receipt-out "$RECEIPT_OUT"

[[ "$(stat -c %a "$PRIVATE_OUT")" == 600 ]] || {
  echo "[source-review] private packet mode differs" >&2
  exit 5
}
test -s "$RECEIPT_OUT"
sha256sum "$RECEIPT_OUT"
