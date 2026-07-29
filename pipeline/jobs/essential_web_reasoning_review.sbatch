#!/bin/bash
#SBATCH -J shohin-essential-review
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 1-00:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/essential_web_reasoning_review_%j.out
#SBATCH --open-mode=append
# Materialize a private exact-document packet only after the immutable
# Essential-Web candidate completes. This job does not admit or relabel data.
set -euo pipefail
export PATH=/usr/bin:/bin

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CODE_ROOT=${CODE_ROOT:?set CODE_ROOT to an isolated pinned review root}
BUILDER=$CODE_ROOT/pipeline/build_general_source_review_packet.py
PROFILE=$CODE_ROOT/pipeline/profile_general_source.py
TOKENIZE=$CODE_ROOT/pipeline/tokenize_shards.py
VERIFY_SHARDS=$CODE_ROOT/pipeline/verify_tokenized_shards.py
BUILDER_SHA256=${BUILDER_SHA256:?set exact review-builder SHA-256}
PROFILE_SHA256=${PROFILE_SHA256:?set exact source-profiler SHA-256}
TOKENIZE_SHA256=${TOKENIZE_SHA256:?set exact tokenizer-pipeline SHA-256}
VERIFY_SHARDS_SHA256=${VERIFY_SHARDS_SHA256:?set exact shard-verifier SHA-256}
SHARD_DIR=${SHARD_DIR:-$BASE/artifacts/shards/essential_web_reasoning_core_1b}
PRIVATE_OUT=${PRIVATE_OUT:-$BASE/scratchpad/private_reviews/essential_web_reasoning_core_1b_1000.jsonl}
RECEIPT_OUT=${RECEIPT_OUT:-$BASE/artifacts/source_reviews/essential_web_reasoning_core_1b_1000.receipt.json}
ROWS=${ROWS:-1000}
MAX_REVIEW_CHARS=${MAX_REVIEW_CHARS:-12000}

[[ -d "$CODE_ROOT" && ! -L "$CODE_ROOT" ]] || {
  echo "[essential-review] isolated code root is unavailable" >&2
  exit 2
}
printf '%s  %s\n' \
  "$BUILDER_SHA256" "$BUILDER" \
  "$PROFILE_SHA256" "$PROFILE" \
  "$TOKENIZE_SHA256" "$TOKENIZE" \
  "$VERIFY_SHARDS_SHA256" "$VERIFY_SHARDS" | sha256sum -c -
[[ -d "$SHARD_DIR" && ! -L "$SHARD_DIR" ]] || {
  echo "[essential-review] candidate directory is unavailable" >&2
  exit 3
}
[[ -f "$SHARD_DIR/manifest.json" && ! -L "$SHARD_DIR/manifest.json" ]] || {
  echo "[essential-review] candidate manifest is unavailable" >&2
  exit 3
}
for path in "$PRIVATE_OUT" "$RECEIPT_OUT"; do
  [[ "$path" == /* && ! -e "$path" && ! -L "$path" ]] || {
    echo "[essential-review] output path differs: $path" >&2
    exit 4
  }
done
mkdir -p "$(dirname "$PRIVATE_OUT")" "$(dirname "$RECEIPT_OUT")"
chmod 0700 "$(dirname "$PRIVATE_OUT")"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
export PYTHONPATH=$CODE_ROOT
cd "$CODE_ROOT"
"$PY" -u "$BUILDER" \
  --shard-dir "$SHARD_DIR" \
  --selection-code "$TOKENIZE" \
  --rows "$ROWS" \
  --max-review-chars "$MAX_REVIEW_CHARS" \
  --private-out "$PRIVATE_OUT" \
  --receipt-out "$RECEIPT_OUT"

[[ "$(stat -c %a "$PRIVATE_OUT")" == 600 ]] || {
  echo "[essential-review] private packet mode differs" >&2
  exit 5
}
test -s "$RECEIPT_OUT"
sha256sum "$RECEIPT_OUT"
echo "[essential-review] private packet built; corpus remains quarantined"
