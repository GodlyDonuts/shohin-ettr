#!/bin/bash
#SBATCH -J shohin-source-profile
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/general_source_profile_%j.out
#SBATCH --open-mode=append
# Profile-only CPU job. Review excerpts remain private and are never training
# data. No model, tokenizer, shard, checkpoint, or GPU is touched.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
DATASET=${DATASET:?set DATASET}
CONFIG=${CONFIG:-}
SPLIT=${SPLIT:-train}
TEXT_FIELD=${TEXT_FIELD:-text}
NESTED_FILES_FIELD=${NESTED_FILES_FIELD:-}
NESTED_TEXT_FIELD=${NESTED_TEXT_FIELD:-content}
NESTED_REVIEW_CONTEXT_FIELD=${NESTED_REVIEW_CONTEXT_FIELD:-}
MAX_FILES_PER_RECORD=${MAX_FILES_PER_RECORD:-8}
SCAN_ROWS=${SCAN_ROWS:-10000}
REVIEW_ROWS=${REVIEW_ROWS:-100}
MAX_REVIEW_CHARS=${MAX_REVIEW_CHARS:-16000}
SHUFFLE_SEED=${SHUFFLE_SEED:-20260728}
SHUFFLE_BUFFER=${SHUFFLE_BUFFER:-10000}
SLUG=${SLUG:?set a stable SLUG}
REVISION=${REVISION:-}
START_DELAY_SECONDS=${START_DELAY_SECONDS:-0}
OUT=${OUT:-$BASE/artifacts/source_profiles/${SLUG}.profile.json}
REVIEW_OUT=${REVIEW_OUT:-$BASE/private/source_reviews/${SLUG}.review.jsonl}

case "$START_DELAY_SECONDS" in
  ''|*[!0-9]*) echo "START_DELAY_SECONDS must be a nonnegative integer" >&2; exit 2 ;;
esac
if [ "$START_DELAY_SECONDS" -gt 3600 ]; then
  echo "START_DELAY_SECONDS may not exceed one hour" >&2
  exit 2
fi

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

cd "$BASE"
if [ "$START_DELAY_SECONDS" -gt 0 ]; then
  echo "[profile] delaying ${START_DELAY_SECONDS}s to avoid upstream request bursts"
  sleep "$START_DELAY_SECONDS"
fi
ARGS=(
  --dataset "$DATASET"
  --split "$SPLIT"
  --text-field "$TEXT_FIELD"
  --scan-rows "$SCAN_ROWS"
  --review-rows "$REVIEW_ROWS"
  --max-review-chars "$MAX_REVIEW_CHARS"
  --shuffle-seed "$SHUFFLE_SEED"
  --shuffle-buffer "$SHUFFLE_BUFFER"
  --evals-dir "$BASE/artifacts/evals"
  --out "$OUT"
  --review-out "$REVIEW_OUT"
)
if [ -n "$CONFIG" ]; then
  ARGS+=(--config "$CONFIG")
fi
if [ -n "$REVISION" ]; then
  ARGS+=(--revision "$REVISION")
fi
if [ -n "$NESTED_FILES_FIELD" ]; then
  ARGS+=(
    --nested-files-field "$NESTED_FILES_FIELD"
    --nested-text-field "$NESTED_TEXT_FIELD"
    --max-files-per-record "$MAX_FILES_PER_RECORD"
  )
  if [ -n "$NESTED_REVIEW_CONTEXT_FIELD" ]; then
    ARGS+=(--nested-review-context-field "$NESTED_REVIEW_CONTEXT_FIELD")
  fi
fi
"$PY" pipeline/profile_general_source.py "${ARGS[@]}"
