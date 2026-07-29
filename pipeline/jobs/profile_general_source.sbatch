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
CODE_ROOT=${CODE_ROOT:-$BASE}
DATASET=${DATASET:?set DATASET}
CONFIG=${CONFIG:-}
SPLIT=${SPLIT:-train}
TEXT_FIELD=${TEXT_FIELD:-text}
NESTED_FILES_FIELD=${NESTED_FILES_FIELD:-}
NESTED_TEXT_FIELD=${NESTED_TEXT_FIELD:-content}
NESTED_REVIEW_CONTEXT_FIELD=${NESTED_REVIEW_CONTEXT_FIELD:-}
PROFILE_FIELDS=${PROFILE_FIELDS:-}
PROFILE_NUMERIC_FIELDS=${PROFILE_NUMERIC_FIELDS:-}
MAX_FILES_PER_RECORD=${MAX_FILES_PER_RECORD:-8}
SCAN_ROWS=${SCAN_ROWS:-10000}
REVIEW_ROWS=${REVIEW_ROWS:-100}
MAX_REVIEW_CHARS=${MAX_REVIEW_CHARS:-16000}
SHUFFLE_SEED=${SHUFFLE_SEED:-20260728}
SHUFFLE_BUFFER=${SHUFFLE_BUFFER:-10000}
SLUG=${SLUG:?set a stable SLUG}
REVISION=${REVISION:-}
START_DELAY_SECONDS=${START_DELAY_SECONDS:-0}
LOCAL_JSONL=${LOCAL_JSONL:-}
LOCAL_SOURCE_SHA256=${LOCAL_SOURCE_SHA256:-}
LOCAL_DATASET_CARD=${LOCAL_DATASET_CARD:-}
LOCAL_DATASET_CARD_SHA256=${LOCAL_DATASET_CARD_SHA256:-}
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
[[ "$CODE_ROOT" == /* && -d "$CODE_ROOT" && ! -L "$CODE_ROOT" ]] || {
  echo "CODE_ROOT must be an absolute physical directory" >&2
  exit 2
}
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
if [ -n "$PROFILE_FIELDS" ]; then
  IFS=',' read -r -a PROFILE_FIELD_ARRAY <<< "$PROFILE_FIELDS"
  for FIELD in "${PROFILE_FIELD_ARRAY[@]}"; do
    ARGS+=(--profile-field "$FIELD")
  done
fi
if [ -n "$PROFILE_NUMERIC_FIELDS" ]; then
  IFS=',' read -r -a PROFILE_NUMERIC_FIELD_ARRAY <<< "$PROFILE_NUMERIC_FIELDS"
  for FIELD in "${PROFILE_NUMERIC_FIELD_ARRAY[@]}"; do
    ARGS+=(--profile-numeric-field "$FIELD")
  done
fi
if [ -n "$LOCAL_JSONL" ]; then
  if [ -z "$LOCAL_SOURCE_SHA256" ] || [ -z "$LOCAL_DATASET_CARD" ] || \
     [ -z "$LOCAL_DATASET_CARD_SHA256" ] || [ -z "$REVISION" ] || \
     [ -z "$CONFIG" ]; then
    echo "local profile mode requires file/card hashes, revision, and config" >&2
    exit 2
  fi
  ARGS+=(
    --local-jsonl "$LOCAL_JSONL"
    --local-source-sha256 "$LOCAL_SOURCE_SHA256"
    --local-dataset-card "$LOCAL_DATASET_CARD"
    --local-dataset-card-sha256 "$LOCAL_DATASET_CARD_SHA256"
  )
elif [ -n "$LOCAL_SOURCE_SHA256" ] || [ -n "$LOCAL_DATASET_CARD" ] || \
     [ -n "$LOCAL_DATASET_CARD_SHA256" ]; then
  echo "incomplete local profile mode" >&2
  exit 2
fi
"$PY" "$CODE_ROOT/pipeline/profile_general_source.py" "${ARGS[@]}"
