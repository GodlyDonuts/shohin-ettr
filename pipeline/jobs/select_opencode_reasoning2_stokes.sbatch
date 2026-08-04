#!/bin/bash
#SBATCH --job-name=ocr2-select
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --array=0-69%8
#SBATCH --output=logs/ocr2_select_%A_%a.out
#SBATCH --error=logs/ocr2_select_%A_%a.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${DATASET_REVISION:?DATASET_REVISION is required}"

PYTHON=${PYTHON:-/home/sa305415/shohin/miniforge3/bin/python}
SHARD_COUNT=${SHARD_COUNT:-70}
MAX_RESPONSE_CHARS=${MAX_RESPONSE_CHARS:-24000}
MAX_CODE_CHARS=${MAX_CODE_CHARS:-20000}
INDEX=${SLURM_ARRAY_TASK_ID:?array task id is required}

test -x "$PYTHON"
test -f "$RUNTIME/pipeline/select_opencode_reasoning2_candidates.py"
test "$SHARD_COUNT" -eq 70

mkdir -p logs "$OUTPUT_ROOT/shards"
printf -v PADDED '%05d' "$INDEX"
NAME="train-${PADDED}-of-00070.parquet"
URL="https://huggingface.co/datasets/nvidia/OpenCodeReasoning-2/resolve/${DATASET_REVISION}/train/python/${NAME}"
LOCAL_ROOT=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
mkdir -p "$LOCAL_ROOT"
PARQUET="$LOCAL_ROOT/$NAME"
OUT="$OUTPUT_ROOT/shards/part-${PADDED}.jsonl"
REPORT="$OUTPUT_ROOT/shards/part-${PADDED}.report.json"
test ! -e "$OUT"
test ! -e "$REPORT"

curl --fail --location --retry 20 --retry-delay 15 \
  --connect-timeout 30 --output "$PARQUET" "$URL"
test -s "$PARQUET"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
"$PYTHON" "$RUNTIME/pipeline/select_opencode_reasoning2_candidates.py" scan \
  --input "$PARQUET" \
  --output "$OUT" \
  --report "$REPORT" \
  --dataset-revision "$DATASET_REVISION" \
  --shard-index "$INDEX" \
  --max-response-chars "$MAX_RESPONSE_CHARS" \
  --max-code-chars "$MAX_CODE_CHARS"

rm -f "$PARQUET"
