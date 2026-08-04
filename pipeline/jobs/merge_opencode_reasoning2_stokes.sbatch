#!/bin/bash
#SBATCH --job-name=ocr2-merge
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ocr2_merge_%j.out
#SBATCH --error=logs/ocr2_merge_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${DATASET_REVISION:?DATASET_REVISION is required}"

PYTHON=${PYTHON:-/home/sa305415/shohin/miniforge3/bin/python}
test -x "$PYTHON"
test -f "$RUNTIME/pipeline/select_opencode_reasoning2_candidates.py"
mkdir -p logs

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
cd "$OUTPUT_ROOT"
"$PYTHON" "$RUNTIME/pipeline/select_opencode_reasoning2_candidates.py" merge \
  --inputs-glob 'shards/part-*.jsonl' \
  --expected-shards 70 \
  --output candidates.jsonl \
  --report candidates.report.json \
  --dataset-revision "$DATASET_REVISION"
