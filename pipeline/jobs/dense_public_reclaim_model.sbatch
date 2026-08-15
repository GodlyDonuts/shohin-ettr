#!/bin/bash
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --no-requeue

set -euo pipefail
: "${REPOSITORY_ROOT:?}" "${PYTHON:?}" "${MODEL_ROOT:?}" "${MODEL_RECEIPT:?}" "${OUTPUT:?}"
PYTHONPATH="$REPOSITORY_ROOT/pipeline:$REPOSITORY_ROOT/train" "$PYTHON" \
  "$REPOSITORY_ROOT/pipeline/reclaim_dense_public_model.py" \
  --model-root "$MODEL_ROOT" \
  --model-receipt "$MODEL_RECEIPT" \
  --output "$OUTPUT"
chmod 444 "$OUTPUT"
