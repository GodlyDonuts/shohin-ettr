#!/bin/bash
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --no-requeue

set -euo pipefail
: "${REPOSITORY_ROOT:?}" "${PYTHON:?}" "${SCORE_ROOT:?}" "${OUTPUT:?}"
export PYTHONDONTWRITEBYTECODE=1
PYTHONPATH="$REPOSITORY_ROOT/pipeline:$REPOSITORY_ROOT/train" "$PYTHON" \
  "$REPOSITORY_ROOT/pipeline/analyze_dense_public_screen.py" \
  --score "$SCORE_ROOT/mmlu_pro.json" \
  --score "$SCORE_ROOT/ifeval.json" \
  --score "$SCORE_ROOT/musr.json" \
  --output "$OUTPUT"
chmod 444 "$OUTPUT"
