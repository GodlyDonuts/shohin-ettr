#!/bin/bash
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${REPOSITORY_ROOT:?}" "${PYTHON:?}" "${BENCHMARK:?}" "${GENERATION_ROOT:?}"
: "${ASSESSORS:?}" "${IFEVAL_ROOT:?}" "${OUTPUT:?}" "${SHARD_COUNT:?}"
arguments=()
index=0
while [ "$index" -lt "$SHARD_COUNT" ]; do
  arguments+=(--generation-report "$GENERATION_ROOT/shard_${index}/report.json")
  index=$((index + 1))
done
export PYTHONDONTWRITEBYTECODE=1
PYTHONPATH="$REPOSITORY_ROOT/pipeline:$REPOSITORY_ROOT/train" "$PYTHON" \
  "$REPOSITORY_ROOT/pipeline/score_dense_public_benchmark.py" \
  --benchmark "$BENCHMARK" \
  "${arguments[@]}" \
  --assessors "$ASSESSORS" \
  --ifeval-root "$IFEVAL_ROOT" \
  --output "$OUTPUT"
chmod 444 "$OUTPUT"
