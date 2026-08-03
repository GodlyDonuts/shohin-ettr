#!/bin/bash
#SBATCH --job-name=gpqa-diamond-fetch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/gpqa_diamond_fetch_%j.out
#SBATCH --error=logs/gpqa_diamond_fetch_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${OUTPUT:?OUTPUT is required}"
: "${REPORT:?REPORT is required}"

PYTHON=${PYTHON:-python3}
SEED=${SEED:-20260802}
test -f "$RUNTIME/pipeline/fetch_gpqa_diamond_eval.py"
test ! -e "$OUTPUT"
test ! -e "$REPORT"
mkdir -p logs "$(dirname "$OUTPUT")" "$(dirname "$REPORT")"

"$PYTHON" "$RUNTIME/pipeline/fetch_gpqa_diamond_eval.py" \
  --output "$OUTPUT" \
  --report "$REPORT" \
  --seed "$SEED"
