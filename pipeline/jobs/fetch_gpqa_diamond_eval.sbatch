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

PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603-r2/bin/python}
SEED=${SEED:-20260802}
test -x "$PYTHON"
test -f "$RUNTIME/pipeline/fetch_gpqa_diamond_eval.py"
test ! -e "$OUTPUT"
test ! -e "$REPORT"
mkdir -p logs "$(dirname "$OUTPUT")" "$(dirname "$REPORT")"

"$PYTHON" "$RUNTIME/pipeline/fetch_gpqa_diamond_eval.py" \
  --output "$OUTPUT" \
  --report "$REPORT" \
  --seed "$SEED"
