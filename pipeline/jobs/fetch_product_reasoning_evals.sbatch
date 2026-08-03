#!/bin/bash
#SBATCH --job-name=product-eval-fetch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/product_eval_fetch_%j.out
#SBATCH --error=logs/product_eval_fetch_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
CACHE_DIR=${CACHE_DIR:-/lustre/fs1/home/sa305415/shohin/artifacts/external/hf_cache}
PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603-r2/bin/python}

test -x "$PYTHON"
test -f "$RUNTIME/pipeline/fetch_product_reasoning_evals.py"
mkdir -p logs "$CACHE_DIR" "$(dirname "$OUTPUT_DIR")"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PYTHON" "$RUNTIME/pipeline/fetch_product_reasoning_evals.py" \
  --output-dir "$OUTPUT_DIR" \
  --cache-dir "$CACHE_DIR"
