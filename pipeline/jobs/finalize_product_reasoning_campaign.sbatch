#!/bin/bash
#SBATCH --job-name=product-decision
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/product_decision_%j.out
#SBATCH --error=logs/product_decision_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${ROOT:?ROOT is required}"
: "${MATH_VERIFY:?MATH_VERIFY is required}"
: "${OUTPUT:?OUTPUT is required}"

PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603-r2/bin/python}
test -x "$PYTHON"
test -f "$RUNTIME/pipeline/rescore_product_reasoning_report.py"
test -f "$RUNTIME/pipeline/aggregate_product_reasoning_campaign.py"
test ! -e "$OUTPUT"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$MATH_VERIFY:$RUNTIME/train:$RUNTIME/pipeline"

for arm in baseline ettr_residual dense_residual; do
  prefix="$ROOT/${arm}_short_u200_ce6e532"
  for task in gsm8k math500 humaneval mbpp gpqa bbh_logic; do
    input="${prefix}_${task}_dev_v4.json"
    output="${prefix}_${task}_dev_v5.json"
    test -s "$input"
    test ! -e "$output"
    "$PYTHON" "$RUNTIME/pipeline/rescore_product_reasoning_report.py" \
      --input "$input" --output "$output"
  done
done

"$PYTHON" "$RUNTIME/pipeline/aggregate_product_reasoning_campaign.py" \
  --baseline "B1=$ROOT/baseline_short_u200_ce6e532" \
  --treatment "T2=$ROOT/ettr_residual_short_u200_ce6e532" \
  --control "C2=$ROOT/dense_residual_short_u200_ce6e532" \
  --suffix _dev_v5.json \
  --output "$OUTPUT"
