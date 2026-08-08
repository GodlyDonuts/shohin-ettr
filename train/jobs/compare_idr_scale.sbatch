#!/bin/bash
#SBATCH --job-name=idr-scale-compare
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/idr_scale_compare_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/idr_scale_compare_%j.err

set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

: "${RUNTIME:?RUNTIME is required}"
: "${RUNTIME_MANIFEST_SHA256:?RUNTIME_MANIFEST_SHA256 is required}"
: "${TRAINED_EVAL_ROOT:?TRAINED_EVAL_ROOT is required}"
: "${CONTROL_EVAL_ROOT:?CONTROL_EVAL_ROOT is required}"
: "${COMPARISON_OUTPUT:?COMPARISON_OUTPUT is required}"

test -x "$PYTHON"
test -f "$RUNTIME/pipeline/compare_idr_scale.py"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test ! -e "$COMPARISON_OUTPUT"

export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
"$PYTHON" "$RUNTIME/pipeline/compare_idr_scale.py" \
  --trained-development "$TRAINED_EVAL_ROOT/development/merged/report.json" \
  --control-development "$CONTROL_EVAL_ROOT/development/merged/report.json" \
  --trained-holdout "$TRAINED_EVAL_ROOT/holdout/merged/report.json" \
  --control-holdout "$CONTROL_EVAL_ROOT/holdout/merged/report.json" \
  --output "$COMPARISON_OUTPUT"

sha256sum "$COMPARISON_OUTPUT"
chmod a-w "$COMPARISON_OUTPUT"
