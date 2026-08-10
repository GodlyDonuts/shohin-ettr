#!/bin/bash
#SBATCH -J wgp1-holdout-data
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/wgp1_holdout_data_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
OUT=${OUT:?set isolated output root}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
SOURCE=${SOURCE:-$BASE/artifacts/rg_v4/rg_eval.jsonl}
EXPECTED_SOURCE_SHA256=${EXPECTED_SOURCE_SHA256:-35a2625a44bd42161ac9ee562fb57a6f69ab9771747ef4166d0042d925b46435}
EXPECTED_SPLIT=${EXPECTED_SPLIT:-eval_seed}
PROTECTED=$BASE/artifacts/product_reasoning/btt1_byte_supervision_cf4f9cb_r1

test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
test "$(sha256sum "$SOURCE" | awk '{print $1}')" = "$EXPECTED_SOURCE_SHA256"
export PYTHONPATH="$CODE_ROOT/pipeline"
"$PYTHON" "$CODE_ROOT/pipeline/build_wgp1_holdout.py" \
  --source "$SOURCE" \
  --expected-source-sha256 "$EXPECTED_SOURCE_SHA256" \
  --expected-split "$EXPECTED_SPLIT" \
  --protected "$PROTECTED/train.jsonl" \
  --protected "$PROTECTED/development.jsonl" \
  --output-root "$OUT"
find "$OUT" -type f -exec chmod a-w {} +
