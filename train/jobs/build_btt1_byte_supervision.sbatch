#!/bin/bash
#SBATCH -J btt1-data
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/btt1_data_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
OUT=${OUT:?set isolated output root}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
SOURCE=$BASE/artifacts/product_reasoning/mltc1_lexical_supervision_2dabf32_r1

test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
test "$(sha256sum "$SOURCE/train.jsonl" | awk '{print $1}')" = \
  33b8eb7bf154f3e938de00f06aebd7adb720eb4a1b6bd3f81554f9eb0b11bf3f
test "$(sha256sum "$SOURCE/development.jsonl" | awk '{print $1}')" = \
  8e867e7cdcc47015096979314450790cd86c804fd841b34d3df411a236b33679
export PYTHONPATH="$CODE_ROOT/pipeline"
"$PYTHON" "$CODE_ROOT/pipeline/build_btt1_byte_supervision.py" \
  --train "$SOURCE/train.jsonl" \
  --development "$SOURCE/development.jsonl" \
  --expected-train-sha256 33b8eb7bf154f3e938de00f06aebd7adb720eb4a1b6bd3f81554f9eb0b11bf3f \
  --expected-development-sha256 8e867e7cdcc47015096979314450790cd86c804fd841b34d3df411a236b33679 \
  --output-root "$OUT"
find "$OUT" -type f -exec chmod a-w {} +
