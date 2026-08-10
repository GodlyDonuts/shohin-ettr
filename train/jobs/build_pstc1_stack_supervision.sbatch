#!/bin/bash
#SBATCH -J pstc1-data
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/pstc1_data_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
OUT=${OUT:?set isolated output root}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
TRAIN=$BASE/artifacts/product_reasoning/structured_ledger_rg_v4_r1/train.jsonl
DEV=$BASE/artifacts/product_reasoning/structured_ledger_rg_v4_r1/development.jsonl

test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
test "$(sha256sum "$TRAIN" | awk '{print $1}')" = \
  02bd92260538ed75c7b1e82aa7cdb0bed4029ed06c8e982ef1b059e5605699df
test "$(sha256sum "$DEV" | awk '{print $1}')" = \
  f5d4a7150c384da046b3e86c76f9b73e1fdc15d4b56add29c3cf0fa83450bdee
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PYTHON" "$CODE_ROOT/pipeline/build_pstc1_stack_supervision.py" \
  --train "$TRAIN" \
  --development "$DEV" \
  --expected-train-sha256 02bd92260538ed75c7b1e82aa7cdb0bed4029ed06c8e982ef1b059e5605699df \
  --expected-development-sha256 f5d4a7150c384da046b3e86c76f9b73e1fdc15d4b56add29c3cf0fa83450bdee \
  --output-root "$OUT"
find "$OUT" -type f -exec chmod a-w {} +
