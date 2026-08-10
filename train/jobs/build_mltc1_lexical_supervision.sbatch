#!/bin/bash
#SBATCH -J mltc1-data
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/mltc1_data_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
OUT=${OUT:?set isolated output root}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
SOURCE=$BASE/artifacts/product_reasoning/pstc1_stack_supervision_ca5abd6_r1

test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
test "$(sha256sum "$SOURCE/train.jsonl" | awk '{print $1}')" = \
  b522d58b3403064987a47d7b22657ef862945f18dfb352a78625fbcc27ebf008
test "$(sha256sum "$SOURCE/development.jsonl" | awk '{print $1}')" = \
  bf90044cd17e5e63cdeb7f91bffaae6e4bc59e992fb95cb8da9d2b29da00c914
export PYTHONPATH="$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PYTHON" "$CODE_ROOT/pipeline/build_mltc1_lexical_supervision.py" \
  --train "$SOURCE/train.jsonl" \
  --development "$SOURCE/development.jsonl" \
  --expected-train-sha256 b522d58b3403064987a47d7b22657ef862945f18dfb352a78625fbcc27ebf008 \
  --expected-development-sha256 bf90044cd17e5e63cdeb7f91bffaae6e4bc59e992fb95cb8da9d2b29da00c914 \
  --output-root "$OUT"
find "$OUT" -type f -exec chmod a-w {} +
