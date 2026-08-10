#!/bin/bash
#SBATCH -J cte1-data
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/cte1_data_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
OUT=${OUT:?set isolated output root}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
TRAIN=$BASE/artifacts/product_reasoning/nmc1/data_bd30f2a_r1/program_train.jsonl
DEV=$BASE/artifacts/product_reasoning/nmc1/data_bd30f2a_r1/development.jsonl

test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
test "$(sha256sum "$TRAIN" | awk '{print $1}')" = \
  71010bac117d9fbca2a28e7f9a63f24456b519a7b4bc172ddd772ff27e851512
test "$(sha256sum "$DEV" | awk '{print $1}')" = \
  981b83016d9a895af3016b3629c5852b868c80e364dbe97e3901a8d3e8ced4bd

export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PYTHON" "$CODE_ROOT/pipeline/build_cte1_canonical_transactions.py" \
  --train "$TRAIN" \
  --expected-train-sha256 71010bac117d9fbca2a28e7f9a63f24456b519a7b4bc172ddd772ff27e851512 \
  --development "$DEV" \
  --expected-development-sha256 981b83016d9a895af3016b3629c5852b868c80e364dbe97e3901a8d3e8ced4bd \
  --output "$OUT"
find "$OUT" -type f -exec chmod a-w {} +
