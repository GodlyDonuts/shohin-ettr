#!/bin/bash
#SBATCH -J dtc1-eval
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/dtc1_eval_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
OUT=${OUT:?set isolated output report}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
DATA=$BASE/artifacts/product_reasoning/nmc1/data_bd30f2a_r1/development.jsonl
DRAFT_REPORT=$BASE/artifacts/product_reasoning/nmc1/development_f24a237_r2/direct_normal.json
LAM=$BASE/artifacts/product_reasoning/lam1/full_6b1c1f3_r1/microcode.pt

test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
test "$(sha256sum "$DATA" | awk '{print $1}')" = \
  981b83016d9a895af3016b3629c5852b868c80e364dbe97e3901a8d3e8ced4bd
test "$(sha256sum "$DRAFT_REPORT" | awk '{print $1}')" = \
  234a029a7479fa056a3bdacd286dcc6e5a1d4cf7b472028b77fec7859ada2a49
test "$(sha256sum "$LAM" | awk '{print $1}')" = \
  baab62ec4a256e89950a257616d9f023b8f5e5c8c98b927052f7e722e815c7c4

export PYTHONPATH="$CODE_ROOT/train:$BASE/envs/idr08-eval-r1/lib/python3.11/site-packages"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PYTHON" "$CODE_ROOT/train/eval_dtc1_development.py" \
  --data "$DATA" \
  --expected-data-sha256 981b83016d9a895af3016b3629c5852b868c80e364dbe97e3901a8d3e8ced4bd \
  --draft-report "$DRAFT_REPORT" \
  --expected-draft-report-sha256 234a029a7479fa056a3bdacd286dcc6e5a1d4cf7b472028b77fec7859ada2a49 \
  --lam-checkpoint "$LAM" \
  --output "$OUT"
chmod a-w "$OUT"
