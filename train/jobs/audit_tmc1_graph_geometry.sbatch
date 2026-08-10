#!/bin/bash
#SBATCH -J tmc1-audit
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/tmc1_audit_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
OUT=${OUT:?set isolated output report}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
DATA=$BASE/artifacts/product_reasoning/nmc1/data_bd30f2a_r1
LAM=$BASE/artifacts/product_reasoning/lam1/full_6b1c1f3_r1/microcode.pt

test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
test "$(sha256sum "$DATA/program_train.jsonl" | awk '{print $1}')" = \
  71010bac117d9fbca2a28e7f9a63f24456b519a7b4bc172ddd772ff27e851512
test "$(sha256sum "$DATA/development.jsonl" | awk '{print $1}')" = \
  981b83016d9a895af3016b3629c5852b868c80e364dbe97e3901a8d3e8ced4bd
test "$(sha256sum "$LAM" | awk '{print $1}')" = \
  baab62ec4a256e89950a257616d9f023b8f5e5c8c98b927052f7e722e815c7c4

export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1
"$PYTHON" "$CODE_ROOT/pipeline/audit_tmc1_graph_geometry.py" \
  --train "$DATA/program_train.jsonl" \
  --expected-train-sha256 71010bac117d9fbca2a28e7f9a63f24456b519a7b4bc172ddd772ff27e851512 \
  --development "$DATA/development.jsonl" \
  --expected-development-sha256 981b83016d9a895af3016b3629c5852b868c80e364dbe97e3901a8d3e8ced4bd \
  --lam-checkpoint "$LAM" \
  --output "$OUT"
chmod a-w "$OUT"
