#!/bin/bash
#SBATCH -J lam1-microcode
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/lam1_microcode_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
OUT=${OUT:?set isolated output root}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
DATA=$BASE/artifacts/product_reasoning/btt1_byte_supervision_cf4f9cb_r1

test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$CODE_ROOT/train"
"$PYTHON" "$CODE_ROOT/train/train_lam1_microcode.py" \
  --train "$DATA/train.jsonl" \
  --development "$DATA/development.jsonl" \
  --expected-train-sha256 8deabd1ad1cdd93466cf9a53bde3012886fd63ff327914897bd70cbd9e13bc6e \
  --expected-development-sha256 de16d5dc8dd3676f4c3f4a69c306ec19a0ea8a42e9750bf93a2aaf16f4be0bf8 \
  --steps 32 \
  --learning-rate 1.0 \
  --seed 2026081041 \
  --output "$OUT"
find "$OUT" -type f -exec chmod a-w {} +
