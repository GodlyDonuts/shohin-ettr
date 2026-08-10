#!/bin/bash
#SBATCH -J lam1-compose
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/lam1_compose_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
OUT=${OUT:?set isolated output report}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
BTT=$BASE/artifacts/product_reasoning/btt1/full_0d5b75e_r1/compiler.pt
LAM=$BASE/artifacts/product_reasoning/lam1/full_6b1c1f3_r1/microcode.pt
DATA=$BASE/artifacts/product_reasoning/btt1_byte_supervision_cf4f9cb_r1

test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
test "$(sha256sum "$BTT" | awk '{print $1}')" = 2283c86b1a640c9d5c02ffbc70b4646a0a1ad3538c4e1d4ab4ea3b164363e278
test "$(sha256sum "$LAM" | awk '{print $1}')" = baab62ec4a256e89950a257616d9f023b8f5e5c8c98b927052f7e722e815c7c4
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
nvidia-smi
"$PYTHON" "$CODE_ROOT/train/eval_lam1_composition.py" \
  --btt-checkpoint "$BTT" \
  --lam-checkpoint "$LAM" \
  --data "$DATA/development.jsonl" \
  --expected-train-sha256 8deabd1ad1cdd93466cf9a53bde3012886fd63ff327914897bd70cbd9e13bc6e \
  --expected-data-sha256 de16d5dc8dd3676f4c3f4a69c306ec19a0ea8a42e9750bf93a2aaf16f4be0bf8 \
  --expected-rows 3917 \
  --beam-width 64 \
  --batch-size 128 \
  --output "$OUT"
chmod a-w "$OUT"
