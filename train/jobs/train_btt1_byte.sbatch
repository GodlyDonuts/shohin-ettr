#!/bin/bash
#SBATCH -J btt1-fit
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/btt1_fit_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
OUT=${OUT:?set isolated output root}
UPDATES=${UPDATES:-1024}
BATCH_SIZE=${BATCH_SIZE:-64}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
DATA=$BASE/artifacts/product_reasoning/btt1_byte_supervision_cf4f9cb_r1/train.jsonl

test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
test "$(sha256sum "$DATA" | awk '{print $1}')" = \
  8deabd1ad1cdd93466cf9a53bde3012886fd63ff327914897bd70cbd9e13bc6e
export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/train_btt1_byte.py" \
  --data "$DATA" \
  --expected-data-sha256 8deabd1ad1cdd93466cf9a53bde3012886fd63ff327914897bd70cbd9e13bc6e \
  --output "$OUT" --updates "$UPDATES" --batch-size "$BATCH_SIZE" \
  --width 256 --encoder-layers 6 --heads 8 --learning-rate 3e-4 \
  --seed 2026081061 --data-seed 2026081062 --log-interval 10
find "$OUT" -type f -exec chmod a-w {} +
