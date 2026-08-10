#!/bin/bash
#SBATCH -J btt1-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/btt1_eval_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
CHECKPOINT=${CHECKPOINT:?set BTT1 checkpoint}
OUT=${OUT:?set isolated output report}
CONTROL=${CONTROL:?set normal, source_shuffled, zero_bytes, or flat_executor}
PROJECTION=${PROJECTION:-none}
EXPECTED_ROWS=${EXPECTED_ROWS:-3917}
HOLDOUT=${HOLDOUT:-0}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
TRAIN=$BASE/artifacts/product_reasoning/btt1_byte_supervision_cf4f9cb_r1/train.jsonl
DATA=$BASE/artifacts/product_reasoning/btt1_byte_supervision_cf4f9cb_r1/development.jsonl

test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
test "$(sha256sum "$TRAIN" | awk '{print $1}')" = \
  8deabd1ad1cdd93466cf9a53bde3012886fd63ff327914897bd70cbd9e13bc6e
test "$(sha256sum "$DATA" | awk '{print $1}')" = \
  de16d5dc8dd3676f4c3f4a69c306ec19a0ea8a42e9750bf93a2aaf16f4be0bf8
export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
nvidia-smi -L
HOLDOUT_ARG=()
if [ "$HOLDOUT" = 1 ]; then HOLDOUT_ARG=(--holdout); fi
"$PYTHON" "$CODE_ROOT/train/eval_btt1_byte.py" \
  --checkpoint "$CHECKPOINT" \
  --expected-train-sha256 8deabd1ad1cdd93466cf9a53bde3012886fd63ff327914897bd70cbd9e13bc6e \
  --data "$DATA" \
  --expected-data-sha256 de16d5dc8dd3676f4c3f4a69c306ec19a0ea8a42e9750bf93a2aaf16f4be0bf8 \
  --output "$OUT" --control "$CONTROL" --batch-size 128 --projection "$PROJECTION" \
  --expected-rows "$EXPECTED_ROWS" "${HOLDOUT_ARG[@]}"
chmod a-w "$OUT"
