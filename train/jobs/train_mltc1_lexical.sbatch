#!/bin/bash
#SBATCH -J mltc1-fit
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/mltc1_fit_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
OUT=${OUT:?set isolated output root}
UPDATES=${UPDATES:-1024}
BATCH_SIZE=${BATCH_SIZE:-32}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL=$BASE/artifacts/external/qwen3.5-0.8b-2fc0636
DATA=$BASE/artifacts/product_reasoning/mltc1_lexical_supervision_2dabf32_r1/train.jsonl

test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
test "$(sha256sum "$DATA" | awk '{print $1}')" = \
  33b8eb7bf154f3e938de00f06aebd7adb720eb4a1b6bd3f81554f9eb0b11bf3f
test "$(sha256sum "$MODEL/tokenizer.json" | awk '{print $1}')" = \
  5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42
export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline:$BASE/envs/idr08-eval-r1/lib/python3.11/site-packages"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/train_mltc1_lexical.py" \
  --model-root "$MODEL" \
  --model-revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --data "$DATA" \
  --expected-data-sha256 33b8eb7bf154f3e938de00f06aebd7adb720eb4a1b6bd3f81554f9eb0b11bf3f \
  --output "$OUT" \
  --updates "$UPDATES" \
  --batch-size "$BATCH_SIZE" \
  --max-source-tokens 256 \
  --width 384 \
  --encoder-layers 4 \
  --heads 8 \
  --learning-rate 2e-4 \
  --seed 2026081051 \
  --data-seed 2026081052 \
  --log-interval 10
find "$OUT" -type f -exec chmod a-w {} +
