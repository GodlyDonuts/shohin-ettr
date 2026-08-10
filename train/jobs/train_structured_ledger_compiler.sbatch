#!/bin/bash
#SBATCH -J ledger-fit
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/structured_ledger_fit_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
DATA=${DATA:?set immutable ledger SFT JSONL}
OUT=${OUT:?set isolated output root}
UPDATES=${UPDATES:-1024}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL=$BASE/artifacts/external/qwen3.5-0.8b-2fc0636

test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
test "$(sha256sum "$DATA" | awk '{print $1}')" = \
  6a5876f2b8eed1387c31459062102b9bd007bff99d556ee6f63c75613310f671
test "$(sha256sum "$MODEL/tokenizer.json" | awk '{print $1}')" = \
  5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42
export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline:$BASE/envs/idr08-eval-r1/lib/python3.11/site-packages"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/hf_product_reasoning_train.py" \
  --model-root "$MODEL" \
  --model-revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --model-loader auto \
  --data "$DATA" \
  --output "$OUT" \
  --arm baseline \
  --updates "$UPDATES" \
  --batch-size 4 \
  --gradient-accumulation 2 \
  --max-rows 75935 \
  --max-sequence-length 384 \
  --learning-rate 2e-5 \
  --lora-layers 4 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-scope all \
  --quantization none \
  --seed 2026081021 \
  --data-seed 2026081022 \
  --log-interval 10 \
  --checkpoint-interval "$UPDATES"
find "$OUT" -type f -exec chmod a-w {} +
