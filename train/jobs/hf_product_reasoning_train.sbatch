#!/bin/bash
#SBATCH --job-name=hf-product-train
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/hf_product_train_%j.out
#SBATCH --error=logs/hf_product_train_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MODEL_REVISION:?MODEL_REVISION is required}"
: "${DATA:?DATA is required}"
: "${OUTPUT:?OUTPUT is required}"
: "${ARM:?ARM is required}"

PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603-r2/bin/python}
UPDATES=${UPDATES:-200}
BATCH_SIZE=${BATCH_SIZE:-2}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-8}
MAX_ROWS=${MAX_ROWS:-100000}
MAX_SEQUENCE_LENGTH=${MAX_SEQUENCE_LENGTH:-1024}

test -x "$PYTHON"
test -f "$RUNTIME/train/hf_product_reasoning_train.py"
test -f "$RUNTIME/train/integrated_reasoning_workspace.py"
test -d "$MODEL_ROOT"
test -f "$DATA"
test ! -e "$OUTPUT"

mkdir -p logs "$(dirname "$OUTPUT")"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON" "$RUNTIME/train/hf_product_reasoning_train.py" \
  --model-root "$MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --data "$DATA" \
  --output "$OUTPUT" \
  --arm "$ARM" \
  --updates "$UPDATES" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation "$GRADIENT_ACCUMULATION" \
  --max-rows "$MAX_ROWS" \
  --max-sequence-length "$MAX_SEQUENCE_LENGTH"
