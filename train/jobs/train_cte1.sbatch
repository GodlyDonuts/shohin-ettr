#!/bin/bash
#SBATCH -J cte1-fit
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/cte1_fit_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
DATA=${DATA:?set immutable CTE1 train data}
EXPECTED_DATA_SHA256=${EXPECTED_DATA_SHA256:?set CTE1 train SHA-256}
OUT=${OUT:?set isolated output root}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL=$BASE/artifacts/external/qwen3.5-0.8b-2fc0636

test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
test "$(sha256sum "$DATA" | awk '{print $1}')" = "$EXPECTED_DATA_SHA256"
export PYTHONPATH="$CODE_ROOT/train:$BASE/envs/idr08-eval-r1/lib/python3.11/site-packages"
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
  --updates 1024 \
  --batch-size 4 \
  --gradient-accumulation 2 \
  --max-rows 6333 \
  --max-sequence-length 1024 \
  --learning-rate 2e-5 \
  --lora-layers 4 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-scope all \
  --quantization none \
  --seed 2026081051 \
  --data-seed 2026081052 \
  --log-interval 16 \
  --checkpoint-interval 1024
find "$OUT" -type f -exec chmod a-w {} +

