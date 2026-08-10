#!/bin/bash
#SBATCH -J nmc1-fit
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/nmc1_fit_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
ARM=${ARM:?set program or direct}
OUT=${OUT:?set isolated output root}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL=$BASE/artifacts/external/qwen3.5-0.8b-2fc0636
DATA_ROOT=$BASE/artifacts/product_reasoning/nmc1/data_bd30f2a_r1

case "$ARM" in
  program)
    DATA=$DATA_ROOT/program_train.jsonl
    EXPECTED=71010bac117d9fbca2a28e7f9a63f24456b519a7b4bc172ddd772ff27e851512
    ;;
  direct)
    DATA=$DATA_ROOT/direct_train.jsonl
    EXPECTED=b78fa53081114f1b5c08f34c0f26df641dfd975a4975130a6b0c8098a0944757
    ;;
  *) echo "unknown NMC1 arm: $ARM" >&2; exit 2 ;;
esac
test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
test "$(sha256sum "$DATA" | awk '{print $1}')" = "$EXPECTED"
test "$(sha256sum "$MODEL/tokenizer.json" | awk '{print $1}')" = \
  5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42
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
