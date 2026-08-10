#!/bin/bash
#SBATCH -J dtmc1-fit
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 03:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/dtmc1_fit_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
DATA=${DATA:?set immutable merged draft corpus}
EXPECTED_DATA_SHA256=${EXPECTED_DATA_SHA256:?set merged corpus SHA-256}
OUT=${OUT:?set isolated output root}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL=$BASE/artifacts/external/qwen3.5-0.8b-2fc0636
OWNER=$BASE/artifacts/product_reasoning/nmc1/direct_fit_2bad8eb_r1/checkpoint_0001024.pt

test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
test "$(sha256sum "$OWNER" | awk '{print $1}')" = \
  8a2b6550f4083368eff2493f1b14d59a1d9dd2e4b39d4da58ff23cdd1b250b53
test "$(sha256sum "$DATA" | awk '{print $1}')" = "$EXPECTED_DATA_SHA256"
test "$(sha256sum "$MODEL/tokenizer.json" | awk '{print $1}')" = \
  5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42

export PYTHONPATH="$CODE_ROOT/train:$BASE/envs/idr08-eval-r1/lib/python3.11/site-packages"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/train_dtmc1_compiler.py" \
  --model-root "$MODEL" \
  --model-revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --owner-checkpoint "$OWNER" \
  --expected-owner-sha256 8a2b6550f4083368eff2493f1b14d59a1d9dd2e4b39d4da58ff23cdd1b250b53 \
  --expected-owner-data-sha256 b78fa53081114f1b5c08f34c0f26df641dfd975a4975130a6b0c8098a0944757 \
  --data "$DATA" \
  --expected-data-sha256 "$EXPECTED_DATA_SHA256" \
  --output "$OUT" \
  --updates 4096 \
  --batch-size 32 \
  --max-source-tokens 1024 \
  --width 512 \
  --source-layers 2 \
  --decoder-layers 4 \
  --heads 8 \
  --learning-rate 2e-4 \
  --seed 2026081061 \
  --data-seed 2026081062 \
  --log-interval 64
find "$OUT" -type f -exec chmod a-w {} +
