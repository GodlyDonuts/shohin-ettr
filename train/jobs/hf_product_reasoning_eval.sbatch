#!/bin/bash
#SBATCH --job-name=hf-product-eval
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/hf_product_eval_%j.out
#SBATCH --error=logs/hf_product_eval_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MODEL_REVISION:?MODEL_REVISION is required}"
: "${TASK:?TASK is required}"
: "${DATA:?DATA is required}"
: "${OUTPUT:?OUTPUT is required}"

PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603-r2/bin/python}
COUNT=${COUNT:-100}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-768}
GENERATION_MODE=${GENERATION_MODE:-greedy}
ADAPTER_CHECKPOINT=${ADAPTER_CHECKPOINT:-}
ENABLE_THINKING=${ENABLE_THINKING:-true}

test -x "$PYTHON"
test -f "$RUNTIME/train/hf_product_reasoning_eval.py"
test -d "$MODEL_ROOT"
test -f "$DATA"
test ! -e "$OUTPUT"

mkdir -p logs "$(dirname "$OUTPUT")"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"

ADAPTER_ARGS=()
if [[ -n "$ADAPTER_CHECKPOINT" ]]; then
  test -f "$ADAPTER_CHECKPOINT"
  ADAPTER_ARGS+=(--adapter-checkpoint "$ADAPTER_CHECKPOINT")
fi

THINKING_ARG=--enable-thinking
if [[ "$ENABLE_THINKING" == "false" ]]; then
  THINKING_ARG=--no-enable-thinking
elif [[ "$ENABLE_THINKING" != "true" ]]; then
  echo "ENABLE_THINKING must be true or false" >&2
  exit 2
fi

"$PYTHON" "$RUNTIME/train/hf_product_reasoning_eval.py" \
  --model-root "$MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --task "$TASK" \
  --data "$DATA" \
  --output "$OUTPUT" \
  --count "$COUNT" \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --generation-mode "$GENERATION_MODE" \
  "$THINKING_ARG" \
  "${ADAPTER_ARGS[@]}"
