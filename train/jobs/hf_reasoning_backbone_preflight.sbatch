#!/bin/bash
#SBATCH --job-name=hf-reasoning-preflight
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/hf_reasoning_preflight_%j.out
#SBATCH --error=logs/hf_reasoning_preflight_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${MODEL:?MODEL is required}"
: "${REVISION:?REVISION is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${OUTPUT:?OUTPUT is required}"

PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python3.13}
test -x "$PYTHON"
test -d "$RUNTIME"
test -f "$RUNTIME/train/hf_reasoning_backbone_preflight.py"
test ! -e "$OUTPUT"

mkdir -p logs "$MODEL_ROOT" "$(dirname "$OUTPUT")"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-/lustre/fs1/home/sa305415/shohin/.cache/huggingface}
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"

"$PYTHON" "$RUNTIME/train/hf_reasoning_backbone_preflight.py" \
  --model "$MODEL" \
  --revision "$REVISION" \
  --model-root "$MODEL_ROOT" \
  --output "$OUTPUT"
