#!/bin/bash
#SBATCH --job-name=product-token-mix
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/product_token_mix_%j.out
#SBATCH --error=logs/product_token_mix_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${DATA:?DATA is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MODEL_REVISION:?MODEL_REVISION is required}"
: "${OUTPUT:?OUTPUT is required}"

PYTHON=${PYTHON:-python3}
MAX_SEQUENCE_LENGTH=${MAX_SEQUENCE_LENGTH:-1024}
WORKSPACE_SLOTS=${WORKSPACE_SLOTS:-0}
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

test -f "$RUNTIME/pipeline/audit_product_reasoning_token_mix.py"
test -f "$DATA"
test ! -e "$OUTPUT"
mkdir -p logs "$(dirname "$OUTPUT")"

"$PYTHON" "$RUNTIME/pipeline/audit_product_reasoning_token_mix.py" \
  --data "$DATA" --model-root "$MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --max-sequence-length "$MAX_SEQUENCE_LENGTH" \
  --workspace-slots "$WORKSPACE_SLOTS" --output "$OUTPUT"
