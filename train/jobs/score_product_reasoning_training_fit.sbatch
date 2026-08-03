#!/bin/bash
#SBATCH --job-name=product-fit-score
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/product_fit_score_%j.out
#SBATCH --error=logs/product_fit_score_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MODEL_REVISION:?MODEL_REVISION is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${DATA:?DATA is required}"
: "${OUTPUT:?OUTPUT is required}"

PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603-r2/bin/python}
MAX_SEQUENCE_LENGTH=${MAX_SEQUENCE_LENGTH:-512}
STAGE_MODEL=${STAGE_MODEL:-true}

test -x "$PYTHON"
test -f "$RUNTIME/train/score_product_reasoning_training_fit.py"
test -d "$MODEL_ROOT"
test -f "$CHECKPOINT"
test -f "$DATA"
test ! -e "$OUTPUT"

SOURCE_MODEL_ROOT=$MODEL_ROOT
EFFECTIVE_MODEL_ROOT=$MODEL_ROOT
if [[ "$STAGE_MODEL" == "true" ]]; then
  LOCAL_PARENT=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
  EFFECTIVE_MODEL_ROOT=$LOCAL_PARENT/model
  mkdir -p "$EFFECTIVE_MODEL_ROOT"
  cp -a "$SOURCE_MODEL_ROOT"/. "$EFFECTIVE_MODEL_ROOT"/
elif [[ "$STAGE_MODEL" != "false" ]]; then
  echo "STAGE_MODEL must be true or false" >&2
  exit 2
fi

mkdir -p logs "$(dirname "$OUTPUT")"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"

"$PYTHON" "$RUNTIME/train/score_product_reasoning_training_fit.py" \
  --model-root "$EFFECTIVE_MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --checkpoint "$CHECKPOINT" \
  --data "$DATA" \
  --output "$OUTPUT" \
  --max-sequence-length "$MAX_SEQUENCE_LENGTH"
