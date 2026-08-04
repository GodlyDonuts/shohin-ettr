#!/bin/bash
#SBATCH --job-name=hf-product-rlvr
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/hf_product_rlvr_%j.out
#SBATCH --error=logs/hf_product_rlvr_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MODEL_REVISION:?MODEL_REVISION is required}"
: "${DATA:?DATA is required}"
: "${REPLAY_DATA:?REPLAY_DATA is required}"
: "${OUTPUT:?OUTPUT is required}"
: "${WARM_START_CHECKPOINT:?WARM_START_CHECKPOINT is required}"

PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603-r2/bin/python}
UPDATES=${UPDATES:-100}
START_UPDATE=${START_UPDATE:-0}
SCHEDULE_TOTAL_UPDATES=${SCHEDULE_TOTAL_UPDATES:-100}
SAMPLES=${SAMPLES:-4}
GROUPS_PER_UPDATE=${GROUPS_PER_UPDATE:-4}
MAX_ROWS=${MAX_ROWS:-4096}
REPLAY_MAX_ROWS=${REPLAY_MAX_ROWS:-100000}
MAX_SEQUENCE_LENGTH=${MAX_SEQUENCE_LENGTH:-4096}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1536}
LEARNING_RATE=${LEARNING_RATE:-2e-7}
LORA_LAYERS=${LORA_LAYERS:-4}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
UNFREEZE_LAYERS=${UNFREEZE_LAYERS:-2}
REPLAY_WEIGHT=${REPLAY_WEIGHT:-0.25}
SEED=${SEED:-20260804}
DATA_SEED=${DATA_SEED:-20260804}
REPLAY_DATA_SEED=${REPLAY_DATA_SEED:-20260802}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-25}
STAGE_MODEL=${STAGE_MODEL:-true}

test -x "$PYTHON"
test -f "$RUNTIME/train/hf_product_reasoning_rlvr_train.py"
test -f "$RUNTIME/train/hf_product_reasoning_rollouts.py"
test -f "$RUNTIME/train/hf_product_reasoning_eval.py"
test -f "$RUNTIME/train/hf_product_reasoning_train.py"
test -d "$MODEL_ROOT"
test -f "$DATA"
test -f "$REPLAY_DATA"
test -f "$WARM_START_CHECKPOINT"
test ! -e "$OUTPUT"

SOURCE_MODEL_ROOT=$MODEL_ROOT
EFFECTIVE_MODEL_ROOT=$MODEL_ROOT
if [[ "$STAGE_MODEL" == "true" ]]; then
  LOCAL_PARENT=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
  EFFECTIVE_MODEL_ROOT=$LOCAL_PARENT/model
  mkdir -p "$EFFECTIVE_MODEL_ROOT"
  cp -a "$SOURCE_MODEL_ROOT"/. "$EFFECTIVE_MODEL_ROOT"/
  test -f "$EFFECTIVE_MODEL_ROOT/config.json"
  test -f "$EFFECTIVE_MODEL_ROOT/model.safetensors.index.json"
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON" "$RUNTIME/train/hf_product_reasoning_rlvr_train.py" \
  --model-root "$EFFECTIVE_MODEL_ROOT" \
  --model-source-root "$SOURCE_MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --data "$DATA" \
  --replay-data "$REPLAY_DATA" \
  --output "$OUTPUT" \
  --warm-start-checkpoint "$WARM_START_CHECKPOINT" \
  --arm baseline \
  --start-update "$START_UPDATE" \
  --updates "$UPDATES" \
  --schedule-total-updates "$SCHEDULE_TOTAL_UPDATES" \
  --samples "$SAMPLES" \
  --groups-per-update "$GROUPS_PER_UPDATE" \
  --max-rows "$MAX_ROWS" \
  --replay-max-rows "$REPLAY_MAX_ROWS" \
  --max-sequence-length "$MAX_SEQUENCE_LENGTH" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --learning-rate "$LEARNING_RATE" \
  --lora-layers "$LORA_LAYERS" \
  --lora-rank "$LORA_RANK" \
  --lora-alpha "$LORA_ALPHA" \
  --unfreeze-layers "$UNFREEZE_LAYERS" \
  --replay-weight "$REPLAY_WEIGHT" \
  --seed "$SEED" \
  --data-seed "$DATA_SEED" \
  --replay-data-seed "$REPLAY_DATA_SEED" \
  --checkpoint-interval "$CHECKPOINT_INTERVAL"
