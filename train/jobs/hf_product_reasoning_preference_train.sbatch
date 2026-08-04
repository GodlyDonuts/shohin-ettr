#!/bin/bash
#SBATCH --job-name=hf-product-pref
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/hf_product_pref_%j.out
#SBATCH --error=logs/hf_product_pref_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MODEL_REVISION:?MODEL_REVISION is required}"
: "${DATA:?DATA is required}"
: "${REPLAY_DATA:?REPLAY_DATA is required}"
: "${OUTPUT:?OUTPUT is required}"
: "${WARM_START_CHECKPOINT:?WARM_START_CHECKPOINT is required}"

PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603-r2/bin/python}
UPDATES=${UPDATES:-200}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-16}
MAX_ROWS=${MAX_ROWS:-100000}
REPLAY_MAX_ROWS=${REPLAY_MAX_ROWS:-100000}
MAX_SEQUENCE_LENGTH=${MAX_SEQUENCE_LENGTH:-4096}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
LORA_LAYERS=${LORA_LAYERS:-4}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
UNFREEZE_LAYERS=${UNFREEZE_LAYERS:-2}
BETA=${BETA:-2.0}
MARGIN=${MARGIN:-0.0}
SFT_WEIGHT=${SFT_WEIGHT:-0.25}
REPLAY_WEIGHT=${REPLAY_WEIGHT:-0.5}
SEED=${SEED:-31}
DATA_SEED=${DATA_SEED:-20260806}
REPLAY_DATA_SEED=${REPLAY_DATA_SEED:-20260802}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-100}
STAGE_MODEL=${STAGE_MODEL:-true}

test -x "$PYTHON"
test -f "$RUNTIME/train/hf_product_reasoning_preference_train.py"
test -f "$RUNTIME/train/hf_product_reasoning_train.py"
test -f "$RUNTIME/train/integrated_reasoning_workspace.py"
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

"$PYTHON" "$RUNTIME/train/hf_product_reasoning_preference_train.py" \
  --model-root "$EFFECTIVE_MODEL_ROOT" \
  --model-source-root "$SOURCE_MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --data "$DATA" \
  --replay-data "$REPLAY_DATA" \
  --output "$OUTPUT" \
  --warm-start-checkpoint "$WARM_START_CHECKPOINT" \
  --arm baseline \
  --updates "$UPDATES" \
  --gradient-accumulation "$GRADIENT_ACCUMULATION" \
  --max-rows "$MAX_ROWS" \
  --replay-max-rows "$REPLAY_MAX_ROWS" \
  --max-sequence-length "$MAX_SEQUENCE_LENGTH" \
  --learning-rate "$LEARNING_RATE" \
  --lora-layers "$LORA_LAYERS" \
  --lora-rank "$LORA_RANK" \
  --lora-alpha "$LORA_ALPHA" \
  --unfreeze-layers "$UNFREEZE_LAYERS" \
  --beta "$BETA" \
  --margin "$MARGIN" \
  --sft-weight "$SFT_WEIGHT" \
  --replay-weight "$REPLAY_WEIGHT" \
  --seed "$SEED" \
  --data-seed "$DATA_SEED" \
  --replay-data-seed "$REPLAY_DATA_SEED" \
  --checkpoint-interval "$CHECKPOINT_INTERVAL"
