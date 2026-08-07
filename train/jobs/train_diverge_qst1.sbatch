#!/bin/bash
#SBATCH --job-name=diverge-qst1
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/diverge_qst1_%j.out
#SBATCH --error=logs/diverge_qst1_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MODEL_REVISION:?MODEL_REVISION is required}"
: "${DATA:?DATA is required}"
: "${OUTPUT:?OUTPUT is required}"

PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603-r2/bin/python}
UPDATES=${UPDATES:-1000}
BATCH_SIZE=${BATCH_SIZE:-2}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-8}
MAX_ROWS=${MAX_ROWS:-100000}
MAX_SEQUENCE_LENGTH=${MAX_SEQUENCE_LENGTH:-1024}
LEARNING_RATE=${LEARNING_RATE:-2e-4}
LORA_LAYERS=${LORA_LAYERS:-4}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
WORKSPACE_WIDTH=${WORKSPACE_WIDTH:-384}
SOURCE_SLOTS=${SOURCE_SLOTS:-8}
STATE_SLOTS=${STATE_SLOTS:-8}
QUERY_SLOTS=${QUERY_SLOTS:-4}
RECURRENT_STEPS=${RECURRENT_STEPS:-8}
ATTENTION_HEADS=${ATTENTION_HEADS:-8}
FF_MULTIPLIER=${FF_MULTIPLIER:-2}
BINDING_TEMPERATURE=${BINDING_TEMPERATURE:-0.10}
BINDING_WEIGHT=${BINDING_WEIGHT:-0.05}
RESET_WEIGHT=${RESET_WEIGHT:-0.02}
HALTING_WEIGHT=${HALTING_WEIGHT:-0.01}
SEED=${SEED:-2026080701}
DATA_SEED=${DATA_SEED:-20260802}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-100}
STAGE_MODEL=${STAGE_MODEL:-true}

test -x "$PYTHON"
test -f "$RUNTIME/train/train_diverge_qst1.py"
test -f "$RUNTIME/train/diverge_qst1_product.py"
test -f "$RUNTIME/train/hf_product_reasoning_train.py"
test -d "$MODEL_ROOT"
test -f "$DATA"
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

nvidia-smi
"$PYTHON" "$RUNTIME/train/train_diverge_qst1.py" \
  --model-root "$EFFECTIVE_MODEL_ROOT" \
  --model-source-root "$SOURCE_MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --data "$DATA" \
  --output "$OUTPUT" \
  --updates "$UPDATES" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation "$GRADIENT_ACCUMULATION" \
  --max-rows "$MAX_ROWS" \
  --max-sequence-length "$MAX_SEQUENCE_LENGTH" \
  --learning-rate "$LEARNING_RATE" \
  --lora-layers "$LORA_LAYERS" \
  --lora-rank "$LORA_RANK" \
  --lora-alpha "$LORA_ALPHA" \
  --workspace-width "$WORKSPACE_WIDTH" \
  --source-slots "$SOURCE_SLOTS" \
  --state-slots "$STATE_SLOTS" \
  --query-slots "$QUERY_SLOTS" \
  --recurrent-steps "$RECURRENT_STEPS" \
  --attention-heads "$ATTENTION_HEADS" \
  --ff-multiplier "$FF_MULTIPLIER" \
  --binding-temperature "$BINDING_TEMPERATURE" \
  --binding-weight "$BINDING_WEIGHT" \
  --reset-weight "$RESET_WEIGHT" \
  --halting-weight "$HALTING_WEIGHT" \
  --seed "$SEED" \
  --data-seed "$DATA_SEED" \
  --checkpoint-interval "$CHECKPOINT_INTERVAL"
