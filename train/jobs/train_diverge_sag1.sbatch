#!/bin/bash
#SBATCH --job-name=diverge-sag1
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/diverge_sag1_%j.out
#SBATCH --error=logs/diverge_sag1_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${RUNTIME_MANIFEST_SHA256:?RUNTIME_MANIFEST_SHA256 is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MODEL_REVISION:?MODEL_REVISION is required}"
: "${MODEL_PREFLIGHT:?MODEL_PREFLIGHT is required}"
: "${MODEL_PREFLIGHT_SHA256:?MODEL_PREFLIGHT_SHA256 is required}"
: "${BASE_CHECKPOINT:?BASE_CHECKPOINT is required}"
: "${BASE_CHECKPOINT_SHA256:?BASE_CHECKPOINT_SHA256 is required}"
: "${DATA:?DATA is required}"
: "${DATA_SHA256:?DATA_SHA256 is required}"
: "${OUTPUT:?OUTPUT is required}"

PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603-r2/bin/python}
UPDATES=${UPDATES:-256}
BATCH_SIZE=${BATCH_SIZE:-1}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-16}
MAX_ROWS=${MAX_ROWS:-100000}
MAX_SEQUENCE_LENGTH=${MAX_SEQUENCE_LENGTH:-1024}
LEARNING_RATE=${LEARNING_RATE:-2e-4}
LORA_LAYERS=${LORA_LAYERS:-4}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
WORKSPACE_WIDTH=${WORKSPACE_WIDTH:-512}
SOURCE_SLOTS=${SOURCE_SLOTS:-8}
QUERY_SLOTS=${QUERY_SLOTS:-4}
RECURRENT_STEPS=${RECURRENT_STEPS:-8}
ATTENTION_HEADS=${ATTENTION_HEADS:-8}
FF_MULTIPLIER=${FF_MULTIPLIER:-2}
POINTER_TEMPERATURE=${POINTER_TEMPERATURE:-0.50}
BINDING_WEIGHT=${BINDING_WEIGHT:-0.05}
COVERAGE_WEIGHT=${COVERAGE_WEIGHT:-0.02}
RESET_WEIGHT=${RESET_WEIGHT:-0.02}
ROUTER_HIDDEN=${ROUTER_HIDDEN:-256}
ADVANTAGE_MARGIN=${ADVANTAGE_MARGIN:-0.02}
ROUTER_THRESHOLD=${ROUTER_THRESHOLD:-0.50}
ROUTER_WEIGHT=${ROUTER_WEIGHT:-0.20}
RISK_WEIGHT=${RISK_WEIGHT:-0.50}
SPARSITY_WEIGHT=${SPARSITY_WEIGHT:-0.01}
SEED=${SEED:-2026080711}
DATA_SEED=${DATA_SEED:-20260802}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-64}
STAGE_MODEL=${STAGE_MODEL:-true}

test -x "$PYTHON"
test -f "$RUNTIME/train/train_diverge_sag1.py"
test -f "$RUNTIME/train/diverge_sag1_product.py"
test -f "$RUNTIME/train/diverge_qpt1_product.py"
test -f "$RUNTIME/train/hf_product_reasoning_train.py"
test -d "$MODEL_ROOT"
test -f "$MODEL_PREFLIGHT"
test -f "$BASE_CHECKPOINT"
test -f "$DATA"
test ! -e "$OUTPUT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_PREFLIGHT" | cut -d' ' -f1)" = "$MODEL_PREFLIGHT_SHA256"
test "$(sha256sum "$BASE_CHECKPOINT" | cut -d' ' -f1)" = "$BASE_CHECKPOINT_SHA256"
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = "$DATA_SHA256"
test "$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$MODEL_PREFLIGHT")" = pass

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
"$PYTHON" "$RUNTIME/train/train_diverge_sag1.py" \
  --model-root "$EFFECTIVE_MODEL_ROOT" \
  --model-source-root "$SOURCE_MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --base-checkpoint-sha256 "$BASE_CHECKPOINT_SHA256" \
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
  --query-slots "$QUERY_SLOTS" \
  --recurrent-steps "$RECURRENT_STEPS" \
  --attention-heads "$ATTENTION_HEADS" \
  --ff-multiplier "$FF_MULTIPLIER" \
  --pointer-temperature "$POINTER_TEMPERATURE" \
  --binding-weight "$BINDING_WEIGHT" \
  --coverage-weight "$COVERAGE_WEIGHT" \
  --reset-weight "$RESET_WEIGHT" \
  --router-hidden "$ROUTER_HIDDEN" \
  --advantage-margin "$ADVANTAGE_MARGIN" \
  --router-threshold "$ROUTER_THRESHOLD" \
  --router-weight "$ROUTER_WEIGHT" \
  --risk-weight "$RISK_WEIGHT" \
  --sparsity-weight "$SPARSITY_WEIGHT" \
  --seed "$SEED" \
  --data-seed "$DATA_SEED" \
  --checkpoint-interval "$CHECKPOINT_INTERVAL"

test -f "$OUTPUT/report.json"
chmod -R a-w "$OUTPUT"
