#!/bin/bash
#SBATCH --job-name=resume-sag1
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/resume_diverge_sag1_%j.out
#SBATCH --error=logs/resume_diverge_sag1_%j.err

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

test -x "$PYTHON"
test -f "$RUNTIME/train/train_diverge_sag1.py"
test -f "$RUNTIME/train/diverge_sag1_product.py"
test -f "$RUNTIME/train/diverge_qpt1_product.py"
test -f "$RUNTIME/train/hf_product_reasoning_train.py"
test -d "$MODEL_ROOT"
test -f "$MODEL_PREFLIGHT"
test -f "$BASE_CHECKPOINT"
test -f "$DATA"
test -d "$OUTPUT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_PREFLIGHT" | cut -d' ' -f1)" = "$MODEL_PREFLIGHT_SHA256"
test "$(sha256sum "$BASE_CHECKPOINT" | cut -d' ' -f1)" = "$BASE_CHECKPOINT_SHA256"
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = "$DATA_SHA256"

if [[ -f "$OUTPUT/report.json" ]]; then
  echo "[sag1-resume] primary run already complete; no action needed"
  exit 0
fi

RESUME_CHECKPOINT=$(find "$OUTPUT" -maxdepth 1 -type f -name 'checkpoint_*.pt' | sort | tail -n 1)
test -n "$RESUME_CHECKPOINT"

SOURCE_MODEL_ROOT=$MODEL_ROOT
LOCAL_PARENT=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
EFFECTIVE_MODEL_ROOT=$LOCAL_PARENT/model
mkdir -p "$EFFECTIVE_MODEL_ROOT" logs
cp -a "$SOURCE_MODEL_ROOT"/. "$EFFECTIVE_MODEL_ROOT"/
test -f "$EFFECTIVE_MODEL_ROOT/config.json"
test -f "$EFFECTIVE_MODEL_ROOT/model.safetensors.index.json"

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
  --resume-checkpoint "$RESUME_CHECKPOINT" \
  --data "$DATA" \
  --output "$OUTPUT" \
  --updates 256 \
  --batch-size 1 \
  --gradient-accumulation 16 \
  --max-rows 100000 \
  --max-sequence-length 1024 \
  --learning-rate 2e-4 \
  --lora-layers 4 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --workspace-width 512 \
  --source-slots 8 \
  --query-slots 4 \
  --recurrent-steps 8 \
  --attention-heads 8 \
  --ff-multiplier 2 \
  --pointer-temperature 0.50 \
  --binding-weight 0.05 \
  --coverage-weight 0.02 \
  --reset-weight 0.02 \
  --router-hidden 256 \
  --advantage-margin 0.02 \
  --router-threshold 0.50 \
  --router-weight 0.20 \
  --risk-weight 0.50 \
  --sparsity-weight 0.01 \
  --seed 2026080711 \
  --data-seed 20260802 \
  --checkpoint-interval 64

test -f "$OUTPUT/report.json"
chmod -R a-w "$OUTPUT"
