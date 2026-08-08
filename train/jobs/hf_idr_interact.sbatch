#!/bin/bash
#SBATCH --job-name=idr-interact
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/idr_interact_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/idr_interact_%j.err

set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

: "${RUNTIME:?RUNTIME is required}"
: "${RUNTIME_MANIFEST_SHA256:?RUNTIME_MANIFEST_SHA256 is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MODEL_REVISION:?MODEL_REVISION is required}"
: "${MODEL_CONFIG_SHA256:?MODEL_CONFIG_SHA256 is required}"
: "${DRAFT_CHECKPOINT:?DRAFT_CHECKPOINT is required}"
: "${DRAFT_CHECKPOINT_SHA256:?DRAFT_CHECKPOINT_SHA256 is required}"
: "${REVISION_CHECKPOINT:?REVISION_CHECKPOINT is required}"
: "${REVISION_CHECKPOINT_SHA256:?REVISION_CHECKPOINT_SHA256 is required}"
: "${QUESTIONS:?QUESTIONS is required}"
: "${QUESTIONS_SHA256:?QUESTIONS_SHA256 is required}"
: "${REPORT:?REPORT is required}"
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-768}
SEED=${SEED:-2026080819}

test -x "$PYTHON"
test -f "$RUNTIME/train/hf_idr_interact.py"
test -f "$MODEL_ROOT/config.json"
test -f "$DRAFT_CHECKPOINT"
test -f "$REVISION_CHECKPOINT"
test -f "$QUESTIONS"
test ! -e "$REPORT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$DRAFT_CHECKPOINT" | cut -d' ' -f1)" = "$DRAFT_CHECKPOINT_SHA256"
test "$(sha256sum "$REVISION_CHECKPOINT" | cut -d' ' -f1)" = "$REVISION_CHECKPOINT_SHA256"
test "$(sha256sum "$QUESTIONS" | cut -d' ' -f1)" = "$QUESTIONS_SHA256"

mkdir -p logs "$(dirname "$REPORT")"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"

nvidia-smi
"$PYTHON" "$RUNTIME/train/hf_idr_interact.py" \
  --model-root "$MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --draft-checkpoint "$DRAFT_CHECKPOINT" \
  --revision-checkpoint "$REVISION_CHECKPOINT" \
  --questions-jsonl "$QUESTIONS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --seed "$SEED" \
  --report "$REPORT"

sha256sum "$REPORT"
chmod a-w "$REPORT"
