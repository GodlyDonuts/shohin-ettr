#!/bin/bash
#SBATCH --job-name=idr-aqc-interact
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/idr_aqc_interact_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/idr_aqc_interact_%j.err

set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

: "${RUNTIME:?RUNTIME is required}"
: "${RUNTIME_MANIFEST_SHA256:?RUNTIME_MANIFEST_SHA256 is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MODEL_CONFIG_SHA256:?MODEL_CONFIG_SHA256 is required}"
: "${RELEASE_ROOT:?RELEASE_ROOT is required}"
: "${RELEASE_MANIFEST_SHA256:?RELEASE_MANIFEST_SHA256 is required}"
: "${RELEASE_SUMS_SHA256:?RELEASE_SUMS_SHA256 is required}"
: "${QUESTIONS:?QUESTIONS is required}"
: "${QUESTIONS_SHA256:?QUESTIONS_SHA256 is required}"
: "${REPORT:?REPORT is required}"
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-768}
SEED=${SEED:-2026080907}
BATCH_PAIRS=${BATCH_PAIRS:-2}

test -x "$PYTHON"
test -f "$RUNTIME/train/hf_idr_aqc_interact.py"
test -f "$MODEL_ROOT/config.json"
test -f "$RELEASE_ROOT/manifest.json"
test -f "$RELEASE_ROOT/SHA256SUMS"
test -f "$QUESTIONS"
test ! -e "$REPORT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = \
  "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = \
  "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$RELEASE_ROOT/manifest.json" | cut -d' ' -f1)" = \
  "$RELEASE_MANIFEST_SHA256"
test "$(sha256sum "$RELEASE_ROOT/SHA256SUMS" | cut -d' ' -f1)" = \
  "$RELEASE_SUMS_SHA256"
(cd "$RELEASE_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$QUESTIONS" | cut -d' ' -f1)" = "$QUESTIONS_SHA256"

LOCAL_PARENT=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
EFFECTIVE_MODEL_ROOT=$LOCAL_PARENT/model
mkdir -p "$EFFECTIVE_MODEL_ROOT" logs "$(dirname "$REPORT")"
cp -a "$MODEL_ROOT"/. "$EFFECTIVE_MODEL_ROOT"/

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"

nvidia-smi
"$PYTHON" "$RUNTIME/train/hf_idr_aqc_interact.py" \
  --release-root "$RELEASE_ROOT" \
  --model-root "$EFFECTIVE_MODEL_ROOT" \
  --questions-jsonl "$QUESTIONS" \
  --report "$REPORT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --seed "$SEED" \
  --batch-pairs "$BATCH_PAIRS"

sha256sum "$REPORT"
chmod a-w "$REPORT"
