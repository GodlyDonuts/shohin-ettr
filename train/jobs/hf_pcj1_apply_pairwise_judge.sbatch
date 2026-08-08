#!/bin/bash
#SBATCH --job-name=pcj1-apply
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/pcj1_apply_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/pcj1_apply_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${RUNTIME_MANIFEST_SHA256:?RUNTIME_MANIFEST_SHA256 is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MODEL_REVISION:?MODEL_REVISION is required}"
: "${MODEL_CONFIG_SHA256:?MODEL_CONFIG_SHA256 is required}"
: "${ADAPTER_CHECKPOINT:?ADAPTER_CHECKPOINT is required}"
: "${ADAPTER_CHECKPOINT_SHA256:?ADAPTER_CHECKPOINT_SHA256 is required}"
: "${JUDGE:?JUDGE is required}"
: "${JUDGE_REPORT:?JUDGE_REPORT is required}"
: "${PAIRS:?PAIRS is required}"
: "${PAIRS_SHA256:?PAIRS_SHA256 is required}"
: "${OUTPUT:?OUTPUT is required}"

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

test -x "$PYTHON"
for path in \
  "$RUNTIME/train/hf_pcj1_apply_pairwise_judge.py" \
  "$RUNTIME/train/hf_pcj1_pairwise_judge.py" \
  "$RUNTIME/train/hf_cvg1_apply_completion_verifier.py" \
  "$RUNTIME/train/hf_cvg1_completion_verifier.py" \
  "$RUNTIME/train/hf_product_reasoning_eval.py" \
  "$RUNTIME/train/hf_product_reasoning_train.py" \
  "$RUNTIME/train/integrated_reasoning_workspace.py" \
  "$MODEL_ROOT/config.json" \
  "$ADAPTER_CHECKPOINT" \
  "$JUDGE" \
  "$JUDGE_REPORT" \
  "$PAIRS"; do
  test -f "$path"
done
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$ADAPTER_CHECKPOINT" | cut -d' ' -f1)" = "$ADAPTER_CHECKPOINT_SHA256"
test "$(sha256sum "$PAIRS" | cut -d' ' -f1)" = "$PAIRS_SHA256"
test ! -e "$OUTPUT"

SOURCE_MODEL_ROOT=$MODEL_ROOT
LOCAL_PARENT=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
EFFECTIVE_MODEL_ROOT=$LOCAL_PARENT/model
mkdir -p "$EFFECTIVE_MODEL_ROOT" logs "$(dirname "$OUTPUT")"
cp -a "$SOURCE_MODEL_ROOT"/. "$EFFECTIVE_MODEL_ROOT"/
test -f "$EFFECTIVE_MODEL_ROOT/config.json"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"

nvidia-smi
"$PYTHON" "$RUNTIME/train/hf_pcj1_apply_pairwise_judge.py" \
  --model-root "$EFFECTIVE_MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --adapter-checkpoint "$ADAPTER_CHECKPOINT" \
  --judge "$JUDGE" \
  --judge-report "$JUDGE_REPORT" \
  --pairs "$PAIRS" \
  --output "$OUTPUT" \
  --batch-rows 2

sha256sum "$OUTPUT"
chmod a-w "$OUTPUT"
