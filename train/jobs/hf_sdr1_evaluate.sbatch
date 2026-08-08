#!/bin/bash
#SBATCH --job-name=sdr1-eval
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/sdr1_eval_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/sdr1_eval_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${RUNTIME_MANIFEST_SHA256:?RUNTIME_MANIFEST_SHA256 is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MODEL_REVISION:?MODEL_REVISION is required}"
: "${MODEL_CONFIG_SHA256:?MODEL_CONFIG_SHA256 is required}"
: "${ADAPTER_CHECKPOINT:?ADAPTER_CHECKPOINT is required}"
: "${ADAPTER_CHECKPOINT_SHA256:?ADAPTER_CHECKPOINT_SHA256 is required}"
: "${DATA:?DATA is required}"
: "${DATA_SHA256:?DATA_SHA256 is required}"
: "${DATA_REPORT:?DATA_REPORT is required}"
: "${DATA_REPORT_SHA256:?DATA_REPORT_SHA256 is required}"
: "${SPLIT:?SPLIT is required}"
: "${CANDIDATES_OUTPUT:?CANDIDATES_OUTPUT is required}"
: "${REPORT:?REPORT is required}"

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

test -x "$PYTHON"
for path in \
  "$RUNTIME/train/hf_sdr1_evaluate.py" \
  "$RUNTIME/train/hf_vcr1_evaluate_reviser.py" \
  "$RUNTIME/train/hf_product_reasoning_eval.py" \
  "$RUNTIME/train/hf_product_reasoning_rollouts.py" \
  "$RUNTIME/train/hf_product_reasoning_train.py" \
  "$RUNTIME/train/integrated_reasoning_workspace.py" \
  "$MODEL_ROOT/config.json" \
  "$ADAPTER_CHECKPOINT" \
  "$DATA" \
  "$DATA_REPORT"; do
  test -f "$path"
done
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$ADAPTER_CHECKPOINT" | cut -d' ' -f1)" = "$ADAPTER_CHECKPOINT_SHA256"
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = "$DATA_SHA256"
test "$(sha256sum "$DATA_REPORT" | cut -d' ' -f1)" = "$DATA_REPORT_SHA256"
test ! -e "$CANDIDATES_OUTPUT"
test ! -e "$REPORT"

SOURCE_MODEL_ROOT=$MODEL_ROOT
LOCAL_PARENT=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
EFFECTIVE_MODEL_ROOT=$LOCAL_PARENT/model
mkdir -p "$EFFECTIVE_MODEL_ROOT" logs "$(dirname "$REPORT")"
cp -a "$SOURCE_MODEL_ROOT"/. "$EFFECTIVE_MODEL_ROOT"/

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"

nvidia-smi
"$PYTHON" "$RUNTIME/train/hf_sdr1_evaluate.py" \
  --model-root "$EFFECTIVE_MODEL_ROOT" \
  --model-source-root "$SOURCE_MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --adapter-checkpoint "$ADAPTER_CHECKPOINT" \
  --data "$DATA" \
  --data-report "$DATA_REPORT" \
  --split "$SPLIT" \
  --candidates-output "$CANDIDATES_OUTPUT" \
  --report "$REPORT" \
  --batch-size 2 \
  --max-new-tokens 768 \
  --seed 2026080816

sha256sum "$CANDIDATES_OUTPUT" "$REPORT"
chmod a-w "$CANDIDATES_OUTPUT" "$REPORT"
