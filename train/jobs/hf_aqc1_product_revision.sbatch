#!/bin/bash
#SBATCH --job-name=aqc1-product-revision
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/aqc1_product_revision_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/aqc1_product_revision_%j.err

set -euo pipefail
: "${RUNTIME:?}" "${RUNTIME_MANIFEST_SHA256:?}" "${MODEL_ROOT:?}" "${MODEL_CONFIG_SHA256:?}"
: "${MODEL_REVISION:?}" "${ADAPTER_CHECKPOINT:?}" "${ADAPTER_CHECKPOINT_SHA256:?}"
: "${DATA:?}" "${DATA_SHA256:?}" "${DATA_REPORT:?}" "${DATA_REPORT_SHA256:?}"
: "${SKIP:?}" "${COUNT:?}" "${CANDIDATES_OUTPUT:?}" "${REPORT:?}"
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$ADAPTER_CHECKPOINT" | cut -d' ' -f1)" = "$ADAPTER_CHECKPOINT_SHA256"
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = "$DATA_SHA256"
test "$(sha256sum "$DATA_REPORT" | cut -d' ' -f1)" = "$DATA_REPORT_SHA256"
test ! -e "$CANDIDATES_OUTPUT"; test ! -e "$REPORT"
LOCAL_PARENT=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
EFFECTIVE_MODEL_ROOT=$LOCAL_PARENT/model
mkdir -p "$EFFECTIVE_MODEL_ROOT" "$(dirname "$REPORT")"
cp -a "$MODEL_ROOT"/. "$EFFECTIVE_MODEL_ROOT"/
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
nvidia-smi
"$PYTHON" "$RUNTIME/train/hf_aqc1_product_revision.py" \
  --model-root "$EFFECTIVE_MODEL_ROOT" --model-revision "$MODEL_REVISION" \
  --adapter-checkpoint "$ADAPTER_CHECKPOINT" --data "$DATA" --data-report "$DATA_REPORT" \
  --skip "$SKIP" --count "$COUNT" --candidates-output "$CANDIDATES_OUTPUT" --report "$REPORT"
sha256sum "$CANDIDATES_OUTPUT" "$REPORT"
chmod a-w "$CANDIDATES_OUTPUT" "$REPORT"
