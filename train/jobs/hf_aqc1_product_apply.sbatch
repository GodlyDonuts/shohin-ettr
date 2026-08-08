#!/bin/bash
#SBATCH --job-name=aqc1-product-apply
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/aqc1_product_apply_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/aqc1_product_apply_%j.err

set -euo pipefail
: "${RUNTIME:?}" "${RUNTIME_MANIFEST_SHA256:?}" "${MODEL_ROOT:?}" "${MODEL_CONFIG_SHA256:?}"
: "${ADAPTER_CHECKPOINT:?}" "${ADAPTER_CHECKPOINT_SHA256:?}" "${COMMIT:?}" "${COMMIT_SHA256:?}"
: "${COMMIT_REPORT:?}" "${COMMIT_REPORT_SHA256:?}" "${PAIRS:?}" "${PAIRS_SHA256:?}"
: "${PAIRS_REPORT:?}" "${PAIRS_REPORT_SHA256:?}" "${ARM:?}" "${CANDIDATES_OUTPUT:?}" "${REPORT:?}"
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
for spec in \
  "$MODEL_ROOT/config.json:$MODEL_CONFIG_SHA256" "$ADAPTER_CHECKPOINT:$ADAPTER_CHECKPOINT_SHA256" \
  "$COMMIT:$COMMIT_SHA256" "$COMMIT_REPORT:$COMMIT_REPORT_SHA256" \
  "$PAIRS:$PAIRS_SHA256" "$PAIRS_REPORT:$PAIRS_REPORT_SHA256"; do
  path=${spec%%:*}; expected=${spec#*:}; test "$(sha256sum "$path" | cut -d' ' -f1)" = "$expected"
done
test ! -e "$CANDIDATES_OUTPUT"; test ! -e "$REPORT"
LOCAL_PARENT=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
EFFECTIVE_MODEL_ROOT=$LOCAL_PARENT/model
mkdir -p "$EFFECTIVE_MODEL_ROOT" "$(dirname "$REPORT")"
cp -a "$MODEL_ROOT"/. "$EFFECTIVE_MODEL_ROOT"/
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
nvidia-smi
"$PYTHON" "$RUNTIME/train/hf_aqc1_product_apply.py" \
  --arm "$ARM" --model-root "$EFFECTIVE_MODEL_ROOT" --adapter-checkpoint "$ADAPTER_CHECKPOINT" \
  --commit "$COMMIT" --commit-report "$COMMIT_REPORT" --pairs "$PAIRS" \
  --pairs-report "$PAIRS_REPORT" --candidates-output "$CANDIDATES_OUTPUT" --report "$REPORT"
sha256sum "$CANDIDATES_OUTPUT" "$REPORT"
chmod a-w "$CANDIDATES_OUTPUT" "$REPORT"
