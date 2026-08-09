#!/bin/bash
#SBATCH --job-name=mtr1-router
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/mtr1_router_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/mtr1_router_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 ARM MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 MODEL_MANIFEST_SHA256 DATA DATA_SHA256 DATA_REPORT \
  DATA_REPORT_SHA256 OUTPUT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
if [[ "$ARM" == treatment ]]; then
  : "${ADAPTER_CHECKPOINT:?treatment checkpoint required}"
  : "${ADAPTER_CHECKPOINT_SHA256:?treatment checkpoint hash required}"
elif [[ "$ARM" != unchanged ]]; then
  echo "ARM differs" >&2; exit 2
fi
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$MODEL_ROOT/SHA256SUMS" | cut -d' ' -f1)" = "$MODEL_MANIFEST_SHA256"
(cd "$MODEL_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = "$DATA_SHA256"
test "$(sha256sum "$DATA_REPORT" | cut -d' ' -f1)" = "$DATA_REPORT_SHA256"
test ! -e "$OUTPUT"

SOURCE_MODEL_ROOT=$MODEL_ROOT
LOCAL_PARENT=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
EFFECTIVE_MODEL_ROOT=$LOCAL_PARENT/model
mkdir -p "$EFFECTIVE_MODEL_ROOT" "$(dirname "$OUTPUT")"
cp -a "$SOURCE_MODEL_ROOT"/. "$EFFECTIVE_MODEL_ROOT"/
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
args=()
if [[ "$ARM" == treatment ]]; then
  test "$(sha256sum "$ADAPTER_CHECKPOINT" | cut -d' ' -f1)" = "$ADAPTER_CHECKPOINT_SHA256"
  args+=(--adapter-checkpoint "$ADAPTER_CHECKPOINT")
fi
nvidia-smi
"$PYTHON" "$RUNTIME/train/mtr1_router_accounting.py" \
  --arm "$ARM" --model-root "$EFFECTIVE_MODEL_ROOT" \
  --model-source-root "$SOURCE_MODEL_ROOT" --model-revision "$MODEL_REVISION" \
  --data "$DATA" --data-report "$DATA_REPORT" --split development \
  --rows-per-task 29 --batch-size 1 --output "$OUTPUT" "${args[@]}"
sha256sum "$OUTPUT"; chmod a-w "$OUTPUT"
