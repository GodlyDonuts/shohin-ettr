#!/bin/bash
#SBATCH --job-name=ndr1-drafts
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/ndr1_drafts_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/ndr1_drafts_%j.err

set -euo pipefail

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 MODEL_MANIFEST_SHA256 ADAPTER_CHECKPOINT \
  ADAPTER_CHECKPOINT_SHA256 SOURCE SOURCE_SHA256 SOURCE_REPORT \
  SOURCE_REPORT_SHA256 OUTPUT REPORT SHARD_INDEX SHARD_COUNT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test -x "$PYTHON"
for path in "$RUNTIME/train/hf_ndr1_generate_drafts.py" "$SOURCE" \
  "$SOURCE_REPORT" "$MODEL_ROOT/config.json" "$ADAPTER_CHECKPOINT"; do
  test -f "$path"
done
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = \
  "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = \
  "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$MODEL_ROOT/SHA256SUMS" | cut -d' ' -f1)" = \
  "$MODEL_MANIFEST_SHA256"
test "$(sha256sum "$ADAPTER_CHECKPOINT" | cut -d' ' -f1)" = \
  "$ADAPTER_CHECKPOINT_SHA256"
test "$(sha256sum "$SOURCE" | cut -d' ' -f1)" = "$SOURCE_SHA256"
test "$(sha256sum "$SOURCE_REPORT" | cut -d' ' -f1)" = \
  "$SOURCE_REPORT_SHA256"
test ! -e "$OUTPUT"
test ! -e "$REPORT"

SOURCE_MODEL_ROOT=$MODEL_ROOT
LOCAL_PARENT=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
EFFECTIVE_MODEL_ROOT=$LOCAL_PARENT/model
mkdir -p "$EFFECTIVE_MODEL_ROOT" "$(dirname "$OUTPUT")"
cp -a "$SOURCE_MODEL_ROOT"/. "$EFFECTIVE_MODEL_ROOT"/
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
nvidia-smi
"$PYTHON" "$RUNTIME/train/hf_ndr1_generate_drafts.py" \
  --source "$SOURCE" --source-report "$SOURCE_REPORT" \
  --model-root "$EFFECTIVE_MODEL_ROOT" --model-source-root "$SOURCE_MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" --model-loader multimodal \
  --adapter-checkpoint "$ADAPTER_CHECKPOINT" --output "$OUTPUT" \
  --report "$REPORT" --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT" --batch-size 4 --max-new-tokens 768 \
  --seed 2026080919
sha256sum "$OUTPUT" "$REPORT"
chmod a-w "$OUTPUT" "$REPORT"
