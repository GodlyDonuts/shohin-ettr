#!/bin/bash
#SBATCH --job-name=tcs1-semantic
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/tcs1_semantic_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/tcs1_semantic_%j.err

set -euo pipefail

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 MODEL_MANIFEST_SHA256 ADAPTER_CHECKPOINT \
  ADAPTER_CHECKPOINT_SHA256 TRAIN TRAIN_SHA256 DEVELOPMENT \
  DEVELOPMENT_SHA256 SHAPE_REPORT SHAPE_REPORT_SHA256 OUTPUT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
UPDATES=${UPDATES:-256}
MECHANICS_ONLY=${MECHANICS_ONLY:-false}
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

test ! -e "$OUTPUT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$MODEL_ROOT/SHA256SUMS" | cut -d' ' -f1)" = "$MODEL_MANIFEST_SHA256"
test "$(sha256sum "$ADAPTER_CHECKPOINT" | cut -d' ' -f1)" = "$ADAPTER_CHECKPOINT_SHA256"
test "$(sha256sum "$TRAIN" | cut -d' ' -f1)" = "$TRAIN_SHA256"
test "$(sha256sum "$DEVELOPMENT" | cut -d' ' -f1)" = "$DEVELOPMENT_SHA256"
test "$(sha256sum "$SHAPE_REPORT" | cut -d' ' -f1)" = "$SHAPE_REPORT_SHA256"

SOURCE_MODEL_ROOT=$MODEL_ROOT
LOCAL_PARENT=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
EFFECTIVE_MODEL_ROOT=$LOCAL_PARENT/model
mkdir -p "$EFFECTIVE_MODEL_ROOT"
cp -a "$SOURCE_MODEL_ROOT"/. "$EFFECTIVE_MODEL_ROOT"/

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"

mechanics_args=()
if [ "$MECHANICS_ONLY" = true ]; then mechanics_args+=(--mechanics-only); fi
nvidia-smi
"$PYTHON" "$RUNTIME/train/hf_tcs1_semantic_selector.py" \
  --model-root "$EFFECTIVE_MODEL_ROOT" \
  --model-source-root "$SOURCE_MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --adapter-checkpoint "$ADAPTER_CHECKPOINT" \
  --train "$TRAIN" \
  --development "$DEVELOPMENT" \
  --shape-report "$SHAPE_REPORT" \
  --output "$OUTPUT" \
  --updates "$UPDATES" \
  --gradient-accumulation 8 \
  --head-width 512 \
  --max-sequence-length 2048 \
  --seed 2026080903 \
  "${mechanics_args[@]}"

test -f "$OUTPUT/selector.pt"
test -f "$OUTPUT/report.json"
sha256sum "$OUTPUT/selector.pt" "$OUTPUT/report.json"
chmod -R a-w "$OUTPUT"
