#!/bin/bash
#SBATCH --job-name=vfr1-eval
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/vfr1_eval_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/vfr1_eval_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${RUNTIME_MANIFEST_SHA256:?RUNTIME_MANIFEST_SHA256 is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MODEL_REVISION:?MODEL_REVISION is required}"
: "${MODEL_CONFIG_SHA256:?MODEL_CONFIG_SHA256 is required}"
: "${MODEL_MANIFEST_SHA256:?MODEL_MANIFEST_SHA256 is required}"
: "${ADAPTER_CHECKPOINT:?ADAPTER_CHECKPOINT is required}"
: "${ADAPTER_CHECKPOINT_SHA256:?ADAPTER_CHECKPOINT_SHA256 is required}"
: "${DATA:?DATA is required}"
: "${DATA_SHA256:?DATA_SHA256 is required}"
: "${DATA_REPORT:?DATA_REPORT is required}"
: "${DATA_REPORT_SHA256:?DATA_REPORT_SHA256 is required}"
: "${CANDIDATES_OUTPUT:?CANDIDATES_OUTPUT is required}"
: "${REPORT:?REPORT is required}"
SHARD_INDEX=${SHARD_INDEX:-0}
SHARD_COUNT=${SHARD_COUNT:-1}
MODEL_LOADER=${MODEL_LOADER:-multimodal}
STAGE_MODEL=${STAGE_MODEL:-true}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1536}

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test -x "$PYTHON"
for path in "$RUNTIME/train/hf_vfr1_evaluate_reviser.py" \
  "$RUNTIME/train/hf_vfr1_generate_traces.py" \
  "$RUNTIME/train/hf_product_reasoning_eval.py" \
  "$RUNTIME/train/hf_product_reasoning_rollouts.py" \
  "$RUNTIME/train/hf_product_reasoning_train.py" \
  "$MODEL_ROOT/config.json" "$MODEL_ROOT/SHA256SUMS" \
  "$ADAPTER_CHECKPOINT" "$DATA" "$DATA_REPORT"; do test -f "$path"; done
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$MODEL_ROOT/SHA256SUMS" | cut -d' ' -f1)" = "$MODEL_MANIFEST_SHA256"
(cd "$MODEL_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$ADAPTER_CHECKPOINT" | cut -d' ' -f1)" = "$ADAPTER_CHECKPOINT_SHA256"
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = "$DATA_SHA256"
test "$(sha256sum "$DATA_REPORT" | cut -d' ' -f1)" = "$DATA_REPORT_SHA256"
test ! -e "$CANDIDATES_OUTPUT"; test ! -e "$REPORT"

SOURCE_MODEL_ROOT=$MODEL_ROOT
if [[ "$STAGE_MODEL" == "true" && -n "${SLURM_TMPDIR:-}" ]]; then
  EFFECTIVE_MODEL_ROOT=$SLURM_TMPDIR/model
  mkdir -p "$EFFECTIVE_MODEL_ROOT"
  cp -a "$SOURCE_MODEL_ROOT"/. "$EFFECTIVE_MODEL_ROOT"/
  export TMPDIR=$SLURM_TMPDIR/tmp
elif [[ "$STAGE_MODEL" == "false" || -z "${SLURM_TMPDIR:-}" ]]; then
  EFFECTIVE_MODEL_ROOT=$SOURCE_MODEL_ROOT
  export TMPDIR=$BASE/tmp/$SLURM_JOB_ID
  trap 'rm -rf "$TMPDIR"' EXIT
else
  echo "STAGE_MODEL must be true or false" >&2; exit 2
fi
mkdir -p "$TMPDIR" logs "$(dirname "$REPORT")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
nvidia-smi
"$PYTHON" "$RUNTIME/train/hf_vfr1_evaluate_reviser.py" \
  --model-root "$EFFECTIVE_MODEL_ROOT" --model-source-root "$SOURCE_MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" --model-loader "$MODEL_LOADER" \
  --adapter-checkpoint "$ADAPTER_CHECKPOINT" --data "$DATA" \
  --data-report "$DATA_REPORT" --candidates-output "$CANDIDATES_OUTPUT" \
  --report "$REPORT" --batch-size 2 --max-new-tokens "$MAX_NEW_TOKENS" \
  --seed 2026080913 --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT"
sha256sum "$CANDIDATES_OUTPUT" "$REPORT"
chmod a-w "$CANDIDATES_OUTPUT" "$REPORT"
