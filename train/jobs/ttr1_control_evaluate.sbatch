#!/bin/bash
#SBATCH --job-name=ttr1-control-eval
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/ttr1_control_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/ttr1_control_%j.err

set -euo pipefail

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 CONTROL MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 DATA DATA_SHA256 DATA_REPORT DATA_REPORT_SHA256 SPLIT \
  CANDIDATES_OUTPUT REPORT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
SHARD_INDEX=${SHARD_INDEX:-0}
SHARD_COUNT=${SHARD_COUNT:-1}

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test -x "$PYTHON"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
if [[ "$CONTROL" == "independent_commitment" ]]; then
  : "${ADAPTER_CHECKPOINT:?independent checkpoint is required}"
  : "${ADAPTER_CHECKPOINT_SHA256:?independent checkpoint hash is required}"
fi
if [[ -n "${ADAPTER_CHECKPOINT:-}" ]]; then
  test "$(sha256sum "$ADAPTER_CHECKPOINT" | cut -d' ' -f1)" = \
    "$ADAPTER_CHECKPOINT_SHA256"
fi
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = "$DATA_SHA256"
test "$(sha256sum "$DATA_REPORT" | cut -d' ' -f1)" = "$DATA_REPORT_SHA256"
test ! -e "$CANDIDATES_OUTPUT"
test ! -e "$REPORT"

SOURCE_MODEL_ROOT=$MODEL_ROOT
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  EFFECTIVE_MODEL_ROOT=$SLURM_TMPDIR/model
  mkdir -p "$EFFECTIVE_MODEL_ROOT"
  cp -a "$SOURCE_MODEL_ROOT"/. "$EFFECTIVE_MODEL_ROOT"/
  export TMPDIR=$SLURM_TMPDIR/tmp
else
  EFFECTIVE_MODEL_ROOT=$SOURCE_MODEL_ROOT
  export TMPDIR=$BASE/tmp/$SLURM_JOB_ID
  trap 'rm -rf "$TMPDIR"' EXIT
fi
mkdir -p "$TMPDIR" "$(dirname "$REPORT")"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
nvidia-smi
ADAPTER_ARGS=()
if [[ -n "${ADAPTER_CHECKPOINT:-}" ]]; then
  ADAPTER_ARGS+=(--adapter-checkpoint "$ADAPTER_CHECKPOINT")
fi
"$PYTHON" "$RUNTIME/train/ttr1_control_evaluate.py" \
  --control "$CONTROL" \
  --model-root "$EFFECTIVE_MODEL_ROOT" \
  --model-source-root "$SOURCE_MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --model-loader auto \
  --data "$DATA" \
  --data-report "$DATA_REPORT" \
  --split "$SPLIT" \
  --candidates-output "$CANDIDATES_OUTPUT" \
  --report "$REPORT" \
  --batch-size 2 \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT" \
  "${ADAPTER_ARGS[@]}"
chmod a-w "$CANDIDATES_OUTPUT" "$REPORT"
sha256sum "$CANDIDATES_OUTPUT" "$REPORT"
