#!/bin/bash
#SBATCH --job-name=vfr1-teacher
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/vfr1_teacher_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/vfr1_teacher_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${RUNTIME_MANIFEST_SHA256:?RUNTIME_MANIFEST_SHA256 is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MODEL_REVISION:?MODEL_REVISION is required}"
: "${MODEL_CONFIG_SHA256:?MODEL_CONFIG_SHA256 is required}"
: "${ADAPTER_CHECKPOINT:?ADAPTER_CHECKPOINT is required}"
: "${ADAPTER_SHA256:?ADAPTER_SHA256 is required}"
: "${REQUESTS:?REQUESTS is required}"
: "${REQUESTS_SHA256:?REQUESTS_SHA256 is required}"
: "${REQUEST_REPORT:?REQUEST_REPORT is required}"
: "${REQUEST_REPORT_SHA256:?REQUEST_REPORT_SHA256 is required}"
: "${OUTPUT:?OUTPUT is required}"
: "${REPORT:?REPORT is required}"

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL_LOADER=${MODEL_LOADER:-multimodal}
SHARD_INDEX=${SHARD_INDEX:-0}
SHARD_COUNT=${SHARD_COUNT:-1}
MAX_ROWS=${MAX_ROWS:-0}
BATCH_SIZE=${BATCH_SIZE:-2}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
SEED=${SEED:-2026080911}

test -x "$PYTHON"
for path in "$RUNTIME/train/hf_vfr1_generate_traces.py" \
  "$RUNTIME/pipeline/build_vfr1_teacher_requests.py" \
  "$MODEL_ROOT/config.json" "$ADAPTER_CHECKPOINT" "$REQUESTS" "$REQUEST_REPORT"; do
  test -f "$path"
done
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$ADAPTER_CHECKPOINT" | cut -d' ' -f1)" = "$ADAPTER_SHA256"
test "$(sha256sum "$REQUESTS" | cut -d' ' -f1)" = "$REQUESTS_SHA256"
test "$(sha256sum "$REQUEST_REPORT" | cut -d' ' -f1)" = "$REQUEST_REPORT_SHA256"
test ! -e "$OUTPUT"
test ! -e "$REPORT"

LOCAL_PARENT=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
EFFECTIVE_MODEL_ROOT=$LOCAL_PARENT/model
mkdir -p "$EFFECTIVE_MODEL_ROOT" "$(dirname "$OUTPUT")" "$(dirname "$REPORT")" logs
cp -a "$MODEL_ROOT"/. "$EFFECTIVE_MODEL_ROOT"/

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"

nvidia-smi
"$PYTHON" "$RUNTIME/train/hf_vfr1_generate_traces.py" \
  --model-root "$EFFECTIVE_MODEL_ROOT" \
  --model-source-root "$MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --model-loader "$MODEL_LOADER" \
  --adapter-checkpoint "$ADAPTER_CHECKPOINT" \
  --requests "$REQUESTS" \
  --request-report "$REQUEST_REPORT" \
  --output "$OUTPUT" \
  --report "$REPORT" \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT" \
  --max-rows "$MAX_ROWS" \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --seed "$SEED"

chmod a-w "$OUTPUT" "$REPORT"
