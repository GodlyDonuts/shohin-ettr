#!/bin/bash
#SBATCH --job-name=ectr0-eval
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/ectr0_eval_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/ectr0_eval_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${RUNTIME_MANIFEST_SHA256:?RUNTIME_MANIFEST_SHA256 is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MODEL_REVISION:?MODEL_REVISION is required}"
: "${MODEL_CONFIG_SHA256:?MODEL_CONFIG_SHA256 is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${CHECKPOINT_SHA256:?CHECKPOINT_SHA256 is required}"
: "${DATA:?DATA is required}"
: "${DATA_SHA256:?DATA_SHA256 is required}"
: "${CTF_REPORT:?CTF_REPORT is required}"
: "${CTF_REPORT_SHA256:?CTF_REPORT_SHA256 is required}"
: "${CONTROL:?CONTROL is required}"
: "${SHARD_INDEX:?SHARD_INDEX is required}"
: "${SHARD_COUNT:?SHARD_COUNT is required}"
: "${OUTPUT:?OUTPUT is required}"

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

test -x "$PYTHON"
test ! -e "$OUTPUT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)" = "$CHECKPOINT_SHA256"
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = "$DATA_SHA256"
test "$(sha256sum "$CTF_REPORT" | cut -d' ' -f1)" = "$CTF_REPORT_SHA256"

LOCAL_PARENT=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
EFFECTIVE_MODEL_ROOT=$LOCAL_PARENT/model
mkdir -p "$EFFECTIVE_MODEL_ROOT" "$(dirname "$OUTPUT")" logs
cp -a "$MODEL_ROOT"/. "$EFFECTIVE_MODEL_ROOT"/

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$RUNTIME/train:$RUNTIME/pipeline

nvidia-smi
"$PYTHON" "$RUNTIME/train/eval_ectr0_executor_revision.py" \
  --control "$CONTROL" \
  --model-root "$EFFECTIVE_MODEL_ROOT" \
  --model-source-root "$MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --adapter-checkpoint "$CHECKPOINT" \
  --expected-adapter-sha256 "$CHECKPOINT_SHA256" \
  --data "$DATA" \
  --expected-data-sha256 "$DATA_SHA256" \
  --ctf-report "$CTF_REPORT" \
  --expected-ctf-sha256 "$CTF_REPORT_SHA256" \
  --max-new-tokens 512 \
  --max-sequence-length 4096 \
  --batch-size 4 \
  --seed 2026081061 \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT" \
  --output "$OUTPUT"

chmod a-w "$OUTPUT"
sha256sum "$OUTPUT"
