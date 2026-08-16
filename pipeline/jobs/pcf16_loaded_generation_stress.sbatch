#!/usr/bin/env bash
#SBATCH --job-name=pcf16-loaded-generation-stress
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --array=0-1%2
#SBATCH --no-requeue

set -euo pipefail

: "${REPOSITORY_ROOT:?REPOSITORY_ROOT is required}"
: "${RUNTIME:?RUNTIME is required}"
: "${PYTHON:?PYTHON is required}"
: "${DATA:?DATA is required}"
: "${ROW:?ROW is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${ADAPTER_CHECKPOINT:?ADAPTER_CHECKPOINT is required}"
: "${ITERATIONS:?ITERATIONS is required}"

scratch="/tmp/pcf16-loaded-stress-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
case "$scratch" in
  /tmp/pcf16-loaded-stress-[0-9]*-[01]) ;;
  *) printf 'pcf16: unsafe scratch path\n' >&2; exit 2 ;;
esac
[[ ! -e "$scratch" && ! -L "$scratch" ]]
mkdir -m 700 "$scratch"
cleanup() {
  chmod -R u+w "$scratch" 2>/dev/null || true
  rm -rf --one-file-system -- "$scratch"
}
trap cleanup EXIT

export SLURM_TMPDIR="$scratch"
export TMPDIR="$scratch"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

/usr/bin/nvidia-smi -L
"$PYTHON" -P -s -B \
  "$REPOSITORY_ROOT/pipeline/diagnose_pcf1_sandbox_launch.py" \
  --data "$DATA" \
  --row "$ROW" \
  --iterations "$ITERATIONS" \
  --model-root "$MODEL_ROOT" \
  --adapter-checkpoint "$ADAPTER_CHECKPOINT" \
  --generate-before-score

