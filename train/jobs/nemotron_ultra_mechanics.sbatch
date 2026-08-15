#!/bin/bash
#SBATCH --job-name=nemotron-ultra-mechanics
#SBATCH --partition=highgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:8
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/nemotron_ultra_mechanics_%j.out
#SBATCH --error=logs/nemotron_ultra_mechanics_%j.err

set -euo pipefail

: "${PYTHON:?PYTHON is required}"
: "${RUNTIME:?RUNTIME is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MODEL_MANIFEST:?MODEL_MANIFEST is required}"
: "${MODEL_MANIFEST_SHA256:?MODEL_MANIFEST_SHA256 is required}"
: "${OVERLAY_ROOT:?OVERLAY_ROOT is required}"
: "${OVERLAY_MANIFEST:?OVERLAY_MANIFEST is required}"
: "${CAUSAL_CONV_ROOT:?CAUSAL_CONV_ROOT is required}"
: "${OUTPUT:?OUTPUT is required}"

[[ -x "$PYTHON" ]]
[[ -d "$RUNTIME" && ! -L "$RUNTIME" ]]
[[ -d "$MODEL_ROOT" && ! -L "$MODEL_ROOT" ]]
[[ -f "$MODEL_MANIFEST" && ! -L "$MODEL_MANIFEST" ]]
[[ -d "$OVERLAY_ROOT" && ! -L "$OVERLAY_ROOT" ]]
[[ -f "$OVERLAY_MANIFEST" && ! -L "$OVERLAY_MANIFEST" ]]
[[ -d "$CAUSAL_CONV_ROOT" && ! -L "$CAUSAL_CONV_ROOT" ]]
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]]
[[ "${SLURM_GPUS_ON_NODE:-}" == "8" ]]
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
mkdir -p logs "$(dirname "$OUTPUT")"

export PATH=/apps/slurm/current/bin:/usr/bin:/bin
export PYTHONPATH="$RUNTIME/train:$OVERLAY_ROOT:$CAUSAL_CONV_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$SLURM_TMPDIR/hf" XDG_CACHE_HOME="$SLURM_TMPDIR/cache"
export TMPDIR="$SLURM_TMPDIR" TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

"$PYTHON" -P -s -B "$RUNTIME/train/hf_nemotron_ultra_mechanics.py" \
  --model-root "$MODEL_ROOT" \
  --model-manifest "$MODEL_MANIFEST" \
  --expected-model-manifest-sha256 "$MODEL_MANIFEST_SHA256" \
  --overlay-root "$OVERLAY_ROOT" \
  --overlay-manifest "$OVERLAY_MANIFEST" \
  --output "$OUTPUT"

checkpoint=${OUTPUT%.json}.checkpoint.pt
[[ -f "$OUTPUT" && ! -L "$OUTPUT" ]]
[[ -f "$checkpoint" && ! -L "$checkpoint" ]]
chmod a-w "$OUTPUT" "$checkpoint"
