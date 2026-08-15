#!/bin/bash
#SBATCH --job-name=nemotron-super-mechanics
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:nvidia_h100_pcie:2
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc33,evc34,evc37,evc38,evc43,evc46,evc50
#SBATCH --output=logs/nemotron_super_mechanics_%j.out
#SBATCH --error=logs/nemotron_super_mechanics_%j.err

set -euo pipefail

: "${PYTHON:?PYTHON is required}"
: "${RUNTIME:?RUNTIME is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MODEL_MANIFEST:?MODEL_MANIFEST is required}"
: "${OVERLAY_ROOT:?OVERLAY_ROOT is required}"
: "${OVERLAY_MANIFEST:?OVERLAY_MANIFEST is required}"
: "${CAUSAL_CONV_ROOT:?CAUSAL_CONV_ROOT is required}"
: "${OUTPUT:?OUTPUT is required}"
: "${SLURM_TMPDIR:?SLURM_TMPDIR is required}"

[[ -x "$PYTHON" ]]
[[ -d "$RUNTIME" && ! -L "$RUNTIME" ]]
[[ -d "$MODEL_ROOT" && ! -L "$MODEL_ROOT" ]]
[[ -f "$MODEL_MANIFEST" && ! -L "$MODEL_MANIFEST" ]]
[[ -d "$OVERLAY_ROOT" && ! -L "$OVERLAY_ROOT" ]]
[[ -f "$OVERLAY_MANIFEST" && ! -L "$OVERLAY_MANIFEST" ]]
[[ -d "$CAUSAL_CONV_ROOT" && ! -L "$CAUSAL_CONV_ROOT" ]]
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]]
[[ "${SLURM_GPUS_ON_NODE:-}" == "2" ]]
mkdir -p logs "$(dirname "$OUTPUT")"

export PATH=/apps/slurm/current/bin:/usr/bin:/bin
export PYTHONPATH="$RUNTIME/train:$OVERLAY_ROOT:$CAUSAL_CONV_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$SLURM_TMPDIR/hf" XDG_CACHE_HOME="$SLURM_TMPDIR/cache"
export TMPDIR="$SLURM_TMPDIR" TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

"$PYTHON" -P -s -B "$RUNTIME/train/hf_nemotron_super_mechanics.py" \
  --model-root "$MODEL_ROOT" \
  --model-manifest "$MODEL_MANIFEST" \
  --overlay-root "$OVERLAY_ROOT" \
  --overlay-manifest "$OVERLAY_MANIFEST" \
  --output "$OUTPUT"

checkpoint=${OUTPUT%.json}.checkpoint.pt
[[ -f "$OUTPUT" && ! -L "$OUTPUT" ]]
[[ -f "$checkpoint" && ! -L "$checkpoint" ]]
chmod a-w "$OUTPUT" "$checkpoint"
