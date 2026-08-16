#!/bin/bash
#SBATCH --job-name=nemotron-ultra-direct-revision
#SBATCH --partition=highgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:8
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/nemotron_ultra_train_%j.out
#SBATCH --error=logs/nemotron_ultra_train_%j.err

set -euo pipefail

required=(
  PYTHON RUNTIME MODEL_ROOT MODEL_MANIFEST MODEL_MANIFEST_SHA256 OVERLAY_ROOT
  OVERLAY_MANIFEST CAUSAL_CONV_ROOT MECHANICS_REPORT DATA OUTPUT
)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
[[ "${SLURM_GPUS_ON_NODE:-}" == "8" ]]
[[ -x "$PYTHON" ]]
for path in "$RUNTIME" "$MODEL_ROOT" "$OVERLAY_ROOT" "$CAUSAL_CONV_ROOT"; do
  [[ -d "$path" && ! -L "$path" ]]
done
for path in "$MODEL_MANIFEST" "$OVERLAY_MANIFEST" "$MECHANICS_REPORT" "$DATA"; do
  [[ -f "$path" && ! -L "$path" ]]
done
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]]
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

"$PYTHON" -P -s -B "$RUNTIME/train/hf_nemotron_ultra_train_revision.py" \
  --model-root "$MODEL_ROOT" \
  --model-manifest "$MODEL_MANIFEST" \
  --expected-model-manifest-sha256 "$MODEL_MANIFEST_SHA256" \
  --overlay-root "$OVERLAY_ROOT" \
  --overlay-manifest "$OVERLAY_MANIFEST" \
  --causal-conv-root "$CAUSAL_CONV_ROOT" \
  --mechanics-report "$MECHANICS_REPORT" \
  --data "$DATA" \
  --output "$OUTPUT"

checkpoint="$OUTPUT/checkpoint_0000256.pt"
report="$OUTPUT/report.json"
[[ -f "$checkpoint" && ! -L "$checkpoint" ]]
[[ -f "$report" && ! -L "$report" ]]
chmod a-w "$checkpoint" "$report" "$OUTPUT"
