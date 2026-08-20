#!/bin/bash
#SBATCH --job-name=mixtral-8x22b-revision
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:nvidia_h100_pcie:2
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc33,evc34,evc37,evc38,evc50
#SBATCH --output=logs/mixtral_8x22b_revision_%j.out
#SBATCH --error=logs/mixtral_8x22b_revision_%j.err

set -euo pipefail

: "${PYTHON:?PYTHON is required}"
: "${RUNTIME:?RUNTIME is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MODEL_MANIFEST:?MODEL_MANIFEST is required}"
: "${EXPECTED_MODEL_MANIFEST_SHA256:?EXPECTED_MODEL_MANIFEST_SHA256 is required}"
: "${MECHANICS_REPORT:?MECHANICS_REPORT is required}"
: "${DATA:?DATA is required}"
: "${OUTPUT:?OUTPUT is required}"

[[ -x "$PYTHON" ]]
for path in "$RUNTIME" "$MODEL_ROOT"; do
  [[ -d "$path" && ! -L "$path" ]]
done
for path in "$MODEL_MANIFEST" "$MECHANICS_REPORT" "$DATA"; do
  [[ -f "$path" && ! -L "$path" ]]
done
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]]
[[ "${SLURM_GPUS_ON_NODE:-}" == "2" ]]
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
mkdir -p logs "$(dirname "$OUTPUT")"

export PATH=/apps/slurm/current/bin:/usr/bin:/bin
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$SLURM_TMPDIR/hf" XDG_CACHE_HOME="$SLURM_TMPDIR/cache"
export TMPDIR="$SLURM_TMPDIR" TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

"$PYTHON" -P -s -B "$RUNTIME/train/hf_mixtral_8x22b_train_revision.py" \
  --model-root "$MODEL_ROOT" \
  --model-manifest "$MODEL_MANIFEST" \
  --expected-model-manifest-sha256 "$EXPECTED_MODEL_MANIFEST_SHA256" \
  --mechanics-report "$MECHANICS_REPORT" \
  --data "$DATA" \
  --output "$OUTPUT"

checkpoint="$OUTPUT/checkpoint_0000256.pt"
report="$OUTPUT/report.json"
[[ -f "$checkpoint" && ! -L "$checkpoint" ]]
[[ -f "$report" && ! -L "$report" ]]
chmod a-w "$checkpoint" "$report" "$OUTPUT"
