#!/bin/bash
#SBATCH --job-name=upward-moe-drafts
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:nvidia_h100_pcie:2
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc33,evc34,evc37,evc38,evc43,evc46,evc50
#SBATCH --output=logs/upward_moe_drafts_%A_%a.out
#SBATCH --error=logs/upward_moe_drafts_%A_%a.err

set -euo pipefail

required=(
  HOST PYTHON RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_MANIFEST
  MECHANICS_REPORT TRAIN_SOURCE DEVELOPMENT_SOURCE FREEZE_REPORT
  OWNER_CHECKPOINT SHARD_ROOT
)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
[[ "$HOST" == nemotron-super || "$HOST" == mixtral-8x22b ]]
[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^([0-9]|1[0-5])$ ]]
[[ "${SLURM_GPUS_ON_NODE:-}" == 2 ]]
[[ -x "$PYTHON" && -d "$RUNTIME" && ! -L "$RUNTIME" ]]
for path in "$MODEL_MANIFEST" "$MECHANICS_REPORT" "$TRAIN_SOURCE" \
  "$DEVELOPMENT_SOURCE" "$FREEZE_REPORT" "$OWNER_CHECKPOINT"; do
  [[ -f "$path" && ! -L "$path" ]]
done
OUTPUT=$SHARD_ROOT/$(printf 'shard_%02d/candidates.jsonl' "$SLURM_ARRAY_TASK_ID")
REPORT=$SHARD_ROOT/$(printf 'shard_%02d/report.json' "$SLURM_ARRAY_TASK_ID")
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" && ! -e "$REPORT" && ! -L "$REPORT" ]]

source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_runtime
mkdir -p "$(dirname "$OUTPUT")" logs

host_args=(--host "$HOST")
if [[ "$HOST" == nemotron-super ]]; then
  : "${OVERLAY_ROOT:?OVERLAY_ROOT is required for Nemotron Super}"
  : "${OVERLAY_MANIFEST:?OVERLAY_MANIFEST is required for Nemotron Super}"
  : "${CAUSAL_CONV_ROOT:?CAUSAL_CONV_ROOT is required for Nemotron Super}"
  host_args+=(--overlay-root "$OVERLAY_ROOT" --overlay-manifest "$OVERLAY_MANIFEST")
  export PYTHONPATH="$RUNTIME/train:$OVERLAY_ROOT:$CAUSAL_CONV_ROOT"
else
  : "${EXPECTED_MODEL_MANIFEST_SHA256:?EXPECTED_MODEL_MANIFEST_SHA256 is required for Mixtral}"
  [[ "$EXPECTED_MODEL_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]
  host_args+=(--expected-model-manifest-sha256 "$EXPECTED_MODEL_MANIFEST_SHA256")
  export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
fi
export PATH=/apps/slurm/current/bin:/usr/bin:/bin
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$SLURM_TMPDIR/hf" XDG_CACHE_HOME="$SLURM_TMPDIR/cache"
export TMPDIR="$SLURM_TMPDIR" TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

"$PYTHON" -P -s -B "$RUNTIME/train/hf_upward_moe_generate_drafts.py" \
  "${host_args[@]}" \
  --model-root "$MODEL_ROOT" \
  --model-manifest "$MODEL_MANIFEST" \
  --mechanics-report "$MECHANICS_REPORT" \
  --train-source "$TRAIN_SOURCE" \
  --development-source "$DEVELOPMENT_SOURCE" \
  --freeze-report "$FREEZE_REPORT" \
  --owner-checkpoint "$OWNER_CHECKPOINT" \
  --output "$OUTPUT" --report "$REPORT" \
  --shard-index "$SLURM_ARRAY_TASK_ID" --shard-count 16 \
  --batch-size 1 --max-new-tokens 768 --seed 2026080818

[[ -f "$OUTPUT" && ! -L "$OUTPUT" && -f "$REPORT" && ! -L "$REPORT" ]]
chmod a-w "$OUTPUT" "$REPORT"
