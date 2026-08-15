#!/bin/bash
#SBATCH --job-name=upward-moe-temporal
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:nvidia_h100_pcie:2
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc33,evc34,evc37,evc38,evc43,evc46,evc50
#SBATCH --output=logs/upward_moe_temporal_%j.out
#SBATCH --error=logs/upward_moe_temporal_%j.err

set -euo pipefail
required=(
  HOST PYTHON RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_MANIFEST
  MECHANICS_REPORT DATA DATA_REPORT OWNER_CHECKPOINT
  REVISION_CHECKPOINT OUTPUT
)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
[[ "$HOST" == nemotron-super || "$HOST" == mixtral-8x22b ]]
[[ "${SLURM_GPUS_ON_NODE:-}" == 2 ]]
[[ -x "$PYTHON" && -d "$RUNTIME" && ! -L "$RUNTIME" ]]
for path in "$MODEL_MANIFEST" "$MECHANICS_REPORT" "$DATA" "$DATA_REPORT" \
  "$OWNER_CHECKPOINT" "$REVISION_CHECKPOINT"; do
  [[ -f "$path" && ! -L "$path" ]]
done
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]]

source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_runtime
EXPECTED_DATA_SHA256=$(PYTHONPATH="$RUNTIME/pipeline:$RUNTIME/train" \
  "$PYTHON" -P -s -B "$RUNTIME/pipeline/read_upward_moe_data_receipt.py" \
  --host "$HOST" --report "$DATA_REPORT" --kind revision_train --data "$DATA")
[[ "$EXPECTED_DATA_SHA256" =~ ^[0-9a-f]{64}$ ]]
q36_verify_sha256 "$DATA" "$EXPECTED_DATA_SHA256"
mkdir -p logs "$(dirname "$OUTPUT")"

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON" -P -s -B "$RUNTIME/train/hf_upward_moe_train_temporal_gate.py" \
  "${host_args[@]}" --model-root "$MODEL_ROOT" \
  --model-manifest "$MODEL_MANIFEST" --mechanics-report "$MECHANICS_REPORT" \
  --data "$DATA" --expected-data-sha256 "$EXPECTED_DATA_SHA256" \
  --owner-checkpoint "$OWNER_CHECKPOINT" \
  --revision-checkpoint "$REVISION_CHECKPOINT" --output "$OUTPUT" \
  --seed 2026081511 --learning-rate 0.0002 \
  --initial-revision-weight 0.1 --causal-loss-weight 1.0 \
  --routing-supervision-weight 0.0

[[ -f "$OUTPUT/checkpoint_0000256.pt" && ! -L "$OUTPUT/checkpoint_0000256.pt" ]]
[[ -f "$OUTPUT/report.json" && ! -L "$OUTPUT/report.json" ]]
chmod -R a-w "$OUTPUT"
