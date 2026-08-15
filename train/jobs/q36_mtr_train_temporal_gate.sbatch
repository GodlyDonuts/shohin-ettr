#!/bin/bash
#SBATCH --job-name=q36-mtr-temporal-gate
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc33,evc34,evc37,evc38,evc43,evc46,evc50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_mtr_temporal_gate_%j.out
#SBATCH --error=logs/q36_mtr_temporal_gate_%j.err

set -euo pipefail

required=(
  RUNTIME RUNTIME_MANIFEST_SHA256 PYTHON MODEL_ROOT MODEL_MANIFEST
  MODEL_MANIFEST_SHA256 MODEL_REVISION MODEL_CONFIG_SHA256 DATA
  OWNER_CHECKPOINT REVISION_CHECKPOINT OUTPUT ENVIRONMENT_RECEIPT
  ENVIRONMENT_RECEIPT_SHA256 ENVIRONMENT_TREE_SHA256
)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done

source "$RUNTIME/train/jobs/q36_mtr_common.sh"
ARCHITECTURE=${ARCHITECTURE:-temporal}
[[ "$ARCHITECTURE" == temporal || "$ARCHITECTURE" == multi_trajectory || \
  "$ARCHITECTURE" == tri_trajectory || "$ARCHITECTURE" == tri_geometry || \
  "$ARCHITECTURE" == tri_hierarchical ]] || \
  q36_die "temporal gate architecture differs"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_runtime
q36_verify_model
q36_verify_environment
q36_verify_sha256 "$DATA" "$(q36_sha256 "$DATA")"
q36_verify_sha256 "$OWNER_CHECKPOINT" "$(q36_sha256 "$OWNER_CHECKPOINT")"
q36_verify_sha256 "$REVISION_CHECKPOINT" "$(q36_sha256 "$REVISION_CHECKPOINT")"
architecture_args=(--architecture "$ARCHITECTURE")
if [[ "$ARCHITECTURE" == multi_trajectory || "$ARCHITECTURE" == tri_trajectory || \
  "$ARCHITECTURE" == tri_geometry || "$ARCHITECTURE" == tri_hierarchical ]]; then
  [[ -n "${DRAFT_HIDDEN_CHECKPOINT:-}" ]] || \
    q36_die "multi-trajectory draft-hidden checkpoint is required"
  q36_verify_sha256 "$DRAFT_HIDDEN_CHECKPOINT" "$(q36_sha256 "$DRAFT_HIDDEN_CHECKPOINT")"
  architecture_args+=(--draft-hidden-checkpoint "$DRAFT_HIDDEN_CHECKPOINT")
  if [[ "$ARCHITECTURE" == tri_trajectory || "$ARCHITECTURE" == tri_geometry || \
    "$ARCHITECTURE" == tri_hierarchical ]]; then
    architecture_args+=(
      --max-rows 1189
      --data-seed 2026081517
      --initial-branch-weights 0.1 0.8 0.1
    )
  else
    architecture_args+=(
      --max-rows 1167
      --data-seed 2026081514
      --initial-branch-weights 0.9 0.1
    )
  fi
  [[ "${ROUTING_SUPERVISION_WEIGHT:-0.1}" == 0.1 ]] || \
    q36_die "multi-trajectory routing supervision differs"
  ROUTING_SUPERVISION_WEIGHT=0.1
fi
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || q36_die "temporal gate output exists"

trainer=$RUNTIME/train/hf_q36_mtr_train_temporal_gate.py
q36_require_file "$trainer"
routing=()
if [[ -n "${ROUTING_SUPERVISION_WEIGHT:-}" ]]; then
  [[ "$ROUTING_SUPERVISION_WEIGHT" == "0.1" ]] || \
    q36_die "temporal routing supervision differs"
  routing=(--routing-supervision-weight "$ROUTING_SUPERVISION_WEIGHT")
fi
causal=()
if [[ -n "${CAUSAL_LOSS_WEIGHT:-}" ]]; then
  [[ "$CAUSAL_LOSS_WEIGHT" == "0.0" || "$CAUSAL_LOSS_WEIGHT" == "1.0" ]] || \
    q36_die "temporal causal loss weight differs"
  [[ "$ARCHITECTURE" == multi_trajectory || "$ARCHITECTURE" == tri_trajectory || \
    "$ARCHITECTURE" == tri_geometry || \
    "$ARCHITECTURE" == tri_hierarchical || \
    "$CAUSAL_LOSS_WEIGHT" == "1.0" ]] || \
    q36_die "temporal causal loss cannot be disabled"
  causal=(--causal-loss-weight "$CAUSAL_LOSS_WEIGHT")
fi
model=$(q36_stage_model)
mkdir -p "$(dirname "$OUTPUT")" logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
q36_export_pythonpath
nvidia-smi
"$PYTHON" "$trainer" \
  --model-root "$model" \
  --model-source-root "$MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --model-config-sha256 "$MODEL_CONFIG_SHA256" \
  "${architecture_args[@]}" \
  --data "$DATA" \
  --owner-checkpoint "$OWNER_CHECKPOINT" \
  --revision-checkpoint "$REVISION_CHECKPOINT" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --environment-receipt-sha256 "$ENVIRONMENT_RECEIPT_SHA256" \
  --environment-tree-sha256 "$ENVIRONMENT_TREE_SHA256" \
  "${routing[@]}" \
  "${causal[@]}" \
  --output "$OUTPUT"

test -f "$OUTPUT/checkpoint_0000256.pt" && test -f "$OUTPUT/report.json"
chmod -R a-w "$OUTPUT"
