#!/bin/bash
#SBATCH --job-name=q36-mtr-temporal-eval
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc33,evc34,evc37,evc38,evc43,evc46,evc50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_mtr_temporal_eval_%A_%a.out
#SBATCH --error=logs/q36_mtr_temporal_eval_%A_%a.err

set -euo pipefail

required=(
  RUNTIME RUNTIME_MANIFEST_SHA256 PYTHON MODEL_ROOT MODEL_MANIFEST
  MODEL_MANIFEST_SHA256 MODEL_REVISION MODEL_CONFIG_SHA256 SOURCE SOURCE_SHA256
  DRAFT_CANDIDATES OWNER_CHECKPOINT REVISION_CHECKPOINT GATE_CHECKPOINT
  OUTPUT_ROOT EXPECTED_ROWS SHARD_COUNT ENVIRONMENT_RECEIPT
  ENVIRONMENT_RECEIPT_SHA256 ENVIRONMENT_TREE_SHA256
)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
[[ "$SHARD_COUNT" =~ ^(4|16)$ ]] || { printf 'shard count differs\n' >&2; exit 2; }
[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ ]] || { printf 'array index differs\n' >&2; exit 2; }
(( SLURM_ARRAY_TASK_ID < SHARD_COUNT )) || { printf 'array index differs\n' >&2; exit 2; }

source "$RUNTIME/train/jobs/q36_mtr_common.sh"
ARCHITECTURE=${ARCHITECTURE:-temporal}
[[ "$ARCHITECTURE" == temporal || "$ARCHITECTURE" == multi_trajectory || \
  "$ARCHITECTURE" == tri_trajectory || "$ARCHITECTURE" == tri_geometry || \
  "$ARCHITECTURE" == tri_hierarchical ]] || \
  q36_die "temporal evaluation architecture differs"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_runtime
q36_verify_model
q36_verify_environment
q36_verify_sha256 "$SOURCE" "$SOURCE_SHA256"
q36_require_file "$OWNER_CHECKPOINT"
q36_require_file "$REVISION_CHECKPOINT"
q36_require_file "$GATE_CHECKPOINT"
architecture_args=(--architecture "$ARCHITECTURE")
arm=temporal_gate
if [[ "$ARCHITECTURE" == multi_trajectory || "$ARCHITECTURE" == tri_trajectory || \
  "$ARCHITECTURE" == tri_geometry || "$ARCHITECTURE" == tri_hierarchical ]]; then
  [[ -n "${DRAFT_HIDDEN_CHECKPOINT:-}" ]] || \
    q36_die "multi-trajectory draft-hidden checkpoint is required"
  q36_require_file "$DRAFT_HIDDEN_CHECKPOINT"
  architecture_args+=(--draft-hidden-checkpoint "$DRAFT_HIDDEN_CHECKPOINT")
  arm=multi_trajectory_gate
fi
IFS=: read -r -a draft_paths <<< "$DRAFT_CANDIDATES"
[[ ${#draft_paths[@]} -eq SHARD_COUNT ]] || q36_die "draft candidate geometry differs"
draft_args=()
for path in "${draft_paths[@]}"; do
  q36_require_file "$path"
  draft_args+=(--draft-candidates "$path")
done

shard=$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
output=$OUTPUT_ROOT/$arm/shard_$shard/candidates.jsonl
report=$OUTPUT_ROOT/$arm/shard_$shard/report.json
[[ ! -e "$output" && ! -L "$output" && ! -e "$report" && ! -L "$report" ]] || \
  q36_die "temporal evaluation output exists"
mkdir -p "$(dirname "$output")" logs
model=$(q36_stage_model)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
q36_export_pythonpath
nvidia-smi
"$PYTHON" "$RUNTIME/train/hf_q36_mtr_evaluate_temporal_gate.py" \
  --model-root "$model" \
  --model-source-root "$MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  "${architecture_args[@]}" \
  --owner-checkpoint "$OWNER_CHECKPOINT" \
  --revision-checkpoint "$REVISION_CHECKPOINT" \
  --gate-checkpoint "$GATE_CHECKPOINT" \
  --source "$SOURCE" \
  --source-sha256 "$SOURCE_SHA256" \
  "${draft_args[@]}" \
  --candidates-output "$output" \
  --report "$report" \
  --expected-rows "$EXPECTED_ROWS" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --shard-count "$SHARD_COUNT" \
  --batch-size 2
test -f "$output" && test -f "$report"
chmod a-w "$output" "$report"
