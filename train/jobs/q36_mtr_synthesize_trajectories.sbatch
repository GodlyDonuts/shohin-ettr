#!/bin/bash
#SBATCH --job-name=q36-mtr-synthesis
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc33,evc34,evc37,evc38,evc43,evc46,evc50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_mtr_synthesis_%j.out
#SBATCH --error=logs/q36_mtr_synthesis_%j.err

set -euo pipefail

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE_COMMIT PYTHON TRAIN_SOURCE TRAIN_SOURCE_SHA256 DEVELOPMENT_SOURCE DEVELOPMENT_SOURCE_SHA256 FREEZE_REPORT FREEZE_REPORT_SHA256 ALIGNED_CHECKPOINT CURRENT_CANDIDATES OWNER71_CANDIDATES OWNER8_CANDIDATES SHARD_ROOT ENVIRONMENT_RECEIPT ENVIRONMENT_RECEIPT_SHA256 ENVIRONMENT_TREE_SHA256; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^([0-9]|1[0-5])$ ]] || { printf 'SLURM_ARRAY_TASK_ID differs\n' >&2; exit 2; }
SHARD_INDEX=$SLURM_ARRAY_TASK_ID
OUTPUT=$SHARD_ROOT/$(printf 'shard_%02d/candidates.jsonl' "$SHARD_INDEX")
REPORT=$SHARD_ROOT/$(printf 'shard_%02d/report.json' "$SHARD_INDEX")
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_runtime
q36_require_authorization
q36_verify_model
q36_verify_environment
q36_verify_sha256 "$TRAIN_SOURCE" "$TRAIN_SOURCE_SHA256"
q36_verify_sha256 "$DEVELOPMENT_SOURCE" "$DEVELOPMENT_SOURCE_SHA256"
q36_verify_sha256 "$FREEZE_REPORT" "$FREEZE_REPORT_SHA256"
trainer=${SYNTHESIS_SCRIPT:-$RUNTIME/train/hf_q36_mtr_synthesize_trajectories.py}
[[ -f "$trainer" && ! -L "$trainer" ]] || q36_die "synthesis script differs"
if [[ -n "${SYNTHESIS_SCRIPT:-}" ]]; then
  q36_require SYNTHESIS_SCRIPT_SHA256
  q36_verify_sha256 "$trainer" "$SYNTHESIS_SCRIPT_SHA256"
fi
rotation=${SYNTHESIS_ROTATION_OFFSET:-0}
[[ "$rotation" =~ ^[012]$ ]] || q36_die "synthesis rotation offset differs"
[[ -f "$ALIGNED_CHECKPOINT" && ! -L "$ALIGNED_CHECKPOINT" ]] || q36_die "aligned checkpoint differs"
for owner in CURRENT OWNER71 OWNER8; do
  name="${owner}_CANDIDATES"; IFS=: read -r -a paths <<< "${!name}"
  [[ ${#paths[@]} -eq 16 ]] || q36_die "synthesis candidate shard geometry differs"
done
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" && ! -e "$REPORT" && ! -L "$REPORT" ]] || q36_die "synthesis output exists"
model=$(q36_stage_model)
mkdir -p "$(dirname "$OUTPUT")" logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
q36_export_pythonpath
args=()
for owner in current owner71 owner8; do
  name="${owner^^}_CANDIDATES"; IFS=: read -r -a paths <<< "${!name}"
  for path in "${paths[@]}"; do args+=(--"$owner"-candidates "$path"); done
done
nvidia-smi
"$PYTHON" "$trainer" \
  --train-source "$TRAIN_SOURCE" --development-source "$DEVELOPMENT_SOURCE" \
  --freeze-report "$FREEZE_REPORT" --model-root "$model" \
  --model-revision "$MODEL_REVISION" --aligned-checkpoint "$ALIGNED_CHECKPOINT" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --environment-receipt-sha256 "$ENVIRONMENT_RECEIPT_SHA256" \
  --environment-tree-sha256 "$ENVIRONMENT_TREE_SHA256" \
  "${args[@]}" --output "$OUTPUT" --report "$REPORT" \
  --shard-index "$SHARD_INDEX" --shard-count 16 --batch-size 2 \
  --max-new-tokens 768 --seed 2026081421 --rotation-offset "$rotation"
test -f "$OUTPUT" && test -f "$REPORT"
chmod a-w "$OUTPUT" "$REPORT"
