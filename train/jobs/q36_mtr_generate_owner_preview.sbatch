#!/bin/bash
#SBATCH --job-name=q36-mtr-owner-preview
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc33,evc34,evc37,evc38,evc43,evc46,evc50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_mtr_owner_preview_%j.out
#SBATCH --error=logs/q36_mtr_owner_preview_%j.err

set -euo pipefail

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE_COMMIT PYTHON PREVIEW_SCRIPT PREVIEW_SCRIPT_SHA256 TRAIN_SOURCE TRAIN_SOURCE_SHA256 DEVELOPMENT_SOURCE DEVELOPMENT_SOURCE_SHA256 FREEZE_REPORT FREEZE_REPORT_SHA256 OWNER_CHECKPOINT ENVIRONMENT_RECEIPT ENVIRONMENT_RECEIPT_SHA256 ENVIRONMENT_TREE_SHA256 OUTPUT REPORT; do
  [[ -n "$(printenv "$variable")" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_runtime
q36_verify_model
q36_verify_environment
q36_verify_sha256 "$PREVIEW_SCRIPT" "$PREVIEW_SCRIPT_SHA256"
q36_verify_sha256 "$TRAIN_SOURCE" "$TRAIN_SOURCE_SHA256"
q36_verify_sha256 "$DEVELOPMENT_SOURCE" "$DEVELOPMENT_SOURCE_SHA256"
q36_verify_sha256 "$FREEZE_REPORT" "$FREEZE_REPORT_SHA256"
[[ -f "$OWNER_CHECKPOINT" && ! -L "$OWNER_CHECKPOINT" ]] || q36_die "owner preview checkpoint differs"
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" && ! -e "$REPORT" && ! -L "$REPORT" ]] || q36_die "owner preview output exists"
model=$(q36_stage_model)
mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$REPORT")" logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
q36_export_pythonpath
nvidia-smi
"$PYTHON" "$PREVIEW_SCRIPT" \
  --train-source "$TRAIN_SOURCE" \
  --development-source "$DEVELOPMENT_SOURCE" \
  --freeze-report "$FREEZE_REPORT" \
  --model-root "$model" \
  --owner-checkpoint "$OWNER_CHECKPOINT" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --environment-receipt-sha256 "$ENVIRONMENT_RECEIPT_SHA256" \
  --environment-tree-sha256 "$ENVIRONMENT_TREE_SHA256" \
  --canonical-shard-index 0 \
  --batch-size 4 \
  --max-new-tokens 768 \
  --seed 2026080818 \
  --output "$OUTPUT" \
  --report "$REPORT"
test -f "$OUTPUT" && test -f "$REPORT"
chmod a-w "$OUTPUT" "$REPORT"
