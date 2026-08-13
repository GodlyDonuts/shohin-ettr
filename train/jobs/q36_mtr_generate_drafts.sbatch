#!/bin/bash
#SBATCH --job-name=q36-mtr-drafts
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc33,evc37,evc38,evc46
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=02:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_mtr_drafts_%j.out
#SBATCH --error=logs/q36_mtr_drafts_%j.err

set -euo pipefail

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE_COMMIT PYTHON TRAIN_SOURCE TRAIN_SOURCE_SHA256 DEVELOPMENT_SOURCE DEVELOPMENT_SOURCE_SHA256 FREEZE_REPORT FREEZE_REPORT_SHA256 OWNER_CHECKPOINT SHARD_ROOT SHARD_COUNT ENVIRONMENT_RECEIPT ENVIRONMENT_RECEIPT_SHA256 ENVIRONMENT_TREE_SHA256; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^([0-9]|1[0-5])$ ]] || { printf 'SLURM_ARRAY_TASK_ID differs\n' >&2; exit 2; }
SHARD_INDEX=$SLURM_ARRAY_TASK_ID
OUTPUT=$SHARD_ROOT/$(printf 'shard_%02d/candidates.jsonl' "$SHARD_INDEX")
REPORT=$SHARD_ROOT/$(printf 'shard_%02d/report.json' "$SHARD_INDEX")
export OUTPUT
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
OWNER_CHECKPOINT_SHA256=$(q36_sha256 "$OWNER_CHECKPOINT")
q36_verify_sha256 "$OWNER_CHECKPOINT" "$OWNER_CHECKPOINT_SHA256"
[[ "$SHARD_COUNT" == 16 ]] || q36_die "draft shard count differs"
[[ "$SHARD_INDEX" =~ ^([0-9]|1[0-5])$ ]] || q36_die "draft shard index differs"
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" && ! -e "$REPORT" && ! -L "$REPORT" ]] || \
  q36_die "draft output exists"
model=$(q36_stage_model)
mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$REPORT")" logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
q36_export_pythonpath
nvidia-smi
"$PYTHON" "$RUNTIME/train/hf_q36_mtr_generate_drafts.py" \
  --train-source "$TRAIN_SOURCE" \
  --development-source "$DEVELOPMENT_SOURCE" \
  --freeze-report "$FREEZE_REPORT" \
  --model-root "$model" \
  --model-revision "$MODEL_REVISION" \
  --owner-checkpoint "$OWNER_CHECKPOINT" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --environment-receipt-sha256 "$ENVIRONMENT_RECEIPT_SHA256" \
  --environment-tree-sha256 "$ENVIRONMENT_TREE_SHA256" \
  --output "$OUTPUT" \
  --report "$REPORT" \
  --shard-index "$SHARD_INDEX" \
  --shard-count 16 \
  --batch-size 4 \
  --max-new-tokens 768 \
  --seed 2026080818
test -f "$OUTPUT" && test -f "$REPORT"
chmod a-w "$OUTPUT" "$REPORT"
