#!/bin/bash
#SBATCH --job-name=q36-mtr-commit
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc33,evc34,evc37,evc38,evc43,evc46,evc50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_mtr_commit_%j.out
#SBATCH --error=logs/q36_mtr_commit_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE_COMMIT PYTHON ALIGNED_CHECKPOINT PAIRS PAIRS_REPORT DEVELOPMENT_PAIRS DEVELOPMENT_PAIRS_REPORT OUTPUT ENVIRONMENT_RECEIPT ENVIRONMENT_RECEIPT_SHA256 ENVIRONMENT_TREE_SHA256; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_runtime
q36_require_authorization
q36_verify_model
q36_verify_environment
ALIGNED_CHECKPOINT_SHA256=$(q36_sha256 "$ALIGNED_CHECKPOINT")
PAIRS_SHA256=$(q36_sha256 "$PAIRS")
PAIRS_REPORT_SHA256=$(q36_sha256 "$PAIRS_REPORT")
DEVELOPMENT_PAIRS_SHA256=$(q36_sha256 "$DEVELOPMENT_PAIRS")
DEVELOPMENT_PAIRS_REPORT_SHA256=$(q36_sha256 "$DEVELOPMENT_PAIRS_REPORT")
q36_verify_sha256 "$ALIGNED_CHECKPOINT" "$ALIGNED_CHECKPOINT_SHA256"
q36_verify_sha256 "$PAIRS" "$PAIRS_SHA256"
q36_verify_sha256 "$PAIRS_REPORT" "$PAIRS_REPORT_SHA256"
q36_verify_sha256 "$DEVELOPMENT_PAIRS" "$DEVELOPMENT_PAIRS_SHA256"
q36_verify_sha256 "$DEVELOPMENT_PAIRS_REPORT" "$DEVELOPMENT_PAIRS_REPORT_SHA256"
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || q36_die "commit output exists"
model=$(q36_stage_model)
mkdir -p "$(dirname "$OUTPUT")" logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
q36_export_pythonpath
nvidia-smi
"$PYTHON" -P -s -B "$RUNTIME/train/hf_q36_mtr_train_commit.py" \
  --model-root "$model" \
  --model-source-root "$MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --model-loader causal \
  --adapter-checkpoint "$ALIGNED_CHECKPOINT" \
  --pairs "$PAIRS" \
  --pairs-report "$PAIRS_REPORT" \
  --development-pairs "$DEVELOPMENT_PAIRS" \
  --development-pairs-report "$DEVELOPMENT_PAIRS_REPORT" \
  --output "$OUTPUT" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --environment-receipt-sha256 "$ENVIRONMENT_RECEIPT_SHA256" \
  --environment-tree-sha256 "$ENVIRONMENT_TREE_SHA256" \
  --updates 128 --gradient-accumulation 8 --head-width 512 \
  --max-sequence-length 3072 --evaluation-batch-pairs 2 \
  --backbone-learning-rate 2e-6 --head-learning-rate 2e-4 \
  --tie-loss-weight 0.25 --weight-decay 0.01 --max-gradient-norm 1.0 \
  --seed 2026080822
test -f "$OUTPUT/commit.pt" && test -f "$OUTPUT/report.json"
test -f "$OUTPUT/development_selections.jsonl" && test -f "$OUTPUT/application_report.json"
chmod -R a-w "$OUTPUT"
