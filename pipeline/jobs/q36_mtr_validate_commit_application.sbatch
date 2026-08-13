#!/bin/bash
#SBATCH --job-name=q36-mtr-commit-validate
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_mtr_commit_validate_%j.out
#SBATCH --error=logs/q36_mtr_commit_validate_%j.err

set -euo pipefail
required=(RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE_COMMIT PYTHON COMMIT_CHECKPOINT COMMIT_TRAINING_REPORT DEVELOPMENT_PAIRS DEVELOPMENT_PAIRS_REPORT APPLICATION_REPORT SELECTIONS OUTPUT)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_verify_runtime
q36_require_authorization
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || q36_die "commit validation exists"
mkdir -p "$(dirname "$OUTPUT")" logs
q36_export_pythonpath
"$PYTHON" -P -s -B "$RUNTIME/pipeline/validate_q36_mtr_commit_application.py" \
  --commit-checkpoint "$COMMIT_CHECKPOINT" \
  --commit-training-report "$COMMIT_TRAINING_REPORT" \
  --development-pairs "$DEVELOPMENT_PAIRS" \
  --development-pairs-report "$DEVELOPMENT_PAIRS_REPORT" \
  --application-report "$APPLICATION_REPORT" --selections "$SELECTIONS" \
  --output "$OUTPUT"
test -f "$OUTPUT" && chmod a-w "$OUTPUT"
