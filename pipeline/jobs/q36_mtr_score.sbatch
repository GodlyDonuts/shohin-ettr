#!/bin/bash
#SBATCH --job-name=q36-mtr-score
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_mtr_score_%j.out
#SBATCH --error=logs/q36_mtr_score_%j.err

set -euo pipefail
required=(RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE_COMMIT PYTHON DEVELOPMENT_DATA DATA_REPORT ASSESSOR_BOARD ASSESSOR_RECEIPT CANDIDATES_ROOT REVISION_REPORT REVISION_CANDIDATES UNCHANGED_REPORT UNCHANGED_CANDIDATES SELF_REFINEMENT_REPORT SELF_REFINEMENT_CANDIDATES DRAFT_HIDDEN_REPORT DRAFT_HIDDEN_CANDIDATES APPLICATION_REPORT SELECTIONS COMMIT_TRAINING_REPORT PRECOMPUTE_CUSTODY ENVIRONMENT_RECEIPT GRAPH_CONTRACT SCORE_AUTHORIZATION SANDBOX_RECEIPT_OUTPUT OUTPUT)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_verify_runtime
q36_require_authorization
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" && ! -e "$SANDBOX_RECEIPT_OUTPUT" && ! -L "$SANDBOX_RECEIPT_OUTPUT" ]] || q36_die "score output exists"
consumption="$(dirname "$OUTPUT")/$(basename "$OUTPUT").score-authorization-consumed.json"
terminal_failure="$(dirname "$OUTPUT")/$(basename "$OUTPUT").terminal-failure.json"
freeze_score_evidence() {
  for path in "$SANDBOX_RECEIPT_OUTPUT" "$consumption" "$terminal_failure" "$OUTPUT"; do
    [[ ! -e "$path" ]] || chmod -R a-w "$path" || true
  done
}
trap freeze_score_evidence EXIT
mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$SANDBOX_RECEIPT_OUTPUT")" logs
export PYTHONDONTWRITEBYTECODE=1
q36_export_pythonpath
"$PYTHON" -P -s -B "$RUNTIME/pipeline/score_q36_mtr.py" \
  --development-data "$DEVELOPMENT_DATA" --data-report "$DATA_REPORT" \
  --assessor-board "$ASSESSOR_BOARD" --assessor-receipt "$ASSESSOR_RECEIPT" \
  --candidates-root "$CANDIDATES_ROOT" \
  --revision-report "$REVISION_REPORT" --revision-candidates "$REVISION_CANDIDATES" \
  --unchanged-report "$UNCHANGED_REPORT" --unchanged-candidates "$UNCHANGED_CANDIDATES" \
  --self-refinement-report "$SELF_REFINEMENT_REPORT" --self-refinement-candidates "$SELF_REFINEMENT_CANDIDATES" \
  --draft-hidden-report "$DRAFT_HIDDEN_REPORT" --draft-hidden-candidates "$DRAFT_HIDDEN_CANDIDATES" \
  --application-report "$APPLICATION_REPORT" --selections "$SELECTIONS" \
  --commit-training-report "$COMMIT_TRAINING_REPORT" \
  --precompute-custody "$PRECOMPUTE_CUSTODY" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" --graph-contract "$GRAPH_CONTRACT" \
  --score-authorization "$SCORE_AUTHORIZATION" \
  --sandbox-receipt-output "$SANDBOX_RECEIPT_OUTPUT" --output "$OUTPUT"
test -f "$OUTPUT/report.json" && test -f "$OUTPUT/outcomes.jsonl" && test -f "$consumption"
