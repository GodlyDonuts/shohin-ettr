#!/bin/bash
#SBATCH --job-name=q36-mtr-final-custody
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_mtr_final_custody_%j.out
#SBATCH --error=logs/q36_mtr_final_custody_%j.err

set -euo pipefail
required=(RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE_COMMIT PYTHON PRECOMPUTE_CUSTODY SCORE_REPORT SCORE_CONSUMPTION SCHEDULER_ACCOUNTING EVIDENCE_MIRROR GRAPH_CONTRACT LEARNED_COMMIT_REPORT TRAINED_REVISION_REPORT UNCHANGED_REPORT SELF_REFINEMENT_REPORT DRAFT_HIDDEN_REPORT OUTPUT)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_verify_runtime
q36_require_authorization
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || q36_die "final custody output exists"
mkdir -p "$(dirname "$OUTPUT")" logs
q36_export_pythonpath
"$PYTHON" -P -s -B "$RUNTIME/pipeline/build_q36_mtr_custody.py" final \
  --precompute-custody "$PRECOMPUTE_CUSTODY" --score-report "$SCORE_REPORT" \
  --score-consumption "$SCORE_CONSUMPTION" --scheduler-accounting "$SCHEDULER_ACCOUNTING" \
  --evidence-mirror "$EVIDENCE_MIRROR" --graph-contract "$GRAPH_CONTRACT" \
  --arm-report "learned_commit=$LEARNED_COMMIT_REPORT" \
  --arm-report "trained_revision=$TRAINED_REVISION_REPORT" \
  --arm-report "unchanged=$UNCHANGED_REPORT" \
  --arm-report "self_refinement=$SELF_REFINEMENT_REPORT" \
  --arm-report "draft_hidden=$DRAFT_HIDDEN_REPORT" --output "$OUTPUT"
test -f "$OUTPUT" && chmod a-w "$OUTPUT"
