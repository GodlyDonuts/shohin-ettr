#!/bin/bash
#SBATCH --job-name=q36-mtr-final-gate
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_mtr_compare_%j.out
#SBATCH --error=logs/q36_mtr_compare_%j.err

set -euo pipefail
required=(RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE_COMMIT PYTHON GRAPH_CONTRACT FINAL_CUSTODY PRETERMINAL_EVIDENCE_MIRROR AUTHORIZED_EVIDENCE_ROOT TERMINAL_EVIDENCE_ROOT LEARNED_COMMIT_REPORT TRAINED_REVISION_REPORT UNCHANGED_REPORT SELF_REFINEMENT_REPORT DRAFT_HIDDEN_REPORT OUTPUT)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_verify_runtime
q36_require_authorization
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || q36_die "final comparison exists"
mkdir -p "$(dirname "$OUTPUT")" logs
q36_export_pythonpath
"$PYTHON" -P -s -B "$RUNTIME/pipeline/compare_q36_mtr.py" \
  --learned-commit-report "$LEARNED_COMMIT_REPORT" \
  --trained-revision-report "$TRAINED_REVISION_REPORT" \
  --unchanged-report "$UNCHANGED_REPORT" \
  --self-refinement-report "$SELF_REFINEMENT_REPORT" \
  --draft-hidden-report "$DRAFT_HIDDEN_REPORT" \
  --final-custody "$FINAL_CUSTODY" --graph-contract "$GRAPH_CONTRACT" \
  --output "$OUTPUT"
test -f "$OUTPUT" && chmod a-w "$OUTPUT"
[[ -d "$AUTHORIZED_EVIDENCE_ROOT" && ! -L "$AUTHORIZED_EVIDENCE_ROOT" ]] || q36_die "authorized evidence root differs"
"$PYTHON" -P -s -B "$RUNTIME/pipeline/seal_q36_mtr_terminal_evidence.py" \
  --graph-contract "$GRAPH_CONTRACT" \
  --preterminal-evidence "$PRETERMINAL_EVIDENCE_MIRROR" \
  --final-custody "$FINAL_CUSTODY" --final-result "$OUTPUT" \
  --authorized-root "$AUTHORIZED_EVIDENCE_ROOT" \
  --output-root "$TERMINAL_EVIDENCE_ROOT"
test -f "$TERMINAL_EVIDENCE_ROOT/manifest.json"
