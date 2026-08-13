#!/bin/bash
#SBATCH --job-name=q36-mtr-authorize
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_mtr_authorize_%j.out
#SBATCH --error=logs/q36_mtr_authorize_%j.err

set -euo pipefail
required=(RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE_COMMIT PYTHON PRECOMPUTE_CUSTODY GRAPH_CONTRACT SCORE_OUTPUT_ROOT OUTPUT)
artifacts=(APPLICATION_REPORT ASSESSOR_RECEIPT COMMIT_TRAINING_REPORT DATA_REPORT DEVELOPMENT_DATA DRAFT_HIDDEN_CANDIDATES DRAFT_HIDDEN_EVALUATION_REPORT ENVIRONMENT_RECEIPT PRESCORE_ACCOUNTING REVISION_CANDIDATES REVISION_REPORT SELECTIONS SELF_REFINEMENT_CANDIDATES SELF_REFINEMENT_REPORT UNCHANGED_CANDIDATES UNCHANGED_REPORT)
for variable in "${required[@]}" "${artifacts[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_verify_runtime
q36_require_authorization
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" && ! -e "$SCORE_OUTPUT_ROOT" && ! -L "$SCORE_OUTPUT_ROOT" ]] || q36_die "score authorization output exists"
args=()
for variable in "${artifacts[@]}"; do
  name=$(printf '%s' "$variable" | tr '[:upper:]' '[:lower:]')
  args+=(--artifact "$name=${!variable}")
done
mkdir -p "$(dirname "$OUTPUT")" logs
q36_export_pythonpath
"$PYTHON" -P -s -B "$RUNTIME/pipeline/build_q36_mtr_custody.py" authorize \
  --precompute-custody "$PRECOMPUTE_CUSTODY" \
  --graph-contract "$GRAPH_CONTRACT" "${args[@]}" \
  --score-output-root "$SCORE_OUTPUT_ROOT" --output "$OUTPUT"
test -f "$OUTPUT" && chmod a-w "$OUTPUT"
