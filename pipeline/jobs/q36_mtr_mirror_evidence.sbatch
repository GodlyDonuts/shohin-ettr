#!/bin/bash
#SBATCH --job-name=q36-mtr-evidence
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_mtr_evidence_%j.out
#SBATCH --error=logs/q36_mtr_evidence_%j.err

set -euo pipefail
required=(RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE_COMMIT PYTHON RUN_ID GRAPH_CONTRACT PRECOMPUTE_CUSTODY PRESCORE_ACCOUNTING SCORE_AUTHORIZATION SCORE_CONSUMPTION SCORE_REPORT SCORE_OUTCOMES SCORE_SANDBOX_RECEIPT SCHEDULER_ACCOUNTING PLAN DISPATCH_RECEIPT MODEL_MANIFEST AUTHORIZED_EVIDENCE_ROOT OUTPUT_ROOT LEARNED_COMMIT_ARM_REPORT TRAINED_REVISION_ARM_REPORT UNCHANGED_ARM_REPORT SELF_REFINEMENT_ARM_REPORT DRAFT_HIDDEN_ARM_REPORT)
artifacts=(ALIGNED_CHECKPOINT ALIGNED_REPORT APPLICATION_REPORT APPLICATION_VALIDATION ASSESSOR_RECEIPT CALIBRATION_PAIRS CALIBRATION_PAIRS_REPORT CALIBRATION_REVISION_CANDIDATES CALIBRATION_REVISION_REPORT CALIBRATION_UNCHANGED_CANDIDATES CALIBRATION_UNCHANGED_REPORT COMMIT_CHECKPOINT COMMIT_TRAINING_REPORT DATA_REPORT DEVELOPMENT_PAIRS DEVELOPMENT_PAIRS_REPORT DRAFT_HIDDEN_CANDIDATES DRAFT_HIDDEN_CHECKPOINT DRAFT_HIDDEN_EVALUATION_REPORT DRAFT_HIDDEN_REPORT DRAFT_REPORT DRAFTS ENVIRONMENT_RECEIPT FREEZE_REPORT LIVE_PREFLIGHT MECHANICS_REPORT OWNER_CHECKPOINT OWNER_REPORT REVISION_CANDIDATES REVISION_REPORT SELECTIONS SELF_REFINEMENT_CANDIDATES SELF_REFINEMENT_REPORT UNCHANGED_CANDIDATES UNCHANGED_REPORT)
for variable in "${required[@]}" "${artifacts[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_verify_runtime
q36_require_authorization
[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] || q36_die "evidence output exists"
args=()
for variable in "${artifacts[@]}"; do
  name=$(printf '%s' "$variable" | tr '[:upper:]' '[:lower:]')
  args+=(--precompute-artifact "$name=${!variable}")
done
[[ -d "$AUTHORIZED_EVIDENCE_ROOT" && ! -L "$AUTHORIZED_EVIDENCE_ROOT" ]] || q36_die "authorized evidence root differs"
mkdir -p logs
q36_export_pythonpath
"$PYTHON" -P -s -B "$RUNTIME/pipeline/mirror_q36_mtr_evidence.py" \
  --run-id "$RUN_ID" --source-commit "$SOURCE_COMMIT" \
  --graph-contract "$GRAPH_CONTRACT" --precompute-custody "$PRECOMPUTE_CUSTODY" \
  --prescore-accounting "$PRESCORE_ACCOUNTING" \
  --score-authorization "$SCORE_AUTHORIZATION" --score-consumption "$SCORE_CONSUMPTION" \
  --score-report "$SCORE_REPORT" --score-outcomes "$SCORE_OUTCOMES" \
  --score-sandbox-receipt "$SCORE_SANDBOX_RECEIPT" \
  --scheduler-accounting "$SCHEDULER_ACCOUNTING" --plan "$PLAN" \
  --dispatch-receipt "$DISPATCH_RECEIPT" --model-manifest "$MODEL_MANIFEST" \
  --runtime-manifest "$RUNTIME/SHA256SUMS" \
  --arm-report "learned_commit=$LEARNED_COMMIT_ARM_REPORT" \
  --arm-report "trained_revision=$TRAINED_REVISION_ARM_REPORT" \
  --arm-report "unchanged=$UNCHANGED_ARM_REPORT" \
  --arm-report "self_refinement=$SELF_REFINEMENT_ARM_REPORT" \
  --arm-report "draft_hidden=$DRAFT_HIDDEN_ARM_REPORT" \
  "${args[@]}" --authorized-root "$AUTHORIZED_EVIDENCE_ROOT" \
  --output-root "$OUTPUT_ROOT"
test -f "$OUTPUT_ROOT/manifest.json" && test -d "$OUTPUT_ROOT/artifacts"
