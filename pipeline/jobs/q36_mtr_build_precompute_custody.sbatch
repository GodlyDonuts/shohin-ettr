#!/bin/bash
#SBATCH --job-name=q36-mtr-precompute
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_mtr_precompute_%j.out
#SBATCH --error=logs/q36_mtr_precompute_%j.err

set -euo pipefail
required=(RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE_COMMIT PYTHON RUN_ID GRAPH_CONTRACT OUTPUT MODEL_ROOT MODEL_MANIFEST MODEL_MANIFEST_SHA256 MODEL_REVISION MODEL_CONFIG_SHA256)
artifacts=(ALIGNED_CHECKPOINT ALIGNED_REPORT APPLICATION_REPORT APPLICATION_VALIDATION ASSESSOR_RECEIPT CALIBRATION_DATA CALIBRATION_PAIRS CALIBRATION_PAIRS_REPORT CALIBRATION_REVISION_CANDIDATES CALIBRATION_REVISION_REPORT CALIBRATION_UNCHANGED_CANDIDATES CALIBRATION_UNCHANGED_REPORT COMMIT_CHECKPOINT COMMIT_TRAINING_REPORT DATA_REPORT DEVELOPMENT_DATA DEVELOPMENT_PAIRS DEVELOPMENT_PAIRS_REPORT DEVELOPMENT_SOURCES DRAFT_HIDDEN_CANDIDATES DRAFT_HIDDEN_CHECKPOINT DRAFT_HIDDEN_EVALUATION_REPORT DRAFT_HIDDEN_REPORT DRAFT_REPORT DRAFTS ENVIRONMENT_RECEIPT FREEZE_REPORT MECHANICS_REPORT OWNER_CHECKPOINT OWNER_DATA OWNER_REPORT REVISION_CANDIDATES REVISION_REPORT REVISION_TRAINING_DATA SELECTIONS SELF_REFINEMENT_CANDIDATES SELF_REFINEMENT_REPORT TRAIN_SOURCES UNCHANGED_CANDIDATES UNCHANGED_REPORT)
for variable in "${required[@]}" "${artifacts[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_verify_runtime
q36_require_authorization
q36_verify_model
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || q36_die "precompute output exists"
args=()
for variable in "${artifacts[@]}"; do
  name=$(printf '%s' "$variable" | tr '[:upper:]' '[:lower:]')
  args+=(--artifact "$name=${!variable}")
done
mkdir -p "$(dirname "$OUTPUT")" logs
q36_export_pythonpath
"$PYTHON" -P -s -B "$RUNTIME/pipeline/build_q36_mtr_custody.py" precompute \
  --run-id "$RUN_ID" --graph-contract "$GRAPH_CONTRACT" \
  --runtime-root "$RUNTIME" --runtime-manifest "$RUNTIME/SHA256SUMS" \
  --model-root "$MODEL_ROOT" --model-manifest "$MODEL_MANIFEST" \
  "${args[@]}" --output "$OUTPUT"
test -f "$OUTPUT" && chmod a-w "$OUTPUT"
