#!/bin/bash
#SBATCH --job-name=pcf1-authorize-score
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc33,evc38,evc46
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/pcf1_authorize_score_%j.out
#SBATCH --error=logs/pcf1_authorize_score_%j.err

set -euo pipefail

: "${RUNTIME:?}" "${RUNTIME_MANIFEST_SHA256:?}"
test -f "$RUNTIME/SHA256SUMS"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
source "$RUNTIME/train/jobs/pcf1_common.sh"
pcf1_verify_runtime
for variable in RUN_ID CONFIRMATION_DATA CONFIRMATION_ASSESSOR_RECEIPT REVISION_REPORT REVISION_CANDIDATES UNCHANGED_REPORT UNCHANGED_CANDIDATES SELF_REFINEMENT_REPORT SELF_REFINEMENT_CANDIDATES CANDIDATES_ROOT CONFIRMATION_PAIRS CONFIRMATION_PAIR_REPORT CONFIRMATION_SELECTIONS COMMIT_APPLICATION_REPORT COMMIT_TRAINING_REPORT MECHANICS_REPORT SANDBOX_RECEIPT DATA_CUSTODY MODEL_CUSTODY RUNTIME_CUSTODY PRESCORE_DISPATCH_RECEIPT PRESCORE_ACCOUNTING_RECEIPT SCORE_OUTPUT_ROOT OUTPUT; do
  pcf1_require "$variable"
done
pcf1_reject_protected_path "$CANDIDATES_ROOT"
pcf1_require_dir "$CANDIDATES_ROOT"
for path in "$CONFIRMATION_DATA" "$CONFIRMATION_ASSESSOR_RECEIPT" "$REVISION_REPORT" "$REVISION_CANDIDATES" "$UNCHANGED_REPORT" "$UNCHANGED_CANDIDATES" "$SELF_REFINEMENT_REPORT" "$SELF_REFINEMENT_CANDIDATES" "$CONFIRMATION_PAIRS" "$CONFIRMATION_PAIR_REPORT" "$CONFIRMATION_SELECTIONS" "$COMMIT_APPLICATION_REPORT" "$COMMIT_TRAINING_REPORT" "$MECHANICS_REPORT" "$DATA_CUSTODY" "$MODEL_CUSTODY" "$RUNTIME_CUSTODY" "$PRESCORE_DISPATCH_RECEIPT" "$PRESCORE_ACCOUNTING_RECEIPT"; do
  pcf1_require_safe_input "$path"
done
pcf1_require_fresh "$OUTPUT"
python=$(pcf1_python)
sandbox_receipt_sha256=$(pcf1_validate_sandbox_receipt "$SANDBOX_RECEIPT")
pcf1_export_runtime
"$python" "$RUNTIME/pipeline/build_pcf1_custody.py" authorize-score \
  --run-id "$RUN_ID" \
  --confirmation-data "$CONFIRMATION_DATA" \
  --confirmation-assessor-receipt "$CONFIRMATION_ASSESSOR_RECEIPT" \
  --revision-report "$REVISION_REPORT" \
  --revision-candidates "$REVISION_CANDIDATES" \
  --unchanged-report "$UNCHANGED_REPORT" \
  --unchanged-candidates "$UNCHANGED_CANDIDATES" \
  --self-refinement-report "$SELF_REFINEMENT_REPORT" \
  --self-refinement-candidates "$SELF_REFINEMENT_CANDIDATES" \
  --candidates-root "$CANDIDATES_ROOT" \
  --confirmation-pairs "$CONFIRMATION_PAIRS" \
  --confirmation-pair-report "$CONFIRMATION_PAIR_REPORT" \
  --confirmation-selections "$CONFIRMATION_SELECTIONS" \
  --commit-application-report "$COMMIT_APPLICATION_REPORT" \
  --commit-training-report "$COMMIT_TRAINING_REPORT" \
  --mechanics-report "$MECHANICS_REPORT" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --environment-receipt-sha256 "$ENVIRONMENT_RECEIPT_SHA256" \
  --sandbox-receipt "$SANDBOX_RECEIPT" \
  --sandbox-receipt-sha256 "$sandbox_receipt_sha256" \
  --data-custody "$DATA_CUSTODY" \
  --model-custody "$MODEL_CUSTODY" \
  --runtime-custody "$RUNTIME_CUSTODY" \
  --prescore-dispatch-receipt "$PRESCORE_DISPATCH_RECEIPT" \
  --prescore-accounting-receipt "$PRESCORE_ACCOUNTING_RECEIPT" \
  --score-output-root "$SCORE_OUTPUT_ROOT" \
  --output "$OUTPUT"
pcf1_validate_json "$OUTPUT" shohin-pcf1-score-authorization-v1
chmod a-w "$OUTPUT"
