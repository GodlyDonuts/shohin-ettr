#!/bin/bash
#SBATCH --job-name=pcf1-commit-score
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc33,evc37,evc38,evc46
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pcf1_commit_score_%j.out
#SBATCH --error=logs/pcf1_commit_score_%j.err

set -euo pipefail

: "${RUNTIME:?}" "${RUNTIME_MANIFEST_SHA256:?}"
test -f "$RUNTIME/SHA256SUMS"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
source "$RUNTIME/train/jobs/pcf1_common.sh"
pcf1_verify_runtime
for variable in CONFIRMATION_DATA CONFIRMATION_ASSESSORS CONFIRMATION_ASSESSOR_RECEIPT DATA_REPORT REVISION_REPORT REVISION_CANDIDATES UNCHANGED_REPORT UNCHANGED_CANDIDATES SELF_REFINEMENT_REPORT SELF_REFINEMENT_CANDIDATES CANDIDATES_ROOT CONFIRMATION_PAIRS CONFIRMATION_PAIRS_REPORT SELECTIONS APPLICATION_REPORT TRAINING_REPORT MECHANICS_REPORT SANDBOX_RECEIPT DATA_CUSTODY MODEL_CUSTODY RUNTIME_CUSTODY PRESCORE_DISPATCH_RECEIPT PRESCORE_ACCOUNTING_RECEIPT PRESCORE_AUTHORIZATION OUTPUT_ROOT; do
  pcf1_require "$variable"
done
pcf1_reject_protected_path "$CANDIDATES_ROOT"
pcf1_require_dir "$CANDIDATES_ROOT"
for path in "$CONFIRMATION_DATA" "$CONFIRMATION_ASSESSORS" "$CONFIRMATION_ASSESSOR_RECEIPT" "$DATA_REPORT" "$REVISION_REPORT" "$REVISION_CANDIDATES" "$UNCHANGED_REPORT" "$UNCHANGED_CANDIDATES" "$SELF_REFINEMENT_REPORT" "$SELF_REFINEMENT_CANDIDATES" "$CONFIRMATION_PAIRS" "$CONFIRMATION_PAIRS_REPORT" "$SELECTIONS" "$APPLICATION_REPORT" "$TRAINING_REPORT" "$MECHANICS_REPORT" "$SANDBOX_RECEIPT" "$DATA_CUSTODY" "$MODEL_CUSTODY" "$RUNTIME_CUSTODY" "$PRESCORE_DISPATCH_RECEIPT" "$PRESCORE_ACCOUNTING_RECEIPT" "$PRESCORE_AUTHORIZATION"; do
  pcf1_require_safe_input "$path"
done
pcf1_require_fresh "$OUTPUT_ROOT"
score_consumption=${OUTPUT_ROOT}.score-authorization-consumed.json
sandbox_probe=${OUTPUT_ROOT}.sandbox-probe.json
terminal_failure=${OUTPUT_ROOT}.terminal-failure.json
pcf1_require_fresh "$score_consumption"
pcf1_require_fresh "$sandbox_probe"
pcf1_require_fresh "$terminal_failure"
python=$(pcf1_python)
sandbox_receipt_sha256=$(pcf1_validate_sandbox_receipt "$SANDBOX_RECEIPT")
"$python" - "$MECHANICS_REPORT" "$SANDBOX_RECEIPT" "$sandbox_receipt_sha256" <<'PY'
import json
from pathlib import Path
import sys

mechanics_path, sandbox_path = map(Path, sys.argv[1:3])
expected_sha256 = sys.argv[3]
mechanics = json.loads(mechanics_path.read_text(encoding="utf-8"))
if (
    Path(str(mechanics.get("sandbox_receipt", ""))).resolve()
    != sandbox_path.resolve()
    or mechanics.get("sandbox_receipt_sha256") != expected_sha256
    or mechanics.get("sandbox_isolation_passed") is not True
):
    raise SystemExit("PCF1 score sandbox admission differs")
PY
pcf1_initialize_scratch
freeze_score_claims() {
  local status=$?
  trap - EXIT
  [[ ! -d "$OUTPUT_ROOT" || -L "$OUTPUT_ROOT" ]] || pcf1_freeze_tree "$OUTPUT_ROOT"
  [[ ! -f "$sandbox_probe" ]] || chmod a-w "$sandbox_probe"
  [[ ! -f "$score_consumption" ]] || chmod a-w "$score_consumption"
  [[ ! -f "$terminal_failure" ]] || chmod a-w "$terminal_failure"
  if ! pcf1_cleanup_scratch; then
    [[ "$status" -ne 0 ]] || status=1
  fi
  exit "$status"
}
trap freeze_score_claims EXIT
pcf1_export_runtime
"$python" "$RUNTIME/pipeline/score_pcf1_commit.py" \
  --confirmation-data "$CONFIRMATION_DATA" \
  --confirmation-assessors "$CONFIRMATION_ASSESSORS" \
  --confirmation-assessor-receipt "$CONFIRMATION_ASSESSOR_RECEIPT" \
  --data-report "$DATA_REPORT" \
  --revision-report "$REVISION_REPORT" \
  --revision-candidates "$REVISION_CANDIDATES" \
  --unchanged-report "$UNCHANGED_REPORT" \
  --unchanged-candidates "$UNCHANGED_CANDIDATES" \
  --self-refinement-report "$SELF_REFINEMENT_REPORT" \
  --self-refinement-candidates "$SELF_REFINEMENT_CANDIDATES" \
  --candidates-root "$CANDIDATES_ROOT" \
  --confirmation-pairs "$CONFIRMATION_PAIRS" \
  --confirmation-pairs-report "$CONFIRMATION_PAIRS_REPORT" \
  --selections "$SELECTIONS" \
  --application-report "$APPLICATION_REPORT" \
  --training-report "$TRAINING_REPORT" \
  --mechanics-report "$MECHANICS_REPORT" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --environment-receipt-sha256 "$ENVIRONMENT_RECEIPT_SHA256" \
  --data-custody "$DATA_CUSTODY" \
  --model-custody "$MODEL_CUSTODY" \
  --runtime-custody "$RUNTIME_CUSTODY" \
  --prescore-dispatch-receipt "$PRESCORE_DISPATCH_RECEIPT" \
  --prescore-accounting-receipt "$PRESCORE_ACCOUNTING_RECEIPT" \
  --prescore-authorization "$PRESCORE_AUTHORIZATION" \
  --sandbox-probe-output "$sandbox_probe" \
  --output-root "$OUTPUT_ROOT"
pcf1_validate_json "$OUTPUT_ROOT/report.json" shohin-pcf1-commit-result-v1
pcf1_validate_json "$score_consumption" shohin-pcf1-score-consumption-v1
score_sandbox_receipt_sha256=$(pcf1_validate_sandbox_receipt "$sandbox_probe")
"$python" - "$OUTPUT_ROOT/report.json" "$sandbox_probe" "$score_sandbox_receipt_sha256" <<'PY'
import json
from pathlib import Path
import sys

report_path, sandbox_path = map(Path, sys.argv[1:3])
sandbox_receipt_sha256 = sys.argv[3]
report = json.loads(report_path.read_text(encoding="utf-8"))
sandbox = json.loads(sandbox_path.read_text(encoding="utf-8"))
if (
    Path(str(report.get("sandbox_receipt", ""))).resolve()
    != sandbox_path.resolve()
    or report.get("sandbox_receipt_sha256") != sandbox_receipt_sha256
    or report.get("code_sandbox_probe_sha256") != sandbox_receipt_sha256
    or report.get("code_sandbox_probe_result_sha256")
    != sandbox.get("probe_sha256")
    or report.get("code_sandbox_probe_passed") is not True
):
    raise SystemExit("PCF1 final score sandbox binding differs")
PY
pcf1_freeze_tree "$OUTPUT_ROOT"
chmod a-w "$score_consumption"
chmod a-w "$sandbox_probe"
pcf1_cleanup_scratch
trap - EXIT
