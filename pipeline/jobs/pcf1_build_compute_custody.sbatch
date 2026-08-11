#!/bin/bash
#SBATCH --job-name=pcf1-compute-custody
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc38,evc46
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/pcf1_compute_custody_%j.out
#SBATCH --error=logs/pcf1_compute_custody_%j.err

set -euo pipefail

: "${RUNTIME:?}" "${RUNTIME_MANIFEST_SHA256:?}"
test -f "$RUNTIME/SHA256SUMS"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
source "$RUNTIME/train/jobs/pcf1_common.sh"
pcf1_verify_runtime
for variable in RUN_ID DATA_CUSTODY MODEL_CUSTODY RUNTIME_CUSTODY SANDBOX_RECEIPT SCORE_SANDBOX_PROBE NORMALIZED_ROOT SCORE_CONSUMPTION DISPATCH_RECEIPT ACCOUNTING_RECEIPT OUTPUT; do
  pcf1_require "$variable"
done
for path in "$DATA_CUSTODY" "$MODEL_CUSTODY" "$RUNTIME_CUSTODY" "$SCORE_CONSUMPTION" "$SCORE_SANDBOX_PROBE" "$DISPATCH_RECEIPT" "$ACCOUNTING_RECEIPT"; do
  pcf1_require_safe_input "$path"
done
pcf1_require_dir "$NORMALIZED_ROOT"
pcf1_require_fresh "$OUTPUT"
python=$(pcf1_python)
sandbox_receipt_sha256=$(pcf1_validate_sandbox_receipt "$SANDBOX_RECEIPT")
pcf1_export_runtime
"$python" "$RUNTIME/pipeline/build_pcf1_custody.py" compute \
  --run-id "$RUN_ID" \
  --data-custody "$DATA_CUSTODY" \
  --model-custody "$MODEL_CUSTODY" \
  --runtime-custody "$RUNTIME_CUSTODY" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --environment-receipt-sha256 "$ENVIRONMENT_RECEIPT_SHA256" \
  --sandbox-receipt "$SANDBOX_RECEIPT" \
  --sandbox-receipt-sha256 "$sandbox_receipt_sha256" \
  --score-sandbox-probe "$SCORE_SANDBOX_PROBE" \
  --normalized-root "$NORMALIZED_ROOT" \
  --score-consumption "$SCORE_CONSUMPTION" \
  --dispatch-receipt "$DISPATCH_RECEIPT" \
  --accounting-receipt "$ACCOUNTING_RECEIPT" \
  --output "$OUTPUT"
pcf1_validate_json "$OUTPUT" shohin-pcf1-compute-custody-v1
chmod a-w "$OUTPUT"
