#!/bin/bash
#SBATCH --job-name=q36-mtr-data
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_mtr_data_%j.out
#SBATCH --error=logs/q36_mtr_data_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE_COMMIT PYTHON SOURCE_ROOT DRAFTS DRAFT_REPORT ASSESSOR_RECEIPT OUTPUT PHASE_AUTHORIZATION PHASE_AUTHORIZATION_SHA256; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_verify_runtime
q36_require_authorization
DRAFTS_SHA256=$(q36_sha256 "$DRAFTS")
DRAFT_REPORT_SHA256=$(q36_sha256 "$DRAFT_REPORT")
ASSESSOR_RECEIPT_SHA256=$(q36_sha256 "$ASSESSOR_RECEIPT")
q36_verify_sha256 "$DRAFTS" "$DRAFTS_SHA256"
q36_verify_sha256 "$DRAFT_REPORT" "$DRAFT_REPORT_SHA256"
q36_verify_sha256 "$ASSESSOR_RECEIPT" "$ASSESSOR_RECEIPT_SHA256"
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || q36_die "materialized output exists"
"$PYTHON" "$RUNTIME/pipeline/build_q36_mtr_data.py" \
  --source-root "$SOURCE_ROOT" \
  --drafts "$DRAFTS" \
  --draft-report "$DRAFT_REPORT" \
  --assessor-receipt "$ASSESSOR_RECEIPT" \
  --output "$OUTPUT"
test -f "$OUTPUT/report.json"
chmod -R a-w "$OUTPUT"
