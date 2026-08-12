#!/bin/bash
#SBATCH --job-name=pcf1-confirmation-pairs
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc33,evc37,evc38,evc46
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/pcf1_confirmation_pairs_%j.out
#SBATCH --error=logs/pcf1_confirmation_pairs_%j.err

set -euo pipefail

: "${RUNTIME:?}" "${RUNTIME_MANIFEST_SHA256:?}"
test -f "$RUNTIME/SHA256SUMS"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
source "$RUNTIME/train/jobs/pcf1_common.sh"
pcf1_verify_runtime
for variable in CONFIRMATION_DATA REVISION_REPORT REVISION_CANDIDATES UNCHANGED_REPORT UNCHANGED_CANDIDATES CANDIDATES_ROOT OUTPUT REPORT; do
  pcf1_require "$variable"
done
pcf1_reject_protected_path "$CANDIDATES_ROOT"
pcf1_require_dir "$CANDIDATES_ROOT"
pcf1_require_safe_input "$CONFIRMATION_DATA"
pcf1_require_safe_input "$REVISION_REPORT"
pcf1_require_safe_input "$REVISION_CANDIDATES"
pcf1_require_safe_input "$UNCHANGED_REPORT"
pcf1_require_safe_input "$UNCHANGED_CANDIDATES"
pcf1_require_fresh "$OUTPUT"
pcf1_require_fresh "$REPORT"
python=$(pcf1_python)
pcf1_export_runtime
"$python" "$RUNTIME/pipeline/build_pcf1_confirmation_pairs.py" \
  --confirmation-data "$CONFIRMATION_DATA" \
  --revision-report "$REVISION_REPORT" \
  --revision-candidates "$REVISION_CANDIDATES" \
  --unchanged-report "$UNCHANGED_REPORT" \
  --unchanged-candidates "$UNCHANGED_CANDIDATES" \
  --candidates-root "$CANDIDATES_ROOT" \
  --output "$OUTPUT" \
  --report "$REPORT"
pcf1_validate_json "$REPORT" shohin-pcf1-confirmation-pair-report-v1
chmod a-w "$OUTPUT" "$REPORT"
