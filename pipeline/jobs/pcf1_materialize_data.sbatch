#!/bin/bash
#SBATCH --job-name=pcf1-materialize
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc38,evc46
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/pcf1_materialize_%j.out
#SBATCH --error=logs/pcf1_materialize_%j.err

set -euo pipefail

: "${RUNTIME:?}" "${RUNTIME_MANIFEST_SHA256:?}"
test -f "$RUNTIME/SHA256SUMS"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
source "$RUNTIME/train/jobs/pcf1_common.sh"
pcf1_verify_runtime
pcf1_bind_prepared_inputs
for variable in DRAFTS DRAFTS_REPORT ASSESSOR_RECEIPT OUTPUT; do
  pcf1_require "$variable"
done
pcf1_reject_protected_path "$SOURCE_ROOT"
pcf1_require_file "$SOURCE_ROOT/report.json"
pcf1_require_safe_input "$DRAFTS"
pcf1_require_safe_input "$DRAFTS_REPORT"
pcf1_validate_json "$DRAFTS_REPORT" shohin-pcf1-merged-drafts-v1
pcf1_require_fresh "$OUTPUT"
python=$(pcf1_python)
pcf1_require_safe_input "$ASSESSOR_RECEIPT"
"$python" - "$DRAFTS" "$DRAFTS_REPORT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

drafts, report_path = map(Path, sys.argv[1:])
report = json.loads(report_path.read_text(encoding="utf-8"))
if report.get("output_sha256") != hashlib.sha256(drafts.read_bytes()).hexdigest():
    raise SystemExit("PCF1 merged draft binding differs")
PY
pcf1_export_runtime
"$python" "$RUNTIME/pipeline/build_pcf1_data.py" materialize \
  --source-root "$SOURCE_ROOT" \
  --drafts "$DRAFTS" \
  --assessor-receipt "$ASSESSOR_RECEIPT" \
  --output "$OUTPUT"
pcf1_validate_json "$OUTPUT/materialization_report.json" shohin-pcf1-data-report-v1
pcf1_freeze_tree "$OUTPUT"
