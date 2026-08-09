#!/bin/bash
#SBATCH --job-name=transfer-fit-dispatch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/transfer_fit_dispatch_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/transfer_fit_dispatch_%j.err

set -euo pipefail

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 MODEL_MANIFEST_SHA256 DATA_ROOT TREATMENT_OUTPUT \
  INDEPENDENT_OUTPUT RECEIPT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
test ! -e "$TREATMENT_OUTPUT"; test ! -e "$INDEPENDENT_OUTPUT"; test ! -e "$RECEIPT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$MODEL_ROOT/SHA256SUMS" | cut -d' ' -f1)" = "$MODEL_MANIFEST_SHA256"
(cd "$MODEL_ROOT" && sha256sum -c SHA256SUMS >/dev/null)

DATA=$DATA_ROOT/train.jsonl
DATA_REPORT=$DATA_ROOT/report.json
test -f "$DATA"; test -f "$DATA_REPORT"
DATA_SHA256=$(sha256sum "$DATA" | cut -d' ' -f1)
DATA_REPORT_SHA256=$(sha256sum "$DATA_REPORT" | cut -d' ' -f1)
python3 - "$DATA" "$DATA_REPORT" <<'PY'
import hashlib, json, sys
from pathlib import Path
data, report_path = map(Path, sys.argv[1:])
report=json.loads(report_path.read_text())
expected=report.get("outputs",{}).get("train",{})
if report.get("schema") != "shohin-idr1-revision-data-report-v1":
    raise SystemExit("revision data schema differs")
if expected.get("rows") != 9655 or expected.get("sha256") != hashlib.sha256(data.read_bytes()).hexdigest():
    raise SystemExit("revision data receipt differs")
PY

COMMON="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
COMMON+=",MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION"
COMMON+=",MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256,MODEL_MANIFEST_SHA256=$MODEL_MANIFEST_SHA256"
COMMON+=",DATA=$DATA,DATA_SHA256=$DATA_SHA256,DATA_REPORT=$DATA_REPORT"
COMMON+=",DATA_REPORT_SHA256=$DATA_REPORT_SHA256"
treatment_job=$(sbatch --parsable --exclude=evc33,evc38,evc46 \
  --export="$COMMON,OUTPUT=$TREATMENT_OUTPUT,MASK_INTERNAL_DRAFT=false" \
  "$RUNTIME/train/jobs/train_transfer_reviser.sbatch")
independent_job=$(sbatch --parsable --exclude=evc33,evc38,evc46 \
  --export="$COMMON,OUTPUT=$INDEPENDENT_OUTPUT,MASK_INTERNAL_DRAFT=true" \
  "$RUNTIME/train/jobs/train_transfer_reviser.sbatch")

mkdir -p "$(dirname "$RECEIPT")"
temporary=$RECEIPT.partial.$$
cat >"$temporary" <<EOF
{"data_report_sha256":"$DATA_REPORT_SHA256","data_sha256":"$DATA_SHA256","independent_job":"$independent_job","model_manifest_sha256":"$MODEL_MANIFEST_SHA256","runtime_manifest_sha256":"$RUNTIME_MANIFEST_SHA256","status":"submitted","treatment_job":"$treatment_job"}
EOF
mv "$temporary" "$RECEIPT"; chmod a-w "$RECEIPT"
printf 'treatment_job=%s independent_job=%s\n' "$treatment_job" "$independent_job"

