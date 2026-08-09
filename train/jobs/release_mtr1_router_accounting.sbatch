#!/bin/bash
#SBATCH --job-name=mtr1-router-release
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/mtr1_router_release_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/mtr1_router_release_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 MODEL_MANIFEST_SHA256 DATA_ROOT TREATMENT_FIT \
  FIT_RECEIPT OUTPUT_ROOT DISPATCH_RECEIPT RELEASE_RECEIPT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
test ! -e "$RELEASE_RECEIPT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
read -r treatment_job < <(python3 - "$FIT_RECEIPT" <<'PY'
import json, sys
report=json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "submitted":
    raise SystemExit("fit dispatch receipt differs")
print(report["treatment_job"])
PY
)
exports="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
exports+=",MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION"
exports+=",MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256,MODEL_MANIFEST_SHA256=$MODEL_MANIFEST_SHA256"
exports+=",DATA_ROOT=$DATA_ROOT,TREATMENT_FIT=$TREATMENT_FIT,FIT_RECEIPT=$FIT_RECEIPT"
exports+=",OUTPUT_ROOT=$OUTPUT_ROOT,DISPATCH_RECEIPT=$DISPATCH_RECEIPT"
dispatch_job=$(sbatch --parsable --dependency=afterok:"$treatment_job" \
  --export="$exports" "$RUNTIME/train/jobs/dispatch_mtr1_router_accounting.sbatch")
dispatch_job=${dispatch_job%%;*}
mkdir -p "$(dirname "$RELEASE_RECEIPT")"

temporary=$RELEASE_RECEIPT.partial.$$
cat >"$temporary" <<EOF
{"accounting_dispatch_job":"$dispatch_job","runtime_manifest_sha256":"$RUNTIME_MANIFEST_SHA256","status":"submitted","treatment_fit_job":"$treatment_job"}
EOF
mv "$temporary" "$RELEASE_RECEIPT"; chmod a-w "$RELEASE_RECEIPT"
cat "$RELEASE_RECEIPT"
