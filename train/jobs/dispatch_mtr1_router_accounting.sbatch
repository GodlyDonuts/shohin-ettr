#!/bin/bash
#SBATCH --job-name=mtr1-router-dispatch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/mtr1_router_dispatch_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/mtr1_router_dispatch_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 MODEL_MANIFEST_SHA256 DATA_ROOT TREATMENT_FIT \
  FIT_RECEIPT OUTPUT_ROOT DISPATCH_RECEIPT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
test ! -e "$DISPATCH_RECEIPT"; test ! -e "$OUTPUT_ROOT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
DATA=$DATA_ROOT/development.jsonl
DATA_REPORT=$DATA_ROOT/report.json
CHECKPOINT=$TREATMENT_FIT/checkpoint_0000256.pt
for path in "$DATA" "$DATA_REPORT" "$CHECKPOINT" "$TREATMENT_FIT/report.json"; do
  test -f "$path"
done
DATA_SHA256=$(sha256sum "$DATA" | cut -d' ' -f1)
DATA_REPORT_SHA256=$(sha256sum "$DATA_REPORT" | cut -d' ' -f1)
CHECKPOINT_SHA256=$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)
mkdir -p "$OUTPUT_ROOT" "$(dirname "$DISPATCH_RECEIPT")"
common="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
common+=",MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION"
common+=",MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256,MODEL_MANIFEST_SHA256=$MODEL_MANIFEST_SHA256"
common+=",DATA=$DATA,DATA_SHA256=$DATA_SHA256,DATA_REPORT=$DATA_REPORT"
common+=",DATA_REPORT_SHA256=$DATA_REPORT_SHA256"
unchanged_job=$(sbatch --parsable --exclude=evc33,evc38,evc46 \
  --export="$common,ARM=unchanged,OUTPUT=$OUTPUT_ROOT/unchanged.json" \
  "$RUNTIME/train/jobs/mtr1_router_accounting.sbatch")
unchanged_job=${unchanged_job%%;*}
treatment_job=$(sbatch --parsable --exclude=evc33,evc38,evc46 \
  --export="$common,ARM=treatment,ADAPTER_CHECKPOINT=$CHECKPOINT,ADAPTER_CHECKPOINT_SHA256=$CHECKPOINT_SHA256,OUTPUT=$OUTPUT_ROOT/treatment.json" \
  "$RUNTIME/train/jobs/mtr1_router_accounting.sbatch")
treatment_job=${treatment_job%%;*}
temporary=$DISPATCH_RECEIPT.partial.$$
cat >"$temporary" <<EOF
{"runtime_manifest_sha256":"$RUNTIME_MANIFEST_SHA256","status":"submitted","treatment_job":"$treatment_job","unchanged_job":"$unchanged_job"}
EOF
mv "$temporary" "$DISPATCH_RECEIPT"; chmod a-w "$DISPATCH_RECEIPT"
cat "$DISPATCH_RECEIPT"
