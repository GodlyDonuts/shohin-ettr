#!/bin/bash
#SBATCH --job-name=ndr1-fit-arm
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/ndr1_fit_arm_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/ndr1_fit_arm_%j.err

set -euo pipefail

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 DRAFT_DISPATCH_RECEIPT \
  MODEL_ROOT MODEL_REVISION MODEL_CONFIG_SHA256 WARM_START WARM_START_SHA256 \
  DATA_ROOT OUTPUT_ROOT FIT_RECEIPT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
test -f "$DRAFT_DISPATCH_RECEIPT"
test ! -e "$OUTPUT_ROOT"
test ! -e "$FIT_RECEIPT"

merge_job=$(
  python3 - "$DRAFT_DISPATCH_RECEIPT" <<'PY'
import json,sys
value=json.load(open(sys.argv[1]))
job=str(value.get("merge_job", ""))
if value.get("status")!="submitted" or not job.isdigit(): raise SystemExit(2)
print(job)
PY
)
exports="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
exports+=",MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION"
exports+=",MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256"
exports+=",WARM_START=$WARM_START,WARM_START_SHA256=$WARM_START_SHA256"
exports+=",DATA_ROOT=$DATA_ROOT,OUTPUT_ROOT=$OUTPUT_ROOT,RECEIPT=$FIT_RECEIPT"
dispatcher=$(sbatch --parsable --dependency=afterok:"$merge_job" \
  --export="$exports" "$RUNTIME/train/jobs/dispatch_ndr1_fit.sbatch")
printf 'FIT_DISPATCH=%s\nDATA_MERGE=%s\n' "$dispatcher" "$merge_job"
