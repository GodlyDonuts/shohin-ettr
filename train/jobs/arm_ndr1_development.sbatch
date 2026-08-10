#!/bin/bash
#SBATCH --job-name=ndr1-eval-arm
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/ndr1_eval_arm_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/ndr1_eval_arm_%j.err

set -euo pipefail

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 FIT_RECEIPT MODEL_ROOT \
  MODEL_REVISION MODEL_CONFIG_SHA256 MODEL_MANIFEST_SHA256 DEVELOPMENT \
  DEVELOPMENT_SHA256 DEVELOPMENT_REPORT DEVELOPMENT_REPORT_SHA256 \
  NDR_DATA_REPORT ALIGNED_TRAIN_ROOT SHUFFLED_TRAIN_ROOT OUTPUT_ROOT \
  RECEIPT ARM_RECEIPT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done

POLL_SECONDS=${POLL_SECONDS:-15}
WAIT_SECONDS=${WAIT_SECONDS:-25200}
SHARD_COUNT=${SHARD_COUNT:-4}
test "$POLL_SECONDS" -ge 1
test "$WAIT_SECONDS" -ge "$POLL_SECONDS"
test ! -e "$OUTPUT_ROOT"
test ! -e "$RECEIPT"
test ! -e "$ARM_RECEIPT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = \
  "$RUNTIME_MANIFEST_SHA256"

deadline=$((SECONDS + WAIT_SECONDS))
while [[ ! -s "$FIT_RECEIPT" ]]; do
  if (( SECONDS >= deadline )); then
    echo "timed out waiting for $FIT_RECEIPT" >&2
    exit 3
  fi
  sleep "$POLL_SECONDS"
done

readarray -t fit_values < <(
  python3 - "$FIT_RECEIPT" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
if value.get("status") != "submitted":
    raise SystemExit(2)
for key in ("aligned_job", "shuffled_job"):
    job = str(value.get(key, ""))
    if not job.isdigit():
        raise SystemExit(2)
    print(job)
PY
)
aligned_job=${fit_values[0]}
shuffled_job=${fit_values[1]}

test -s "$NDR_DATA_REPORT"
ndr_data_report_sha256=$(sha256sum "$NDR_DATA_REPORT" | cut -d' ' -f1)
exports="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
exports+=",MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION"
exports+=",MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256"
exports+=",MODEL_MANIFEST_SHA256=$MODEL_MANIFEST_SHA256"
exports+=",DEVELOPMENT=$DEVELOPMENT,DEVELOPMENT_SHA256=$DEVELOPMENT_SHA256"
exports+=",DEVELOPMENT_REPORT=$DEVELOPMENT_REPORT"
exports+=",DEVELOPMENT_REPORT_SHA256=$DEVELOPMENT_REPORT_SHA256"
exports+=",NDR_DATA_REPORT=$NDR_DATA_REPORT"
exports+=",NDR_DATA_REPORT_SHA256=$ndr_data_report_sha256"
exports+=",ALIGNED_TRAIN_ROOT=$ALIGNED_TRAIN_ROOT"
exports+=",SHUFFLED_TRAIN_ROOT=$SHUFFLED_TRAIN_ROOT"
exports+=",OUTPUT_ROOT=$OUTPUT_ROOT,RECEIPT=$RECEIPT"
exports+=",SHARD_COUNT=$SHARD_COUNT"

dispatcher=$(sbatch --parsable \
  --dependency=afterok:"$aligned_job":"$shuffled_job" \
  --export="$exports" "$RUNTIME/train/jobs/dispatch_ndr1_development.sbatch")

temporary=$ARM_RECEIPT.partial.$$
python3 - "$temporary" "$dispatcher" "$aligned_job" "$shuffled_job" \
  "$ndr_data_report_sha256" <<'PY'
import json
from pathlib import Path
import sys

path, dispatcher, aligned_job, shuffled_job, data_sha = sys.argv[1:]
Path(path).write_text(
    json.dumps(
        {
            "aligned_job": aligned_job,
            "development_dispatcher": dispatcher,
            "holdout_opened": False,
            "ndr_data_report_sha256": data_sha,
            "shuffled_job": shuffled_job,
            "status": "submitted",
        },
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
mv "$temporary" "$ARM_RECEIPT"
chmod a-w "$ARM_RECEIPT"
cat "$ARM_RECEIPT"
