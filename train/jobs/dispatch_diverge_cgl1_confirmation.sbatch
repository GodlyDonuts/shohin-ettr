#!/bin/bash
#SBATCH -J dispatch-cgl1-confirm
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 00:05:00
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/dispatch_diverge_cgl1_confirmation_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE_CODE_ROOT=${BASE_CODE_ROOT:?set immutable qualified base code root}
OVERLAY_ROOT=${OVERLAY_ROOT:?set immutable CGL1 confirmation overlay}
ASSESSMENT=${ASSESSMENT:?set passing CGL1 assessment}
RESULTS_ROOT=${RESULTS_ROOT:?set immutable CGL1 development results root}
DATA=${DATA:?set fresh balanced CGL1 confirmation board}
DATA_SHA256=${DATA_SHA256:?set CGL1 confirmation board SHA-256}
OUTPUT=${OUTPUT:?set fresh CGL1 confirmation report}
RECEIPT=${RECEIPT:?set fresh CGL1 dispatch receipt}
PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603-r2/bin/python}
CONFIRM_JOB=$OVERLAY_ROOT/train/jobs/confirm_diverge_cgl1.sbatch

for path in "$BASE_CODE_ROOT" "$OVERLAY_ROOT" "$RESULTS_ROOT"; do
  test -d "$path" || { echo "missing CGL1 directory: $path" >&2; exit 2; }
done
for path in "$ASSESSMENT" "$DATA" "$CONFIRM_JOB"; do
  test -f "$path" || { echo "missing CGL1 input: $path" >&2; exit 2; }
done
test "$(sha256sum "$DATA" | awk '{print $1}')" = "$DATA_SHA256"
test ! -e "$OUTPUT" || { echo "CGL1 confirmation output already exists" >&2; exit 2; }
test ! -e "$RECEIPT" || { echo "CGL1 dispatch receipt already exists" >&2; exit 2; }

ASSESSMENT_SHA256=$(sha256sum "$ASSESSMENT" | awk '{print $1}')
SELECTED=$(
  "$PYTHON" - "$ASSESSMENT" <<'PY'
import json
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
selected = report.get("selected_treatment")
if report.get("status") != "pass" or selected not in {"shohin", "smollm2"}:
    raise SystemExit("CGL1 assessment did not admit confirmation")
print(selected)
PY
)

CHILD_JOB=$(
  sbatch --parsable \
    --export=ALL,BASE_CODE_ROOT="$BASE_CODE_ROOT",OVERLAY_ROOT="$OVERLAY_ROOT",ASSESSMENT="$ASSESSMENT",ASSESSMENT_SHA256="$ASSESSMENT_SHA256",RESULTS_ROOT="$RESULTS_ROOT",DATA="$DATA",DATA_SHA256="$DATA_SHA256",OUTPUT="$OUTPUT" \
    "$CONFIRM_JOB"
)

TEMPORARY=$RECEIPT.tmp.$SLURM_JOB_ID
"$PYTHON" - "$TEMPORARY" "$CHILD_JOB" "$SELECTED" "$ASSESSMENT" \
  "$ASSESSMENT_SHA256" "$DATA" "$DATA_SHA256" "$OUTPUT" <<'PY'
import json
import os
from pathlib import Path
import sys

path = Path(sys.argv[1])
payload = {
    "schema": "shohin-diverge-cgl1-confirmation-dispatch-v1",
    "child_job_id": sys.argv[2],
    "selected_treatment": sys.argv[3],
    "assessment": sys.argv[4],
    "assessment_sha256": sys.argv[5],
    "data": sys.argv[6],
    "data_sha256": sys.argv[7],
    "output": sys.argv[8],
}
with path.open("x", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
mv "$TEMPORARY" "$RECEIPT"
chmod a-w "$RECEIPT"
printf '[dispatch-cgl1-confirm] child=%s selected=%s assessment_sha256=%s\n' \
  "$CHILD_JOB" "$SELECTED" "$ASSESSMENT_SHA256"
