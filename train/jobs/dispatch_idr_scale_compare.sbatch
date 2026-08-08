#!/bin/bash
#SBATCH --job-name=idr-scale-compare-dispatch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/idr_scale_compare_dispatch_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/idr_scale_compare_dispatch_%j.err

set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

: "${RUNTIME:?RUNTIME is required}"
: "${RUNTIME_MANIFEST_SHA256:?RUNTIME_MANIFEST_SHA256 is required}"
: "${TRAINED_DISPATCH_RECEIPT:?TRAINED_DISPATCH_RECEIPT is required}"
: "${CONTROL_DISPATCH_RECEIPT:?CONTROL_DISPATCH_RECEIPT is required}"
: "${TRAINED_EVAL_ROOT:?TRAINED_EVAL_ROOT is required}"
: "${CONTROL_EVAL_ROOT:?CONTROL_EVAL_ROOT is required}"
: "${COMPARISON_OUTPUT:?COMPARISON_OUTPUT is required}"
: "${DISPATCH_RECEIPT:?DISPATCH_RECEIPT is required}"

test -x "$PYTHON"
test -f "$RUNTIME/train/jobs/compare_idr_scale.sbatch"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test -f "$TRAINED_DISPATCH_RECEIPT"
test -f "$CONTROL_DISPATCH_RECEIPT"
test ! -e "$DISPATCH_RECEIPT"
test ! -e "$COMPARISON_OUTPUT"

dependency=$(
  "$PYTHON" - "$TRAINED_DISPATCH_RECEIPT" "$CONTROL_DISPATCH_RECEIPT" <<'PY'
import json
from pathlib import Path
import re
import sys

jobs = []
for raw in sys.argv[1:]:
    receipt = json.loads(Path(raw).read_text(encoding="utf-8"))
    if receipt.get("status") != "submitted" or receipt.get("shard_count") < 2:
        raise SystemExit("IDR scale evaluation dispatch receipt differs")
    merges = receipt.get("merge_jobs")
    if not isinstance(merges, list) or len(merges) != 2:
        raise SystemExit("IDR scale merge-job inventory differs")
    if any(not re.fullmatch(r"[0-9]+", str(job)) for job in merges):
        raise SystemExit("IDR scale merge job ID is invalid")
    jobs.extend(str(job) for job in merges)
if len(set(jobs)) != 4:
    raise SystemExit("IDR scale merge jobs are duplicated")
print(":".join(jobs))
PY
)

exports="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
exports+=",TRAINED_EVAL_ROOT=$TRAINED_EVAL_ROOT,CONTROL_EVAL_ROOT=$CONTROL_EVAL_ROOT"
exports+=",COMPARISON_OUTPUT=$COMPARISON_OUTPUT"
compare_job=$(sbatch --parsable --dependency=afterok:"$dependency" \
  --export="$exports" "$RUNTIME/train/jobs/compare_idr_scale.sbatch")

mkdir -p "$(dirname "$DISPATCH_RECEIPT")"
temporary=$DISPATCH_RECEIPT.partial.$$
"$PYTHON" - "$temporary" "$compare_job" "$dependency" <<'PY'
import json
from pathlib import Path
import sys

path, compare_job, dependency = sys.argv[1:]
Path(path).write_text(
    json.dumps(
        {
            "status": "submitted",
            "compare_job": compare_job,
            "merge_dependencies": dependency.split(":"),
        },
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
mv "$temporary" "$DISPATCH_RECEIPT"
chmod a-w "$DISPATCH_RECEIPT"
printf 'compare_job=%s merge_dependencies=%s\n' "$compare_job" "$dependency"
