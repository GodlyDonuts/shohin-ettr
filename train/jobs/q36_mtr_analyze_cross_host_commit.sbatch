#!/bin/bash
#SBATCH --job-name=q36-cross-host-analysis
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_cross_host_analysis_%j.out
#SBATCH --error=logs/q36_cross_host_analysis_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE_COMMIT PYTHON HOST SELECTIONS APPLICATION_REPORT SCORE OUTPUT CROSS_HOST_CONTRACT CROSS_HOST_CONTRACT_SHA256 RUN_ROOT; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_runtime
q36_verify_sha256 "$CROSS_HOST_CONTRACT" "$CROSS_HOST_CONTRACT_SHA256"
"$PYTHON" - "$CROSS_HOST_CONTRACT" "$HOST" "$RUN_ROOT" "$OUTPUT" "${REVISION_MARGIN_THRESHOLD:-0.0}" <<'PY'
import json
from pathlib import Path
import sys

contract = json.load(open(sys.argv[1], encoding="utf-8"))
root = Path(sys.argv[3]).resolve(strict=False)
target = Path(sys.argv[4]).resolve(strict=False)
threshold = float(sys.argv[5])
if (
    contract.get("schema") != "shohin-q36-cross-host-model-owned-commit-contract-v1"
    or contract.get("status") != "frozen_before_execution"
    or sys.argv[2] not in contract.get("hosts", {})
    or root in {Path("/"), Path.home().resolve()}
    or threshold
    != float(contract.get("selector", {}).get("revision_margin_threshold", 0.0))
):
    raise SystemExit("cross-host authorization differs")
try:
    relative = target.relative_to(root)
except ValueError as error:
    raise SystemExit("cross-host output escapes run root") from error
if not relative.parts:
    raise SystemExit("cross-host output equals run root")
PY
for path in "$SELECTIONS" "$APPLICATION_REPORT" "$SCORE"; do
  [[ -f "$path" && ! -L "$path" ]] || q36_die "cross-host analysis input differs"
done
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || q36_die "cross-host result exists"
mkdir -p "$(dirname "$OUTPUT")" logs
q36_export_pythonpath
"$PYTHON" -P -s -B "$RUNTIME/pipeline/analyze_q36_cross_host_commit.py" \
  --host "$HOST" \
  --selections "$SELECTIONS" \
  --application-report "$APPLICATION_REPORT" \
  --score "$SCORE" \
  --output "$OUTPUT" \
  --expected-revision-margin-threshold "${REVISION_MARGIN_THRESHOLD:-0.0}"
test -f "$OUTPUT"
chmod a-w "$OUTPUT"
