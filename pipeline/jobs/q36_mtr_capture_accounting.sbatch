#!/bin/bash
#SBATCH --job-name=q36-mtr-accounting
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_mtr_accounting_%j.out
#SBATCH --error=logs/q36_mtr_accounting_%j.err

set -euo pipefail
required=(RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE_COMMIT PYTHON PHASE RUN_ID GRAPH_CONTRACT PLAN DISPATCH_RECEIPT OUTPUT)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_verify_runtime
q36_require_authorization
[[ "$PHASE" = prescore || "$PHASE" = final ]] || q36_die "accounting phase differs"
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || q36_die "accounting output exists"
mkdir -p "$(dirname "$OUTPUT")" logs
q36_export_pythonpath
"$PYTHON" -P -s -B "$RUNTIME/pipeline/capture_q36_mtr_accounting.py" \
  --phase "$PHASE" --run-id "$RUN_ID" --graph-contract "$GRAPH_CONTRACT" \
  --plan "$PLAN" --dispatch-receipt "$DISPATCH_RECEIPT" --output "$OUTPUT"
test -f "$OUTPUT" && chmod a-w "$OUTPUT"
