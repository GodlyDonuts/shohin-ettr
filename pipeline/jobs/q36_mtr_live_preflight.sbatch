#!/bin/bash
#SBATCH --job-name=q36-mtr-live-preflight
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_mtr_live_preflight_%j.out
#SBATCH --error=logs/q36_mtr_live_preflight_%j.err

set -euo pipefail
required=(RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE_COMMIT PYTHON RUN_ID PHASE_AUTHORIZATION PHASE_AUTHORIZATION_SHA256 GRAPH_CONTRACT PLAN ENVIRONMENT_RECEIPT ENVIRONMENT_RECEIPT_SHA256 ENVIRONMENT_TREE_SHA256 SANDBOX_RECEIPT CLUSTER_PREFLIGHT MODEL_ROOT MODEL_MANIFEST MODEL_MANIFEST_SHA256 MODEL_REVISION MODEL_CONFIG_SHA256 OUTPUT)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_verify_runtime
q36_require_authorization
q36_verify_model
q36_verify_environment
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || q36_die "live preflight output exists"
mkdir -p "$(dirname "$OUTPUT")" logs
q36_export_pythonpath
"$PYTHON" -P -s -B "$RUNTIME/pipeline/validate_q36_mtr_live_preflight.py" \
  --run-id "$RUN_ID" --phase-authorization "$PHASE_AUTHORIZATION" \
  --graph-contract "$GRAPH_CONTRACT" --plan "$PLAN" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --sandbox-receipt "$SANDBOX_RECEIPT" --cluster-preflight "$CLUSTER_PREFLIGHT" \
  --output "$OUTPUT"
test -f "$OUTPUT" && chmod a-w "$OUTPUT"
