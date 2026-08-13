#!/bin/bash
# Read-only Q36 admission plus receipt publication; it cannot submit work.

set -euo pipefail
required=(RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE_COMMIT PYTHON REPOSITORY GRAPH_CONTRACT PLAN MODEL_ROOT MODEL_MANIFEST MODEL_MANIFEST_SHA256 MODEL_REVISION MODEL_CONFIG_SHA256 PAIRS MATH LOGIC_SCIENCE CODE B1 USER_NAME QUOTA_FILESYSTEM PREPARE_ROOT RUN_ROOT RUN_ID)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_verify_runtime
q36_verify_model
q36_verify_overlay "$Q36_BNB_ROOT" "$Q36_BNB_MANIFEST_SHA256"
q36_verify_overlay "$Q36_FAST_KERNEL_ROOT" "$Q36_FAST_KERNEL_MANIFEST_SHA256"
[[ "$PREPARE_ROOT" = /* && ! -e "$PREPARE_ROOT" && ! -L "$PREPARE_ROOT" ]] || q36_die "prepare root differs"
[[ "$RUN_ROOT" = /* && ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || q36_die "run root differs"
mkdir -m 700 "$PREPARE_ROOT"
freeze_prepare() {
  chmod -R a-w "$PREPARE_ROOT" 2>/dev/null || true
}
trap freeze_prepare EXIT
environment="$PREPARE_ROOT/environment.json"
sandbox="$PREPARE_ROOT/sandbox.json"
cluster="$PREPARE_ROOT/cluster_preflight.json"
authorization="$PREPARE_ROOT/phase_authorization.json"
export PYTHONDONTWRITEBYTECODE=1
q36_export_pythonpath
"$PYTHON" -P -s -B "$RUNTIME/pipeline/capture_q36_mtr_environment.py" \
  --runtime-root "$RUNTIME" --runtime-manifest-sha256 "$RUNTIME_MANIFEST_SHA256" \
  --output "$environment"
"$PYTHON" -P -s -B "$RUNTIME/train/pcf1_code_sandbox.py" qualify --output "$sandbox"
"$PYTHON" -P -s -B "$RUNTIME/pipeline/capture_q36_mtr_cluster_preflight.py" \
  --user "$USER_NAME" --filesystem "$QUOTA_FILESYSTEM" \
  --graph-contract "$GRAPH_CONTRACT" --plan "$PLAN" --output "$cluster"
"$PYTHON" -P -s -B "$RUNTIME/pipeline/authorize_q36_mtr_phase.py" \
  --run-id "$RUN_ID" --repository "$REPOSITORY" \
  --graph-contract "$GRAPH_CONTRACT" --plan "$PLAN" \
  --runtime-root "$RUNTIME" --runtime-manifest "$RUNTIME/SHA256SUMS" \
  --model-root "$MODEL_ROOT" --model-manifest "$MODEL_MANIFEST" \
  --environment-receipt "$environment" --sandbox-receipt "$sandbox" \
  --cluster-preflight "$cluster" --pairs "$PAIRS" --math "$MATH" \
  --logic-science "$LOGIC_SCIENCE" --code "$CODE" --b1 "$B1" \
  --run-root "$RUN_ROOT" --output "$authorization"
test -f "$authorization" && test ! -e "$RUN_ROOT"
printf '%s\n' "$authorization"
