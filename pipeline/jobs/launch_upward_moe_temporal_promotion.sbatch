#!/bin/bash
#SBATCH --job-name=q36-upward-moe-launch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_upward_moe_launch_%j.out
#SBATCH --error=logs/q36_upward_moe_launch_%j.err

set -euo pipefail

required=(
  PYTHON RUNTIME RUNTIME_MANIFEST_SHA256 PROMOTION RUN_ROOT AUTOMATION_ROOT
)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || {
    printf '%s is required\n' "$variable" >&2
    exit 2
  }
done
[[ -x "$PYTHON" ]]
[[ -d "$RUNTIME" && ! -L "$RUNTIME" ]]
[[ -f "$PROMOTION" && ! -L "$PROMOTION" ]]
[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]]
[[ ! -e "$AUTOMATION_ROOT" && ! -L "$AUTOMATION_ROOT" ]]

source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
finalize() {
  status=$?
  q36_cleanup_local_tmp
  if [[ -d "$AUTOMATION_ROOT" && ! -L "$AUTOMATION_ROOT" ]]; then
    chmod -R a-w "$AUTOMATION_ROOT"
  fi
  exit "$status"
}
trap finalize EXIT
q36_verify_runtime
q36_export_pythonpath
mkdir -p logs

export PATH=/apps/slurm/current/bin:/usr/bin:/bin
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$SLURM_TMPDIR"
"$PYTHON" -P -s -B \
  "$RUNTIME/pipeline/launch_upward_moe_temporal_promotion.py" \
  --promotion "$PROMOTION" \
  --runtime "$RUNTIME" \
  --runtime-manifest-sha256 "$RUNTIME_MANIFEST_SHA256" \
  --python "$PYTHON" \
  --run-root "$RUN_ROOT" \
  --claim "$AUTOMATION_ROOT/claim.json" \
  --dispatch-receipt "$AUTOMATION_ROOT/dispatch.json" \
  --launch-receipt "$AUTOMATION_ROOT/launch.json" \
  --failure-receipt "$AUTOMATION_ROOT/terminal_failure.json" \
  --submit

[[ -d "$AUTOMATION_ROOT" && ! -L "$AUTOMATION_ROOT" ]]
