#!/bin/bash
#SBATCH --job-name=q36-mtr-environment
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_mtr_environment_%j.out
#SBATCH --error=logs/q36_mtr_environment_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE_COMMIT PYTHON OUTPUT; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_verify_runtime
q36_verify_overlay "$Q36_BNB_ROOT" "$Q36_BNB_MANIFEST_SHA256"
q36_verify_overlay "$Q36_FAST_KERNEL_ROOT" "$Q36_FAST_KERNEL_MANIFEST_SHA256"
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || q36_die "environment output exists"
mkdir -p "$(dirname "$OUTPUT")" logs
export PYTHONDONTWRITEBYTECODE=1
q36_export_pythonpath
"$PYTHON" -P -s -B "$RUNTIME/pipeline/capture_q36_mtr_environment.py" \
  --runtime-root "$RUNTIME" \
  --runtime-manifest-sha256 "$RUNTIME_MANIFEST_SHA256" \
  --output "$OUTPUT"
test -f "$OUTPUT" && chmod a-w "$OUTPUT"
