#!/bin/bash
#SBATCH --job-name=q36-upward-moe-curve
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_upward_moe_curve_%j.out
#SBATCH --error=logs/q36_upward_moe_curve_%j.err

set -euo pipefail

required=(
  PYTHON RUNTIME RUNTIME_MANIFEST_SHA256
  QWEN_POINT SUPER_POINT MIXTRAL_POINT OUTPUT
)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || {
    printf '%s is required\n' "$variable" >&2
    exit 2
  }
done
[[ -x "$PYTHON" ]]
[[ -d "$RUNTIME" && ! -L "$RUNTIME" ]]
for point in "$QWEN_POINT" "$SUPER_POINT" "$MIXTRAL_POINT"; do
  [[ -f "$point" && ! -L "$point" ]]
done
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]]

source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_runtime
q36_export_pythonpath
mkdir -p logs "$(dirname "$OUTPUT")"

export PATH=/apps/slurm/current/bin:/usr/bin:/bin
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$SLURM_TMPDIR"
"$PYTHON" -P -s -B "$RUNTIME/pipeline/analyze_upward_moe_scaling.py" \
  --point "$QWEN_POINT" \
  --point "$SUPER_POINT" \
  --point "$MIXTRAL_POINT" \
  --output "$OUTPUT"

[[ -f "$OUTPUT" && ! -L "$OUTPUT" ]]
chmod a-w "$OUTPUT"
