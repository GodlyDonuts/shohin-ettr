#!/bin/bash
#SBATCH --job-name=q36-upward-moe-promote
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_upward_moe_promote_%j.out
#SBATCH --error=logs/q36_upward_moe_promote_%j.err

set -euo pipefail

required=(
  PYTHON RUNTIME RUNTIME_MANIFEST_SHA256
  SUPER_SCORE MIXTRAL_SCORE OUTPUT
)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || {
    printf '%s is required\n' "$variable" >&2
    exit 2
  }
done
[[ -x "$PYTHON" ]]
[[ -d "$RUNTIME" && ! -L "$RUNTIME" ]]
for score in "$SUPER_SCORE" "$MIXTRAL_SCORE"; do
  [[ -f "$score" && ! -L "$score" ]]
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
"$PYTHON" -P -s -B \
  "$RUNTIME/pipeline/select_upward_moe_temporal_promotion.py" \
  --score "$SUPER_SCORE" \
  --score "$MIXTRAL_SCORE" \
  --output "$OUTPUT"

[[ -f "$OUTPUT" && ! -L "$OUTPUT" ]]
chmod a-w "$OUTPUT"
