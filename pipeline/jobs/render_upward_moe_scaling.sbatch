#!/bin/bash
#SBATCH --job-name=q36-upward-moe-figure
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_upward_moe_figure_%j.out
#SBATCH --error=logs/q36_upward_moe_figure_%j.err

set -euo pipefail

for variable in PYTHON RUNTIME RUNTIME_MANIFEST_SHA256 ANALYSIS OUTPUT_ROOT; do
  [[ -n "${!variable:-}" ]] || {
    printf '%s is required\n' "$variable" >&2
    exit 2
  }
done
[[ -x "$PYTHON" ]]
[[ -d "$RUNTIME" && ! -L "$RUNTIME" ]]
[[ -f "$ANALYSIS" && ! -L "$ANALYSIS" ]]
[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]]

source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_runtime
q36_export_pythonpath
mkdir -p logs "$(dirname "$OUTPUT_ROOT")"

export PATH=/apps/slurm/current/bin:/usr/bin:/bin
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$SLURM_TMPDIR"
"$PYTHON" -P -s -B "$RUNTIME/pipeline/render_upward_moe_scaling.py" \
  --analysis "$ANALYSIS" \
  --output-root "$OUTPUT_ROOT"

[[ -d "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]]
[[ -f "$OUTPUT_ROOT/manifest.json" && ! -L "$OUTPUT_ROOT/manifest.json" ]]
[[ -f "$OUTPUT_ROOT/shohin-upward-moe-scaling.svg" ]]
[[ -f "$OUTPUT_ROOT/shohin-upward-moe-scaling-points.csv" ]]
[[ -z "$(find "$OUTPUT_ROOT" -perm /0222 -print -quit)" ]]
