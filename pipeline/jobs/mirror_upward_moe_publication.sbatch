#!/bin/bash
#SBATCH --job-name=upward-moe-mirror
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=logs/upward_moe_mirror_%j.out
#SBATCH --error=logs/upward_moe_mirror_%j.err

set -euo pipefail

required=(PYTHON RUNTIME RUNTIME_MANIFEST_SHA256 AUTHORIZED_ROOT OUTPUT_ROOT PUBLICATION_RECEIPT ANALYSIS FIGURE_ROOT ACCOUNTING SOURCE_POINTS)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
[[ -x "$PYTHON" ]]
[[ -d "$RUNTIME" && ! -L "$RUNTIME" ]]
[[ "$OUTPUT_ROOT" == "$AUTHORIZED_ROOT"/* && ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]]

source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_runtime
q36_export_pythonpath
mkdir -p logs

IFS=: read -r -a accounting_paths <<< "$ACCOUNTING"
IFS=: read -r -a point_paths <<< "$SOURCE_POINTS"
[[ ${#accounting_paths[@]} -eq 2 && ${#point_paths[@]} -ge 3 ]]
arguments=()
for path in "${accounting_paths[@]}"; do arguments+=(--accounting "$path"); done
for path in "${point_paths[@]}"; do arguments+=(--source-point "$path"); done
export PATH=/apps/slurm/current/bin:/usr/bin:/bin
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$SLURM_TMPDIR"
"$PYTHON" -P -s -B "$RUNTIME/pipeline/mirror_upward_moe_publication.py" \
  --authorized-root "$AUTHORIZED_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --publication-receipt "$PUBLICATION_RECEIPT" \
  --analysis "$ANALYSIS" \
  --figure-root "$FIGURE_ROOT" \
  "${arguments[@]}"

[[ -f "$OUTPUT_ROOT/manifest.json" && ! -L "$OUTPUT_ROOT/manifest.json" ]]
[[ -z "$(find "$OUTPUT_ROOT" -perm /222 -print -quit)" ]]
