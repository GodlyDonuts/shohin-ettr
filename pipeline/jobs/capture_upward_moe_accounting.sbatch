#!/bin/bash
#SBATCH --job-name=upward-moe-accounting
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=logs/upward_moe_accounting_%j.out
#SBATCH --error=logs/upward_moe_accounting_%j.err

set -euo pipefail

for variable in PYTHON RUNTIME RUNTIME_MANIFEST_SHA256 HOST SOURCE_COMMIT ALLOCATIONS OUTPUT; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
[[ -x "$PYTHON" ]]
[[ -d "$RUNTIME" && ! -L "$RUNTIME" ]]
[[ "$OUTPUT" == /* && ! -e "$OUTPUT" && ! -L "$OUTPUT" ]]

source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_runtime
q36_export_pythonpath
mkdir -p logs "$(dirname "$OUTPUT")"

IFS=: read -r -a specifications <<< "$ALLOCATIONS"
(( ${#specifications[@]} > 0 ))
arguments=()
for specification in "${specifications[@]}"; do
  arguments+=(--allocation "$specification")
done
export PATH=/apps/slurm/current/bin:/usr/bin:/bin
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$SLURM_TMPDIR"
"$PYTHON" -P -s -B "$RUNTIME/pipeline/capture_upward_moe_accounting.py" \
  --host "$HOST" \
  --source-commit "$SOURCE_COMMIT" \
  "${arguments[@]}" \
  --output "$OUTPUT"

[[ -f "$OUTPUT" && ! -L "$OUTPUT" ]]
chmod a-w "$OUTPUT"
