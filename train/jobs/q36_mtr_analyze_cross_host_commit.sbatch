#!/bin/bash
#SBATCH --job-name=q36-cross-host-analysis
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_cross_host_analysis_%j.out
#SBATCH --error=logs/q36_cross_host_analysis_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE_COMMIT PYTHON HOST SELECTIONS APPLICATION_REPORT SCORE OUTPUT; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_runtime
for path in "$SELECTIONS" "$APPLICATION_REPORT" "$SCORE"; do
  [[ -f "$path" && ! -L "$path" ]] || q36_die "cross-host analysis input differs"
done
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || q36_die "cross-host result exists"
mkdir -p "$(dirname "$OUTPUT")" logs
q36_export_pythonpath
"$PYTHON" -P -s -B "$RUNTIME/pipeline/analyze_q36_cross_host_commit.py" \
  --host "$HOST" \
  --selections "$SELECTIONS" \
  --application-report "$APPLICATION_REPORT" \
  --score "$SCORE" \
  --output "$OUTPUT"
test -f "$OUTPUT"
chmod a-w "$OUTPUT"
