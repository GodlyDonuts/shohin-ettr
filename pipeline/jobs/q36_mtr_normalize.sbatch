#!/bin/bash
#SBATCH --job-name=q36-mtr-normalize
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_mtr_normalize_%j.out
#SBATCH --error=logs/q36_mtr_normalize_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE_COMMIT PYTHON SCORE_REPORT PRECOMPUTE_CUSTODY OUTPUT; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_verify_runtime
q36_require_authorization
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || q36_die "normalize output exists"
mkdir -p "$(dirname "$OUTPUT")" logs
q36_export_pythonpath
"$PYTHON" -P -s -B "$RUNTIME/pipeline/normalize_q36_mtr_score.py" \
  --score-report "$SCORE_REPORT" --precompute-custody "$PRECOMPUTE_CUSTODY" \
  --output "$OUTPUT"
test -d "$OUTPUT" && test "$(find "$OUTPUT" -type f | wc -l)" -eq 5
chmod -R a-w "$OUTPUT"
