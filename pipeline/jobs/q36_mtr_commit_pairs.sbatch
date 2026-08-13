#!/bin/bash
#SBATCH --job-name=q36-mtr-pairs
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_mtr_pairs_%j.out
#SBATCH --error=logs/q36_mtr_pairs_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE_COMMIT PYTHON SPLIT DATA DATA_SHA256 REVISION_REPORT REVISION_REPORT_SHA256 REVISION_CANDIDATES REVISION_CANDIDATES_SHA256 UNCHANGED_REPORT UNCHANGED_REPORT_SHA256 UNCHANGED_CANDIDATES UNCHANGED_CANDIDATES_SHA256 CANDIDATES_ROOT OUTPUT REPORT; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_verify_runtime
q36_require_authorization
for binding in \
  "$DATA:$DATA_SHA256" \
  "$REVISION_REPORT:$REVISION_REPORT_SHA256" \
  "$REVISION_CANDIDATES:$REVISION_CANDIDATES_SHA256" \
  "$UNCHANGED_REPORT:$UNCHANGED_REPORT_SHA256" \
  "$UNCHANGED_CANDIDATES:$UNCHANGED_CANDIDATES_SHA256"; do
  q36_verify_sha256 "${binding%%:*}" "${binding#*:}"
done
case "$SPLIT" in calibration|development) ;; *) q36_die "pair split differs" ;; esac
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" && ! -e "$REPORT" && ! -L "$REPORT" ]] || \
  q36_die "pair output exists"
mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$REPORT")" logs
q36_export_pythonpath
"$PYTHON" -P -s -B "$RUNTIME/pipeline/build_q36_mtr_commit_pairs.py" \
  --split "$SPLIT" \
  --data "$DATA" \
  --revision-report "$REVISION_REPORT" \
  --revision-candidates "$REVISION_CANDIDATES" \
  --unchanged-report "$UNCHANGED_REPORT" \
  --unchanged-candidates "$UNCHANGED_CANDIDATES" \
  --candidates-root "$CANDIDATES_ROOT" \
  --output "$OUTPUT" \
  --report "$REPORT" \
  --seed 2026080820
test -f "$OUTPUT" && test -f "$REPORT"
chmod a-w "$OUTPUT" "$REPORT"
