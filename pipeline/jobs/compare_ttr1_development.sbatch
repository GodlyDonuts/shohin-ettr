#!/bin/bash
#SBATCH --job-name=ttr1-compare-dev
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/ttr1_compare_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/ttr1_compare_%j.err

set -euo pipefail

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 TREATMENT_REPORT \
  UNCHANGED_REPORT SELF_REFINEMENT_REPORT LONG_SINGLE_REPORT BEST_OF_TWO_REPORT \
  INDEPENDENT_REPORT OUTPUT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
for path in "$TREATMENT_REPORT" "$UNCHANGED_REPORT" "$SELF_REFINEMENT_REPORT" \
  "$LONG_SINGLE_REPORT" "$BEST_OF_TWO_REPORT" "$INDEPENDENT_REPORT"; do
  test -f "$path"
done
test ! -e "$OUTPUT"
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test -x "$PYTHON"
"$PYTHON" "$RUNTIME/pipeline/compare_ttr1_development.py" \
  --treatment "$TREATMENT_REPORT" \
  --control-report "$UNCHANGED_REPORT" \
  --control-report "$SELF_REFINEMENT_REPORT" \
  --control-report "$LONG_SINGLE_REPORT" \
  --control-report "$BEST_OF_TWO_REPORT" \
  --control-report "$INDEPENDENT_REPORT" \
  --output "$OUTPUT"
chmod a-w "$OUTPUT"
sha256sum "$OUTPUT"
