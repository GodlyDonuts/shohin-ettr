#!/bin/bash
#SBATCH --job-name=sctr1-compare
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/sctr1_compare_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/sctr1_compare_%j.err

set -euo pipefail

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 SELECTIVE_REPORT \
  ALWAYS_REVISE_REPORT UNCHANGED_REPORT LONG_REPORT INDEPENDENT_REPORT \
  SHUFFLED_REPORT SCTR_DATA_REPORT STANDARD_DATA_REPORT OUTPUT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = \
  "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
for path in "$SELECTIVE_REPORT" "$ALWAYS_REVISE_REPORT" "$UNCHANGED_REPORT" \
  "$LONG_REPORT" "$INDEPENDENT_REPORT" "$SHUFFLED_REPORT" \
  "$SCTR_DATA_REPORT" "$STANDARD_DATA_REPORT"; do
  test -f "$path"
done
test ! -e "$OUTPUT"
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test -x "$PYTHON"
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
"$PYTHON" "$RUNTIME/pipeline/compare_sctr1_development.py" \
  --selective "$SELECTIVE_REPORT" \
  --always-revise "$ALWAYS_REVISE_REPORT" \
  --unchanged "$UNCHANGED_REPORT" \
  --long "$LONG_REPORT" \
  --independent "$INDEPENDENT_REPORT" \
  --shuffled "$SHUFFLED_REPORT" \
  --sctr-data-report "$SCTR_DATA_REPORT" \
  --standard-data-report "$STANDARD_DATA_REPORT" \
  --output "$OUTPUT"
chmod a-w "$OUTPUT"
sha256sum "$OUTPUT"
