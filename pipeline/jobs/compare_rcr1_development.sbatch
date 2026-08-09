#!/bin/bash
#SBATCH --job-name=rcr1-compare
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/rcr1_compare_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/rcr1_compare_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 ROUTER_REPORT ATTENTION_REPORT \
  UNCHANGED_REPORT PRIOR_TREATMENT_REPORT ROUTER_FIT_REPORT \
  ATTENTION_FIT_REPORT OUTPUT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test ! -e "$OUTPUT"
"$PYTHON" "$RUNTIME/pipeline/compare_rcr1_development.py" \
  --router-report "$ROUTER_REPORT" --attention-report "$ATTENTION_REPORT" \
  --unchanged-report "$UNCHANGED_REPORT" \
  --prior-treatment-report "$PRIOR_TREATMENT_REPORT" \
  --router-fit-report "$ROUTER_FIT_REPORT" \
  --attention-fit-report "$ATTENTION_FIT_REPORT" --output "$OUTPUT"
sha256sum "$OUTPUT"
chmod a-w "$OUTPUT"

