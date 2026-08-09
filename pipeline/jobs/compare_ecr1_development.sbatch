#!/bin/bash
#SBATCH --job-name=ecr1-compare
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/ecr1_compare_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/ecr1_compare_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 TREATMENT_REPORT SHARED_REPORT \
  DRAFT_REPORT ZERO_REPORT MEAN_REPORT PERMUTATION_REPORT UNCHANGED_REPORT \
  MTR_REPORT TREATMENT_FIT SHARED_FIT DRAFT_FIT SEMANTIC_REPORT OUTPUT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test ! -e "$OUTPUT"
"$PYTHON" "$RUNTIME/pipeline/compare_ecr1_development.py" \
  --treatment-report "$TREATMENT_REPORT" --shared-report "$SHARED_REPORT" \
  --draft-report "$DRAFT_REPORT" --zero-report "$ZERO_REPORT" \
  --mean-report "$MEAN_REPORT" --permutation-report "$PERMUTATION_REPORT" \
  --unchanged-report "$UNCHANGED_REPORT" --mtr-report "$MTR_REPORT" \
  --treatment-fit "$TREATMENT_FIT" --shared-fit "$SHARED_FIT" \
  --draft-fit "$DRAFT_FIT" --semantic-report "$SEMANTIC_REPORT" \
  --output "$OUTPUT"
sha256sum "$OUTPUT"
chmod a-w "$OUTPUT"
