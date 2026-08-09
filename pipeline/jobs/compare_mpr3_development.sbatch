#!/bin/bash
#SBATCH --job-name=mpr3-compare
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/mpr3_compare_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/mpr3_compare_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 ALIGNED_REPORT SHUFFLED_REPORT HIDDEN_REPORT OWNER_REPORT ALIGNED_FIT SHUFFLED_FIT HIDDEN_FIT DATA_REPORT SEMANTIC_ATTRIBUTION OUTPUT; do
  test -n "${!variable:-}" || exit 2
done
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
"$PYTHON" "$RUNTIME/pipeline/compare_mpr3_development.py" \
  --aligned-report "$ALIGNED_REPORT" --shuffled-report "$SHUFFLED_REPORT" \
  --hidden-report "$HIDDEN_REPORT" --owner-report "$OWNER_REPORT" \
  --aligned-fit "$ALIGNED_FIT" --shuffled-fit "$SHUFFLED_FIT" \
  --hidden-fit "$HIDDEN_FIT" --data-report "$DATA_REPORT" \
  --semantic-attribution "$SEMANTIC_ATTRIBUTION" --output "$OUTPUT"
chmod a-w "$OUTPUT"
