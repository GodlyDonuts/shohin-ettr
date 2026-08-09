#!/bin/bash
#SBATCH --job-name=ser1-compare
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/ser1_compare_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/ser1_compare_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 TREATMENT_REPORT \
  SHARED_FLOP_REPORT SHARED_PARAMETER_REPORT TREATMENT_FIT SHARED_FLOP_FIT \
  SHARED_PARAMETER_FIT OUTPUT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603-r2/bin/python}
"$PYTHON" "$RUNTIME/pipeline/compare_ser1_development.py" \
  --treatment-report "$TREATMENT_REPORT" \
  --shared-flop-report "$SHARED_FLOP_REPORT" \
  --shared-parameter-report "$SHARED_PARAMETER_REPORT" \
  --treatment-fit "$TREATMENT_FIT" --shared-flop-fit "$SHARED_FLOP_FIT" \
  --shared-parameter-fit "$SHARED_PARAMETER_FIT" --output "$OUTPUT"
sha256sum "$OUTPUT"
chmod a-w "$OUTPUT"
