#!/bin/bash
#SBATCH --job-name=obr1-compare
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/obr1_compare_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/obr1_compare_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 EVALUATION_REPORT FIT_REPORT DATA_REPORT OUTPUT; do
  test -n "${!variable:-}" || exit 2
done
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
"$PYTHON" "$RUNTIME/pipeline/compare_obr1_development.py" \
  --evaluation-report "$EVALUATION_REPORT" --fit-report "$FIT_REPORT" \
  --data-report "$DATA_REPORT" --output "$OUTPUT"
chmod a-w "$OUTPUT"
