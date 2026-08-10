#!/bin/bash
#SBATCH --job-name=gset1-compare
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/gset1_compare_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/gset1_compare_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 ALIGNED SWAPPED HIDDEN OUTPUT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test ! -e "$OUTPUT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
set +e
"$PYTHON" "$RUNTIME/pipeline/compare_gset1_stage0.py" \
  --aligned "$ALIGNED" --swapped "$SWAPPED" --hidden "$HIDDEN" --output "$OUTPUT"
status=$?
set -e
test "$status" -eq 0 -o "$status" -eq 3
test -f "$OUTPUT"
chmod a-w "$OUTPUT"
exit "$status"
