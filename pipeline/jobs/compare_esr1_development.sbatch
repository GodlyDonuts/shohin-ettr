#!/bin/bash
#SBATCH --job-name=esr1-compare
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/esr1_compare_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/esr1_compare_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 SYNDROME_REPORT ETTR_REPORT \
  ALWAYS_REPORT SYNDROME_FIT_REPORT ETTR_FIT_REPORT OUTPUT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
for path in "$SYNDROME_REPORT" "$ETTR_REPORT" "$ALWAYS_REPORT" \
  "$SYNDROME_FIT_REPORT" "$ETTR_FIT_REPORT"; do test -f "$path"; done
test ! -e "$OUTPUT"
PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603-r2/bin/python}
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
"$PYTHON" "$RUNTIME/pipeline/compare_esr1_development.py" \
  --syndrome-report "$SYNDROME_REPORT" --ettr-report "$ETTR_REPORT" \
  --always-report "$ALWAYS_REPORT" --syndrome-fit-report "$SYNDROME_FIT_REPORT" \
  --ettr-fit-report "$ETTR_FIT_REPORT" --output "$OUTPUT"
chmod a-w "$OUTPUT"
sha256sum "$OUTPUT"
