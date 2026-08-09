#!/bin/bash
#SBATCH --job-name=dset-q35t-cmp
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/dset_q35t_cmp_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/dset_q35t_cmp_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 ALIGNED HIDDEN OUTPUT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test ! -e "$OUTPUT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
"$PYTHON" "$RUNTIME/pipeline/compare_dset_q35t.py" \
  --aligned "$ALIGNED" --hidden "$HIDDEN" --output "$OUTPUT"
