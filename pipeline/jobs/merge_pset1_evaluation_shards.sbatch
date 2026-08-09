#!/bin/bash
#SBATCH --job-name=pset1-merge
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/pset1_merge_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/pset1_merge_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 INPUTS OUTPUT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test ! -e "$OUTPUT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
IFS='|' read -r -a paths <<< "$INPUTS"
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
"$PYTHON" "$RUNTIME/pipeline/merge_pset1_evaluation_shards.py" --inputs "${paths[@]}" --output "$OUTPUT"
test -f "$OUTPUT"
chmod a-w "$OUTPUT"
