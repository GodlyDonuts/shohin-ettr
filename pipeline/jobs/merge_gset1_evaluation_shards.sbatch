#!/bin/bash
#SBATCH --job-name=gset1-merge
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/gset1_merge_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/gset1_merge_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 INPUT_DIR OUTPUT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test ! -e "$OUTPUT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
mapfile -t INPUTS < <(find "$INPUT_DIR" -maxdepth 1 -type f -name 'shard_*.json' | sort)
test "${#INPUTS[@]}" -eq 8
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
"$PYTHON" "$RUNTIME/pipeline/merge_gset1_evaluation_shards.py" \
  --inputs "${INPUTS[@]}" --output "$OUTPUT"
test -f "$OUTPUT"
chmod a-w "$OUTPUT"
