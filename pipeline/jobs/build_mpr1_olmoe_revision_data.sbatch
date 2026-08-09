#!/bin/bash
#SBATCH --job-name=mpr1-data
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/mpr1_data_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/mpr1_data_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE SOURCE_REPORT PAIRS \
  MODEL_ROOT OUTPUT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test ! -e "$OUTPUT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
"$PYTHON" "$RUNTIME/pipeline/build_mpr1_olmoe_revision_data.py" \
  --source "$SOURCE" --source-report "$SOURCE_REPORT" --pairs "$PAIRS" \
  --model-root "$MODEL_ROOT" --output "$OUTPUT"
sha256sum "$OUTPUT"/train_aligned.jsonl "$OUTPUT"/train_shuffled.jsonl "$OUTPUT"/report.json
chmod -R a-w "$OUTPUT"

