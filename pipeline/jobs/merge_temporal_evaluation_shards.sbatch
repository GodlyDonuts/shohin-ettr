#!/bin/bash
#SBATCH --job-name=temporal-merge
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/temporal_merge_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/temporal_merge_%j.err

set -euo pipefail

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 EVALUATOR DATA DATA_SHA256 \
  SPLIT SHARD_REPORTS CANDIDATES_OUTPUT REPORT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = \
  "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = "$DATA_SHA256"
test ! -e "$CANDIDATES_OUTPUT"
test ! -e "$REPORT"

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test -x "$PYTHON"
IFS=: read -r -a reports <<<"$SHARD_REPORTS"
arguments=()
for path in "${reports[@]}"; do
  test -f "$path"
  arguments+=(--shard-report "$path")
done
case "$EVALUATOR" in
  sctr1) program=$RUNTIME/pipeline/merge_sctr1_evaluation_shards.py ;;
  idr1) program=$RUNTIME/pipeline/merge_idr1_evaluation_shards.py ;;
  *) echo "unknown temporal evaluator: $EVALUATOR" >&2; exit 4 ;;
esac
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
"$PYTHON" "$program" \
  --data "$DATA" \
  --split "$SPLIT" \
  "${arguments[@]}" \
  --candidates-output "$CANDIDATES_OUTPUT" \
  --report "$REPORT"
chmod a-w "$CANDIDATES_OUTPUT" "$REPORT"
sha256sum "$CANDIDATES_OUTPUT" "$REPORT"
