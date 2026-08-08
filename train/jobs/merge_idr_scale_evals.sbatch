#!/bin/bash
#SBATCH --job-name=idr-scale-merge
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/idr_scale_merge_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/idr_scale_merge_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${RUNTIME_MANIFEST_SHA256:?RUNTIME_MANIFEST_SHA256 is required}"
: "${DATA:?DATA is required}"
: "${DATA_SHA256:?DATA_SHA256 is required}"
: "${SPLIT:?SPLIT is required}"
: "${SHARD_ROOT:?SHARD_ROOT is required}"
: "${SHARD_COUNT:?SHARD_COUNT is required}"
: "${CANDIDATES_OUTPUT:?CANDIDATES_OUTPUT is required}"
: "${REPORT:?REPORT is required}"

test "$SHARD_COUNT" -ge 2
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = "$DATA_SHA256"
test ! -e "$CANDIDATES_OUTPUT"
test ! -e "$REPORT"

reports=()
for shard in $(seq 0 $((SHARD_COUNT - 1))); do
  shard_report="$SHARD_ROOT/shard_${shard}/report.json"
  test -f "$shard_report"
  reports+=(--shard-report "$shard_report")
done
test "${#reports[@]}" -eq $((SHARD_COUNT * 2))

mkdir -p "$(dirname "$REPORT")"
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
python3 "$RUNTIME/pipeline/merge_idr1_evaluation_shards.py" \
  --data "$DATA" \
  --split "$SPLIT" \
  "${reports[@]}" \
  --candidates-output "$CANDIDATES_OUTPUT" \
  --report "$REPORT"
sha256sum "$CANDIDATES_OUTPUT" "$REPORT"
chmod a-w "$CANDIDATES_OUTPUT" "$REPORT"
