#!/bin/bash
#SBATCH --job-name=ttr1-control-merge
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/ttr1_control_merge_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/ttr1_control_merge_%j.err

set -euo pipefail

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 DATA DATA_SHA256 SPLIT \
  SHARD_REPORTS CANDIDATES_OUTPUT REPORT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = "$DATA_SHA256"
test ! -e "$CANDIDATES_OUTPUT"
test ! -e "$REPORT"

IFS=, read -r -a reports <<<"$SHARD_REPORTS"
arguments=()
for path in "${reports[@]}"; do
  test -f "$path"
  arguments+=(--shard-report "$path")
done
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
python3 "$RUNTIME/pipeline/merge_ttr1_control_shards.py" \
  --data "$DATA" \
  --split "$SPLIT" \
  "${arguments[@]}" \
  --candidates-output "$CANDIDATES_OUTPUT" \
  --report "$REPORT"
chmod a-w "$CANDIDATES_OUTPUT" "$REPORT"
sha256sum "$CANDIDATES_OUTPUT" "$REPORT"
