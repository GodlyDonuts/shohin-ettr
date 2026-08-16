#!/bin/bash
#SBATCH --job-name=merge-upward-moe-drafts
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=logs/merge_upward_moe_drafts_%j.out
#SBATCH --error=logs/merge_upward_moe_drafts_%j.err

set -euo pipefail
for variable in HOST PYTHON RUNTIME TRAIN_SOURCE DEVELOPMENT_SOURCE FREEZE_REPORT SHARD_ROOT OUTPUT REPORT; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
[[ "$HOST" == nemotron-super || "$HOST" == mixtral-8x22b || \
  "$HOST" == nemotron-ultra ]]
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" && ! -e "$REPORT" && ! -L "$REPORT" ]]
args=()
for index in $(seq 0 15); do
  report=$SHARD_ROOT/$(printf 'shard_%02d/report.json' "$index")
  candidates=$SHARD_ROOT/$(printf 'shard_%02d/candidates.jsonl' "$index")
  [[ -f "$report" && ! -L "$report" && -f "$candidates" && ! -L "$candidates" ]]
  args+=(--shard-report "$report" --shard-candidates "$candidates")
done
mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$REPORT")" logs
PYTHONPATH="$RUNTIME/pipeline:$RUNTIME/train" "$PYTHON" -P -s -B \
  "$RUNTIME/pipeline/merge_upward_moe_drafts.py" \
  --host "$HOST" --train-source "$TRAIN_SOURCE" \
  --development-source "$DEVELOPMENT_SOURCE" --freeze-report "$FREEZE_REPORT" \
  "${args[@]}" --output "$OUTPUT" --report "$REPORT"
chmod a-w "$OUTPUT" "$REPORT"
