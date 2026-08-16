#!/bin/bash
#SBATCH --job-name=q36-mtr-merge-owner-select
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_mtr_merge_owner_select_%j.out
#SBATCH --error=logs/q36_mtr_merge_owner_select_%j.err

set -euo pipefail

for variable in PYTHON RUNTIME MERGE_SCRIPT MERGE_SCRIPT_SHA256 TRAIN_SOURCE DEVELOPMENT_SOURCE FREEZE_REPORT SHARD_ZERO_ROOT SHARD_REST_ROOT OUTPUT REPORT; do
  [[ -n "$(printenv "$variable")" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
[[ "$MERGE_SCRIPT_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ "$(sha256sum "$MERGE_SCRIPT" | cut -d' ' -f1)" == "$MERGE_SCRIPT_SHA256" ]] || exit 2
for input in "$TRAIN_SOURCE" "$DEVELOPMENT_SOURCE" "$FREEZE_REPORT"; do
  [[ -f "$input" && ! -L "$input" ]] || exit 2
done
[[ -d "$SHARD_ZERO_ROOT" && ! -L "$SHARD_ZERO_ROOT" ]] || exit 2
[[ -d "$SHARD_REST_ROOT" && ! -L "$SHARD_REST_ROOT" ]] || exit 2
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" && ! -e "$REPORT" && ! -L "$REPORT" ]] || exit 2
export PATH=/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export PYTHONPATH=$RUNTIME/train:$RUNTIME/pipeline
arguments=(
  --selection-report "$SHARD_ZERO_ROOT/selection_report.json"
  --selected-candidates "$SHARD_ZERO_ROOT/candidates.jsonl"
)
for index in $(seq -w 1 15); do
  arguments+=(
    --selection-report "$SHARD_REST_ROOT/shard_$index/selection_report.json"
    --selected-candidates "$SHARD_REST_ROOT/shard_$index/candidates.jsonl"
  )
done
"$PYTHON" "$MERGE_SCRIPT" \
  --train-source "$TRAIN_SOURCE" \
  --development-source "$DEVELOPMENT_SOURCE" \
  --freeze-report "$FREEZE_REPORT" \
  "${arguments[@]}" \
  --output "$OUTPUT" \
  --report "$REPORT"
test -f "$OUTPUT" && test -f "$REPORT"
chmod a-w "$OUTPUT" "$REPORT"
