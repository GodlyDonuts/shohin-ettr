#!/bin/bash
#SBATCH --job-name=q36-mtr-merge-drafts
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_mtr_merge_drafts_%j.out
#SBATCH --error=logs/q36_mtr_merge_drafts_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE_COMMIT PYTHON TRAIN_SOURCE TRAIN_SOURCE_SHA256 DEVELOPMENT_SOURCE DEVELOPMENT_SOURCE_SHA256 FREEZE_REPORT FREEZE_REPORT_SHA256 SHARD_ROOT OUTPUT REPORT PHASE_AUTHORIZATION PHASE_AUTHORIZATION_SHA256; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_verify_runtime
q36_require_authorization
q36_verify_sha256 "$TRAIN_SOURCE" "$TRAIN_SOURCE_SHA256"
q36_verify_sha256 "$DEVELOPMENT_SOURCE" "$DEVELOPMENT_SOURCE_SHA256"
q36_verify_sha256 "$FREEZE_REPORT" "$FREEZE_REPORT_SHA256"
[[ -d "$SHARD_ROOT" && ! -L "$SHARD_ROOT" ]] || q36_die "draft shard root differs"
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" && ! -e "$REPORT" && ! -L "$REPORT" ]] || \
  q36_die "draft merge output exists"
q36_export_pythonpath
arguments=()
for ((index=0; index<16; index++)); do
  root=$SHARD_ROOT/$(printf 'shard_%02d' "$index")
  arguments+=(--shard-report "$root/report.json" --shard-candidates "$root/candidates.jsonl")
done
"$PYTHON" "$RUNTIME/pipeline/merge_q36_mtr_drafts.py" \
  --train-source "$TRAIN_SOURCE" \
  --development-source "$DEVELOPMENT_SOURCE" \
  --freeze-report "$FREEZE_REPORT" \
  "${arguments[@]}" --output "$OUTPUT" --report "$REPORT"
test -f "$OUTPUT" && test -f "$REPORT"
chmod a-w "$OUTPUT" "$REPORT"
