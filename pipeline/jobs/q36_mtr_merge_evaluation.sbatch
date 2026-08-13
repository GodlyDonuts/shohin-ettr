#!/bin/bash
#SBATCH --job-name=q36-mtr-merge-eval
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_mtr_merge_eval_%j.out
#SBATCH --error=logs/q36_mtr_merge_eval_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE_COMMIT PYTHON ARM SPLIT DATA DATA_REPORT SHARD_ROOT OUTPUT REPORT PHASE_AUTHORIZATION PHASE_AUTHORIZATION_SHA256; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_verify_runtime
q36_require_authorization
DATA_SHA256=$(q36_sha256 "$DATA")
DATA_REPORT_SHA256=$(q36_sha256 "$DATA_REPORT")
q36_verify_sha256 "$DATA" "$DATA_SHA256"
q36_verify_sha256 "$DATA_REPORT" "$DATA_REPORT_SHA256"
[[ -d "$SHARD_ROOT" && ! -L "$SHARD_ROOT" ]] || q36_die "shard root differs"
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" && ! -e "$REPORT" && ! -L "$REPORT" ]] || \
  q36_die "merge output exists"
case "$SPLIT" in
  calibration) count=4 ;;
  development) count=8 ;;
  *) q36_die "merge split differs" ;;
esac
arguments=()
for ((index=0; index<count; index++)); do
  root=$SHARD_ROOT/$(printf 'shard_%02d' "$index")
  arguments+=(--shard-report "$root/report.json" --shard-candidates "$root/candidates.jsonl")
  if [[ "$SPLIT" == calibration ]]; then
    arguments+=(--shard-sandbox-probe "$root/sandbox.json")
  fi
done
"$PYTHON" "$RUNTIME/pipeline/merge_q36_mtr_evaluations.py" \
  --arm "$ARM" --split "$SPLIT" \
  --data "$DATA" --data-report "$DATA_REPORT" \
  "${arguments[@]}" --output "$OUTPUT" --report "$REPORT"
test -f "$OUTPUT" && test -f "$REPORT"
chmod a-w "$OUTPUT" "$REPORT"
