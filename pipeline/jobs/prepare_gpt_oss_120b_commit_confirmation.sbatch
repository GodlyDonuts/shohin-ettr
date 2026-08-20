#!/bin/bash
#SBATCH --job-name=gpt-oss-confirm-prep
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=logs/gpt_oss_confirm_prep_%j.out
#SBATCH --error=logs/gpt_oss_confirm_prep_%j.err

set -euo pipefail
for variable in PYTHON RUNTIME RUNTIME_MANIFEST_SHA256 QUESTIONS EXCLUDED_QUESTIONS PRIOR_Q36_SOURCE OUTPUT_ROOT; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_runtime
for path in "$QUESTIONS" "$EXCLUDED_QUESTIONS" "$PRIOR_Q36_SOURCE"; do
  [[ -f "$path" && ! -L "$path" ]] || q36_die "confirmation preparation input differs"
done
[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] || q36_die "confirmation preparation root exists"
mkdir -m 700 "$OUTPUT_ROOT" logs
q36_export_pythonpath
"$PYTHON" -P -s -B "$RUNTIME/pipeline/prepare_gpt_oss_120b_commit_confirmation.py" \
  --questions "$QUESTIONS" \
  --excluded-questions "$EXCLUDED_QUESTIONS" \
  --prior-q36-source "$PRIOR_Q36_SOURCE" \
  --source-output "$OUTPUT_ROOT/source.jsonl" \
  --receipt "$OUTPUT_ROOT/receipt.json"
test -f "$OUTPUT_ROOT/source.jsonl" && test -f "$OUTPUT_ROOT/receipt.json"
chmod a-w "$OUTPUT_ROOT"/*.json "$OUTPUT_ROOT"/*.jsonl "$OUTPUT_ROOT"
