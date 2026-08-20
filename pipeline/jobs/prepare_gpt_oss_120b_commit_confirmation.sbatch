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
for variable in PYTHON RUNTIME RUNTIME_MANIFEST_SHA256 QUESTIONS EXCLUDED_QUESTIONS PRIOR_Q36_SOURCE SELECTION_SEED OUTPUT_ROOT; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
prior_confirmation_spec=${PRIOR_CONFIRMATION_SOURCES:-${PRIOR_CONFIRMATION_SOURCE:-}}
[[ -n "$prior_confirmation_spec" ]] || { printf 'PRIOR_CONFIRMATION_SOURCES is required\n' >&2; exit 2; }
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_runtime
IFS=: read -r -a prior_confirmation_sources <<< "$prior_confirmation_spec"
[[ ${#prior_confirmation_sources[@]} -gt 0 ]]
for path in "$QUESTIONS" "$EXCLUDED_QUESTIONS" "$PRIOR_Q36_SOURCE" "${prior_confirmation_sources[@]}"; do
  [[ -f "$path" && ! -L "$path" ]] || q36_die "confirmation preparation input differs"
done
[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] || q36_die "confirmation preparation root exists"
mkdir -m 700 "$OUTPUT_ROOT"
mkdir -p logs
q36_export_pythonpath
prior_confirmation_args=()
for path in "${prior_confirmation_sources[@]}"; do
  prior_confirmation_args+=(--prior-confirmation-source "$path")
done
"$PYTHON" -P -s -B "$RUNTIME/pipeline/prepare_gpt_oss_120b_commit_confirmation.py" \
  --questions "$QUESTIONS" \
  --excluded-questions "$EXCLUDED_QUESTIONS" \
  --prior-q36-source "$PRIOR_Q36_SOURCE" \
  "${prior_confirmation_args[@]}" \
  --selection-seed "$SELECTION_SEED" \
  --source-output "$OUTPUT_ROOT/source.jsonl" \
  --receipt "$OUTPUT_ROOT/receipt.json"
test -f "$OUTPUT_ROOT/source.jsonl" && test -f "$OUTPUT_ROOT/receipt.json"
chmod a-w "$OUTPUT_ROOT"/*.json "$OUTPUT_ROOT"/*.jsonl "$OUTPUT_ROOT"
