#!/bin/bash
#SBATCH --job-name=q36-external-score
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_external_score_%j.out
#SBATCH --error=logs/q36_external_score_%j.err

set -euo pipefail

required=(RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE_COMMIT PYTHON SCORE_SCRIPT SCORE_SCRIPT_SHA256 ASSESSORS ASSESSORS_SHA256 INPUT_ROOT OUTPUT SANDBOX_RECEIPT SPLIT EXPECTED_ROWS SHARD_COUNT)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_runtime
q36_verify_sha256 "$SCORE_SCRIPT" "$SCORE_SCRIPT_SHA256"
q36_verify_sha256 "$ASSESSORS" "$ASSESSORS_SHA256"
[[ -d "$INPUT_ROOT" && ! -L "$INPUT_ROOT" ]] || q36_die "external input root differs"
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" && ! -e "$SANDBOX_RECEIPT" && ! -L "$SANDBOX_RECEIPT" ]] || q36_die "external score output exists"
args=()
for arm in unchanged self_refinement revision draft_hidden interpolation; do
  option=${arm//_/-}
  for ((index=0; index<SHARD_COUNT; index++)); do
    printf -v shard '%02d' "$index"
    path=$INPUT_ROOT/$arm/shard_$shard/candidates.jsonl
    [[ -f "$path" && ! -L "$path" ]] || q36_die "external candidate differs"
    args+=(--$option-candidates "$path")
  done
done
score_mode_args=()
if [[ -n "${FIXED_RULE:-}" ]]; then
  score_mode_args+=(--fixed-rule "$FIXED_RULE")
fi
mkdir -p "$(dirname "$OUTPUT")" logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
q36_export_pythonpath
"$PYTHON" "$SCORE_SCRIPT" --assessors "$ASSESSORS" "${args[@]}" \
  --split "$SPLIT" --expected-rows "$EXPECTED_ROWS" --shard-count "$SHARD_COUNT" \
  "${score_mode_args[@]}" \
  --sandbox-receipt "$SANDBOX_RECEIPT" --output "$OUTPUT"
test -f "$OUTPUT" && test -f "$SANDBOX_RECEIPT"
chmod a-w "$OUTPUT" "$SANDBOX_RECEIPT"
