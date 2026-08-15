#!/bin/bash
#SBATCH --job-name=q36-external-forest
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_external_forest_%j.out
#SBATCH --error=logs/q36_external_forest_%j.err

set -euo pipefail

required=(RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE_COMMIT PYTHON FOREST_SCRIPT FOREST_SCRIPT_SHA256 DEVELOPMENT_UNCHANGED_ROOT DEVELOPMENT_SELF_REFINEMENT_ROOT DEVELOPMENT_CONTROL_SCORE_ROOT DEVELOPMENT_REVISION_CANDIDATES DEVELOPMENT_REVISION_SCORE DEVELOPMENT_HIDDEN_CANDIDATES DEVELOPMENT_HIDDEN_SCORE DEVELOPMENT_INTERPOLATION_ROOT TARGET_ROOT ASSESSORS ASSESSORS_SHA256 SPLIT EXPECTED_ROWS SHARD_COUNT OUTPUT SELECTIONS SANDBOX_RECEIPT)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_runtime
q36_verify_sha256 "$FOREST_SCRIPT" "$FOREST_SCRIPT_SHA256"
q36_verify_sha256 "$ASSESSORS" "$ASSESSORS_SHA256"
for path in "$DEVELOPMENT_REVISION_CANDIDATES" "$DEVELOPMENT_REVISION_SCORE" "$DEVELOPMENT_HIDDEN_CANDIDATES" "$DEVELOPMENT_HIDDEN_SCORE"; do
  [[ -f "$path" && ! -L "$path" ]] || q36_die "external forest development input differs"
done
for path in "$DEVELOPMENT_UNCHANGED_ROOT" "$DEVELOPMENT_SELF_REFINEMENT_ROOT" "$DEVELOPMENT_CONTROL_SCORE_ROOT" "$DEVELOPMENT_INTERPOLATION_ROOT" "$TARGET_ROOT"; do
  [[ -d "$path" && ! -L "$path" ]] || q36_die "external forest input root differs"
done
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" && ! -e "$SELECTIONS" && ! -L "$SELECTIONS" && ! -e "$SANDBOX_RECEIPT" && ! -L "$SANDBOX_RECEIPT" ]] || q36_die "external forest output exists"

args=()
for index in {0..7}; do
  printf -v shard '%02d' "$index"
  args+=(--development-unchanged-candidates "$DEVELOPMENT_UNCHANGED_ROOT/shard_$shard/candidates.jsonl")
  args+=(--development-self-refinement-candidates "$DEVELOPMENT_SELF_REFINEMENT_ROOT/shard_$shard/candidates.jsonl")
  args+=(--development-unchanged-scores "$DEVELOPMENT_CONTROL_SCORE_ROOT/unchanged/shard_$shard.json")
  args+=(--development-self-refinement-scores "$DEVELOPMENT_CONTROL_SCORE_ROOT/self_refinement/shard_$shard.json")
done
args+=(--development-revision-candidates "$DEVELOPMENT_REVISION_CANDIDATES")
args+=(--development-revision-scores "$DEVELOPMENT_REVISION_SCORE")
args+=(--development-draft-hidden-candidates "$DEVELOPMENT_HIDDEN_CANDIDATES")
args+=(--development-draft-hidden-scores "$DEVELOPMENT_HIDDEN_SCORE")
for index in {0..15}; do
  printf -v shard '%02d' "$index"
  args+=(--development-interpolation-candidates "$DEVELOPMENT_INTERPOLATION_ROOT/shard_$shard/candidates.jsonl")
  args+=(--development-interpolation-scores "$DEVELOPMENT_INTERPOLATION_ROOT/shard_$shard/development_score.json")
done
for arm in unchanged self_refinement revision draft_hidden interpolation; do
  option=${arm//_/-}
  for ((index=0; index<SHARD_COUNT; index++)); do
    printf -v shard '%02d' "$index"
    path=$TARGET_ROOT/$arm/shard_$shard/candidates.jsonl
    [[ -f "$path" && ! -L "$path" ]] || q36_die "external forest target differs"
    args+=(--target-$option-candidates "$path")
  done
done
mkdir -p "$(dirname "$OUTPUT")" logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
q36_export_pythonpath
"$PYTHON" "$FOREST_SCRIPT" "${args[@]}" --assessors "$ASSESSORS" \
  --split "$SPLIT" --expected-rows "$EXPECTED_ROWS" --shard-count "$SHARD_COUNT" \
  --selections "$SELECTIONS" --sandbox-receipt "$SANDBOX_RECEIPT" --output "$OUTPUT"
test -f "$OUTPUT" && test -f "$SELECTIONS" && test -f "$SANDBOX_RECEIPT"
chmod a-w "$OUTPUT" "$SELECTIONS" "$SANDBOX_RECEIPT"
