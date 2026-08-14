#!/bin/bash
#SBATCH --job-name=q36-mtr-multi-commit
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc33,evc34,evc37,evc38,evc43,evc46,evc50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_mtr_multi_commit_%j.out
#SBATCH --error=logs/q36_mtr_multi_commit_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE_COMMIT PYTHON APPLY_SCRIPT APPLY_SCRIPT_SHA256 ALIGNED_CHECKPOINT COMMIT_CHECKPOINT DEVELOPMENT_SOURCE CURRENT_CANDIDATES OWNER71_CANDIDATES OWNER8_CANDIDATES OUTPUT SELECTIONS REPORT ENVIRONMENT_RECEIPT ENVIRONMENT_RECEIPT_SHA256 ENVIRONMENT_TREE_SHA256; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_runtime
q36_require_authorization
q36_verify_model
q36_verify_environment
q36_verify_sha256 "$APPLY_SCRIPT" "$APPLY_SCRIPT_SHA256"
q36_verify_sha256 "$ALIGNED_CHECKPOINT" "$(q36_sha256 "$ALIGNED_CHECKPOINT")"
q36_verify_sha256 "$COMMIT_CHECKPOINT" "$(q36_sha256 "$COMMIT_CHECKPOINT")"
q36_verify_sha256 "$DEVELOPMENT_SOURCE" "$(q36_sha256 "$DEVELOPMENT_SOURCE")"
IFS=: read -r -a current_paths <<< "$CURRENT_CANDIDATES"
IFS=: read -r -a owner71_paths <<< "$OWNER71_CANDIDATES"
IFS=: read -r -a owner8_paths <<< "$OWNER8_CANDIDATES"
[[ ${#current_paths[@]} -eq 16 && ${#owner71_paths[@]} -eq 16 && ${#owner8_paths[@]} -eq 16 ]] || q36_die "multi-owner shard geometry differs"
for path in "${current_paths[@]}" "${owner71_paths[@]}" "${owner8_paths[@]}"; do
  q36_verify_sha256 "$path" "$(q36_sha256 "$path")"
done
for path in "$OUTPUT" "$SELECTIONS" "$REPORT"; do
  [[ ! -e "$path" && ! -L "$path" ]] || q36_die "multi-owner output exists"
done
model=$(q36_stage_model)
mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$SELECTIONS")" "$(dirname "$REPORT")" logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
q36_export_pythonpath
arguments=()
for path in "${current_paths[@]}"; do arguments+=(--current-candidates "$path"); done
for path in "${owner71_paths[@]}"; do arguments+=(--owner71-candidates "$path"); done
for path in "${owner8_paths[@]}"; do arguments+=(--owner8-candidates "$path"); done
nvidia-smi
"$PYTHON" -P -s -B "$APPLY_SCRIPT" \
  --model-root "$model" \
  --model-revision "$MODEL_REVISION" \
  --model-loader causal \
  --adapter-checkpoint "$ALIGNED_CHECKPOINT" \
  --commit-checkpoint "$COMMIT_CHECKPOINT" \
  --development-source "$DEVELOPMENT_SOURCE" \
  "${arguments[@]}" \
  --output "$OUTPUT" \
  --selections "$SELECTIONS" \
  --report "$REPORT" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --environment-receipt-sha256 "$ENVIRONMENT_RECEIPT_SHA256" \
  --environment-tree-sha256 "$ENVIRONMENT_TREE_SHA256" \
  --max-sequence-length 3072 --batch-identities 2
test -f "$OUTPUT" && test -f "$SELECTIONS" && test -f "$REPORT"
chmod a-w "$OUTPUT" "$SELECTIONS" "$REPORT"
