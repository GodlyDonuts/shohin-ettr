#!/bin/bash
#SBATCH --job-name=q36-external-evaluate
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc33,evc34,evc37,evc38,evc43,evc46,evc50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_external_evaluate_%A_%a.out
#SBATCH --error=logs/q36_external_evaluate_%A_%a.err

set -euo pipefail

required=(RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE_COMMIT PYTHON MODEL_ROOT MODEL_MANIFEST MODEL_MANIFEST_SHA256 MODEL_REVISION MODEL_CONFIG_SHA256 ENVIRONMENT_RECEIPT ENVIRONMENT_RECEIPT_SHA256 ENVIRONMENT_TREE_SHA256 EXTERNAL_SCRIPT EXTERNAL_SCRIPT_SHA256 ARM ADAPTER_CHECKPOINT SOURCE SOURCE_SHA256 OUTPUT_ROOT)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
EXPECTED_ROWS=${EXPECTED_ROWS:-256}
SHARD_COUNT=${SHARD_COUNT:-4}
case "$EXPECTED_ROWS:$SHARD_COUNT" in
  256:4) [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-3]$ ]] ;;
  1279:16) [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^([0-9]|1[0-5])$ ]] ;;
  *) false ;;
esac || { printf 'external geometry differs\n' >&2; exit 2; }
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_runtime
q36_verify_model
q36_verify_environment
q36_verify_sha256 "$EXTERNAL_SCRIPT" "$EXTERNAL_SCRIPT_SHA256"
q36_verify_sha256 "$SOURCE" "$SOURCE_SHA256"
[[ -f "$ADAPTER_CHECKPOINT" && ! -L "$ADAPTER_CHECKPOINT" ]] || q36_die "adapter checkpoint differs"
draft_args=()
if [[ "$ARM" != unchanged ]]; then
  [[ -n "${DRAFT_CANDIDATES:-}" ]] || q36_die "draft candidates are required"
  IFS=: read -r -a draft_paths <<< "$DRAFT_CANDIDATES"
  [[ ${#draft_paths[@]} -eq $SHARD_COUNT ]] || q36_die "draft candidate geometry differs"
  for path in "${draft_paths[@]}"; do
    [[ -f "$path" && ! -L "$path" ]] || q36_die "draft candidate differs"
    draft_args+=(--draft-candidates "$path")
  done
fi
shard=$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
output=$OUTPUT_ROOT/$ARM/shard_$shard/candidates.jsonl
report=$OUTPUT_ROOT/$ARM/shard_$shard/report.json
[[ ! -e "$output" && ! -L "$output" && ! -e "$report" && ! -L "$report" ]] || q36_die "external output exists"
mkdir -p "$(dirname "$output")" logs
model=$(q36_stage_model)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
q36_export_pythonpath
nvidia-smi
"$PYTHON" "$EXTERNAL_SCRIPT" \
  --arm "$ARM" --model-root "$model" --model-revision "$MODEL_REVISION" \
  --adapter-checkpoint "$ADAPTER_CHECKPOINT" --source "$SOURCE" \
  --source-sha256 "$SOURCE_SHA256" "${draft_args[@]}" \
  --candidates-output "$output" --report "$report" --expected-rows "$EXPECTED_ROWS" \
  --shard-index "$SLURM_ARRAY_TASK_ID" --shard-count "$SHARD_COUNT" --batch-size 2
test -f "$output" && test -f "$report"
chmod a-w "$output" "$report"
