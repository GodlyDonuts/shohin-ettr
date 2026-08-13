#!/bin/bash
#SBATCH --job-name=q36-mtr-eval
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc33,evc37,evc38,evc43,evc46
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=02:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_mtr_eval_%j.out
#SBATCH --error=logs/q36_mtr_eval_%j.err

set -euo pipefail

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE_COMMIT PYTHON ARM SPLIT CHECKPOINT DATA DATA_REPORT SHARD_ROOT SHARD_COUNT ENVIRONMENT_RECEIPT ENVIRONMENT_RECEIPT_SHA256 ENVIRONMENT_TREE_SHA256; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-7]$ ]] || { printf 'SLURM_ARRAY_TASK_ID differs\n' >&2; exit 2; }
SHARD_INDEX=$SLURM_ARRAY_TASK_ID
OUTPUT=$SHARD_ROOT/$(printf 'shard_%02d/candidates.jsonl' "$SHARD_INDEX")
REPORT=$SHARD_ROOT/$(printf 'shard_%02d/report.json' "$SHARD_INDEX")
export OUTPUT
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_runtime
q36_require_authorization
q36_verify_model
CHECKPOINT_SHA256=$(q36_sha256 "$CHECKPOINT")
DATA_SHA256=$(q36_sha256 "$DATA")
DATA_REPORT_SHA256=$(q36_sha256 "$DATA_REPORT")
q36_verify_sha256 "$CHECKPOINT" "$CHECKPOINT_SHA256"
q36_verify_sha256 "$DATA" "$DATA_SHA256"
q36_verify_sha256 "$DATA_REPORT" "$DATA_REPORT_SHA256"
q36_verify_environment
case "$SPLIT:$ARM:$SHARD_COUNT" in
  calibration:revision:4|calibration:unchanged:4|development:revision:8|development:unchanged:8|development:self_refinement:8|development:draft_hidden:8) ;;
  *) q36_die "evaluation arm/split/shard geometry differs" ;;
esac
[[ "$SHARD_INDEX" =~ ^[0-7]$ && "$SHARD_INDEX" -lt "$SHARD_COUNT" ]] || \
  q36_die "evaluation shard index differs"
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" && ! -e "$REPORT" && ! -L "$REPORT" ]] || \
  q36_die "evaluation output exists"
sandbox=()
if [[ "$SPLIT" == calibration ]]; then
  SANDBOX_PROBE_OUTPUT=$SHARD_ROOT/$(printf 'shard_%02d/sandbox.json' "$SHARD_INDEX")
  [[ ! -e "$SANDBOX_PROBE_OUTPUT" && ! -L "$SANDBOX_PROBE_OUTPUT" ]] || \
    q36_die "sandbox probe output exists"
  sandbox=(--sandbox-probe-output "$SANDBOX_PROBE_OUTPUT")
elif [[ -n "${SANDBOX_PROBE_OUTPUT:-}" ]]; then
  q36_die "development must not receive a sandbox output"
fi
model=$(q36_stage_model)
mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$REPORT")" logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
q36_export_pythonpath
nvidia-smi
"$PYTHON" "$RUNTIME/train/hf_q36_mtr_evaluate.py" \
  --arm "$ARM" \
  --split "$SPLIT" \
  --model-root "$model" \
  --model-revision "$MODEL_REVISION" \
  --adapter-checkpoint "$CHECKPOINT" \
  --data "$DATA" \
  --data-report "$DATA_REPORT" \
  --candidates-output "$OUTPUT" \
  --report "$REPORT" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --environment-receipt-sha256 "$ENVIRONMENT_RECEIPT_SHA256" \
  --environment-tree-sha256 "$ENVIRONMENT_TREE_SHA256" \
  --batch-size 1 \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT" \
  --seed 2026080816 \
  "${sandbox[@]}"
test -f "$OUTPUT" && test -f "$REPORT"
chmod a-w "$OUTPUT" "$REPORT"
if [[ "$SPLIT" == calibration ]]; then
  test -f "$SANDBOX_PROBE_OUTPUT"
  chmod a-w "$SANDBOX_PROBE_OUTPUT"
fi
