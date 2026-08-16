#!/bin/bash
#SBATCH --job-name=q36-mtr-role
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc33,evc34,evc37,evc38,evc43,evc46,evc50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_mtr_role_%j.out
#SBATCH --error=logs/q36_mtr_role_%j.err

set -euo pipefail

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE_COMMIT PYTHON ROLE DATA OUTPUT ENVIRONMENT_RECEIPT ENVIRONMENT_RECEIPT_SHA256 ENVIRONMENT_TREE_SHA256; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_runtime
q36_require_authorization
q36_verify_model
DATA_SHA256=$(q36_sha256 "$DATA")
q36_verify_sha256 "$DATA" "$DATA_SHA256"
q36_verify_environment
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || q36_die "role output exists"
trainer=${TRAIN_SCRIPT:-$RUNTIME/train/hf_q36_mtr_train_role.py}
[[ -f "$trainer" && ! -L "$trainer" ]] || q36_die "role trainer differs"
if [[ -n "${TRAIN_SCRIPT:-}" ]]; then
  q36_require TRAIN_SCRIPT_SHA256
  q36_verify_sha256 "$trainer" "$TRAIN_SCRIPT_SHA256"
fi
sequence=()
if [[ -n "${MAX_SEQUENCE_LENGTH:-}" ]]; then
  [[ "$MAX_SEQUENCE_LENGTH" == "4224" && "$ROLE" == "aligned" ]] || \
    q36_die "engineering sequence extension differs"
  sequence=(
    --max-sequence-length "$MAX_SEQUENCE_LENGTH"
    --engineering-sequence-extension
    --engineering-loss-chunk-size 512
  )
fi
case "$ROLE" in
  owner)
    [[ -z "${WARM_START_CHECKPOINT:-}" && -z "${WARM_START_CHECKPOINT_SHA256:-}" ]] || \
      q36_die "owner must not warm start"
    warm=()
    ;;
  aligned|draft_hidden)
    q36_require WARM_START_CHECKPOINT
    WARM_START_CHECKPOINT_SHA256=$(q36_sha256 "$WARM_START_CHECKPOINT")
    q36_verify_sha256 "$WARM_START_CHECKPOINT" "$WARM_START_CHECKPOINT_SHA256"
    warm=(--warm-start-checkpoint "$WARM_START_CHECKPOINT")
    ;;
  *) q36_die "role differs" ;;
esac
model=$(q36_stage_model)
mkdir -p "$(dirname "$OUTPUT")" logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
# The longest bounded synthesis presentations approach the H100 allocation
# ceiling. Grow CUDA segments instead of stranding reserved holes; this changes
# allocator mechanics, not examples, update budget, or loss.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
q36_export_pythonpath
nvidia-smi
"$PYTHON" "$trainer" \
  --role "$ROLE" \
  --model-root "$model" \
  --model-source-root "$MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --model-config-sha256 "$MODEL_CONFIG_SHA256" \
  --quantization nf4 \
  --data "$DATA" \
  --output "$OUTPUT" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --environment-receipt-sha256 "$ENVIRONMENT_RECEIPT_SHA256" \
  --environment-tree-sha256 "$ENVIRONMENT_TREE_SHA256" \
  "${sequence[@]}" \
  "${warm[@]}"
test -f "$OUTPUT/checkpoint_0000256.pt" && test -f "$OUTPUT/report.json"
chmod -R a-w "$OUTPUT"
