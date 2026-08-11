#!/bin/bash
#SBATCH --job-name=pcf1-commit-apply
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc38,evc46
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/pcf1_commit_apply_%j.out
#SBATCH --error=logs/pcf1_commit_apply_%j.err

set -euo pipefail

: "${RUNTIME:?}" "${RUNTIME_MANIFEST_SHA256:?}"
test -f "$RUNTIME/SHA256SUMS"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
source "$RUNTIME/train/jobs/pcf1_common.sh"
pcf1_assert_gpu_environment
pcf1_verify_runtime
pcf1_bind_prepared_inputs
pcf1_verify_model
for variable in B1_CHECKPOINT B1_RECEIPT COMMIT_CHECKPOINT PAIRS PAIRS_REPORT SELECTIONS REPORT; do
  pcf1_require "$variable"
done
pcf1_require_safe_input "$B1_CHECKPOINT"
pcf1_validate_b1_receipt "$B1_RECEIPT" "$B1_CHECKPOINT"
pcf1_require_safe_input "$COMMIT_CHECKPOINT"
pcf1_require_safe_input "$PAIRS"
pcf1_require_safe_input "$PAIRS_REPORT"
pcf1_validate_json "$PAIRS_REPORT" shohin-pcf1-confirmation-pair-report-v1
pcf1_require_fresh "$SELECTIONS"
pcf1_require_fresh "$REPORT"
python=$(pcf1_python)
pcf1_initialize_scratch
model=$(pcf1_stage_model)
pcf1_export_runtime
nvidia-smi
"$python" "$RUNTIME/train/hf_pcf1_apply_commit.py" \
  --model-root "$model" \
  --model-source-root "$MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --model-loader multimodal \
  --adapter-checkpoint "$B1_CHECKPOINT" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --environment-receipt-sha256 "$ENVIRONMENT_RECEIPT_SHA256" \
  --commit-checkpoint "$COMMIT_CHECKPOINT" \
  --pairs "$PAIRS" \
  --pairs-report "$PAIRS_REPORT" \
  --selections "$SELECTIONS" \
  --report "$REPORT" \
  --max-sequence-length 3072 \
  --batch-pairs 2
pcf1_validate_json "$REPORT" shohin-pcf1-commit-application-report-v1
chmod a-w "$SELECTIONS" "$REPORT"
