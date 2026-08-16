#!/bin/bash
#SBATCH --job-name=pcf1-precompute-custody
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc33,evc37,evc38,evc46
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pcf1_precompute_custody_%j.out
#SBATCH --error=logs/pcf1_precompute_custody_%j.err

set -euo pipefail

: "${RUNTIME:?}" "${RUNTIME_MANIFEST_SHA256:?}"
test -f "$RUNTIME/SHA256SUMS"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
source "$RUNTIME/train/jobs/pcf1_common.sh"
pcf1_verify_runtime
pcf1_bind_prepared_inputs
pcf1_verify_model
for variable in RUN_ID SOURCE_FREEZE_REPORT TRAIN_SOURCES DEVELOPMENT_SOURCES REFERENCE_PREFLIGHT_ROWS REFERENCE_SANDBOX_RECEIPT MERGED_DRAFTS MERGED_DRAFTS_REPORT REVISION_TRAINING_DATA CALIBRATION_DATA CONFIRMATION_DATA CONFIRMATION_ASSESSOR_RECEIPT DATA_REPORT CALIBRATION_PAIRS CALIBRATION_PAIR_REPORT CONFIRMATION_PAIRS CONFIRMATION_PAIR_REPORT CALIBRATION_REVISION_REPORT CALIBRATION_UNCHANGED_REPORT CALIBRATION_REVISION_SHARD_ROOT CALIBRATION_UNCHANGED_SHARD_ROOT REVISION_REPORT UNCHANGED_REPORT SELF_REFINEMENT_REPORT B1_CHECKPOINT B1_TRAINING_REPORT REVISION_CHECKPOINT REVISION_TRAINING_REPORT COMMIT_CHECKPOINT COMMIT_TRAINING_REPORT COMMIT_APPLICATION_REPORT CONFIRMATION_SELECTIONS MECHANICS_REPORT COMPUTE_HOST_RECEIPT SANDBOX_RECEIPT OUTPUT_ROOT; do
  pcf1_require "$variable"
done
for path in "$SOURCE_FREEZE_REPORT" "$TRAIN_SOURCES" "$DEVELOPMENT_SOURCES" "$REFERENCE_PREFLIGHT_ROWS" "$REFERENCE_SANDBOX_RECEIPT" "$MERGED_DRAFTS" "$MERGED_DRAFTS_REPORT" "$REVISION_TRAINING_DATA" "$CALIBRATION_DATA" "$CONFIRMATION_DATA" "$CONFIRMATION_ASSESSOR_RECEIPT" "$DATA_REPORT" "$CALIBRATION_PAIRS" "$CALIBRATION_PAIR_REPORT" "$CONFIRMATION_PAIRS" "$CONFIRMATION_PAIR_REPORT" "$CALIBRATION_REVISION_REPORT" "$CALIBRATION_UNCHANGED_REPORT" "$REVISION_REPORT" "$UNCHANGED_REPORT" "$SELF_REFINEMENT_REPORT" "$B1_CHECKPOINT" "$B1_TRAINING_REPORT" "$REVISION_CHECKPOINT" "$REVISION_TRAINING_REPORT" "$COMMIT_CHECKPOINT" "$COMMIT_TRAINING_REPORT" "$COMMIT_APPLICATION_REPORT" "$CONFIRMATION_SELECTIONS" "$MECHANICS_REPORT" "$COMPUTE_HOST_RECEIPT"; do
  pcf1_require_safe_input "$path"
done
pcf1_require_fresh "$OUTPUT_ROOT"
python=$(pcf1_python)
sandbox_receipt_sha256=$(pcf1_validate_sandbox_receipt "$SANDBOX_RECEIPT")
calibration_sandbox_arguments=()
for ((index = 0; index < 4; index++)); do
  revision_probe=$CALIBRATION_REVISION_SHARD_ROOT/shards/$index/sandbox_probe.json
  unchanged_probe=$CALIBRATION_UNCHANGED_SHARD_ROOT/shards/$index/sandbox_probe.json
  pcf1_require_safe_input "$revision_probe"
  pcf1_require_safe_input "$unchanged_probe"
  calibration_sandbox_arguments+=(
    --calibration-revision-sandbox-probe "$revision_probe"
    --calibration-unchanged-sandbox-probe "$unchanged_probe"
  )
done
pcf1_export_runtime
"$python" "$RUNTIME/pipeline/build_pcf1_custody.py" precompute \
  --run-id "$RUN_ID" \
  --source-freeze-report "$SOURCE_FREEZE_REPORT" \
  --train-sources "$TRAIN_SOURCES" \
  --development-sources "$DEVELOPMENT_SOURCES" \
  --reference-preflight-rows "$REFERENCE_PREFLIGHT_ROWS" \
  --reference-sandbox-receipt "$REFERENCE_SANDBOX_RECEIPT" \
  --merged-drafts "$MERGED_DRAFTS" \
  --merged-drafts-report "$MERGED_DRAFTS_REPORT" \
  --revision-training-data "$REVISION_TRAINING_DATA" \
  --calibration-data "$CALIBRATION_DATA" \
  --confirmation-data "$CONFIRMATION_DATA" \
  --confirmation-assessor-receipt "$CONFIRMATION_ASSESSOR_RECEIPT" \
  --data-report "$DATA_REPORT" \
  --calibration-pairs "$CALIBRATION_PAIRS" \
  --calibration-pair-report "$CALIBRATION_PAIR_REPORT" \
  --confirmation-pairs "$CONFIRMATION_PAIRS" \
  --confirmation-pair-report "$CONFIRMATION_PAIR_REPORT" \
  --calibration-revision-report "$CALIBRATION_REVISION_REPORT" \
  --calibration-unchanged-report "$CALIBRATION_UNCHANGED_REPORT" \
  "${calibration_sandbox_arguments[@]}" \
  --revision-report "$REVISION_REPORT" \
  --unchanged-report "$UNCHANGED_REPORT" \
  --self-refinement-report "$SELF_REFINEMENT_REPORT" \
  --b1-checkpoint "$B1_CHECKPOINT" \
  --b1-training-report "$B1_TRAINING_REPORT" \
  --revision-checkpoint "$REVISION_CHECKPOINT" \
  --revision-training-report "$REVISION_TRAINING_REPORT" \
  --commit-checkpoint "$COMMIT_CHECKPOINT" \
  --commit-training-report "$COMMIT_TRAINING_REPORT" \
  --commit-application-report "$COMMIT_APPLICATION_REPORT" \
  --confirmation-selections "$CONFIRMATION_SELECTIONS" \
  --mechanics-report "$MECHANICS_REPORT" \
  --compute-host-receipt "$COMPUTE_HOST_RECEIPT" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --environment-receipt-sha256 "$ENVIRONMENT_RECEIPT_SHA256" \
  --sandbox-receipt "$SANDBOX_RECEIPT" \
  --sandbox-receipt-sha256 "$sandbox_receipt_sha256" \
  --model-root "$MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --model-manifest "$MODEL_MANIFEST" \
  --model-manifest-sha256 "$MODEL_MANIFEST_SHA256" \
  --model-config-sha256 "$MODEL_CONFIG_SHA256" \
  --runtime-root "$RUNTIME" \
  --runtime-manifest "$RUNTIME/SHA256SUMS" \
  --runtime-manifest-sha256 "$RUNTIME_MANIFEST_SHA256" \
  --output "$OUTPUT_ROOT"
pcf1_validate_json "$OUTPUT_ROOT/data_custody.json" shohin-pcf1-data-custody-v1
pcf1_validate_json "$OUTPUT_ROOT/model_custody.json" shohin-pcf1-model-custody-v1
pcf1_validate_json "$OUTPUT_ROOT/runtime_custody.json" shohin-pcf1-runtime-custody-v1
pcf1_freeze_tree "$OUTPUT_ROOT"
