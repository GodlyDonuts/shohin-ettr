#!/bin/bash
#SBATCH --job-name=pcf1-drafts
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc38,evc46
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/pcf1_drafts_%A_%a.out
#SBATCH --error=logs/pcf1_drafts_%A_%a.err

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
for variable in B1_CHECKPOINT B1_REPORT B1_RECEIPT DRAFT_ROOT DRAFT_SHARDS; do
  pcf1_require "$variable"
done
[[ "$DRAFT_SHARDS" == 16 ]] || pcf1_die "DRAFT_SHARDS must equal frozen value 16"
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || pcf1_die "draft job must run as an array"
((SLURM_ARRAY_TASK_ID < DRAFT_SHARDS)) || pcf1_die "draft shard index differs"
pcf1_reject_protected_path "$SOURCE_ROOT"
pcf1_require_file "$SOURCE_ROOT/report.json"
pcf1_require_safe_input "$B1_CHECKPOINT"
pcf1_require_safe_input "$B1_REPORT"
pcf1_validate_json "$B1_REPORT" shohin-hf-product-reasoning-training-v1
pcf1_validate_b1_receipt "$B1_RECEIPT" "$B1_CHECKPOINT"
python=$(pcf1_python)
"$python" - "$B1_REPORT" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    report.get("updates") != 256
    or report.get("gradient_accumulation") != 16
    or report.get("max_sequence_length") != 1024
    or report.get("learning_rate") != 2e-4
    or report.get("lora_scope") != "token_mixer"
    or report.get("lora_layer_indices") != [30, 31, 32, 33]
    or report.get("seed") != 2026080711
    or report.get("data_seed") != 20260802
):
    raise SystemExit("PCF1 B1 schedule differs")
PY
shard_root=$DRAFT_ROOT/shards/$SLURM_ARRAY_TASK_ID
pcf1_require_fresh "$shard_root/candidates.jsonl"
pcf1_require_fresh "$shard_root/report.json"
model=$(pcf1_stage_model)
pcf1_export_runtime
nvidia-smi
"$python" "$RUNTIME/train/hf_pcf1_generate_drafts.py" \
  --model-root "$model" \
  --model-source-root "$MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --model-loader multimodal \
  --adapter-checkpoint "$B1_CHECKPOINT" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --environment-receipt-sha256 "$ENVIRONMENT_RECEIPT_SHA256" \
  --source-root "$SOURCE_ROOT" \
  --candidates-output "$shard_root/candidates.jsonl" \
  --report "$shard_root/report.json" \
  --batch-size 2 \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --shard-count "$DRAFT_SHARDS" \
  --seed 2026080818
pcf1_validate_json "$shard_root/report.json" shohin-pcf1-draft-shard-v1
pcf1_freeze_tree "$shard_root"
