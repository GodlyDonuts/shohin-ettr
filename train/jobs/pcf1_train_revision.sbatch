#!/bin/bash
#SBATCH --job-name=pcf1-revision
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc33,evc38,evc46
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/pcf1_revision_%j.out
#SBATCH --error=logs/pcf1_revision_%j.err

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
for variable in DATA DATA_REPORT B1_CHECKPOINT B1_RECEIPT OUTPUT; do
  pcf1_require "$variable"
done
pcf1_require_safe_input "$DATA"
pcf1_require_safe_input "$DATA_REPORT"
pcf1_require_safe_input "$B1_CHECKPOINT"
pcf1_validate_b1_receipt "$B1_RECEIPT" "$B1_CHECKPOINT"
pcf1_validate_json "$DATA_REPORT" shohin-pcf1-data-report-v1
pcf1_require_fresh "$OUTPUT"
python=$(pcf1_python)
"$python" - "$DATA" "$DATA_REPORT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

data, report_path = map(Path, sys.argv[1:])
report = json.loads(report_path.read_text(encoding="utf-8"))
receipt = report.get("outputs", {}).get("revision_train", {})
if (
    Path(str(receipt.get("path", ""))).resolve() != data.resolve()
    or receipt.get("sha256") != hashlib.sha256(data.read_bytes()).hexdigest()
):
    raise SystemExit("PCF1 revision data binding differs")
PY
pcf1_initialize_scratch
model=$(pcf1_stage_model)
pcf1_export_runtime
nvidia-smi
"$python" "$RUNTIME/train/hf_product_reasoning_train.py" \
  --model-root "$model" \
  --model-source-root "$MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --model-loader multimodal \
  --data "$DATA" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --environment-receipt-sha256 "$ENVIRONMENT_RECEIPT_SHA256" \
  --output "$OUTPUT" \
  --warm-start-checkpoint "$B1_CHECKPOINT" \
  --arm baseline \
  --updates 256 \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --max-rows 10000 \
  --max-sequence-length 4096 \
  --learning-rate 2e-5 \
  --lora-layers 4 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-scope token_mixer \
  --seed 2026080815 \
  --data-seed 2026080814 \
  --log-interval 8 \
  --checkpoint-interval 256
pcf1_require_file "$OUTPUT/checkpoint_0000256.pt"
pcf1_validate_json "$OUTPUT/report.json" shohin-hf-product-reasoning-training-v1
"$python" - "$OUTPUT/report.json" "$ENVIRONMENT_RECEIPT_SHA256" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    report.get("environment_receipt_sha256") != sys.argv[2]
    or not isinstance(report.get("environment_tree_sha256"), str)
    or len(report["environment_tree_sha256"]) != 64
):
    raise SystemExit("PCF1 revision training environment differs")
PY
pcf1_freeze_tree "$OUTPUT"
