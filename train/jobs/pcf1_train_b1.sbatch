#!/bin/bash
#SBATCH --job-name=pcf1-b1
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc33,evc37,evc38,evc46
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/pcf1_b1_%j.out
#SBATCH --error=logs/pcf1_b1_%j.err

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
for variable in MECHANICS_REPORT SANDBOX_RECEIPT COMPUTE_HOST_RECEIPT OUTPUT; do
  pcf1_require "$variable"
done
[[ "$B1_DATA_SHA256" == 2461d6f70b44a142854d56c24e1fb42d600065e5788a2c4e055ba47b12696549 ]] || \
  pcf1_die "B1 corpus differs from the frozen PCF1 contract"
pcf1_require_safe_input "$B1_DATA"
pcf1_verify_sha256 "$B1_DATA" "$B1_DATA_SHA256"
pcf1_require_safe_input "$MECHANICS_REPORT"
pcf1_require_safe_input "$COMPUTE_HOST_RECEIPT"
pcf1_validate_json "$MECHANICS_REPORT" shohin-pcf1-mechanics-v1 pass
sandbox_receipt_sha256=$(pcf1_validate_sandbox_receipt "$SANDBOX_RECEIPT")
pcf1_require_fresh "$OUTPUT"
python=$(pcf1_python)
"$python" - "$MECHANICS_REPORT" "$SANDBOX_RECEIPT" "$sandbox_receipt_sha256" "$COMPUTE_HOST_RECEIPT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
sandbox = Path(sys.argv[2])
sandbox_sha256 = sys.argv[3]
compute_host = Path(sys.argv[4])
if (
    report.get("capability_scored") is not False
    or report.get("rows") != 24
    or report.get("commit_ab_order_checks") != 24
    or report.get("commit_ab_serialization_exact") is not True
    or report.get("commit_forward_swapped_exact") is not True
    or report.get("optimizer_updates") != 1
    or report.get("optimizer_presentations") != 24
    or report.get("lora_layer_indices") != [30, 31, 32, 33]
    or report.get("task_router_used") is not False
    or report.get("revision_prompt_parameters") != ["source_prompt", "draft"]
    or report.get("supervisor_fields_visible_to_model") is not False
    or Path(str(report.get("sandbox_receipt", ""))).resolve() != sandbox.resolve()
    or report.get("sandbox_receipt_sha256") != sandbox_sha256
    or report.get("sandbox_isolation_passed") is not True
    or Path(str(report.get("compute_host_receipt", ""))).resolve()
    != compute_host.resolve()
    or report.get("compute_host_receipt_sha256")
    != hashlib.sha256(compute_host.read_bytes()).hexdigest()
    or not isinstance(report.get("compute_host_receipt_sha256"), str)
    or len(report["compute_host_receipt_sha256"]) != 64
    or not isinstance(report.get("nvidia_smi_sha256"), str)
    or len(report["nvidia_smi_sha256"]) != 64
    or report.get("qualified_gpu_name") != "NVIDIA H100 PCIe"
):
    raise SystemExit("PCF1 mechanics gate differs")
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
  --data "$B1_DATA" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --environment-receipt-sha256 "$ENVIRONMENT_RECEIPT_SHA256" \
  --output "$OUTPUT" \
  --arm baseline \
  --updates 256 \
  --batch-size 1 \
  --gradient-accumulation 16 \
  --max-rows 100000 \
  --max-sequence-length 1024 \
  --learning-rate 2e-4 \
  --lora-layers 4 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-scope token_mixer \
  --seed 2026080711 \
  --data-seed 20260802 \
  --log-interval 8 \
  --checkpoint-interval 256
pcf1_require_file "$OUTPUT/checkpoint_0000256.pt"
pcf1_validate_json "$OUTPUT/report.json" shohin-hf-product-reasoning-training-v1
pcf1_require_fresh "$OUTPUT/pcf1_receipt.json"
"$python" - "$B1_DATA" "$OUTPUT/checkpoint_0000256.pt" "$OUTPUT/report.json" "$ENVIRONMENT_RECEIPT" "$ENVIRONMENT_RECEIPT_SHA256" "$OUTPUT/pcf1_receipt.json" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

data, checkpoint, training_report, environment, environment_sha, output = sys.argv[1:]
data, checkpoint, training_report, environment, output = map(
    Path, (data, checkpoint, training_report, environment, output)
)
training = json.loads(training_report.read_text(encoding="utf-8"))
if training.get("lora_layer_indices") != [30, 31, 32, 33]:
    raise SystemExit("PCF1 B1 final-four layers differ")
if hashlib.sha256(environment.read_bytes()).hexdigest() != environment_sha:
    raise SystemExit("PCF1 B1 environment receipt differs")
if (
    training.get("environment_receipt_sha256") != environment_sha
    or not isinstance(training.get("environment_tree_sha256"), str)
    or len(training["environment_tree_sha256"]) != 64
):
    raise SystemExit("PCF1 B1 training environment differs")
payload = {
    "schema": "shohin-pcf1-b1-training-receipt-v1",
    "status": "complete",
    "data": str(data.resolve()),
    "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
    "checkpoint": str(checkpoint.resolve()),
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "training_report": str(training_report.resolve()),
    "training_report_sha256": hashlib.sha256(training_report.read_bytes()).hexdigest(),
    "updates": 256,
    "gradient_accumulation": 16,
    "max_sequence_length": 1024,
    "learning_rate": 2e-4,
    "lora_layer_indices": [30, 31, 32, 33],
    "environment_receipt": str(environment.resolve()),
    "environment_receipt_sha256": environment_sha,
    "environment_tree_sha256": training["environment_tree_sha256"],
    "seed": 2026080711,
    "data_seed": 20260802,
    "legacy_control_plane_schema_exact_hash_exception": True,
    "legacy_schema_model_visible": False,
    "qualified_lineage_preserved": True,
    "sealed_access": {"holdout": 0, "product": 0, "public": 0},
}
temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
with temporary.open("xb") as handle:
    handle.write(encoded)
    handle.flush()
    os.fsync(handle.fileno())
os.link(temporary, output)
temporary.unlink()
PY
pcf1_validate_b1_receipt "$OUTPUT/pcf1_receipt.json" "$OUTPUT/checkpoint_0000256.pt"
pcf1_freeze_tree "$OUTPUT"
