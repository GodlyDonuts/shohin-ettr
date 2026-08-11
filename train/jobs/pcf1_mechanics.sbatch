#!/bin/bash
#SBATCH --job-name=pcf1-mechanics
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc38,evc46
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/pcf1_mechanics_%j.out
#SBATCH --error=logs/pcf1_mechanics_%j.err

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
for variable in MECHANICS_ROOT; do
  pcf1_require "$variable"
done
[[ "$B1_DATA_SHA256" == 2461d6f70b44a142854d56c24e1fb42d600065e5788a2c4e055ba47b12696549 ]] || \
  pcf1_die "B1 corpus differs from the frozen PCF1 contract"
pcf1_require_safe_input "$B1_DATA"
pcf1_verify_sha256 "$B1_DATA" "$B1_DATA_SHA256"
pcf1_reject_protected_path "$SOURCE_ROOT"
pcf1_require_file "$SOURCE_ROOT/report.json"
pcf1_require_fresh "$MECHANICS_ROOT"
pcf1_initialize_scratch
freeze_mechanics_evidence() {
  local status=$?
  trap - EXIT
  [[ ! -e "$MECHANICS_ROOT" ]] || chmod -R a-w "$MECHANICS_ROOT"
  if ! pcf1_cleanup_scratch; then
    [[ "$status" -ne 0 ]] || status=1
  fi
  exit "$status"
}
trap freeze_mechanics_evidence EXIT
python=$(pcf1_python)
pcf1_export_runtime

nvidia_smi=$(command -v nvidia-smi || true)
[[ "$nvidia_smi" == /* ]] || pcf1_die "nvidia-smi is absent on the compute allocation"
compute_host_receipt=$MECHANICS_ROOT/compute_host_receipt.json
"$python" - "$nvidia_smi" "$compute_host_receipt" "$PCF1_EXCLUDED_NODES" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

invoked = Path(sys.argv[1])
output = Path(sys.argv[2])
excluded = set(sys.argv[3].split(","))
resolved = invoked.resolve(strict=True)
if not resolved.is_file():
    raise SystemExit("PCF1 nvidia-smi binary differs")
binary_sha256 = hashlib.sha256(resolved.read_bytes()).hexdigest()
version = subprocess.run(
    [str(invoked), "--version"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
query = subprocess.run(
    [
        str(invoked),
        "--query-gpu=name,driver_version,pci.bus_id",
        "--format=csv,noheader,nounits",
    ],
    check=True,
    capture_output=True,
    text=True,
).stdout.splitlines()
if len(query) != 1:
    raise SystemExit("PCF1 mechanics requires exactly one visible GPU")
fields = [field.strip() for field in query[0].split(",")]
if len(fields) != 3 or fields[0].casefold() != "nvidia h100 pcie":
    raise SystemExit("PCF1 mechanics GPU model differs")
node = os.environ.get("SLURMD_NODENAME", "")
partition = os.environ.get("SLURM_JOB_PARTITION", "")
if not node or node in excluded or partition != "normal":
    raise SystemExit("PCF1 mechanics allocation identity differs")
payload = {
    "schema": "shohin-pcf1-compute-host-receipt-v1",
    "status": "complete",
    "partition": partition,
    "node": node,
    "excluded_nodes": sorted(excluded),
    "nvidia_smi_invoked_path": str(invoked),
    "nvidia_smi_resolved_path": str(resolved),
    "nvidia_smi_sha256": binary_sha256,
    "nvidia_smi_version": version,
    "visible_gpu_count": 1,
    "gpu_name": fields[0],
    "driver_version": fields[1],
    "pci_bus_id": fields[2],
}
output.parent.mkdir(parents=True, exist_ok=True)
temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
with temporary.open("xb") as handle:
    handle.write(encoded)
    handle.flush()
    os.fsync(handle.fileno())
try:
    os.link(temporary, output)
finally:
    temporary.unlink(missing_ok=True)
parent = os.open(output.parent, os.O_RDONLY)
try:
    os.fsync(parent)
finally:
    os.close(parent)
PY
compute_host_receipt_sha256=$(pcf1_sha256 "$compute_host_receipt")

sandbox_receipt=$MECHANICS_ROOT/sandbox_receipt.json
"$python" "$RUNTIME/train/pcf1_code_sandbox.py" qualify \
  --output "$sandbox_receipt"
sandbox_receipt_sha256=$(pcf1_validate_sandbox_receipt "$sandbox_receipt")

pcf1_initialize_scratch
model=$(pcf1_stage_model)
"$nvidia_smi"
adapter_root=$MECHANICS_ROOT/ephemeral_adapter
"$python" "$RUNTIME/train/hf_product_reasoning_train.py" \
  --model-root "$model" \
  --model-source-root "$MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --model-loader multimodal \
  --data "$B1_DATA" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --environment-receipt-sha256 "$ENVIRONMENT_RECEIPT_SHA256" \
  --output "$adapter_root" \
  --arm baseline \
  --updates 1 \
  --batch-size 1 \
  --gradient-accumulation 24 \
  --max-rows 100000 \
  --max-sequence-length 1024 \
  --learning-rate 2e-4 \
  --lora-layers 4 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-scope token_mixer \
  --seed 2026080711 \
  --data-seed 20260802 \
  --log-interval 1 \
  --checkpoint-interval 1

checkpoint=$adapter_root/checkpoint_0000001.pt
pcf1_require_file "$checkpoint"
checkpoint_sha256=$(pcf1_sha256 "$checkpoint")
"$python" "$RUNTIME/train/hf_pcf1_mechanics.py" \
  --model-root "$model" \
  --model-source-root "$MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --model-config-sha256 "$MODEL_CONFIG_SHA256" \
  --model-manifest-sha256 "$MODEL_MANIFEST_SHA256" \
  --runtime-manifest-sha256 "$RUNTIME_MANIFEST_SHA256" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --environment-receipt-sha256 "$ENVIRONMENT_RECEIPT_SHA256" \
  --sandbox-receipt "$sandbox_receipt" \
  --sandbox-receipt-sha256 "$sandbox_receipt_sha256" \
  --compute-host-receipt "$compute_host_receipt" \
  --compute-host-receipt-sha256 "$compute_host_receipt_sha256" \
  --model-loader multimodal \
  --adapter-checkpoint "$checkpoint" \
  --adapter-checkpoint-sha256 "$checkpoint_sha256" \
  --training-report "$adapter_root/report.json" \
  --source-root "$SOURCE_ROOT" \
  --report "$MECHANICS_ROOT/report.json"
pcf1_validate_json "$MECHANICS_ROOT/report.json" shohin-pcf1-mechanics-v1 pass
pcf1_freeze_tree "$MECHANICS_ROOT"
pcf1_cleanup_scratch
trap - EXIT
