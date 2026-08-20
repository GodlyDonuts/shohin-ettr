#!/bin/bash
#SBATCH --job-name=nemotron-super-fp8-probe
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc33,evc34,evc37,evc38,evc50
#SBATCH --output=logs/nemotron_super_fp8_probe_%j.out
#SBATCH --error=logs/nemotron_super_fp8_probe_%j.err

set -euo pipefail

: "${PYTHON:?PYTHON is required}"
: "${RUNTIME:?RUNTIME is required}"
: "${OVERLAY_ROOT:?OVERLAY_ROOT is required}"
: "${OVERLAY_MANIFEST_SHA256:?OVERLAY_MANIFEST_SHA256 is required}"
: "${CAUSAL_CONV_ROOT:?CAUSAL_CONV_ROOT is required}"
: "${OUTPUT:?OUTPUT is required}"

[[ -x "$PYTHON" ]]
[[ -d "$RUNTIME" && ! -L "$RUNTIME" ]]
[[ -d "$OVERLAY_ROOT" && ! -L "$OVERLAY_ROOT" ]]
[[ -d "$CAUSAL_CONV_ROOT" && ! -L "$CAUSAL_CONV_ROOT" ]]
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]]
[[ "${SLURM_GPUS_ON_NODE:-}" == "1" ]]
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_overlay "$OVERLAY_ROOT" "$OVERLAY_MANIFEST_SHA256"
q36_export_nemotron_cuda_toolchain
mkdir -p logs "$(dirname "$OUTPUT")"

export PYTHONPATH="$OVERLAY_ROOT:$CAUSAL_CONV_ROOT"
export PYTHONDONTWRITEBYTECODE=1 CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

"$PYTHON" -P -s -B - "$OUTPUT" <<'PY'
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import torch
from modelopt.torch.quantization.extensions import get_cuda_ext_fp8

output = Path(sys.argv[1])
extension = get_cuda_ext_fp8(raise_if_failed=True)
extension_path = Path(extension.__file__).resolve(strict=True)
inputs = torch.tensor([0.0, 1.0, -1.0, 0.5], device="cuda", dtype=torch.float32)
amax = torch.tensor([1.0], device="cuda", dtype=torch.float32)
result = extension.fake_e4m3fy(inputs, amax)
if (
    result.shape != inputs.shape
    or result.device.type != "cuda"
    or not bool(torch.isfinite(result).all().item())
    or bool(torch.count_nonzero(result).item()) is False
):
    raise SystemExit("Nemotron FP8 extension functional result differs")
values = [float(value) for value in result.detach().cpu().tolist()]
payload = {
    "schema": "shohin-nemotron-super-fp8-extension-probe-v1",
    "status": "pass",
    "slurm_job_id": int(os.environ["SLURM_JOB_ID"]),
    "node": os.environ["SLURMD_NODENAME"],
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "modelopt": importlib.metadata.version("nvidia-modelopt"),
    "cuda_home": os.environ["CUDA_HOME"],
    "nvcc_sha256": hashlib.sha256(
        Path(os.environ["CUDA_HOME"], "bin", "nvcc").read_bytes()
    ).hexdigest(),
    "cc_sha256": hashlib.sha256(Path(os.environ["CC"]).read_bytes()).hexdigest(),
    "cxx_sha256": hashlib.sha256(Path(os.environ["CXX"]).read_bytes()).hexdigest(),
    "torch_cuda_arch_list": os.environ["TORCH_CUDA_ARCH_LIST"],
    "extension_path": str(extension_path),
    "extension_sha256": hashlib.sha256(extension_path.read_bytes()).hexdigest(),
    "functional_values": values,
    "functional_values_sha256": hashlib.sha256(
        json.dumps(values, separators=(",", ":")).encode()
    ).hexdigest(),
    "scientific_rows_read": 0,
    "assessor_accessed": False,
}
if (
    payload["torch"] != "2.6.0+cu124"
    or payload["torch_cuda"] != "12.4"
    or payload["modelopt"] != "0.43.0"
):
    raise SystemExit("Nemotron FP8 extension package binding differs")
data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(fd, "wb") as handle:
    handle.write(data)
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps(payload, sort_keys=True), flush=True)
PY

[[ -f "$OUTPUT" && ! -L "$OUTPUT" ]]
chmod a-w "$OUTPUT"
