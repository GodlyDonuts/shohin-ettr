#!/bin/bash
# No-model H100 admission for Slurm export-metadata scrubbing.
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc38,evc46
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=00:05:00

set -euo pipefail

: "${RUNTIME:?}" "${RUNTIME_MANIFEST_SHA256:?}" "${PYTHON:?}" "${OUTPUT:?}"
test -f "$RUNTIME/SHA256SUMS"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = \
  "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
source "$RUNTIME/train/jobs/pcf1_common.sh"
pcf1_require_fresh "$OUTPUT"
[[ "${SLURM_EXPORT_ENV+x}" == x && -n "$SLURM_EXPORT_ENV" ]] || \
  pcf1_die "Slurm export metadata is absent from the canary"
metadata_sha256=$(printf '%s' "$SLURM_EXPORT_ENV" | sha256sum | cut -d' ' -f1)
pcf1_assert_gpu_environment
[[ "${SLURM_EXPORT_ENV+x}" != x ]] || \
  pcf1_die "Slurm export metadata survived scrubbing"
nvidia_smi=$(command -v nvidia-smi || true)
[[ "$nvidia_smi" == /* ]] || pcf1_die "nvidia-smi is absent"
gpu_name=$($nvidia_smi --query-gpu=name --format=csv,noheader,nounits)
[[ "$gpu_name" == "NVIDIA H100 PCIe" ]] || pcf1_die "canary GPU differs"
"$PYTHON" - "$OUTPUT" "$metadata_sha256" "$gpu_name" <<'PY'
import json
import os
from pathlib import Path
import sys

output = Path(sys.argv[1])
payload = {
    "schema": "shohin-pcf6-gpu-environment-canary-v1",
    "status": "pass",
    "job_id": os.environ["SLURM_JOB_ID"],
    "node": os.environ["SLURMD_NODENAME"],
    "partition": os.environ["SLURM_JOB_PARTITION"],
    "gpu_name": sys.argv[3],
    "slurm_export_env_present_before": True,
    "slurm_export_env_sha256": sys.argv[2],
    "slurm_export_env_present_after": False,
    "actual_gpu_environment_scanned": True,
    "model_opened": False,
    "scientific_score_emitted": False,
}
temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
with temporary.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.rename(temporary, output)
parent = os.open(output.parent, os.O_RDONLY)
try:
    os.fsync(parent)
finally:
    os.close(parent)
PY
pcf1_validate_json "$OUTPUT" shohin-pcf6-gpu-environment-canary-v1 pass
chmod a-w "$OUTPUT"
