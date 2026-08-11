#!/bin/bash
# Non-scientific allocation-local scratch qualification for the PCF2 successor.
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc38,evc46
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00

set -euo pipefail

source "${RUNTIME:?}/train/jobs/pcf1_common.sh"
pcf1_require PYTHON
pcf1_require OUTPUT
pcf1_require ROLE
[[ "$ROLE" == cpu || "$ROLE" == h100 ]] || pcf1_die "scratch canary role differs"
pcf1_require_fresh "$OUTPUT"
[[ -z "${SLURM_TMPDIR:-}" ]] || pcf1_die "scratch canary received ambient SLURM_TMPDIR"

pcf1_export_runtime
scratch=$SLURM_TMPDIR
available_bytes=$PCF1_SCRATCH_AVAILABLE_BYTES
available_inodes=$PCF1_SCRATCH_AVAILABLE_INODES
device=$PCF1_SCRATCH_DEVICE
probe=$scratch/pcf2-scratch-canary.bin
renamed=$scratch/pcf2-scratch-canary-renamed.bin
dd if=/dev/zero of="$probe" bs=1M count=8 status=none
sync "$probe"
mv "$probe" "$renamed"
sync "$scratch"
rm -f "$renamed"

gpu_name=none
if [[ "$ROLE" == h100 ]]; then
  gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)
  [[ "$gpu_name" == "NVIDIA H100 PCIe" ]] || pcf1_die "scratch canary GPU differs"
  [[ "$(printf '%s\n' "$gpu_name" | wc -l | tr -d ' ')" == 1 ]] || \
    pcf1_die "scratch canary GPU count differs"
fi

pcf1_cleanup_scratch
trap - EXIT INT TERM
[[ ! -e "$scratch" && ! -L "$scratch" ]] || pcf1_die "scratch canary cleanup failed"

"$PYTHON" - "$OUTPUT" "$ROLE" "$scratch" "$available_bytes" \
  "$available_inodes" "$device" "$gpu_name" <<'PY'
import json
import os
from pathlib import Path
import sys

output = Path(sys.argv[1])
role, scratch = sys.argv[2:4]
available_bytes, available_inodes, device = map(int, sys.argv[4:7])
gpu_name = sys.argv[7]
payload = {
    "schema": "shohin-pcf2-scratch-canary-v1",
    "status": "pass",
    "scientific_work": False,
    "model_opened": False,
    "data_opened": False,
    "assessor_opened": False,
    "role": role,
    "job_id": int(os.environ["SLURM_JOB_ID"]),
    "partition": os.environ.get("SLURM_JOB_PARTITION"),
    "node": os.environ.get("SLURMD_NODENAME"),
    "scratch_path": scratch,
    "scratch_parent": "/tmp",
    "scratch_available_bytes": available_bytes,
    "scratch_available_inodes": available_inodes,
    "scratch_device": device,
    "scratch_minimum_bytes": 128 * 1024 * 1024 * 1024,
    "scratch_minimum_inodes": 150000,
    "write_rename_fsync_delete_passed": True,
    "cleanup_verified": True,
    "gpu_name": gpu_name,
}
output.parent.mkdir(parents=True, exist_ok=True)
temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
with temporary.open("xb") as handle:
    handle.write(encoded)
    handle.flush()
    os.fsync(handle.fileno())
os.link(temporary, output)
temporary.unlink()
parent = os.open(output.parent, os.O_RDONLY)
try:
    os.fsync(parent)
finally:
    os.close(parent)
PY
chmod a-w "$OUTPUT"
