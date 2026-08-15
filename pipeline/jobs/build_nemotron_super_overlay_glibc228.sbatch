#!/bin/bash
#SBATCH --job-name=nemotron-super-overlay-glibc228
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/nemotron_super_overlay_glibc228_%j.out
#SBATCH --error=logs/nemotron_super_overlay_glibc228_%j.err

set -euo pipefail
umask 077

: "${BASE_PYTHON:?BASE_PYTHON is required}"
: "${CAUSAL_CONV_ROOT:?CAUSAL_CONV_ROOT is required}"
: "${PARENT_OVERLAY:?PARENT_OVERLAY is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

readonly SOURCE_REPOSITORY=https://github.com/state-spaces/mamba.git
readonly SOURCE_TAG=v2.3.2.post1
readonly SOURCE_COMMIT=a14b1dff0454a3bc27d9eb31355dc01e4b2490ec
readonly PARENT_MANIFEST_SHA256=e52a0095628e0c4eb58f69c4ebb6f4d1c7a929792eb9a55046bda670013a6ea2
readonly CUDA_HOME=/apps/cuda/cuda-12.4.0

[[ -x "$BASE_PYTHON" ]]
[[ -d "$CAUSAL_CONV_ROOT" && ! -L "$CAUSAL_CONV_ROOT" ]]
[[ -d "$PARENT_OVERLAY" && ! -L "$PARENT_OVERLAY" ]]
[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]]
[[ "$(sha256sum "$PARENT_OVERLAY/SHA256SUMS" | cut -d' ' -f1)" == "$PARENT_MANIFEST_SHA256" ]]
(cd "$PARENT_OVERLAY" && sha256sum -c SHA256SUMS)

partial="$OUTPUT_ROOT.partial.$SLURM_JOB_ID"
[[ ! -e "$partial" && ! -L "$partial" ]]
mkdir -p logs "$(dirname "$OUTPUT_ROOT")"
mkdir -m 700 "$partial"
scratch=$(mktemp -d "/tmp/nemotron-super-glibc228-${SLURM_JOB_ID}.XXXXXX")
cleanup() {
  if [[ -d "$scratch" ]]; then
    rm -rf --one-file-system -- "$scratch"
  fi
  if [[ -d "$partial" ]]; then
    rm -rf --one-file-system -- "$partial"
  fi
}
trap cleanup EXIT

git clone --quiet --depth 1 --branch "$SOURCE_TAG" "$SOURCE_REPOSITORY" "$scratch/mamba"
[[ "$(git -C "$scratch/mamba" rev-parse HEAD)" == "$SOURCE_COMMIT" ]]

export CUDA_HOME
export PATH="$CUDA_HOME/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export MAMBA_FORCE_BUILD=TRUE MAMBA_SKIP_CUDA_BUILD=FALSE
export MAMBA_FORCE_CXX11_ABI=FALSE MAX_JOBS=16
export PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$BASE_PYTHON" -m pip wheel --no-build-isolation --no-deps \
  --wheel-dir "$scratch/wheels" "$scratch/mamba"
mapfile -t wheels < <(find "$scratch/wheels" -maxdepth 1 -type f -name 'mamba_ssm-*.whl')
[[ "${#wheels[@]}" == 1 ]]
wheel=${wheels[0]}
wheel_sha256=$(sha256sum "$wheel" | cut -d' ' -f1)

cp -a "$PARENT_OVERLAY/." "$partial/"
chmod -R u+w "$partial"
rm -rf --one-file-system -- \
  "$partial/mamba_ssm" \
  "$partial/mamba_ssm-2.3.2.post1.dist-info"
find "$partial" -maxdepth 1 -type f -name 'selective_scan_cuda*.so' -delete
"$BASE_PYTHON" -m pip install --no-input --no-deps --force-reinstall \
  --target "$partial" "$wheel"

env -i \
  PATH=/usr/bin:/bin \
  LD_LIBRARY_PATH="$CUDA_HOME/lib64" \
  PYTHONPATH="$partial:$CAUSAL_CONV_ROOT" \
  PYTHONDONTWRITEBYTECODE=1 \
  "$BASE_PYTHON" -P -s -B - \
  "$partial" "$CAUSAL_CONV_ROOT" "$SOURCE_COMMIT" "$wheel_sha256" \
  "$PARENT_MANIFEST_SHA256" <<'PY'
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys

import modelopt
import torch

root = Path(sys.argv[1]).resolve(strict=True)
causal_root = Path(sys.argv[2]).resolve(strict=True)
source_commit, wheel_sha256, parent_manifest_sha256 = sys.argv[3:]
import mamba_ssm
import selective_scan_cuda

if (
    importlib.metadata.version("mamba-ssm") != "2.3.2.post1"
    or importlib.metadata.version("nvidia-modelopt") != "0.43.0"
    or importlib.metadata.version("causal-conv1d") != "1.6.2.post1"
    or torch.__version__ != "2.6.0+cu124"
    or torch.version.cuda != "12.4"
    or not Path(mamba_ssm.__file__).resolve().is_relative_to(root)
    or not Path(modelopt.__file__).resolve().is_relative_to(root)
):
    raise SystemExit("Nemotron Super rebuilt overlay package binding differs")
causal_spec = importlib.util.find_spec("causal_conv1d")
if (
    causal_spec is None
    or causal_spec.origin is None
    or not Path(causal_spec.origin).resolve().is_relative_to(causal_root)
):
    raise SystemExit("Nemotron Super causal-conv binding differs")

extension = Path(selective_scan_cuda.__file__).resolve(strict=True)
if not extension.is_relative_to(root):
    raise SystemExit("selective-scan extension origin differs")
version_info = subprocess.check_output(
    ["/usr/bin/readelf", "--version-info", str(extension)], text=True
)
versions = {
    tuple(map(int, value.split(".")))
    for value in re.findall(r"GLIBC_(\d+(?:\.\d+)+)", version_info)
}
if versions and max(versions) > (2, 28):
    raise SystemExit("rebuilt extension exceeds Newton GLIBC 2.28")
libc_name, libc_version = platform.libc_ver()
if libc_name != "glibc" or tuple(map(int, libc_version.split("."))) != (2, 28):
    raise SystemExit("overlay build host GLIBC differs")

receipt = {
    "schema": "shohin-nemotron-super-environment-overlay-v2",
    "status": "complete",
    "parent_manifest_sha256": parent_manifest_sha256,
    "mamba_ssm": "2.3.2.post1",
    "mamba_source_commit": source_commit,
    "mamba_source_wheel_sha256": wheel_sha256,
    "selective_scan_cuda_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
    "maximum_required_glibc": ".".join(map(str, max(versions))) if versions else None,
    "build_host_glibc": libc_version,
    "nvidia_modelopt": "0.43.0",
    "causal_conv1d": "1.6.2.post1",
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "scientific_rows_read": 0,
    "gpu_requested": False,
    "kernel_import_passed": True,
}
(root / "overlay_receipt.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n"
)
rows = []
for path in sorted(root.rglob("*")):
    mode = path.lstat().st_mode
    if stat.S_ISDIR(mode):
        continue
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise SystemExit("rebuilt overlay has a link or special member")
    if path.name == "SHA256SUMS":
        continue
    rows.append(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
        f"{path.relative_to(root).as_posix()}\n"
    )
(root / "SHA256SUMS").write_text("".join(rows))
receipt["manifest_entries"] = len(rows)
print(json.dumps(receipt, sort_keys=True), flush=True)
PY

(cd "$partial" && sha256sum -c SHA256SUMS)
chmod -R a-w "$partial"
mv -- "$partial" "$OUTPUT_ROOT"
trap - EXIT
rm -rf --one-file-system -- "$scratch"
(
  cd "$OUTPUT_ROOT"
  sha256sum -c SHA256SUMS
  sha256sum SHA256SUMS overlay_receipt.json selective_scan_cuda*.so
)
