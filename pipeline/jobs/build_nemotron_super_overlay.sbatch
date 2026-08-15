#!/bin/bash
#SBATCH --job-name=nemotron-super-overlay
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/nemotron_super_overlay_%j.out
#SBATCH --error=logs/nemotron_super_overlay_%j.err

set -euo pipefail

: "${BASE_PYTHON:?BASE_PYTHON is required}"
: "${CAUSAL_CONV_ROOT:?CAUSAL_CONV_ROOT is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

MAMBA_VERSION=2.3.2.post1
MODELOPT_VERSION=0.43.0
MAMBA_WHEEL=mamba_ssm-2.3.2.post1+cu12torch2.6cxx11abiFALSE-cp313-cp313-linux_x86_64.whl
MAMBA_WHEEL_SHA256=8c7739d65bf6477631feb5f69a6cafe207aa8b698411c73594de4168502c16d2
MAMBA_WHEEL_URL="https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/${MAMBA_WHEEL/+/%2B}"

[[ -x "$BASE_PYTHON" ]]
[[ -d "$CAUSAL_CONV_ROOT" && ! -L "$CAUSAL_CONV_ROOT" ]]
[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]]
mkdir -p logs "$(dirname "$OUTPUT_ROOT")"
partial="$OUTPUT_ROOT.partial.$SLURM_JOB_ID"
[[ ! -e "$partial" && ! -L "$partial" ]]
mkdir -m 700 "$partial"

scratch=${SLURM_TMPDIR:-/tmp}
download=$(mktemp -d "$scratch/nemotron-super-overlay.XXXXXX")
cleanup() {
  rm -rf -- "$download"
  if [[ -d "$partial" ]]; then
    rm -rf --one-file-system -- "$partial"
  fi
}
trap cleanup EXIT

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
curl -fL --retry 3 --output "$download/$MAMBA_WHEEL" "$MAMBA_WHEEL_URL"
printf '%s  %s\n' "$MAMBA_WHEEL_SHA256" "$download/$MAMBA_WHEEL" | sha256sum -c -

"$BASE_PYTHON" -m pip install --no-input --no-deps --target "$partial" \
  "$download/$MAMBA_WHEEL"
"$BASE_PYTHON" -m pip install --no-input --no-deps --target "$partial" \
  "nvidia-modelopt==$MODELOPT_VERSION" \
  "ninja==1.13.0" \
  "nvidia-ml-py==13.610.43" \
  "omegaconf==2.3.1" \
  "pulp==3.3.2" \
  "pydantic==2.13.4" \
  "pydantic-core==2.46.4" \
  "scipy==1.18.0" \
  "antlr4-python3-runtime==4.9.3" \
  "annotated-types==0.8.0" \
  "typing-inspection==0.4.4"

PYTHONPATH="$partial:$CAUSAL_CONV_ROOT" "$BASE_PYTHON" -P -s -B - \
  "$partial" "$CAUSAL_CONV_ROOT" "$MAMBA_WHEEL_SHA256" <<'PY'
from __future__ import annotations

import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import stat
import sys

import modelopt
import torch

root = Path(sys.argv[1]).resolve(strict=True)
causal_root = Path(sys.argv[2]).resolve(strict=True)
mamba_spec = importlib.util.find_spec("mamba_ssm")
causal_spec = importlib.util.find_spec("causal_conv1d")
if (
    importlib.metadata.version("mamba-ssm") != "2.3.2.post1"
    or importlib.metadata.version("nvidia-modelopt") != "0.43.0"
    or importlib.metadata.version("causal-conv1d") != "1.6.2.post1"
    or torch.__version__ != "2.6.0+cu124"
    or torch.version.cuda != "12.4"
    or mamba_spec is None
    or mamba_spec.origin is None
    or not Path(mamba_spec.origin).resolve().is_relative_to(root)
    or not Path(modelopt.__file__).resolve().is_relative_to(root)
    or causal_spec is None
    or causal_spec.origin is None
    or not Path(causal_spec.origin).resolve().is_relative_to(causal_root)
):
    raise SystemExit("Nemotron Super overlay package binding differs")

rows = []
for path in sorted(root.rglob("*")):
    mode = path.lstat().st_mode
    if stat.S_ISDIR(mode):
        continue
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise SystemExit("Nemotron Super overlay has a link or special member")
    relative = path.relative_to(root).as_posix()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows.append((relative, digest, path.stat().st_size))
if not rows:
    raise SystemExit("Nemotron Super overlay is empty")
tree_payload = b"".join(
    f"{digest}  {relative}  {size}\n".encode()
    for relative, digest, size in rows
)
receipt = {
    "schema": "shohin-nemotron-super-environment-overlay-v1",
    "status": "complete",
    "mamba_ssm": "2.3.2.post1",
    "mamba_wheel_sha256": sys.argv[3],
    "nvidia_modelopt": "0.43.0",
    "causal_conv1d": "1.6.2.post1",
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "files": len(rows),
    "bytes": sum(size for _, _, size in rows),
    "tree_sha256": hashlib.sha256(tree_payload).hexdigest(),
    "scientific_rows_read": 0,
    "gpu_requested": False,
    "gpu_kernel_import_deferred_to_h100_mechanics": True,
}
(root / "overlay_receipt.json").write_text(
    json.dumps(receipt, sort_keys=True, indent=2) + "\n"
)
manifest_rows = []
for path in sorted(root.rglob("*")):
    if path.is_file() and path.name != "SHA256SUMS":
        manifest_rows.append(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
            f"{path.relative_to(root).as_posix()}\n"
        )
(root / "SHA256SUMS").write_text("".join(manifest_rows))
print(json.dumps(receipt, sort_keys=True), flush=True)
PY

(
  cd "$partial"
  sha256sum -c SHA256SUMS
)
chmod -R a-w "$partial"
mv "$partial" "$OUTPUT_ROOT"
trap - EXIT
rm -rf -- "$download"
