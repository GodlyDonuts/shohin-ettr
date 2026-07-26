#!/bin/bash
# H100 smoke for one externally pinned ETTR claim runtime.
# No training data, optimizer, or checkpoint is read.
#SBATCH -J shohin-ettr-runtime-smoke
#SBATCH -p normal
#SBATCH --gres=gpu:h100:1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 00:30:00
#SBATCH --export=NONE
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/ettr_runtime_smoke_%j.out
#SBATCH --open-mode=append
set -euo pipefail
export PATH=/usr/bin:/bin
unset CDPATH ENV BASH_ENV PYTHONHOME PYTHONPATH PYTHONSTARTUP \
  LD_PRELOAD LD_AUDIT LD_LIBRARY_PATH

BASE=/lustre/fs1/home/sa305415/shohin
RUNTIME_TAR=${RUNTIME_TAR:-$BASE/artifacts/runtime/ettr-claim-runtime-v1.tar}
EXPECTED_SHA256=${EXPECTED_SHA256:?set the verifier-pinned runtime SHA-256}
EXPECTED_INVENTORY_SHA256=${EXPECTED_INVENTORY_SHA256:?set the pinned inventory SHA-256}
EXPECTED_SOURCE_BUNDLE_SHA256=${EXPECTED_SOURCE_BUNDLE_SHA256:?set the independently computed source-bundle SHA-256}
EXPECTED_BWRAP_SHA256=${EXPECTED_BWRAP_SHA256:?set the pinned Bubblewrap SHA-256}
TRUSTED_PYTHON=/usr/bin/python3.11
EXPECTED_TRUSTED_PYTHON_SHA256=${EXPECTED_TRUSTED_PYTHON_SHA256:?set its SHA-256}
TRUSTED_VERIFIER=${TRUSTED_VERIFIER:?set the verifier-owned extractor}
EXPECTED_TRUSTED_VERIFIER_SHA256=${EXPECTED_TRUSTED_VERIFIER_SHA256:?set its SHA-256}
BWRAP=/usr/bin/bwrap
ALLOCATED_GPU_INDEX=${SLURM_JOB_GPUS:-}

[[ $EXPECTED_SHA256 =~ ^[0-9a-f]{64}$ ]] \
  && [[ $EXPECTED_INVENTORY_SHA256 =~ ^[0-9a-f]{64}$ ]] \
  && [[ $EXPECTED_SOURCE_BUNDLE_SHA256 =~ ^[0-9a-f]{64}$ ]] \
  && [[ $EXPECTED_BWRAP_SHA256 =~ ^[0-9a-f]{64}$ ]] \
  && [[ $EXPECTED_TRUSTED_PYTHON_SHA256 =~ ^[0-9a-f]{64}$ ]] \
  && [[ $EXPECTED_TRUSTED_VERIFIER_SHA256 =~ ^[0-9a-f]{64}$ ]] \
  && [[ $ALLOCATED_GPU_INDEX =~ ^[0-9]+$ ]] || {
  echo "[ettr-runtime-smoke] expected digests must be 64-hex" >&2
  exit 2
}
[ -s "$RUNTIME_TAR" ] && [ ! -L "$RUNTIME_TAR" ] \
  && [ ! -w "$RUNTIME_TAR" ] || {
  echo "[ettr-runtime-smoke] runtime archive is not immutable" >&2
  exit 3
}
OBSERVED_SHA256=$(
  /usr/bin/sha256sum "$RUNTIME_TAR" | /usr/bin/awk '{print $1}'
)
[ "$OBSERVED_SHA256" = "$EXPECTED_SHA256" ] || {
  echo "[ettr-runtime-smoke] runtime archive digest differs" >&2
  exit 4
}
[ -x "$BWRAP" ] && [ ! -L "$BWRAP" ] \
  && [ "$(/usr/bin/stat -c %u "$BWRAP")" = 0 ] \
  && [ "$(/usr/bin/stat -c %h "$BWRAP")" = 1 ] || {
  echo "[ettr-runtime-smoke] bubblewrap executable differs" >&2
  exit 5
}
BWRAP_SHA256=$(
  /usr/bin/sha256sum "$BWRAP" | /usr/bin/awk '{print $1}'
)
[ "$BWRAP_SHA256" = "$EXPECTED_BWRAP_SHA256" ] || {
  echo "[ettr-runtime-smoke] bubblewrap digest differs" >&2
  exit 5
}
[ -x "$TRUSTED_PYTHON" ] && [ ! -L "$TRUSTED_PYTHON" ] \
  && [ "$(/usr/bin/stat -c %u "$TRUSTED_PYTHON")" = 0 ] \
  && [ "$(/usr/bin/stat -c %h "$TRUSTED_PYTHON")" = 1 ] \
  && [ "$((8#$(/usr/bin/stat -c %a "$TRUSTED_PYTHON") & 8#022))" = 0 ] \
  && [ "$(/usr/bin/stat -c %u /usr/bin)" = 0 ] \
  && [ "$((8#$(/usr/bin/stat -c %a /usr/bin) & 8#022))" = 0 ] \
  && [ "$(/usr/bin/sha256sum "$TRUSTED_PYTHON" \
    | /usr/bin/awk '{print $1}')" \
    = "$EXPECTED_TRUSTED_PYTHON_SHA256" ] || {
  echo "[ettr-runtime-smoke] trusted Python identity differs" >&2
  exit 5
}
[ -r "$TRUSTED_VERIFIER" ] && [ ! -L "$TRUSTED_VERIFIER" ] || {
  echo "[ettr-runtime-smoke] trusted verifier identity differs" >&2
  exit 5
}
/usr/bin/env -i \
  HOME=/nonexistent \
  PATH=/usr/bin:/bin \
  PYTHONNOUSERSITE=1 \
  "$TRUSTED_PYTHON" -I -S -B - <<'PY'
import os
from pathlib import Path
import stat
import sys
import sysconfig

roots = (
    Path(sys.executable).resolve(),
    Path(sysconfig.get_path("stdlib")).resolve(),
)
for root in roots:
    if not root.exists():
        raise SystemExit("[ettr-runtime-smoke] trusted Python root is absent")
    paths = (root,) if root.is_file() else (root, *root.rglob("*"))
    for path in paths:
        metadata = path.lstat()
        if metadata.st_uid != 0 or stat.S_IMODE(metadata.st_mode) & 0o022:
            raise SystemExit(
                f"[ettr-runtime-smoke] trusted Python closure is mutable: {path}"
            )
        if stat.S_ISLNK(metadata.st_mode):
            resolved = path.resolve(strict=True)
            target = resolved.stat()
            if (
                not resolved.is_relative_to(Path("/usr"))
                or target.st_uid != 0
                or stat.S_IMODE(target.st_mode) & 0o022
            ):
                raise SystemExit(
                    "[ettr-runtime-smoke] trusted Python link escapes "
                    f"root-owned /usr: {path}"
                )
print("[ettr-runtime-smoke] root-owned Python closure pass", flush=True)
PY
[ -e "/dev/nvidia$ALLOCATED_GPU_INDEX" ] || {
  echo "[ettr-runtime-smoke] allocated GPU device is unavailable" >&2
  exit 5
}

RUNTIME_PARENT=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}
SCRATCH=$(
  /usr/bin/mktemp -d "$RUNTIME_PARENT/shohin_ettr_runtime_smoke.XXXXXX"
)
ROOT="$SCRATCH/extracted"
chmod 0700 "$SCRATCH"
cleanup() {
  /bin/rmdir "$SCRATCH" 2>/dev/null || true
}
trap cleanup EXIT

run_trusted_verifier() {
  /usr/bin/env -i \
    HOME=/nonexistent \
    PATH=/usr/bin:/bin \
    PYTHONNOUSERSITE=1 \
    "$TRUSTED_PYTHON" -I -S -B - \
    "$TRUSTED_VERIFIER" \
    "$EXPECTED_TRUSTED_VERIFIER_SHA256" \
    "$@" <<'PY'
import fcntl
import hashlib
import os
import runpy
import stat
import sys

source_path, expected_sha256, *arguments = sys.argv[1:]
flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(source_path, flags)
try:
    before = os.fstat(descriptor)
    payload = bytearray()
    while chunk := os.read(descriptor, 8 * 1024 * 1024):
        payload.extend(chunk)
    after = os.fstat(descriptor)
finally:
    os.close(descriptor)
identity = lambda value: (
    value.st_dev,
    value.st_ino,
    value.st_mode,
    value.st_nlink,
    value.st_size,
    value.st_mtime_ns,
    value.st_ctime_ns,
)
if (
    not stat.S_ISREG(before.st_mode)
    or identity(before) != identity(after)
    or hashlib.sha256(payload).hexdigest() != expected_sha256
):
    raise SystemExit("[ettr-runtime-smoke] trusted verifier bytes differ")
sealed = os.memfd_create(
    "ettr-verifier",
    flags=os.MFD_ALLOW_SEALING,
)
offset = 0
while offset < len(payload):
    offset += os.write(sealed, payload[offset:])
fcntl.fcntl(
    sealed,
    fcntl.F_ADD_SEALS,
    fcntl.F_SEAL_SEAL
    | fcntl.F_SEAL_SHRINK
    | fcntl.F_SEAL_GROW
    | fcntl.F_SEAL_WRITE,
)
sealed_path = f"/proc/self/fd/{sealed}"
sys.argv = [sealed_path, *arguments]
sys.dont_write_bytecode = True
runpy.run_path(sealed_path, run_name="__main__")
PY
}

# Newton's 4.18 kernel has no Landlock. Bubblewrap is therefore the mandatory
# filesystem and network boundary for this deployment profile.
BWRAP_ARGS=(
  --die-with-parent \
  --new-session \
  --unshare-user \
  --unshare-pid \
  --unshare-ipc \
  --unshare-uts \
  --unshare-net \
  --clearenv \
  --ro-bind "{ETTR_RUNTIME_ROOT}" /runtime \
  --proc /proc \
  --dev /dev \
  --tmpfs /tmp \
  --setenv HOME /tmp \
  --setenv PATH /runtime/miniforge3/bin \
  --setenv PYTHONNOUSERSITE 1 \
  --setenv CUDA_VISIBLE_DEVICES 0 \
  --setenv LD_LIBRARY_PATH \
    /runtime/miniforge3/lib:/usr/lib64:/lib64:/usr/lib:/lib \
)
for host_path in /lib64 /usr/lib64 /usr/lib /lib /sys; do
  if [ -e "$host_path" ]; then
    BWRAP_ARGS+=(--ro-bind "$host_path" "$host_path")
  fi
done
for device in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
  if [ -e "$device" ]; then
    BWRAP_ARGS+=(--dev-bind "$device" "$device")
  fi
done
BWRAP_ARGS+=(
  --dev-bind "/dev/nvidia$ALLOCATED_GPU_INDEX" /dev/nvidia0
)

SMOKE_CODE=$(/usr/bin/cat <<'PY'
import os
import pathlib
import socket
import sys

import safetensors
import torch

prefix = pathlib.Path("/runtime/miniforge3").resolve()
if pathlib.Path(sys.prefix).resolve() != prefix:
    raise SystemExit("[ettr-runtime-smoke] interpreter prefix escaped")
if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
    raise SystemExit("[ettr-runtime-smoke] allocated GPU visibility differs")
try:
    socket.create_connection(("1.1.1.1", 53), timeout=0.2)
except OSError:
    pass
else:
    raise SystemExit("[ettr-runtime-smoke] network namespace not isolated")
print("[ettr-runtime-smoke] bubblewrap network isolation pass", flush=True)
for name, module in (("torch", torch), ("safetensors", safetensors)):
    origin = pathlib.Path(module.__file__).resolve()
    if not origin.is_relative_to(prefix):
        raise SystemExit(
            f"[ettr-runtime-smoke] {name} escaped runtime: {origin}"
        )
if not torch.cuda.is_available():
    raise SystemExit("[ettr-runtime-smoke] confined CUDA unavailable")
if torch.cuda.device_count() != 1:
    raise SystemExit("[ettr-runtime-smoke] GPU device confinement differs")
if "H100" not in torch.cuda.get_device_name(0):
    raise SystemExit(
        f"[ettr-runtime-smoke] allocated GPU is not H100: "
        f"{torch.cuda.get_device_name(0)}"
    )
value = torch.ones((1024, 1024), device="cuda", dtype=torch.bfloat16)
result = value @ value
torch.cuda.synchronize()
if not torch.isfinite(result).all():
    raise SystemExit("[ettr-runtime-smoke] CUDA operation is nonfinite")
print(
    "[ettr-runtime-smoke] confined runtime imports/CUDA pass "
    f"python={sys.version.split()[0]} torch={torch.__version__} "
    f"safetensors={safetensors.__version__} "
    f"gpu={torch.cuda.get_device_name(0)}",
    flush=True,
)
PY
)

run_trusted_verifier extract-exec \
  --archive "$RUNTIME_TAR" \
  --destination "$ROOT" \
  --expected-archive-sha256 "$EXPECTED_SHA256" \
  --expected-inventory-sha256 "$EXPECTED_INVENTORY_SHA256" \
  --expected-source-bundle-sha256 "$EXPECTED_SOURCE_BUNDLE_SHA256" \
  -- \
  "$BWRAP" "${BWRAP_ARGS[@]}" -- \
  /runtime/miniforge3/bin/python -I -B -c "$SMOKE_CODE"

echo "[ettr-runtime-smoke] pass archive=$OBSERVED_SHA256 inventory=$EXPECTED_INVENTORY_SHA256 bwrap=$BWRAP_SHA256"
