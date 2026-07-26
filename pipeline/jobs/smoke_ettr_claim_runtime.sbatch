#!/bin/bash
# H100 smoke for one externally pinned ETTR claim runtime.
# No training data, optimizer, or checkpoint is read.
#SBATCH -J shohin-ettr-runtime-smoke
#SBATCH -p normal
#SBATCH --gres=gpu:h100:1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/ettr_runtime_smoke_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
RUNTIME_TAR=${RUNTIME_TAR:-$BASE/artifacts/runtime/ettr-claim-runtime-v1.tar}
EXPECTED_SHA256=${EXPECTED_SHA256:?set the verifier-pinned runtime SHA-256}
EXPECTED_INVENTORY_SHA256=${EXPECTED_INVENTORY_SHA256:?set the pinned inventory SHA-256}
EXPECTED_BWRAP_SHA256=${EXPECTED_BWRAP_SHA256:?set the pinned Bubblewrap SHA-256}
BWRAP=${BWRAP:-/usr/bin/bwrap}

[[ $EXPECTED_SHA256 =~ ^[0-9a-f]{64}$ ]] \
  && [[ $EXPECTED_INVENTORY_SHA256 =~ ^[0-9a-f]{64}$ ]] \
  && [[ $EXPECTED_BWRAP_SHA256 =~ ^[0-9a-f]{64}$ ]] || {
  echo "[ettr-runtime-smoke] expected digests must be 64-hex" >&2
  exit 2
}
[ -s "$RUNTIME_TAR" ] && [ ! -L "$RUNTIME_TAR" ] \
  && [ ! -w "$RUNTIME_TAR" ] || {
  echo "[ettr-runtime-smoke] runtime archive is not immutable" >&2
  exit 3
}
OBSERVED_SHA256=$(sha256sum "$RUNTIME_TAR" | awk '{print $1}')
[ "$OBSERVED_SHA256" = "$EXPECTED_SHA256" ] || {
  echo "[ettr-runtime-smoke] runtime archive digest differs" >&2
  exit 4
}
[ -x "$BWRAP" ] && [ ! -L "$BWRAP" ] \
  && [ "$(stat -c %u "$BWRAP")" = 0 ] \
  && [ "$(stat -c %h "$BWRAP")" = 1 ] || {
  echo "[ettr-runtime-smoke] bubblewrap executable differs" >&2
  exit 5
}
BWRAP_SHA256=$(sha256sum "$BWRAP" | awk '{print $1}')
[ "$BWRAP_SHA256" = "$EXPECTED_BWRAP_SHA256" ] || {
  echo "[ettr-runtime-smoke] bubblewrap digest differs" >&2
  exit 5
}

RUNTIME_PARENT=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}
ROOT=$(mktemp -d "$RUNTIME_PARENT/shohin_ettr_runtime_smoke.XXXXXX")
chmod 0700 "$ROOT"
cleanup() {
  chmod -R u+w "$ROOT" 2>/dev/null || true
  rm -rf -- "$ROOT"
}
trap cleanup EXIT

tar --extract --file "$RUNTIME_TAR" --directory "$ROOT" \
  --no-same-owner --no-same-permissions
PY="$ROOT/runtime/miniforge3/bin/python"
VERIFIER="$ROOT/runtime/app/tools/ettr_claim_runtime.py"
[ -x "$PY" ] && [ -r "$VERIFIER" ] || {
  echo "[ettr-runtime-smoke] extracted runtime layout differs" >&2
  exit 6
}
find "$ROOT/runtime" -type d -exec chmod 0555 {} +
find "$ROOT/runtime" -type f -perm /111 -exec chmod 0555 {} +
find "$ROOT/runtime" -type f ! -perm /111 -exec chmod 0444 {} +

RECEIPT="$ROOT/verification.json"
INVENTORY_SHA256=$(
  "$PY" -I -S -B "$VERIFIER" verify-archive --archive "$RUNTIME_TAR"
)
[ "$INVENTORY_SHA256" = "$EXPECTED_INVENTORY_SHA256" ] || {
  echo "[ettr-runtime-smoke] runtime inventory digest differs" >&2
  exit 7
}
"$PY" -I -S -B "$VERIFIER" verify-tree \
  --archive "$RUNTIME_TAR" \
  --runtime-root "$ROOT/runtime" \
  --receipt "$RECEIPT"

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
  --ro-bind "$ROOT/runtime" /runtime \
  --proc /proc \
  --dev /dev \
  --tmpfs /tmp \
  --setenv HOME /tmp \
  --setenv PATH /runtime/miniforge3/bin \
  --setenv PYTHONNOUSERSITE 1 \
  --setenv LD_LIBRARY_PATH \
    /runtime/miniforge3/lib:/usr/lib64:/lib64:/usr/lib:/lib \
)
for host_path in /lib64 /usr/lib64 /usr/lib /lib /sys; do
  if [ -e "$host_path" ]; then
    BWRAP_ARGS+=(--ro-bind "$host_path" "$host_path")
  fi
done
for device in /dev/nvidia*; do
  if [ -e "$device" ]; then
    BWRAP_ARGS+=(--dev-bind "$device" "$device")
  fi
done

"$BWRAP" "${BWRAP_ARGS[@]}" -- \
  /runtime/miniforge3/bin/python -I -S -B - <<'PY'
import pathlib
import socket
import sys

prefix = pathlib.Path("/runtime/miniforge3").resolve()
if pathlib.Path(sys.prefix).resolve() != prefix:
    raise SystemExit("[ettr-runtime-smoke] interpreter prefix escaped")

try:
    socket.create_connection(("1.1.1.1", 53), timeout=0.2)
except OSError:
    pass
else:
    raise SystemExit("[ettr-runtime-smoke] network namespace not isolated")
print("[ettr-runtime-smoke] bubblewrap network isolation pass", flush=True)
PY

"$BWRAP" "${BWRAP_ARGS[@]}" -- \
  /runtime/miniforge3/bin/python -I -B - <<'PY'
import pathlib
import sys

import safetensors
import torch

prefix = pathlib.Path("/runtime/miniforge3").resolve()
if pathlib.Path(sys.prefix).resolve() != prefix:
    raise SystemExit("[ettr-runtime-smoke] interpreter prefix escaped")
for name, module in (("torch", torch), ("safetensors", safetensors)):
    origin = pathlib.Path(module.__file__).resolve()
    if not origin.is_relative_to(prefix):
        raise SystemExit(
            f"[ettr-runtime-smoke] {name} escaped runtime: {origin}"
        )
if not torch.cuda.is_available():
    raise SystemExit("[ettr-runtime-smoke] confined CUDA unavailable")
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

echo "[ettr-runtime-smoke] pass archive=$OBSERVED_SHA256 inventory=$INVENTORY_SHA256 bwrap=$BWRAP_SHA256"
