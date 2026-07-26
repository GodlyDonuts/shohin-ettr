#!/bin/bash
# Exact synthetic WORLD -> COMMAND -> QUERY ETTR supervisor smoke.
# The fixture is untrained, deterministic, and isolated from training assets.
#SBATCH -J shohin-ettr-supervisor-smoke
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 01:00:00
#SBATCH --exclude=evc33,evc34,evc43,evc44
#SBATCH --export=NONE
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/ettr_supervisor_smoke_%j.out
#SBATCH --open-mode=append
set -euo pipefail
export PATH=/usr/bin:/bin
unset CDPATH ENV BASH_ENV PYTHONHOME PYTHONPATH PYTHONSTARTUP \
  LD_PRELOAD LD_AUDIT LD_LIBRARY_PATH

BASE=/lustre/fs1/home/sa305415/shohin
SOURCE_ROOT=${SOURCE_ROOT:-$BASE}
ASSESSOR_PYTHON=${ASSESSOR_PYTHON:-$BASE/miniforge3/bin/python3.13}
HOST_PYTHON=/usr/bin/python3.11
BWRAP=/usr/bin/bwrap
PREPARER=${PREPARER:-$SOURCE_ROOT/pipeline/prepare_ettr_supervisor_smoke.py}
TRUSTED_EXTRACTOR=${TRUSTED_EXTRACTOR:-$SOURCE_ROOT/train/ettr_claim_runtime.py}
RUNTIME_TAR=${RUNTIME_TAR:-$BASE/artifacts/runtime/ettr-claim-runtime-v2.tar}
RUNTIME_INVENTORY=${RUNTIME_INVENTORY:-$RUNTIME_TAR.inventory.json}
RUNTIME_VERIFICATION_RECEIPT=${RUNTIME_VERIFICATION_RECEIPT:-$RUNTIME_TAR.verification-receipt.json}
RUNTIME_RECEIPT_WORLD=${RUNTIME_RECEIPT_WORLD:-$RUNTIME_TAR.runtime-bundle-world.json}
RUNTIME_RECEIPT_COMMAND=${RUNTIME_RECEIPT_COMMAND:-$RUNTIME_TAR.runtime-bundle-command.json}
RUNTIME_RECEIPT_QUERY=${RUNTIME_RECEIPT_QUERY:-$RUNTIME_TAR.runtime-bundle-query.json}

EXPECTED_PREPARER_SHA256=${EXPECTED_PREPARER_SHA256:?pin the preparer SHA-256}
EXPECTED_TRUSTED_EXTRACTOR_SHA256=${EXPECTED_TRUSTED_EXTRACTOR_SHA256:?pin the extractor SHA-256}
EXPECTED_HOST_PYTHON_SHA256=${EXPECTED_HOST_PYTHON_SHA256:?pin the host Python SHA-256}
EXPECTED_ASSESSOR_PYTHON_SHA256=${EXPECTED_ASSESSOR_PYTHON_SHA256:?pin the assessor Python SHA-256}
EXPECTED_ARCHIVE_SHA256=${EXPECTED_ARCHIVE_SHA256:?pin the runtime archive SHA-256}
EXPECTED_INVENTORY_SHA256=${EXPECTED_INVENTORY_SHA256:?pin the runtime inventory SHA-256}
EXPECTED_SOURCE_BUNDLE_SHA256=${EXPECTED_SOURCE_BUNDLE_SHA256:?pin the runtime source bundle SHA-256}
EXPECTED_BWRAP_SHA256=${EXPECTED_BWRAP_SHA256:?pin Bubblewrap SHA-256}
ALLOCATED_GPU_MINOR=${SLURM_JOB_GPUS:-}

for digest in \
  "$EXPECTED_PREPARER_SHA256" \
  "$EXPECTED_TRUSTED_EXTRACTOR_SHA256" \
  "$EXPECTED_HOST_PYTHON_SHA256" \
  "$EXPECTED_ASSESSOR_PYTHON_SHA256" \
  "$EXPECTED_ARCHIVE_SHA256" \
  "$EXPECTED_INVENTORY_SHA256" \
  "$EXPECTED_SOURCE_BUNDLE_SHA256" \
  "$EXPECTED_BWRAP_SHA256"; do
  [[ $digest =~ ^[0-9a-f]{64}$ ]] || {
    echo "[ettr-supervisor-smoke] a required pin is not 64-hex" >&2
    exit 2
  }
done
[[ $ALLOCATED_GPU_MINOR =~ ^[0-9]+$ ]] || {
  echo "[ettr-supervisor-smoke] allocated GPU minor differs" >&2
  exit 2
}

immutable_file() {
  local path=$1 expected=$2
  [ -f "$path" ] && [ ! -L "$path" ] && [ ! -w "$path" ] \
    && [ "$(/usr/bin/stat -c %h "$path")" = 1 ] \
    && [ "$(/usr/bin/sha256sum "$path" | /usr/bin/awk '{print $1}')" \
      = "$expected" ]
}

pinned_source() {
  local path=$1 expected=$2
  [ -f "$path" ] && [ ! -L "$path" ] \
    && [ "$(/usr/bin/stat -c %h "$path")" = 1 ] \
    && [ "$(/usr/bin/sha256sum "$path" | /usr/bin/awk '{print $1}')" \
      = "$expected" ]
}

pinned_source "$PREPARER" "$EXPECTED_PREPARER_SHA256" || {
  echo "[ettr-supervisor-smoke] preparer identity differs" >&2
  exit 3
}
pinned_source \
  "$TRUSTED_EXTRACTOR" \
  "$EXPECTED_TRUSTED_EXTRACTOR_SHA256" || {
  echo "[ettr-supervisor-smoke] extractor identity differs" >&2
  exit 3
}
immutable_file "$RUNTIME_TAR" "$EXPECTED_ARCHIVE_SHA256" || {
  echo "[ettr-supervisor-smoke] runtime archive identity differs" >&2
  exit 3
}
[ -x "$HOST_PYTHON" ] && [ ! -L "$HOST_PYTHON" ] \
  && [ "$(/usr/bin/stat -c %u "$HOST_PYTHON")" = 0 ] \
  && [ "$(/usr/bin/stat -c %h "$HOST_PYTHON")" = 1 ] \
  && [ "$((8#$(/usr/bin/stat -c %a "$HOST_PYTHON") & 8#022))" = 0 ] \
  && [ "$(/usr/bin/sha256sum "$HOST_PYTHON" \
    | /usr/bin/awk '{print $1}')" = "$EXPECTED_HOST_PYTHON_SHA256" ] || {
  echo "[ettr-supervisor-smoke] host Python identity differs" >&2
  exit 3
}
[ -x "$ASSESSOR_PYTHON" ] && [ ! -L "$ASSESSOR_PYTHON" ] \
  && [ "$(/usr/bin/stat -c %h "$ASSESSOR_PYTHON")" = 1 ] \
  && [ "$(/usr/bin/sha256sum "$ASSESSOR_PYTHON" \
    | /usr/bin/awk '{print $1}')" = "$EXPECTED_ASSESSOR_PYTHON_SHA256" ] \
  && [ "$("$ASSESSOR_PYTHON" -I -B -c \
    'import cryptography; print(cryptography.__version__)')" = 46.0.3 ] || {
  echo "[ettr-supervisor-smoke] assessor Python identity differs" >&2
  exit 3
}
[ -x "$BWRAP" ] && [ ! -L "$BWRAP" ] \
  && [ "$(/usr/bin/stat -c %u "$BWRAP")" = 0 ] \
  && [ "$(/usr/bin/stat -c %h "$BWRAP")" = 1 ] \
  && [ "$(/usr/bin/sha256sum "$BWRAP" \
    | /usr/bin/awk '{print $1}')" = "$EXPECTED_BWRAP_SHA256" ] || {
  echo "[ettr-supervisor-smoke] Bubblewrap identity differs" >&2
  exit 3
}
for path in \
  "$RUNTIME_INVENTORY" \
  "$RUNTIME_VERIFICATION_RECEIPT" \
  "$RUNTIME_RECEIPT_WORLD" \
  "$RUNTIME_RECEIPT_COMMAND" \
  "$RUNTIME_RECEIPT_QUERY"; do
  [ -f "$path" ] && [ ! -L "$path" ] && [ ! -w "$path" ] \
    && [ "$(/usr/bin/stat -c %h "$path")" = 1 ] || {
    echo "[ettr-supervisor-smoke] runtime sidecar differs: $path" >&2
    exit 3
  }
done
[ -e "/dev/nvidia$ALLOCATED_GPU_MINOR" ] || {
  echo "[ettr-supervisor-smoke] allocated GPU device is absent" >&2
  exit 3
}

OUT_PARENT=$BASE/artifacts/ettr-supervisor-smoke
OUT=$OUT_PARENT/$SLURM_JOB_ID
SCRATCH_PARENT=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}
EXTRACTED=$SCRATCH_PARENT/ettr_supervisor_runtime_$SLURM_JOB_ID
mkdir -p "$OUT_PARENT"
[ ! -e "$OUT" ] \
  && [ ! -e "$EXTRACTED" ] || {
  echo "[ettr-supervisor-smoke] output already exists" >&2
  exit 4
}

cleanup_runtime() {
  local path
  for path in "$EXTRACTED"; do
    if [ -d "$path" ] && [ ! -L "$path" ]; then
      /usr/bin/chmod -R u+w "$path" 2>/dev/null || true
      /usr/bin/rm -rf -- "$path"
    fi
  done
}
trap cleanup_runtime EXIT

/usr/bin/env -i \
  HOME=/nonexistent \
  PATH=/usr/bin:/bin \
  PYTHONNOUSERSITE=1 \
  "$ASSESSOR_PYTHON" -I -B "$PREPARER" prepare \
  --source-root "$SOURCE_ROOT" \
  --output-root "$OUT" \
  --runtime-archive "$RUNTIME_TAR" \
  --runtime-inventory "$RUNTIME_INVENTORY" \
  --claim-runtime-verification-receipt \
    "$RUNTIME_VERIFICATION_RECEIPT" \
  --runtime-receipt-world "$RUNTIME_RECEIPT_WORLD" \
  --runtime-receipt-command "$RUNTIME_RECEIPT_COMMAND" \
  --runtime-receipt-query "$RUNTIME_RECEIPT_QUERY" \
  --bwrap "$BWRAP" \
  --expected-archive-sha256 "$EXPECTED_ARCHIVE_SHA256" \
  --expected-inventory-sha256 "$EXPECTED_INVENTORY_SHA256" \
  --expected-source-bundle-sha256 "$EXPECTED_SOURCE_BUNDLE_SHA256" \
  --expected-bwrap-sha256 "$EXPECTED_BWRAP_SHA256"

PLAN=$OUT/fixture-plan.json
PLAN_SHA256=$(/usr/bin/sha256sum "$PLAN" | /usr/bin/awk '{print $1}')

# The root-owned interpreter rechecks, seals, and executes the exact preparer
# bytes.  The run-chain phase imports no third-party package on the host.
/usr/bin/env -i \
  HOME=/nonexistent \
  PATH=/usr/bin:/bin \
  PYTHONNOUSERSITE=1 \
  "$HOST_PYTHON" -I -S -B - \
  "$PREPARER" \
  "$EXPECTED_PREPARER_SHA256" \
  run-chain \
  --plan "$PLAN" \
  --expected-plan-sha256 "$PLAN_SHA256" \
  --trusted-extractor "$TRUSTED_EXTRACTOR" \
  --expected-trusted-extractor-sha256 \
    "$EXPECTED_TRUSTED_EXTRACTOR_SHA256" \
  --trusted-host-python "$HOST_PYTHON" \
  --expected-trusted-host-python-sha256 \
    "$EXPECTED_HOST_PYTHON_SHA256" \
  --extraction-destination "$EXTRACTED" \
  --allocated-gpu-minor "$ALLOCATED_GPU_MINOR" <<'PY'
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
    or before.st_nlink != 1
    or before.st_mode & 0o222
    or identity(before) != identity(after)
    or hashlib.sha256(payload).hexdigest() != expected_sha256
):
    raise SystemExit("[ettr-supervisor-smoke] preparer bytes differ")
sealed = os.memfd_create(
    "ettr-supervisor-smoke",
    flags=os.MFD_ALLOW_SEALING,
)
os.write(sealed, payload)
fcntl.fcntl(
    sealed,
    fcntl.F_ADD_SEALS,
    fcntl.F_SEAL_SEAL
    | fcntl.F_SEAL_SHRINK
    | fcntl.F_SEAL_GROW
    | fcntl.F_SEAL_WRITE,
)
sys.argv = [f"/proc/self/fd/{sealed}", *arguments]
sys.dont_write_bytecode = True
runpy.run_path(sys.argv[0], run_name="__main__")
PY

/usr/bin/env -i \
  HOME=/nonexistent \
  PATH=/usr/bin:/bin \
  PYTHONNOUSERSITE=1 \
  "$ASSESSOR_PYTHON" -I -B "$PREPARER" validate \
  --source-root "$SOURCE_ROOT" \
  --plan "$PLAN" \
  --expected-plan-sha256 "$PLAN_SHA256"

REPORT=$OUT/run/smoke-report.json
[ -s "$REPORT" ] && [ ! -w "$REPORT" ] || {
  echo "[ettr-supervisor-smoke] final report is absent" >&2
  exit 5
}
echo "[ettr-supervisor-smoke] pass report=$REPORT sha256=$(/usr/bin/sha256sum "$REPORT" | /usr/bin/awk '{print $1}')"
