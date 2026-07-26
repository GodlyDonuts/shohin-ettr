#!/bin/bash
# Build one deterministic, recursively measured ETTR claim runtime.
# This is CPU-only packaging and cannot read training data or checkpoints.
#SBATCH -J shohin-ettr-runtime
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 02:00:00
#SBATCH --export=NONE
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/ettr_runtime_%j.out
#SBATCH --open-mode=append
set -euo pipefail
export PATH=/usr/bin:/bin
unset CDPATH ENV BASH_ENV PYTHONHOME PYTHONPATH PYTHONSTARTUP \
  LD_PRELOAD LD_AUDIT LD_LIBRARY_PATH GIT_DIR GIT_WORK_TREE \
  GIT_INDEX_FILE GIT_OBJECT_DIRECTORY GIT_ALTERNATE_OBJECT_DIRECTORIES \
  GIT_CONFIG_GLOBAL GIT_CONFIG_SYSTEM

BASE=/lustre/fs1/home/sa305415/shohin
HOST_PYTHON=/usr/bin/python3.11
EXPECTED_HOST_PYTHON_SHA256=${EXPECTED_HOST_PYTHON_SHA256:?set the root-owned host Python SHA-256}
ENV_ROOT=${ENV_ROOT:-$BASE/miniforge3}
SAFETENSORS_WHEEL=${SAFETENSORS_WHEEL:?set the pinned safetensors wheel}
EXPECTED_SAFETENSORS_WHEEL_SHA256=${EXPECTED_SAFETENSORS_WHEEL_SHA256:?set its SHA-256}
SOURCE_ROOT=${SOURCE_ROOT:-$BASE}
SOURCE_COMMIT=${SOURCE_COMMIT:?set SOURCE_COMMIT to the reviewed 40-hex commit}
OUT=${OUT:-$BASE/artifacts/runtime/ettr-claim-runtime-v1.tar}
STAGING_PARENT=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}

[[ $SOURCE_COMMIT =~ ^[0-9a-f]{40}$ ]] || {
  echo "[ettr-runtime] SOURCE_COMMIT must be exact 40-hex" >&2
  exit 2
}
[[ $EXPECTED_SAFETENSORS_WHEEL_SHA256 =~ ^[0-9a-f]{64}$ ]] || {
  echo "[ettr-runtime] safetensors wheel digest must be 64-hex" >&2
  exit 2
}
[[ $EXPECTED_HOST_PYTHON_SHA256 =~ ^[0-9a-f]{64}$ ]] || {
  echo "[ettr-runtime] host Python digest must be 64-hex" >&2
  exit 2
}
[ -x "$HOST_PYTHON" ] && [ ! -L "$HOST_PYTHON" ] \
  && [ "$(/usr/bin/stat -c %u "$HOST_PYTHON")" = 0 ] \
  && [ "$((8#$(/usr/bin/stat -c %a "$HOST_PYTHON") & 8#022))" = 0 ] \
  && [ "$(/usr/bin/sha256sum "$HOST_PYTHON" \
    | /usr/bin/awk '{print $1}')" = "$EXPECTED_HOST_PYTHON_SHA256" ] || {
  echo "[ettr-runtime] root-owned host Python identity differs" >&2
  exit 3
}
[ -d "$SOURCE_ROOT/.git" ] || {
  echo "[ettr-runtime] source repository unavailable" >&2
  exit 3
}
[ -x "$ENV_ROOT/bin/python" ] || {
  echo "[ettr-runtime] source interpreter unavailable" >&2
  exit 4
}
[ -s "$SAFETENSORS_WHEEL" ] && [ ! -L "$SAFETENSORS_WHEEL" ] \
  && [ ! -w "$SAFETENSORS_WHEEL" ] || {
  echo "[ettr-runtime] safetensors wheel identity differs" >&2
  exit 4
}
exec 9<"$SAFETENSORS_WHEEL"
WHEEL_FD_PATH=/proc/self/fd/9
WHEEL_METADATA_BEFORE=$(
  /usr/bin/stat -Lc '%d:%i:%f:%h:%s:%Y:%Z' "$WHEEL_FD_PATH"
)
WHEEL_SHA256=$(
  /usr/bin/sha256sum "$WHEEL_FD_PATH" | /usr/bin/awk '{print $1}'
)
[ "$WHEEL_SHA256" = "$EXPECTED_SAFETENSORS_WHEEL_SHA256" ] || {
  echo "[ettr-runtime] safetensors wheel digest differs" >&2
  exit 4
}
git -C "$SOURCE_ROOT" cat-file -e "$SOURCE_COMMIT^{commit}"
git -C "$SOURCE_ROOT" diff --quiet
git -C "$SOURCE_ROOT" diff --cached --quiet
[ ! -e "$OUT" ] && [ ! -e "$OUT.sha256" ] \
  && [ ! -e "$OUT.inventory.json" ] \
  && [ ! -e "$OUT.runtime-bundle-world.json" ] \
  && [ ! -e "$OUT.runtime-bundle-command.json" ] \
  && [ ! -e "$OUT.runtime-bundle-query.json" ] \
  && [ ! -e "$OUT.stage-policies.json" ] \
  && [ ! -e "$OUT.source-bundle.sha256" ] || {
  echo "[ettr-runtime] refusing an existing output" >&2
  exit 5
}

STAGING=$(mktemp -d "$STAGING_PARENT/shohin_ettr_runtime.XXXXXX")
cleanup() {
  chmod -R u+w "$STAGING" 2>/dev/null || true
  rm -rf -- "$STAGING"
}
trap cleanup EXIT
mkdir -p \
  "$STAGING/runtime/miniforge3" \
  "$STAGING/runtime/app/candidate/world" \
  "$STAGING/runtime/app/candidate/command" \
  "$STAGING/runtime/app/candidate/query" \
  "$STAGING/runtime/app/tools" \
  "$STAGING/runtime/app/receipts" \
  "$(dirname "$OUT")"

echo "[ettr-runtime] copying complete interpreter, stdlib, and dependency tree"
cp -a "$ENV_ROOT/bin" "$ENV_ROOT/lib" "$STAGING/runtime/miniforge3/"
RUNTIME_PY="$STAGING/runtime/miniforge3/bin/python"
SITE_PACKAGES=("$STAGING/runtime/miniforge3/lib"/python*/site-packages)
[ "${#SITE_PACKAGES[@]}" = 1 ] && [ -d "${SITE_PACKAGES[0]}" ] \
  && [ ! -L "${SITE_PACKAGES[0]}" ] || {
  echo "[ettr-runtime] staged site-packages geometry differs" >&2
  exit 4
}
/usr/bin/env -i \
  HOME=/nonexistent \
  PATH=/usr/bin:/bin \
  PYTHONNOUSERSITE=1 \
  "$HOST_PYTHON" -I -S -B - \
  9 \
  "$EXPECTED_SAFETENSORS_WHEEL_SHA256" \
  "${SITE_PACKAGES[0]}" <<'PY'
import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import zipfile

wheel_fd_text, expected_sha256, target_text = sys.argv[1:]
descriptor = os.dup(int(wheel_fd_text))
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
    raise SystemExit("[ettr-runtime] pinned wheel changed during read")
target = Path(target_text)
target_metadata = target.lstat()
if not stat.S_ISDIR(target_metadata.st_mode) or stat.S_ISLNK(
    target_metadata.st_mode
):
    raise SystemExit("[ettr-runtime] wheel target differs")
with zipfile.ZipFile(io.BytesIO(payload), mode="r") as wheel:
    members = wheel.infolist()
    paths = []
    for member in members:
        path = PurePosixPath(member.filename)
        member_mode = member.external_attr >> 16
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
            or stat.S_ISLNK(member_mode)
        ):
            raise SystemExit("[ettr-runtime] wheel member differs")
        paths.append(path)
    for top in {path.parts[0] for path in paths}:
        candidate = target / top
        if candidate.exists() or candidate.is_symlink():
            raise SystemExit("[ettr-runtime] wheel target already exists")
    wheel.extractall(target)
PY
WHEEL_METADATA_AFTER=$(
  /usr/bin/stat -Lc '%d:%i:%f:%h:%s:%Y:%Z' "$WHEEL_FD_PATH"
)
[ "$WHEEL_METADATA_AFTER" = "$WHEEL_METADATA_BEFORE" ] || {
  echo "[ettr-runtime] safetensors wheel changed during extraction" >&2
  exit 4
}
exec 9<&-

COMMON_CANDIDATE_FILES=(
  endogenous_typed_theory_reactor.py
  ettr_factorial_custody.py
  ettr_state_io.py
  model.py
)
declare -A STAGE_RUNNER=(
  [world]=run_ettr_world_compiler.py
  [command]=run_ettr_state_executor.py
  [query]=run_ettr_late_query.py
)
TOOL_FILES=(
  ettr_claim_runtime.py
  ettr_deployment_contract.py
  ettr_runtime_bundle.py
  landlock_stage_exec.py
  run_ettr_verified_stage.py
)
for stage in world command query; do
  for filename in "${COMMON_CANDIDATE_FILES[@]}" "${STAGE_RUNNER[$stage]}"; do
    git -C "$SOURCE_ROOT" show "$SOURCE_COMMIT:train/$filename" \
      > "$STAGING/runtime/app/candidate/$stage/$filename"
  done
done
for filename in "${TOOL_FILES[@]}"; do
  git -C "$SOURCE_ROOT" show "$SOURCE_COMMIT:train/$filename" \
    > "$STAGING/runtime/app/tools/$filename"
done

verify_reviewed_sources() {
  local stage filename reviewed
  reviewed="$STAGING/reviewed-source"
  for stage in world command query; do
    for filename in "${COMMON_CANDIDATE_FILES[@]}" "${STAGE_RUNNER[$stage]}"; do
      git -C "$SOURCE_ROOT" show "$SOURCE_COMMIT:train/$filename" > "$reviewed"
      /usr/bin/cmp -s "$reviewed" \
        "$STAGING/runtime/app/candidate/$stage/$filename" || {
        echo "[ettr-runtime] candidate source differs from commit" >&2
        return 1
      }
    done
  done
  for filename in "${TOOL_FILES[@]}"; do
    git -C "$SOURCE_ROOT" show "$SOURCE_COMMIT:train/$filename" > "$reviewed"
    /usr/bin/cmp -s "$reviewed" \
      "$STAGING/runtime/app/tools/$filename" || {
      echo "[ettr-runtime] tool source differs from commit" >&2
      return 1
    }
  done
  /bin/rm -f -- "$reviewed"
}

# Bytecode and .pth execution are neither needed nor admitted.
find "$STAGING/runtime" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$STAGING/runtime" -type f \
  \( -name '*.pyc' -o -name '*.pyo' -o -name '*.pth' \) -delete

"$RUNTIME_PY" -I -B - <<'PY'
import pathlib
import sys

import safetensors
import torch

prefix = pathlib.Path(sys.prefix).resolve()
for name, module in (("torch", torch), ("safetensors", safetensors)):
    origin = pathlib.Path(module.__file__).resolve()
    if not origin.is_relative_to(prefix):
        raise SystemExit(
            f"[ettr-runtime] staged {name} escapes runtime: {origin}"
        )
if torch.version.cuda is None:
    raise SystemExit("[ettr-runtime] staged Torch is CPU-only")
print(
    "[ettr-runtime] staged imports pass "
    f"torch={torch.__version__} cuda={torch.version.cuda} "
    f"safetensors={safetensors.__version__}",
    flush=True,
)
PY

for stage in world command query; do
  "$RUNTIME_PY" -I -B \
    "$STAGING/runtime/app/tools/ettr_runtime_bundle.py" \
    --source-root "$STAGING/runtime/app/candidate/$stage" \
    --receipt \
      "$STAGING/runtime/app/receipts/ettr-runtime-bundle-$stage.json" \
    --stage "$stage"
done
"$RUNTIME_PY" -I -S -B \
  "$STAGING/runtime/app/tools/ettr_deployment_contract.py" \
  --print-stage-policy-sha256s \
  > "$STAGING/runtime/app/receipts/ettr-stage-policies.json"
verify_reviewed_sources

# Normalize the exact modes measured by the inventory.
find "$STAGING/runtime" -type d -exec chmod 0555 {} +
find "$STAGING/runtime" -type f -perm /111 -exec chmod 0555 {} +
find "$STAGING/runtime" -type f ! -perm /111 -exec chmod 0444 {} +

echo "[ettr-runtime] building deterministic measured archive"
"$RUNTIME_PY" -I -S -B \
  "$STAGING/runtime/app/tools/ettr_claim_runtime.py" build \
  --runtime-root "$STAGING/runtime" \
  --source-commit "$SOURCE_COMMIT" \
  --archive "$OUT"
verify_reviewed_sources

publish_once() {
  local source=$1 destination=$2
  /usr/bin/env -i \
    HOME=/nonexistent \
    PATH=/usr/bin:/bin \
    PYTHONNOUSERSITE=1 \
    "$HOST_PYTHON" -I -S -B - "$source" "$destination" <<'PY'
import os
from pathlib import Path
import stat
import sys

source = Path(sys.argv[1])
destination = Path(os.path.abspath(sys.argv[2]))
source_flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
source_descriptor = os.open(source, source_flags)
parent_descriptor = os.open(
    destination.parent,
    os.O_RDONLY
    | os.O_CLOEXEC
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0),
)
try:
    before = os.fstat(source_descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise SystemExit("[ettr-runtime] publish source differs")
    output = os.open(
        destination.name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_CLOEXEC
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=parent_descriptor,
    )
    try:
        while chunk := os.read(source_descriptor, 8 * 1024 * 1024):
            offset = 0
            while offset < len(chunk):
                offset += os.write(output, chunk[offset:])
        os.fsync(output)
        os.fchmod(output, 0o444)
    finally:
        os.close(output)
    after = os.fstat(source_descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after):
        raise SystemExit("[ettr-runtime] publish source changed")
finally:
    os.close(source_descriptor)
    os.close(parent_descriptor)
PY
}
for stage in world command query; do
  publish_once \
    "$STAGING/runtime/app/receipts/ettr-runtime-bundle-$stage.json" \
    "$OUT.runtime-bundle-$stage.json"
done
publish_once \
  "$STAGING/runtime/app/receipts/ettr-stage-policies.json" \
  "$OUT.stage-policies.json"

echo "[ettr-runtime] complete"
/usr/bin/sha256sum \
  "$OUT" \
  "$OUT.inventory.json" \
  "$OUT.source-bundle.sha256" \
  "$OUT.runtime-bundle-world.json" \
  "$OUT.runtime-bundle-command.json" \
  "$OUT.runtime-bundle-query.json" \
  "$OUT.stage-policies.json"
/bin/ls -lh \
  "$OUT" \
  "$OUT.inventory.json" \
  "$OUT.source-bundle.sha256" \
  "$OUT.runtime-bundle-world.json" \
  "$OUT.runtime-bundle-command.json" \
  "$OUT.runtime-bundle-query.json" \
  "$OUT.stage-policies.json"
