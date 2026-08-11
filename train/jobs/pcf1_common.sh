#!/bin/bash
# Shared fail-closed helpers for the one PCF1 Slurm graph.

set -euo pipefail

readonly PCF1_PARTITION=normal
readonly PCF1_EXCLUDED_NODES=evc26,evc29,evc31,evc32,evc38,evc46
readonly PCF1_MODEL_ROOT=/lustre/fs1/home/sa305415/shohin/artifacts/external/ministral-3-8b-reasoning-2512-81eaece
readonly PCF1_MODEL_MANIFEST_SHA256=46cc9203a18a414e08a53109662c3802b57c046896185ca9ab31875e8167cf1f
readonly PCF1_MODEL_CONFIG_SHA256=5aae04beb9f2a9949eb1df870cf47ba292012a066bdcdcb115a9ac43425f8086
readonly PCF1_MODEL_SOURCE_REVISION_SHA256=3576c1bfaa0652940d12817ad3267ffe65645dc558ceb9a153ffb72f7211a982
# Sole exact historical-environment control-plane exception.
readonly PCF1_PYTHON_ENTRYPOINT=/lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603-r2/bin/python
readonly PCF1_SCRATCH_PARENT=/tmp
readonly PCF1_SCRATCH_MIN_BYTES=$((128 * 1024 * 1024 * 1024))
readonly PCF1_SCRATCH_MIN_INODES=150000

pcf1_die() {
  printf 'pcf1: %s\n' "$*" >&2
  exit 2
}

pcf1_require() {
  local name=$1
  [[ -n "${!name:-}" ]] || pcf1_die "$name is required"
}

pcf1_assert_gpu_environment() {
  pcf1_require PYTHON
  [[ "$PYTHON" == "$PCF1_PYTHON_ENTRYPOINT" ]] || \
    pcf1_die "GPU Python venv entrypoint differs"
  # Slurm duplicates the complete --export payload into this control-plane
  # variable. Its value contains the pinned historical Python path and must
  # not be inherited by model processes. Actual exported variables remain and
  # are independently scanned below.
  unset SLURM_EXPORT_ENV
  "$PYTHON" - "$PCF1_PYTHON_ENTRYPOINT" <<'PY'
import os
import sys

entrypoint = sys.argv[1]
for name, value in os.environ.items():
    folded = f"{name}\n{value}".casefold()
    if any(term in folded for term in ("assessor", "holdout", "product", "public")):
        if name in {"PYTHON", "_"} and value == entrypoint:
            continue
        raise SystemExit(f"protected GPU environment variable: {name}")
PY
}

pcf1_require_file() {
  local path=$1
  [[ -f "$path" && ! -L "$path" ]] || pcf1_die "missing or linked file: $path"
}

pcf1_require_dir() {
  local path=$1
  [[ -d "$path" && ! -L "$path" ]] || pcf1_die "missing or linked directory: $path"
}

pcf1_sha256() {
  sha256sum "$1" | cut -d' ' -f1
}

pcf1_verify_sha256() {
  local path=$1
  local expected=$2
  pcf1_require_file "$path"
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || pcf1_die "invalid SHA-256: $expected"
  [[ "$(pcf1_sha256 "$path")" == "$expected" ]] || \
    pcf1_die "SHA-256 differs: $path"
}

pcf1_verify_runtime_membership() {
  pcf1_require RUNTIME
  pcf1_require RUNTIME_MANIFEST_SHA256
  pcf1_require_dir "$RUNTIME"
  pcf1_verify_sha256 "$RUNTIME/SHA256SUMS" "$RUNTIME_MANIFEST_SHA256"
  pcf1_require PYTHON
  "$PYTHON" - "$RUNTIME" <<'PY'
import json
from pathlib import Path, PurePosixPath
import stat
import sys

root = Path(sys.argv[1])
manifest = root / "SHA256SUMS"
entries = []
for line in manifest.read_text(encoding="utf-8").splitlines():
    digest, separator, relative = line.partition("  ")
    pure = PurePosixPath(relative)
    if (
        not separator
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or pure.is_absolute()
        or ".." in pure.parts
        or not relative
    ):
        raise SystemExit("PCF1 runtime manifest entry differs")
    entries.append(pure.as_posix())
if entries != sorted(entries) or len(entries) != len(set(entries)):
    raise SystemExit("PCF1 runtime manifest order differs")
actual = set()
actual_dirs = set()
for path in root.rglob("*"):
    relative = path.relative_to(root).as_posix()
    mode = path.lstat().st_mode
    if stat.S_ISREG(mode):
        actual.add(relative)
    elif stat.S_ISDIR(mode):
        actual_dirs.add(relative)
    else:
        raise SystemExit("PCF1 runtime contains a link or special member")
expected = {*entries, "SHA256SUMS"}
expected_dirs = set()
for relative in expected:
    parent = PurePosixPath(relative).parent
    while parent != PurePosixPath("."):
        expected_dirs.add(parent.as_posix())
        parent = parent.parent
if actual != expected or actual_dirs != expected_dirs:
    raise SystemExit("PCF1 runtime exact membership differs")
runtime = json.loads((root / "runtime.json").read_text(encoding="utf-8"))
if (
    runtime.get("schema") != "shohin-pcf1-runtime-v1"
    or runtime.get("status") != "complete"
    or runtime.get("extra_files_permitted") is not False
    or not isinstance(runtime.get("source_commit"), str)
    or len(runtime["source_commit"]) != 40
):
    raise SystemExit("PCF1 runtime receipt differs")
PY
  (cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
}

pcf1_verify_environment() {
  pcf1_require PREPARE_ROOT
  local receipt=$PREPARE_ROOT/receipt.json
  local environment=$PREPARE_ROOT/environment_receipt.json
  local expected
  pcf1_require_file "$receipt"
  pcf1_require_file "$environment"
  expected=$("$PYTHON" - "$receipt" "$environment" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

receipt_path, environment = map(Path, sys.argv[1:])
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
record = receipt.get("outputs", {}).get("environment", {})
digest = hashlib.sha256(environment.read_bytes()).hexdigest()
if (
    receipt.get("schema") != "shohin-pcf1-prepare-receipt-v1"
    or receipt.get("status") != "complete"
    or Path(str(record.get("path", ""))).resolve() != environment.resolve()
    or record.get("sha256") != digest
    or record.get("schema") != "shohin-pcf1-environment-receipt-v1"
):
    raise SystemExit("PCF1 prepared environment binding differs")
print(digest)
PY
)
  ENVIRONMENT_RECEIPT=$environment
  ENVIRONMENT_RECEIPT_SHA256=$expected
  export ENVIRONMENT_RECEIPT ENVIRONMENT_RECEIPT_SHA256
  pcf1_export_runtime
  "$PYTHON" "$RUNTIME/pipeline/capture_pcf1_environment.py" verify \
    --runtime-root "$RUNTIME" \
    --runtime-manifest-sha256 "$RUNTIME_MANIFEST_SHA256" \
    --receipt "$ENVIRONMENT_RECEIPT" >/dev/null
}

pcf1_verify_runtime() {
  pcf1_verify_runtime_membership
  pcf1_verify_environment
}

pcf1_verify_model() {
  for variable in MODEL_ROOT MODEL_REVISION MODEL_CONFIG_SHA256 MODEL_MANIFEST MODEL_MANIFEST_SHA256; do
    pcf1_require "$variable"
  done
  [[ "$MODEL_REVISION" == 81eaece1948f3875421d9a45bc55487d10e2d894 ]] || \
    pcf1_die "model revision differs from the frozen PCF1 host"
  pcf1_require_dir "$MODEL_ROOT"
  [[ "$(realpath "$MODEL_ROOT")" == "$PCF1_MODEL_ROOT" ]] || \
    pcf1_die "model root differs from the authoritative PCF1 host"
  [[ "$MODEL_CONFIG_SHA256" == "$PCF1_MODEL_CONFIG_SHA256" ]] || \
    pcf1_die "model config receipt differs from the authoritative PCF1 host"
  [[ "$MODEL_MANIFEST_SHA256" == "$PCF1_MODEL_MANIFEST_SHA256" ]] || \
    pcf1_die "model manifest receipt differs from the authoritative PCF1 host"
  pcf1_verify_sha256 "$MODEL_ROOT/config.json" "$MODEL_CONFIG_SHA256"
  pcf1_verify_sha256 "$MODEL_ROOT/SOURCE_REVISION" "$PCF1_MODEL_SOURCE_REVISION_SHA256"
  pcf1_verify_sha256 "$MODEL_ROOT/SHA256SUMS" "$PCF1_MODEL_MANIFEST_SHA256"
  pcf1_reject_protected_path "$MODEL_MANIFEST"
  pcf1_verify_sha256 "$MODEL_MANIFEST" "$MODEL_MANIFEST_SHA256"
  cmp -s "$MODEL_ROOT/SHA256SUMS" "$MODEL_MANIFEST" || \
    pcf1_die "safe model manifest copy differs"
  pcf1_verify_model_tree "$MODEL_ROOT" "$MODEL_ROOT/SHA256SUMS" 58 35706515534
}

pcf1_verify_model_tree() {
  local root=$1
  local manifest=$2
  local expected_files=$3
  local expected_bytes=$4
  "$PYTHON" - "$root" "$manifest" "$expected_files" "$expected_bytes" <<'PY'
from pathlib import Path, PurePosixPath
import stat
import sys

root, manifest = map(Path, sys.argv[1:3])
expected_files, expected_bytes = map(int, sys.argv[3:])
entries = []
for line in manifest.read_text(encoding="utf-8").splitlines():
    digest, separator, relative = line.partition("  ")
    pure = PurePosixPath(relative)
    if (
        not separator
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or pure.is_absolute()
        or ".." in pure.parts
        or not relative
    ):
        raise SystemExit("PCF1 model manifest entry differs")
    entries.append(pure.as_posix())
if (
    len(entries) != expected_files
    or entries != sorted(entries)
    or len(entries) != len(set(entries))
):
    raise SystemExit("PCF1 model manifest geometry differs")
expected_regular = {*entries, "SHA256SUMS"}
expected_dirs = set()
for relative in expected_regular:
    parent = PurePosixPath(relative).parent
    while parent != PurePosixPath("."):
        expected_dirs.add(parent.as_posix())
        parent = parent.parent
actual_regular = set()
actual_dirs = set()
covered_bytes = 0
for path in root.rglob("*"):
    relative = path.relative_to(root).as_posix()
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
        raise SystemExit("PCF1 model tree contains a link or special member")
    if stat.S_ISREG(mode):
        actual_regular.add(relative)
        if relative != "SHA256SUMS":
            covered_bytes += path.stat().st_size
    else:
        actual_dirs.add(relative)
if (
    actual_regular != expected_regular
    or actual_dirs != expected_dirs
    or covered_bytes != expected_bytes
):
    raise SystemExit("PCF1 model exact membership differs")
PY
  (cd "$root" && sha256sum -c "$manifest" >/dev/null)
}

pcf1_bind_prepared_inputs() {
  pcf1_require PREPARE_ROOT
  pcf1_reject_protected_path "$PREPARE_ROOT"
  local receipt=$PREPARE_ROOT/receipt.json
  local python
  pcf1_require_file "$receipt"
  python=$(pcf1_python)
  local hashes
  hashes=$("$python" - "$PREPARE_ROOT" "$MODEL_ROOT" "$MODEL_REVISION" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root, model_root = map(Path, sys.argv[1:3])
revision = sys.argv[3]
receipt_path = root / "receipt.json"
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

outputs = receipt.get("outputs", {})
b1 = root / "b1_train.jsonl"
manifest = root / "model_manifest.sha256"
environment = root / "environment_receipt.json"
sources = root / "sources"
if (
    receipt.get("schema") != "shohin-pcf1-prepare-receipt-v1"
    or receipt.get("status") != "complete"
    or receipt.get("model_revision") != revision
    or Path(receipt.get("model_root", "")).resolve() != model_root.resolve()
    or outputs.get("b1", {}).get("path") != str(b1.resolve())
    or outputs.get("b1", {}).get("sha256") != digest(b1)
    or outputs.get("model_manifest", {}).get("path") != str(manifest.resolve())
    or outputs.get("model_manifest", {}).get("sha256") != digest(manifest)
    or outputs.get("environment", {}).get("path") != str(environment.resolve())
    or outputs.get("environment", {}).get("sha256") != digest(environment)
    or outputs.get("sources", {}).get("path") != str(sources.resolve())
    or outputs.get("sources", {}).get("report_sha256")
    != digest(sources / "report.json")
):
    raise SystemExit("PCF1 prepared-input receipt differs")
print(
    outputs["b1"]["sha256"],
    outputs["model_manifest"]["sha256"],
    receipt["model_config_sha256"],
    outputs["environment"]["sha256"],
)
PY
)
  read -r B1_DATA_SHA256 MODEL_MANIFEST_SHA256 MODEL_CONFIG_SHA256 ENVIRONMENT_RECEIPT_SHA256 <<<"$hashes"
  SOURCE_ROOT=$PREPARE_ROOT/sources
  B1_DATA=$PREPARE_ROOT/b1_train.jsonl
  MODEL_MANIFEST=$PREPARE_ROOT/model_manifest.sha256
  ENVIRONMENT_RECEIPT=$PREPARE_ROOT/environment_receipt.json
  export SOURCE_ROOT B1_DATA B1_DATA_SHA256 MODEL_MANIFEST
  export MODEL_MANIFEST_SHA256 MODEL_CONFIG_SHA256
  export ENVIRONMENT_RECEIPT ENVIRONMENT_RECEIPT_SHA256
}

pcf1_reject_protected_path() {
  local path=$1
  local resolved folded resolver
  resolver=${PYTHON:-$(command -v python3)}
  resolved=$(
    "$resolver" - "$path" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve(strict=False))
PY
  )
  folded=$(printf '%s\n%s' "$path" "$resolved" | tr '[:upper:]' '[:lower:]')
  local term
  for term in holdout product public; do
    [[ "$folded" != *"$term"* ]] || pcf1_die "protected path supplied: $path"
  done
}

pcf1_validate_sandbox_receipt() {
  local receipt=$1
  local expected=${2:-}
  local python module_root
  pcf1_require_safe_input "$receipt"
  python=$(pcf1_python)
  if [[ -n "${RUNTIME:-}" ]]; then
    module_root=$RUNTIME/train
  else
    module_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
  fi
  PYTHONPATH=$module_root "$python" - "$receipt" "$expected" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
from pcf1_code_sandbox import PCF1SandboxError, validate_sandbox_receipt_payload

path = Path(sys.argv[1])
expected = sys.argv[2]
digest = hashlib.sha256(path.read_bytes()).hexdigest()
if expected and digest != expected:
    raise SystemExit("PCF1 sandbox receipt hash differs")
receipt = json.loads(path.read_text(encoding="utf-8"))
try:
    validate_sandbox_receipt_payload(receipt)
except (PCF1SandboxError, TypeError, ValueError) as error:
    raise SystemExit("PCF1 sandbox receipt differs") from error
print(digest)
PY
}

pcf1_require_fresh() {
  local path=$1
  pcf1_reject_protected_path "$path"
  [[ ! -e "$path" && ! -L "$path" ]] || pcf1_die "refusing existing output: $path"
}

pcf1_require_safe_input() {
  local path=$1
  pcf1_reject_protected_path "$path"
  pcf1_require_file "$path"
}

pcf1_python() {
  pcf1_require PYTHON
  [[ -x "$PYTHON" ]] || pcf1_die "PYTHON is not executable: $PYTHON"
  printf '%s\n' "$PYTHON"
}

pcf1_stage_model_to() {
  local destination=$1
  local expected_files=$2
  local expected_bytes=$3
  [[ ! -e "$destination" ]] || pcf1_die "staged model path already exists"
  mkdir -p "$destination"
  cp -aL "$MODEL_ROOT"/. "$destination"/
  pcf1_verify_sha256 "$destination/config.json" "$MODEL_CONFIG_SHA256"
  pcf1_verify_model_tree "$destination" "$destination/SHA256SUMS" \
    "$expected_files" "$expected_bytes"
  printf '%s\n' "$destination"
}

pcf1_stage_model() {
  local destination
  [[ "${PCF1_SCRATCH_OWNED:-0}" == 1 ]] || \
    pcf1_die "model staging requires parent-owned scratch"
  destination=$SLURM_TMPDIR/pcf1-model
  pcf1_stage_model_to "$destination" 58 35706515534
}

pcf1_export_runtime() {
  pcf1_initialize_scratch
  local cache_root=$SLURM_TMPDIR/pcf1-cache
  mkdir -p "$cache_root/hf" "$cache_root/transformers" "$cache_root/datasets" \
    "$cache_root/xdg" "$cache_root/tmp"
  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export PYTHONDONTWRITEBYTECODE=1
  export HF_HOME=$cache_root/hf
  export HUGGINGFACE_HUB_CACHE=$cache_root/hf/hub
  export TRANSFORMERS_CACHE=$cache_root/transformers
  export HF_DATASETS_CACHE=$cache_root/datasets
  export XDG_CACHE_HOME=$cache_root/xdg
  export TMPDIR=$cache_root/tmp
  export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
}

pcf1_initialize_scratch_to() {
  local parent=$1
  local minimum_bytes=$2
  local minimum_inodes=$3
  local expected_parent_uid=$4
  pcf1_require PYTHON
  [[ "$minimum_bytes" =~ ^[1-9][0-9]*$ ]] || pcf1_die "scratch byte floor differs"
  [[ "$minimum_inodes" =~ ^[1-9][0-9]*$ ]] || pcf1_die "scratch inode floor differs"
  [[ "$expected_parent_uid" =~ ^[0-9]+$ ]] || pcf1_die "scratch parent UID differs"
  [[ "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ ]] || pcf1_die "SLURM_JOB_ID is required for scratch"
  local task=scalar
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    [[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ ]] || pcf1_die "scratch array task differs"
    task=$SLURM_ARRAY_TASK_ID
  fi
  local expected=$parent/pcf1-$SLURM_JOB_ID-$task
  if [[ "${PCF1_SCRATCH_OWNED:-0}" == 1 ]]; then
    [[ "${SLURM_TMPDIR:-}" == "$expected" ]] || pcf1_die "owned scratch path differs"
    [[ -d "$SLURM_TMPDIR" && ! -L "$SLURM_TMPDIR" ]] || pcf1_die "owned scratch disappeared"
    [[ "$(realpath "$SLURM_TMPDIR")" == "$expected" ]] || pcf1_die "owned scratch resolution differs"
    return
  fi
  [[ -z "${SLURM_TMPDIR:-}" ]] || pcf1_die "ambient SLURM_TMPDIR is not admissible"
  local scratch_record
  scratch_record=$("$PYTHON" - "$parent" "$SLURM_JOB_ID" "$task" \
    "$minimum_bytes" "$minimum_inodes" "$expected_parent_uid" <<'PY'
import os
from pathlib import Path
import stat
import sys

parent = Path(sys.argv[1])
job_id, task = sys.argv[2:4]
minimum_bytes, minimum_inodes, expected_parent_uid = map(int, sys.argv[4:])
parent_status = parent.lstat()
if (
    not stat.S_ISDIR(parent_status.st_mode)
    or stat.S_ISLNK(parent_status.st_mode)
    or parent.resolve(strict=True) != parent
    or parent_status.st_uid != expected_parent_uid
    or stat.S_IMODE(parent_status.st_mode) != 0o1777
):
    raise SystemExit("PCF1 scratch parent differs")
name = f"pcf1-{job_id}-{task}"
if not name.replace("-", "").isalnum() or "/" in name:
    raise SystemExit("PCF1 scratch name differs")
path = parent / name
if path.exists() or path.is_symlink():
    raise SystemExit("PCF1 scratch path already exists")
path.mkdir(mode=0o700)
try:
    status = path.lstat()
    if (
        not stat.S_ISDIR(status.st_mode)
        or stat.S_ISLNK(status.st_mode)
        or status.st_uid != os.getuid()
        or stat.S_IMODE(status.st_mode) != 0o700
        or status.st_dev != parent_status.st_dev
        or path.resolve(strict=True) != path
        or any(path.iterdir())
    ):
        raise SystemExit("PCF1 created scratch geometry differs")
    filesystem = os.statvfs(path)
    available_bytes = filesystem.f_bavail * filesystem.f_frsize
    available_inodes = filesystem.f_favail
    if available_bytes < minimum_bytes or available_inodes < minimum_inodes:
        raise SystemExit("PCF1 allocation-local scratch capacity is unsafe")
    probe = path / ".pcf1-write-probe"
    renamed = path / ".pcf1-write-probe-renamed"
    descriptor = os.open(
        probe,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.write(descriptor, b"PCF1 allocation scratch probe\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    probe.rename(renamed)
    directory = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    renamed.unlink()
    print(
        "\t".join(
            map(
                str,
                (path, available_bytes, available_inodes, status.st_dev),
            )
        )
    )
except BaseException:
    try:
        for child in path.iterdir():
            child.unlink(missing_ok=True)
        path.rmdir()
    finally:
        raise
PY
  )
  IFS=$'\t' read -r SLURM_TMPDIR PCF1_SCRATCH_AVAILABLE_BYTES \
    PCF1_SCRATCH_AVAILABLE_INODES PCF1_SCRATCH_DEVICE <<<"$scratch_record"
  [[ "$SLURM_TMPDIR" == "$expected" ]] || pcf1_die "created scratch path differs"
  PCF1_SCRATCH_PARENT_ACTIVE=$parent
  PCF1_SCRATCH_OWNED=1
  export SLURM_TMPDIR PCF1_SCRATCH_PARENT_ACTIVE PCF1_SCRATCH_OWNED
  export PCF1_SCRATCH_AVAILABLE_BYTES PCF1_SCRATCH_AVAILABLE_INODES PCF1_SCRATCH_DEVICE
  trap pcf1_cleanup_scratch EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
}

pcf1_initialize_scratch() {
  pcf1_initialize_scratch_to "$PCF1_SCRATCH_PARENT" \
    "$PCF1_SCRATCH_MIN_BYTES" "$PCF1_SCRATCH_MIN_INODES" 0
}

pcf1_cleanup_scratch() {
  [[ "${PCF1_SCRATCH_OWNED:-0}" == 1 ]] || return 0
  local task=scalar
  [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || task=$SLURM_ARRAY_TASK_ID
  local expected=${PCF1_SCRATCH_PARENT_ACTIVE:?}/pcf1-${SLURM_JOB_ID:?}-$task
  [[ "${SLURM_TMPDIR:-}" == "$expected" ]] || pcf1_die "scratch cleanup target differs"
  "$PYTHON" - "$PCF1_SCRATCH_PARENT_ACTIVE" "$SLURM_TMPDIR" <<'PY'
from pathlib import Path
import os
import shutil
import stat
import sys

parent, path = map(Path, sys.argv[1:])
parent_status = parent.lstat()
status = path.lstat()
if (
    path.parent != parent
    or path.resolve(strict=True) != path
    or not path.name.startswith("pcf1-")
    or not stat.S_ISDIR(status.st_mode)
    or stat.S_ISLNK(status.st_mode)
    or status.st_uid != os.getuid()
    or status.st_dev != parent_status.st_dev
):
    raise SystemExit("PCF1 scratch cleanup boundary differs")
shutil.rmtree(path)
if path.exists() or path.is_symlink():
    raise SystemExit("PCF1 scratch cleanup failed")
directory = os.open(parent, os.O_RDONLY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PY
  PCF1_SCRATCH_OWNED=0
  unset SLURM_TMPDIR TMPDIR HF_HOME HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE
  unset HF_DATASETS_CACHE XDG_CACHE_HOME
  export PCF1_SCRATCH_OWNED
}

pcf1_validate_json() {
  local path=$1
  local schema=$2
  local status=${3:-complete}
  local python
  python=$(pcf1_python)
  "$python" - "$path" "$schema" "$status" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("schema") != sys.argv[2] or payload.get("status") != sys.argv[3]:
    raise SystemExit(f"PCF1 receipt differs: {path}")
PY
}

pcf1_validate_b1_receipt() {
  local receipt=$1
  local checkpoint=$2
  local python
  pcf1_require_safe_input "$receipt"
  pcf1_require_safe_input "$checkpoint"
  python=$(pcf1_python)
  pcf1_require ENVIRONMENT_RECEIPT
  pcf1_require ENVIRONMENT_RECEIPT_SHA256
  "$python" - "$receipt" "$checkpoint" "$ENVIRONMENT_RECEIPT_SHA256" "$ENVIRONMENT_RECEIPT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

receipt_path, checkpoint = map(Path, sys.argv[1:3])
expected_environment_sha256 = sys.argv[3]
environment_path = Path(sys.argv[4])
environment = json.loads(environment_path.read_text(encoding="utf-8"))
environment_tree_sha256 = environment.get("environment_tree", {}).get("sha256")
report = json.loads(receipt_path.read_text(encoding="utf-8"))
actual = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
if (
    report.get("schema") != "shohin-pcf1-b1-training-receipt-v1"
    or report.get("status") != "complete"
    or Path(report.get("checkpoint", "")).resolve() != checkpoint.resolve()
    or report.get("checkpoint_sha256") != actual
    or report.get("data_sha256")
    != "2461d6f70b44a142854d56c24e1fb42d600065e5788a2c4e055ba47b12696549"
    or report.get("legacy_schema_model_visible") is not False
    or report.get("qualified_lineage_preserved") is not True
    or report.get("lora_layer_indices") != [30, 31, 32, 33]
    or report.get("environment_receipt_sha256")
    != expected_environment_sha256
    or report.get("environment_tree_sha256") != environment_tree_sha256
):
    raise SystemExit("PCF1 B1 wrapper receipt differs")
PY
}

pcf1_freeze_tree() {
  local path=$1
  chmod -R a-w "$path"
}
