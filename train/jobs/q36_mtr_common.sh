#!/bin/bash
# Fail-closed helpers for the exactly-once Q36-MTR graph.

set -euo pipefail

readonly Q36_MODEL_REVISION=995ad96eacd98c81ed38be0c5b274b04031597b0
readonly Q36_MODEL_CONFIG_SHA256=93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99
readonly Q36_MODEL_MANIFEST_SHA256=06c9d8d8419244f2d001cb351e164f356718d9d77138e898b13afee35856f56e
readonly Q36_EXCLUDED_NODES=evc26,evc29,evc31,evc32,evc33,evc34,evc37,evc38,evc43,evc46,evc50
readonly Q36_PYTHON_ENTRYPOINT=/lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603-r2/bin/python
readonly Q36_BNB_ROOT=/lustre/fs1/home/sa305415/shohin/env_targets/bitsandbytes-0.50.0-r1
readonly Q36_BNB_MANIFEST_SHA256=2201774754fb2e0fdd2208b78d34b803b910d8e34c79a43de49b29d7df3a8355
readonly Q36_FAST_KERNEL_ROOT=/lustre/fs1/home/sa305415/shohin/env_targets/qwen36-fastkernels-0.4.2-r5
readonly Q36_FAST_KERNEL_MANIFEST_SHA256=dde2adf539302a321afd7322ded3f2f729ac5f96368113a8af82f64efc0b9e8b
readonly Q36_NEMOTRON_CUDA_HOME=/apps/cuda/cuda-12.4.0
readonly Q36_NEMOTRON_GCC_ROOT=/apps/gcc/gcc-12.2.0
readonly Q36_NEMOTRON_NVCC_SHA256=e701519f13153518f0143cc0c18c66f0226eabf73ddd6a7eca0d36b26ebc976b
readonly Q36_NEMOTRON_GCC_SHA256=b617db0d6e6fade76990baa29f1372255575d3178ee2e8f60ba19980db37100f
readonly Q36_NEMOTRON_GXX_SHA256=6264680f3e8ee209ed3b2c22c4040282e9b63fb0d7ec17df71e81765e53db34d

q36_die() {
  printf 'q36-mtr: %s\n' "$*" >&2
  exit 2
}

q36_require() {
  local name=$1
  [[ -n "${!name:-}" ]] || q36_die "$name is required"
}

q36_require_file() {
  [[ -f "$1" && ! -L "$1" ]] || q36_die "missing or linked file: $1"
}

q36_require_dir() {
  [[ -d "$1" && ! -L "$1" ]] || q36_die "missing or linked directory: $1"
}

q36_sha256() {
  sha256sum "$1" | cut -d' ' -f1
}

q36_verify_sha256() {
  local path=$1 expected=$2
  q36_require_file "$path"
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || q36_die "invalid SHA-256"
  [[ "$(q36_sha256 "$path")" == "$expected" ]] || q36_die "SHA-256 differs: $path"
}

q36_init_local_tmp() {
  q36_require SLURM_JOB_ID
  [[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || q36_die "Slurm job identity differs"
  local suffix=$SLURM_JOB_ID
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    [[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ ]] || q36_die "Slurm array identity differs"
    suffix=${suffix}_${SLURM_ARRAY_TASK_ID}
  fi
  export SLURM_TMPDIR=/tmp/q36-mtr-$suffix
  if [[ -e "$SLURM_TMPDIR" || -L "$SLURM_TMPDIR" ]]; then
    [[ -d "$SLURM_TMPDIR" && ! -L "$SLURM_TMPDIR" ]] || q36_die "local temporary root differs"
    [[ "$(stat -c %U "$SLURM_TMPDIR")" == "$(id -un)" ]] || q36_die "local temporary owner differs"
    [[ -z "$(find "$SLURM_TMPDIR" -mindepth 1 -maxdepth 1 -print -quit)" ]] || q36_die "local temporary root is not empty"
  else
    mkdir -m 700 "$SLURM_TMPDIR"
  fi
}

q36_export_nemotron_cuda_toolchain() {
  [[ -d "$Q36_NEMOTRON_CUDA_HOME" && ! -L "$Q36_NEMOTRON_CUDA_HOME" ]] \
    || q36_die "Nemotron CUDA root differs"
  [[ -d "$Q36_NEMOTRON_GCC_ROOT" && ! -L "$Q36_NEMOTRON_GCC_ROOT" ]] \
    || q36_die "Nemotron compiler root differs"
  q36_verify_sha256 \
    "$Q36_NEMOTRON_CUDA_HOME/bin/nvcc" "$Q36_NEMOTRON_NVCC_SHA256"
  q36_verify_sha256 \
    "$Q36_NEMOTRON_GCC_ROOT/bin/gcc" "$Q36_NEMOTRON_GCC_SHA256"
  q36_verify_sha256 \
    "$Q36_NEMOTRON_GCC_ROOT/bin/g++" "$Q36_NEMOTRON_GXX_SHA256"
  export CUDA_HOME="$Q36_NEMOTRON_CUDA_HOME"
  export CC="$Q36_NEMOTRON_GCC_ROOT/bin/gcc"
  export CXX="$Q36_NEMOTRON_GCC_ROOT/bin/g++"
  export PATH="$CUDA_HOME/bin:$Q36_NEMOTRON_GCC_ROOT/bin:/apps/slurm/current/bin:/usr/bin:/bin"
  export LD_LIBRARY_PATH="$Q36_NEMOTRON_GCC_ROOT/lib64:$CUDA_HOME/lib64"
  export TORCH_CUDA_ARCH_LIST=9.0
  export TORCH_EXTENSIONS_DIR="$SLURM_TMPDIR/torch_extensions"
  export MAX_JOBS=8
  mkdir -m 700 "$TORCH_EXTENSIONS_DIR"
}

q36_cleanup_local_tmp() {
  [[ "${SLURM_TMPDIR:-}" =~ ^/tmp/q36-mtr-[0-9]+(_[0-9]+)?$ ]] || return 0
  if [[ -e "$SLURM_TMPDIR" || -L "$SLURM_TMPDIR" ]]; then
    # Staged model/runtime trees are deliberately frozen read-only. Restore
    # owner traversal on directories inside this exact job-owned /tmp root so
    # cleanup cannot strand a full model copy or obscure the stage exit code.
    /usr/bin/find "$SLURM_TMPDIR" -xdev -type d -exec /bin/chmod u+rwx {} +
    /bin/rm -rf --one-file-system -- "$SLURM_TMPDIR"
  fi
}

q36_require_authorization() {
  q36_require PHASE_AUTHORIZATION
  q36_require PHASE_AUTHORIZATION_SHA256
  q36_verify_sha256 "$PHASE_AUTHORIZATION" "$PHASE_AUTHORIZATION_SHA256"
  "$PYTHON" - "$PHASE_AUTHORIZATION" "$SOURCE_COMMIT" \
    "${RUN_ID:-}" "${OUTPUT:-}" <<'PY'
import json
from pathlib import Path
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
run_id = sys.argv[3]
output = sys.argv[4]
if (
    payload.get("schema") != "shohin-q36-mtr-phase-authorization-v1"
    or payload.get("status") != "authorized"
    or payload.get("scientific_submit_authorized") is not True
    or payload.get("source_commit") != sys.argv[2]
    or payload.get("model_revision")
    != "995ad96eacd98c81ed38be0c5b274b04031597b0"
    or payload.get("gate") != "one_source_disjoint_development_gate"
    or payload.get("automatic_retry") is not False
    or payload.get("automatic_successor") is not False
    or payload.get("automatic_confirmation") is not False
    or payload.get("stop_after_gate") is not True
):
    raise SystemExit("Q36-MTR phase authorization differs")
if run_id and payload.get("run_id") != run_id:
    raise SystemExit("Q36-MTR phase run identity differs")
if output:
    root = Path(str(payload.get("run_root", ""))).resolve(strict=False)
    target = Path(output).resolve(strict=False)
    if not root.is_absolute() or root in {Path("/"), Path.home().resolve()}:
        raise SystemExit("Q36-MTR authorized run root differs")
    try:
        relative = target.relative_to(root)
    except ValueError as error:
        raise SystemExit("Q36-MTR output escapes its authorized run root") from error
    if not relative.parts:
        raise SystemExit("Q36-MTR output equals its authorized run root")
PY
}

q36_verify_runtime() {
  q36_require RUNTIME
  q36_require RUNTIME_MANIFEST_SHA256
  q36_require_dir "$RUNTIME"
  [[ "$PYTHON" == "$Q36_PYTHON_ENTRYPOINT" ]] || q36_die "Python entrypoint differs"
  q36_verify_sha256 "$RUNTIME/SHA256SUMS" "$RUNTIME_MANIFEST_SHA256"
  "$PYTHON" - "$RUNTIME" <<'PY'
from pathlib import Path, PurePosixPath
import stat
import sys

root = Path(sys.argv[1]).resolve(strict=True)
entries = []
for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
    digest, separator, relative = line.partition("  ")
    pure = PurePosixPath(relative)
    if (
        not separator
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or pure.is_absolute()
        or ".." in pure.parts
        or "." in pure.parts
        or not relative
    ):
        raise SystemExit("Q36-MTR runtime manifest entry differs")
    entries.append(pure.as_posix())
if entries != sorted(entries) or len(entries) != len(set(entries)):
    raise SystemExit("Q36-MTR runtime manifest order differs")
actual = set()
for path in root.rglob("*"):
    mode = path.lstat().st_mode
    if stat.S_ISREG(mode):
        actual.add(path.relative_to(root).as_posix())
    elif not stat.S_ISDIR(mode):
        raise SystemExit("Q36-MTR runtime contains a link or special member")
if actual != {*entries, "SHA256SUMS"}:
    raise SystemExit("Q36-MTR runtime exact membership differs")
PY
  (cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
}

q36_verify_overlay() {
  local root=$1 expected=$2
  q36_require_dir "$root"
  q36_verify_sha256 "$root/SHA256SUMS" "$expected"
  "$PYTHON" -P -s -B - "$root" <<'PY'
from pathlib import Path, PurePosixPath
import hashlib
import stat
import sys

root = Path(sys.argv[1]).resolve(strict=True)
manifest = root / "SHA256SUMS"
declared = []
for line in manifest.read_text(encoding="utf-8").splitlines():
    digest, separator, relative = line.partition("  ")
    relative = relative[2:] if relative.startswith("./") else relative
    pure = PurePosixPath(relative)
    if (
        not separator
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or not pure.parts
        or pure.is_absolute()
        or "." in pure.parts
        or ".." in pure.parts
        or pure.as_posix() != relative
        or relative in declared
    ):
        raise SystemExit("Q36-MTR overlay manifest entry differs")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise SystemExit("Q36-MTR overlay member differs")
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    if value.hexdigest() != digest:
        raise SystemExit("Q36-MTR overlay member hash differs")
    declared.append(relative)
if declared != sorted(declared):
    raise SystemExit("Q36-MTR overlay manifest order differs")
actual = set()
for path in root.rglob("*"):
    mode = path.lstat().st_mode
    if stat.S_ISREG(mode):
        actual.add(path.relative_to(root).as_posix())
    elif not stat.S_ISDIR(mode):
        raise SystemExit("Q36-MTR overlay has a link or special member")
if actual != {*declared, "SHA256SUMS"}:
    raise SystemExit("Q36-MTR overlay exact membership differs")
PY
}

q36_verify_environment() {
  q36_require ENVIRONMENT_RECEIPT
  q36_require ENVIRONMENT_RECEIPT_SHA256
  q36_require ENVIRONMENT_TREE_SHA256
  q36_verify_sha256 "$ENVIRONMENT_RECEIPT" "$ENVIRONMENT_RECEIPT_SHA256"
  q36_verify_overlay "$Q36_BNB_ROOT" "$Q36_BNB_MANIFEST_SHA256"
  q36_verify_overlay "$Q36_FAST_KERNEL_ROOT" "$Q36_FAST_KERNEL_MANIFEST_SHA256"
  "$PYTHON" -P -s -B - "$ENVIRONMENT_RECEIPT" "$ENVIRONMENT_TREE_SHA256" \
    "$RUNTIME" "$RUNTIME_MANIFEST_SHA256" <<'PY'
import json
from pathlib import Path
import sys

receipt = json.load(open(sys.argv[1], encoding="utf-8"))
runtime = Path(sys.argv[3]).resolve(strict=True)
if (
    receipt.get("schema") != "shohin-q36-mtr-environment-v1"
    or receipt.get("status") != "pass"
    or receipt.get("model_revision") != "995ad96eacd98c81ed38be0c5b274b04031597b0"
    or receipt.get("model_config_sha256") != "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
    or receipt.get("environment_tree_sha256") != sys.argv[2]
    or receipt.get("runtime_root") != str(runtime)
    or receipt.get("runtime_manifest_sha256") != sys.argv[4]
    or receipt.get("bitsandbytes_overlay", {}).get("manifest_sha256")
       != "2201774754fb2e0fdd2208b78d34b803b910d8e34c79a43de49b29d7df3a8355"
    or receipt.get("fast_kernel_overlay", {}).get("manifest_sha256")
       != "dde2adf539302a321afd7322ded3f2f729ac5f96368113a8af82f64efc0b9e8b"
    or receipt.get("packages", {}).get("bitsandbytes") != "0.50.0"
    or receipt.get("packages", {}).get("flash-linear-attention") != "0.4.2"
    or receipt.get("packages", {}).get("causal-conv1d") != "1.6.2.post1"
    or receipt.get("offline_required") is not True
    or receipt.get("bytecode_writes_permitted") is not False
    or receipt.get("scientific_rows_read") != 0
):
    raise SystemExit("Q36-MTR environment receipt differs")
PY
}

q36_export_pythonpath() {
  export PYTHONPATH="$Q36_FAST_KERNEL_ROOT:$Q36_BNB_ROOT:$RUNTIME/train:$RUNTIME/pipeline"
}

q36_verify_model() {
  for variable in MODEL_ROOT MODEL_MANIFEST MODEL_MANIFEST_SHA256 MODEL_REVISION MODEL_CONFIG_SHA256; do
    q36_require "$variable"
  done
  [[ "$MODEL_REVISION" == "$Q36_MODEL_REVISION" ]] || q36_die "model revision differs"
  [[ "$MODEL_CONFIG_SHA256" == "$Q36_MODEL_CONFIG_SHA256" ]] || q36_die "model config differs"
  [[ "$MODEL_MANIFEST_SHA256" == "$Q36_MODEL_MANIFEST_SHA256" ]] || q36_die "model manifest differs"
  q36_require_dir "$MODEL_ROOT"
  [[ "$(realpath "$MODEL_MANIFEST")" == "$(realpath "$MODEL_ROOT/SHA256SUMS")" ]] || \
    q36_die "model manifest is not rooted in the exact host"
  q36_verify_sha256 "$MODEL_ROOT/config.json" "$MODEL_CONFIG_SHA256"
  q36_verify_sha256 "$MODEL_MANIFEST" "$MODEL_MANIFEST_SHA256"
  "$PYTHON" - "$MODEL_ROOT" <<'PY'
from pathlib import Path, PurePosixPath
import stat
import sys

root = Path(sys.argv[1]).resolve(strict=True)
entries = []
for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
    digest, separator, relative = line.partition("  ")
    pure = PurePosixPath(relative)
    if (
        not separator
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or pure.is_absolute()
        or ".." in pure.parts
        or "." in pure.parts
        or not relative
    ):
        raise SystemExit("Q36-MTR model manifest entry differs")
    entries.append(pure.as_posix())
if entries != sorted(entries) or len(entries) != len(set(entries)):
    raise SystemExit("Q36-MTR model manifest order differs")
actual = set()
for path in root.rglob("*"):
    mode = path.lstat().st_mode
    if stat.S_ISREG(mode):
        actual.add(path.relative_to(root).as_posix())
    elif not stat.S_ISDIR(mode):
        raise SystemExit("Q36-MTR model contains a link or special member")
if actual != {*entries, "SHA256SUMS"}:
    raise SystemExit("Q36-MTR model exact membership differs")
PY
  (cd "$MODEL_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
}

q36_stage_model() {
  q36_require SLURM_TMPDIR
  [[ -d "$SLURM_TMPDIR" && ! -L "$SLURM_TMPDIR" ]] || q36_die "SLURM_TMPDIR differs"
  local staged=$SLURM_TMPDIR/q36-model
  local required_kib available_kib
  [[ ! -e "$staged" ]] || q36_die "staged model already exists"
  required_kib=$(du -sk "$MODEL_ROOT" | awk '{print $1}')
  available_kib=$(df -Pk "$SLURM_TMPDIR" | awk 'NR == 2 {print $4}')
  [[ "$required_kib" =~ ^[0-9]+$ && "$available_kib" =~ ^[0-9]+$ ]] || \
    q36_die "local model staging capacity is unreadable"
  (( available_kib >= required_kib + 2097152 )) || \
    q36_die "local model staging capacity is insufficient"
  mkdir "$staged"
  cp -a "$MODEL_ROOT"/. "$staged"/ || q36_die "model staging copy failed"
  q36_verify_sha256 "$staged/config.json" "$MODEL_CONFIG_SHA256"
  (cd "$staged" && sha256sum -c SHA256SUMS >/dev/null) || \
    q36_die "staged model manifest differs"
  printf '%s\n' "$staged"
}
