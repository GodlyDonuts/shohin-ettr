#!/bin/bash
# Fail-closed helpers for the prospective Q36-MTR graph. No dispatcher exists.

set -euo pipefail

readonly Q36_MODEL_REVISION=995ad96eacd98c81ed38be0c5b274b04031597b0
readonly Q36_MODEL_CONFIG_SHA256=93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99
readonly Q36_EXCLUDED_NODES=evc26,evc29,evc31,evc32,evc33,evc37,evc38,evc46

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

q36_require_authorization() {
  q36_require PHASE_AUTHORIZATION
  q36_require PHASE_AUTHORIZATION_SHA256
  q36_verify_sha256 "$PHASE_AUTHORIZATION" "$PHASE_AUTHORIZATION_SHA256"
  "$PYTHON" - "$PHASE_AUTHORIZATION" "$SOURCE_COMMIT" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
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
):
    raise SystemExit("Q36-MTR phase authorization differs")
PY
}

q36_verify_runtime() {
  q36_require RUNTIME
  q36_require RUNTIME_MANIFEST_SHA256
  q36_require_dir "$RUNTIME"
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

q36_verify_model() {
  for variable in MODEL_ROOT MODEL_MANIFEST MODEL_MANIFEST_SHA256 MODEL_REVISION MODEL_CONFIG_SHA256; do
    q36_require "$variable"
  done
  [[ "$MODEL_REVISION" == "$Q36_MODEL_REVISION" ]] || q36_die "model revision differs"
  [[ "$MODEL_CONFIG_SHA256" == "$Q36_MODEL_CONFIG_SHA256" ]] || q36_die "model config differs"
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
  [[ ! -e "$staged" ]] || q36_die "staged model already exists"
  mkdir "$staged"
  cp -a "$MODEL_ROOT"/. "$staged"/
  q36_verify_sha256 "$staged/config.json" "$MODEL_CONFIG_SHA256"
  printf '%s\n' "$staged"
}
