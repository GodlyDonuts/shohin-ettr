#!/bin/bash
# Resumably copy one immutable ETTR-IL-v3 release from Stokes to Newton.
#
# This script runs on Stokes.  It never accepts a password as an argument:
# sshpass reads SSHPASS from the inherited environment.  The destination is
# published only after a complete SHA-256 inventory and the production ETTR
# streaming verifier both pass on Newton.

set -euo pipefail
export PATH=/usr/bin:/bin
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

SOURCE_RELEASE_ROOT=${SOURCE_RELEASE_ROOT:?set immutable Stokes release root}
SOURCE_DATA_ROOT=${SOURCE_DATA_ROOT:?set immutable Stokes data root}
SOURCE_TOKENIZER=${SOURCE_TOKENIZER:?set exact Stokes tokenizer}
DEST_HOST=${DEST_HOST:?set Newton SSH host}
DEST_ROOT=${DEST_ROOT:?set fresh final Newton destination root}
DEST_CODE_ROOT=${DEST_CODE_ROOT:?set immutable Newton ETTR source root}
DEST_PYTHON=${DEST_PYTHON:?set Newton Python interpreter}
SSHPASS=${SSHPASS:?export the Newton password only for this process}

for path in \
  "$SOURCE_RELEASE_ROOT" \
  "$SOURCE_DATA_ROOT" \
  "$SOURCE_TOKENIZER" \
  "$DEST_ROOT" \
  "$DEST_CODE_ROOT" \
  "$DEST_PYTHON"; do
  if [[ "$path" != /* || "$path" == *$'\n'* || "$path" == *$'\r'* ]]; then
    echo "ETTR transfer paths must be absolute single-line paths" >&2
    exit 2
  fi
done
for path in "$DEST_ROOT" "$DEST_CODE_ROOT" "$DEST_PYTHON"; do
  if [[ "$path" == *[!A-Za-z0-9_./-]* ]]; then
    echo "Newton transfer paths contain unsupported shell characters" >&2
    exit 2
  fi
done
if [[ "$DEST_HOST" == *[!A-Za-z0-9_.@-]* || "$DEST_HOST" != *@* ]]; then
  echo "Newton SSH host differs" >&2
  exit 2
fi
for command in python3 rsync sshpass ssh; do
  command -v "$command" >/dev/null || {
    echo "required transfer command is unavailable: $command" >&2
    exit 2
  }
done
if [[ ! -d "$SOURCE_RELEASE_ROOT" || -L "$SOURCE_RELEASE_ROOT" \
  || ! -d "$SOURCE_DATA_ROOT" || -L "$SOURCE_DATA_ROOT" \
  || ! -f "$SOURCE_TOKENIZER" || -L "$SOURCE_TOKENIZER" ]]; then
  echo "ETTR transfer source differs" >&2
  exit 2
fi

scratch=$(mktemp -d "${TMPDIR:-/tmp}/shohin-ettr-transfer.XXXXXX")
trap 'rm -rf "$scratch"' EXIT
inventory="$scratch/transfer-inventory.json"

python3 - \
  "$SOURCE_RELEASE_ROOT" \
  "$SOURCE_DATA_ROOT" \
  "$SOURCE_TOKENIZER" \
  "$inventory" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import stat
import sys


release_root, data_root, tokenizer, output = map(Path, sys.argv[1:])


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(8 * 1024 * 1024)
            if not block:
                break
            value.update(block)
    return value.hexdigest()


def record(path, relative, *, require_immutable):
    before = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or (require_immutable and before.st_mode & 0o222)
    ):
        raise SystemExit(f"source file is not immutable and regular: {relative}")
    sha256 = digest(path)
    after = path.lstat()
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_nlink,
    )
    if identity(before) != identity(after):
        raise SystemExit(f"source file changed while measured: {relative}")
    return {
        "bytes": before.st_size,
        "path": relative,
        "sha256": sha256,
    }


rows = []
for label, root in (("release", release_root), ("data", data_root)):
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        rows.append(
            record(
                path,
                f"{label}/{path.relative_to(root).as_posix()}",
                require_immutable=True,
            )
        )
rows.append(
    record(
        tokenizer,
        "tokenizer.json",
        require_immutable=False,
    )
)
if not rows or not any(row["path"] == "release/release.json" for row in rows):
    raise SystemExit("release inventory is incomplete")
release = json.loads((release_root / "release.json").read_text(encoding="ascii"))
claimed = release.pop("release_payload_sha256", None)
canonical = json.dumps(
    release,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
).encode("ascii")
if (
    release.get("status") != "pass"
    or not isinstance(claimed, str)
    or hashlib.sha256(canonical).hexdigest() != claimed
):
    raise SystemExit("source release receipt differs")
payload = {
    "files": rows,
    "release_payload_sha256": claimed,
    "schema": "shohin-ettr-il-v3-direct-transfer-inventory-v1",
}
payload["payload_sha256"] = hashlib.sha256(
    json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
).hexdigest()
encoded = json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
).encode("ascii")
descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "wb") as destination:
    destination.write(encoded)
    destination.flush()
    os.fsync(destination.fileno())
PY

partial_root="${DEST_ROOT}.partial"
ssh_options=(
  -o IdentitiesOnly=yes
  -o PreferredAuthentications=password
  -o PubkeyAuthentication=no
  -o NumberOfPasswordPrompts=1
  -o ConnectTimeout=20
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=6
)
export SSHPASS
remote_ssh=(sshpass -e ssh "${ssh_options[@]}")
export RSYNC_RSH
printf -v RSYNC_RSH 'sshpass -e ssh'
for value in "${ssh_options[@]}"; do
  printf -v RSYNC_RSH '%s %q' "$RSYNC_RSH" "$value"
done

"${remote_ssh[@]}" "$DEST_HOST" \
  "test ! -e '$DEST_ROOT' && mkdir -p -m 700 '$partial_root/release' '$partial_root/data'"

rsync \
  --archive \
  --partial \
  --append-verify \
  --chmod=D700,F600 \
  "$SOURCE_RELEASE_ROOT/" \
  "$DEST_HOST:$partial_root/release/"
rsync \
  --archive \
  --partial \
  --append-verify \
  --chmod=D700,F600 \
  "$SOURCE_DATA_ROOT/" \
  "$DEST_HOST:$partial_root/data/"
rsync \
  --archive \
  --partial \
  --append-verify \
  --chmod=F600 \
  "$SOURCE_TOKENIZER" \
  "$DEST_HOST:$partial_root/tokenizer.json"
rsync \
  --archive \
  --chmod=F600 \
  "$inventory" \
  "$DEST_HOST:$partial_root/transfer-inventory.json"

"${remote_ssh[@]}" "$DEST_HOST" \
  "test -x '$DEST_PYTHON' \
    && test -d '$DEST_CODE_ROOT' \
    && test -s '$DEST_CODE_ROOT/SHA256SUMS' \
    && cd '$DEST_CODE_ROOT' \
    && sha256sum -c SHA256SUMS >/dev/null"

"${remote_ssh[@]}" "$DEST_HOST" \
  "$DEST_PYTHON - '$partial_root' '$DEST_ROOT' '$DEST_CODE_ROOT'" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import stat
import sys


partial, final, code_root = map(Path, sys.argv[1:])
inventory_path = partial / "transfer-inventory.json"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(8 * 1024 * 1024)
            if not block:
                break
            value.update(block)
    return value.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()


def writable_tree(root):
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            continue
        path.chmod(0o700 if path.is_dir() else 0o600)
    root.chmod(0o700)


try:
    value = json.loads(inventory_path.read_text(encoding="ascii"))
    claimed = value.pop("payload_sha256", None)
    if (
        value.get("schema")
        != "shohin-ettr-il-v3-direct-transfer-inventory-v1"
        or not isinstance(claimed, str)
        or canonical_sha256(value) != claimed
        or not isinstance(value.get("files"), list)
    ):
        raise RuntimeError("transfer inventory receipt differs")
    expected = set()
    for row in value["files"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"bytes", "path", "sha256"}
            or not isinstance(row["path"], str)
            or row["path"].startswith("/")
            or ".." in Path(row["path"]).parts
        ):
            raise RuntimeError("transfer inventory row differs")
        path = partial / row["path"]
        metadata = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != row["bytes"]
            or digest(path) != row["sha256"]
        ):
            raise RuntimeError(f"transferred file differs: {row['path']}")
        expected.add(row["path"])
    observed = {
        path.relative_to(partial).as_posix()
        for root in (partial / "release", partial / "data")
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    observed.add("tokenizer.json")
    if observed != expected:
        raise RuntimeError("transferred file inventory differs")

    for root in (partial / "release", partial / "data"):
        for path in root.rglob("*"):
            if path.is_symlink():
                raise RuntimeError("transferred tree contains a symlink")
            path.chmod(0o500 if path.is_dir() else 0o400)
        root.chmod(0o500)
    (partial / "tokenizer.json").chmod(0o400)

    sys.path[:0] = [str(code_root / "train"), str(code_root / "pipeline")]
    from ettr_v3_streaming import ETTRV3StreamingRelease

    release_sha256 = digest(partial / "release" / "release.json")
    stream = ETTRV3StreamingRelease(
        partial / "release",
        expected_release_sha256=release_sha256,
        data_root=partial / "data",
        tokenizer_path=partial / "tokenizer.json",
    )
    verification = stream.verify_source_shards()
    receipt = {
        "inventory_payload_sha256": claimed,
        "release_file_sha256": release_sha256,
        "release_payload_sha256": value["release_payload_sha256"],
        "schema": "shohin-ettr-il-v3-direct-transfer-receipt-v1",
        "source_commit": stream.release["source_commit"],
        "source_verification": verification,
        "status": "pass",
    }
    receipt["payload_sha256"] = canonical_sha256(receipt)
    receipt_path = partial / "transfer-receipt.json"
    descriptor = os.open(
        receipt_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    with os.fdopen(descriptor, "wb") as destination:
        destination.write(
            json.dumps(
                receipt,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
        )
        destination.flush()
        os.fsync(destination.fileno())
    inventory_path.chmod(0o400)
    partial.chmod(0o500)
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"refusing existing final destination: {final}")
    os.rename(partial, final)
    descriptor = os.open(final.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps(receipt, sort_keys=True))
except BaseException:
    if partial.exists() and not partial.is_symlink():
        writable_tree(partial)
    raise
PY

printf 'ettr_direct_transfer_complete destination=%s\n' "$DEST_ROOT"
