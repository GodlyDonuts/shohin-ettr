"""Deterministic, recursively measured runtime archive for ETTR claims."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
from typing import BinaryIO, Iterator, Literal, Sequence


CLAIM_RUNTIME_SCHEMA = "ettr-claim-runtime-inventory-v1"
CLAIM_RUNTIME_RECEIPT_SCHEMA = "ettr-claim-runtime-verification-v1"
INVENTORY_NAME = "ETTR_CLAIM_RUNTIME_INVENTORY.json"
RUNTIME_PREFIX = "runtime"
PYTHON_RELATIVE_PATH = "miniforge3/bin/python"
CANDIDATE_SOURCE_RELATIVE_ROOT = "app/candidate"
TOOLS_RELATIVE_ROOT = "app/tools"
BOOTSTRAP_RELATIVE_PATH = "app/tools/run_ettr_verified_stage.py"
LANDLOCK_RELATIVE_PATH = "app/tools/landlock_stage_exec.py"
VERIFIER_RELATIVE_PATH = "app/tools/ettr_claim_runtime.py"
DEPLOYMENT_CONTRACT_RELATIVE_PATH = "app/tools/ettr_deployment_contract.py"
RUNTIME_RECEIPT_TOOL_RELATIVE_PATH = "app/tools/ettr_runtime_bundle.py"
COMMON_CANDIDATE_SOURCE_FILES = (
    "endogenous_typed_theory_reactor.py",
    "ettr_factorial_custody.py",
    "ettr_state_io.py",
    "model.py",
)
STAGE_RUNNERS = {
    "world": "run_ettr_world_compiler.py",
    "command": "run_ettr_state_executor.py",
    "query": "run_ettr_late_query.py",
}
CANDIDATE_SOURCE_FILES = tuple(
    sorted(
        {
            *COMMON_CANDIDATE_SOURCE_FILES,
            *STAGE_RUNNERS.values(),
        }
    )
)
CANDIDATE_SOURCE_PATHS = tuple(
    f"{stage}/{name}"
    for stage in STAGE_RUNNERS
    for name in (
        *COMMON_CANDIDATE_SOURCE_FILES,
        STAGE_RUNNERS[stage],
    )
)
TOOL_FILES = (
    "ettr_claim_runtime.py",
    "ettr_deployment_contract.py",
    "ettr_runtime_bundle.py",
    "landlock_stage_exec.py",
    "run_ettr_verified_stage.py",
)
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9._+@=-]+$")


class ETTRClaimRuntimeError(ValueError):
    """The claim runtime differs from its recursively measured identity."""


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _sha256_stream(handle: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _sha256_file(path: Path) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ETTRClaimRuntimeError(
            f"runtime member cannot be opened: {path}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ETTRClaimRuntimeError(
                f"runtime member is not a regular file: {path}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            digest, size = _sha256_stream(handle)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ETTRClaimRuntimeError(
                f"runtime member changed during hashing: {path}"
            )
        return digest, size
    finally:
        os.close(descriptor)


def _validate_relative_path(value: str, *, label: str) -> PurePosixPath:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or "\\" in value
    ):
        raise ETTRClaimRuntimeError(f"{label} is not a safe relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(
            part in {"", ".", ".."} or _SAFE_SEGMENT.fullmatch(part) is None
            for part in path.parts
        )
    ):
        raise ETTRClaimRuntimeError(f"{label} is not a safe relative path")
    return path


def _safe_symlink_target(member: PurePosixPath, target: str) -> str:
    if (
        not isinstance(target, str)
        or not target
        or "\x00" in target
        or "\\" in target
    ):
        raise ETTRClaimRuntimeError("runtime symlink target is unsafe")
    target_path = PurePosixPath(target)
    if target_path.is_absolute() or target != target_path.as_posix():
        raise ETTRClaimRuntimeError("runtime symlink target is unsafe")
    stack = list(member.parent.parts)
    for part in target_path.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not stack:
                raise ETTRClaimRuntimeError("runtime symlink escapes runtime root")
            stack.pop()
        elif _SAFE_SEGMENT.fullmatch(part) is None:
            raise ETTRClaimRuntimeError("runtime symlink target is unsafe")
        else:
            stack.append(part)
    if not stack:
        raise ETTRClaimRuntimeError("runtime symlink target resolves to root")
    return target


@dataclass(frozen=True, slots=True)
class ETTRClaimRuntimeMember:
    path: str
    kind: Literal["directory", "file", "symlink"]
    mode: int
    size: int
    sha256: str | None
    link_target: str | None

    def validate(self) -> None:
        path = _validate_relative_path(self.path, label="runtime member path")
        if (
            self.kind not in {"directory", "file", "symlink"}
            or isinstance(self.mode, bool)
            or not isinstance(self.mode, int)
            or isinstance(self.size, bool)
            or not isinstance(self.size, int)
            or self.size < 0
        ):
            raise ETTRClaimRuntimeError("runtime member metadata differs")
        if self.kind == "directory":
            if (
                self.mode != 0o555
                or self.size != 0
                or self.sha256 is not None
                or self.link_target is not None
            ):
                raise ETTRClaimRuntimeError("runtime directory metadata differs")
        elif self.kind == "file":
            if (
                self.mode not in {0o444, 0o555}
                or _SHA256.fullmatch(self.sha256 or "") is None
                or self.link_target is not None
            ):
                raise ETTRClaimRuntimeError("runtime file metadata differs")
        else:
            if (
                self.mode != 0o777
                or self.size != 0
                or self.sha256 is not None
                or self.link_target is None
            ):
                raise ETTRClaimRuntimeError("runtime symlink metadata differs")
            _safe_symlink_target(path, self.link_target)


@dataclass(frozen=True, slots=True)
class ETTRClaimRuntimeInventory:
    schema: str
    source_commit: str
    python_relative_path: str
    candidate_source_relative_root: str
    bootstrap_relative_path: str
    landlock_relative_path: str
    verifier_relative_path: str
    runtime_receipt_tool_relative_path: str
    members: tuple[ETTRClaimRuntimeMember, ...]

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(
            {
                **{
                    key: value
                    for key, value in asdict(self).items()
                    if key != "members"
                },
                "members": [asdict(member) for member in self.members],
            }
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def validate(self) -> None:
        if (
            self.schema != CLAIM_RUNTIME_SCHEMA
            or _COMMIT.fullmatch(self.source_commit) is None
            or self.python_relative_path != PYTHON_RELATIVE_PATH
            or self.candidate_source_relative_root
            != CANDIDATE_SOURCE_RELATIVE_ROOT
            or self.bootstrap_relative_path != BOOTSTRAP_RELATIVE_PATH
            or self.landlock_relative_path != LANDLOCK_RELATIVE_PATH
            or self.verifier_relative_path != VERIFIER_RELATIVE_PATH
            or self.runtime_receipt_tool_relative_path
            != RUNTIME_RECEIPT_TOOL_RELATIVE_PATH
            or not self.members
        ):
            raise ETTRClaimRuntimeError("claim runtime inventory identity differs")
        previous = ""
        paths: set[str] = set()
        for member in self.members:
            member.validate()
            path = PurePosixPath(member.path)
            if (
                "__pycache__" in path.parts
                or path.suffix in {".pyc", ".pyo", ".pth"}
                or path.name in {"sitecustomize.py", "usercustomize.py"}
            ):
                raise ETTRClaimRuntimeError(
                    "runtime inventory contains executable import metadata"
                )
            if member.path <= previous or member.path in paths:
                raise ETTRClaimRuntimeError(
                    "runtime members are not uniquely path-sorted"
                )
            previous = member.path
            paths.add(member.path)
        required = {
            PYTHON_RELATIVE_PATH,
            BOOTSTRAP_RELATIVE_PATH,
            LANDLOCK_RELATIVE_PATH,
            VERIFIER_RELATIVE_PATH,
            DEPLOYMENT_CONTRACT_RELATIVE_PATH,
            RUNTIME_RECEIPT_TOOL_RELATIVE_PATH,
            *(
                f"{CANDIDATE_SOURCE_RELATIVE_ROOT}/{path}"
                for path in CANDIDATE_SOURCE_PATHS
            ),
        }
        if not required <= paths:
            raise ETTRClaimRuntimeError(
                "claim runtime required member inventory differs"
            )
        candidate_children = {
            member.path.removeprefix(
                f"{CANDIDATE_SOURCE_RELATIVE_ROOT}/"
            )
            for member in self.members
            if member.path.startswith(
                f"{CANDIDATE_SOURCE_RELATIVE_ROOT}/"
            )
        }
        expected_candidate_children = {
            *STAGE_RUNNERS,
            *CANDIDATE_SOURCE_PATHS,
        }
        if candidate_children != expected_candidate_children:
            raise ETTRClaimRuntimeError(
                "candidate source bundle inventory differs"
            )
        tool_children = {
            member.path.removeprefix(f"{TOOLS_RELATIVE_ROOT}/")
            for member in self.members
            if member.path.startswith(f"{TOOLS_RELATIVE_ROOT}/")
        }
        if tool_children != set(TOOL_FILES):
            raise ETTRClaimRuntimeError("runtime tool inventory differs")

    @classmethod
    def from_bytes(cls, payload: bytes) -> ETTRClaimRuntimeInventory:
        try:
            value = json.loads(payload.decode("ascii"))
            inventory = cls(
                schema=value["schema"],
                source_commit=value["source_commit"],
                python_relative_path=value["python_relative_path"],
                candidate_source_relative_root=value[
                    "candidate_source_relative_root"
                ],
                bootstrap_relative_path=value["bootstrap_relative_path"],
                landlock_relative_path=value["landlock_relative_path"],
                verifier_relative_path=value["verifier_relative_path"],
                runtime_receipt_tool_relative_path=value[
                    "runtime_receipt_tool_relative_path"
                ],
                members=tuple(
                    ETTRClaimRuntimeMember(**member)
                    for member in value["members"]
                ),
            )
        except (
            KeyError,
            TypeError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise ETTRClaimRuntimeError(
                "claim runtime inventory is malformed"
            ) from exc
        inventory.validate()
        if payload != inventory.canonical_bytes():
            raise ETTRClaimRuntimeError(
                "claim runtime inventory is not canonical"
            )
        return inventory


@dataclass(frozen=True, slots=True)
class ETTRClaimRuntimeVerificationReceipt:
    schema: str
    archive_sha256: str
    archive_size: int
    inventory_sha256: str
    source_commit: str
    member_count: int
    python_sha256: str
    bootstrap_sha256: str
    landlock_sha256: str
    verifier_sha256: str

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(asdict(self))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def validate(self) -> None:
        if (
            self.schema != CLAIM_RUNTIME_RECEIPT_SCHEMA
            or any(
                _SHA256.fullmatch(value) is None
                for value in (
                    self.archive_sha256,
                    self.inventory_sha256,
                    self.python_sha256,
                    self.bootstrap_sha256,
                    self.landlock_sha256,
                    self.verifier_sha256,
                )
            )
            or _COMMIT.fullmatch(self.source_commit) is None
            or isinstance(self.archive_size, bool)
            or not isinstance(self.archive_size, int)
            or self.archive_size <= 0
            or isinstance(self.member_count, bool)
            or not isinstance(self.member_count, int)
            or self.member_count <= 0
        ):
            raise ETTRClaimRuntimeError(
                "claim runtime verification receipt differs"
            )


def _iter_tree(root: Path) -> Iterator[tuple[str, Path]]:
    pending = [root]
    rows: list[tuple[str, Path]] = []
    while pending:
        directory = pending.pop()
        try:
            children = tuple(directory.iterdir())
        except OSError as exc:
            raise ETTRClaimRuntimeError(
                f"runtime directory cannot be listed: {directory}"
            ) from exc
        for child in children:
            relative = child.relative_to(root).as_posix()
            _validate_relative_path(relative, label="runtime tree path")
            rows.append((relative, child))
            metadata = child.lstat()
            if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(
                metadata.st_mode
            ):
                pending.append(child)
    yield from sorted(rows, key=lambda row: row[0])


def build_inventory(
    runtime_root: Path,
    *,
    source_commit: str,
) -> ETTRClaimRuntimeInventory:
    runtime_root = runtime_root.resolve(strict=True)
    if (
        _COMMIT.fullmatch(source_commit) is None
        or not runtime_root.is_dir()
        or runtime_root.is_symlink()
    ):
        raise ETTRClaimRuntimeError("claim runtime build root differs")
    members: list[ETTRClaimRuntimeMember] = []
    for relative, path in _iter_tree(runtime_root):
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        if stat.S_ISDIR(metadata.st_mode):
            member = ETTRClaimRuntimeMember(
                path=relative,
                kind="directory",
                mode=mode,
                size=0,
                sha256=None,
                link_target=None,
            )
        elif stat.S_ISREG(metadata.st_mode):
            digest, size = _sha256_file(path)
            member = ETTRClaimRuntimeMember(
                path=relative,
                kind="file",
                mode=mode,
                size=size,
                sha256=digest,
                link_target=None,
            )
        elif stat.S_ISLNK(metadata.st_mode):
            target = _safe_symlink_target(
                PurePosixPath(relative),
                os.readlink(path),
            )
            member = ETTRClaimRuntimeMember(
                path=relative,
                kind="symlink",
                mode=0o777,
                size=0,
                sha256=None,
                link_target=target,
            )
        else:
            raise ETTRClaimRuntimeError(
                f"runtime tree contains unsupported object: {relative}"
            )
        member.validate()
        members.append(member)
    inventory = ETTRClaimRuntimeInventory(
        schema=CLAIM_RUNTIME_SCHEMA,
        source_commit=source_commit,
        python_relative_path=PYTHON_RELATIVE_PATH,
        candidate_source_relative_root=CANDIDATE_SOURCE_RELATIVE_ROOT,
        bootstrap_relative_path=BOOTSTRAP_RELATIVE_PATH,
        landlock_relative_path=LANDLOCK_RELATIVE_PATH,
        verifier_relative_path=VERIFIER_RELATIVE_PATH,
        runtime_receipt_tool_relative_path=(
            RUNTIME_RECEIPT_TOOL_RELATIVE_PATH
        ),
        members=tuple(members),
    )
    inventory.validate()
    return inventory


def validate_runtime_tree(
    runtime_root: Path,
    inventory: ETTRClaimRuntimeInventory,
) -> None:
    inventory.validate()
    observed = build_inventory(
        runtime_root,
        source_commit=inventory.source_commit,
    )
    if observed != inventory:
        raise ETTRClaimRuntimeError("extracted runtime tree differs")


def _tar_info(
    *,
    name: str,
    mode: int,
    kind: bytes,
    size: int = 0,
    linkname: str = "",
) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.mode = mode
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.type = kind
    info.size = size
    info.linkname = linkname
    info.pax_headers = {}
    return info


def build_archive(
    runtime_root: Path,
    archive_path: Path,
    *,
    source_commit: str,
) -> ETTRClaimRuntimeInventory:
    inventory = build_inventory(
        runtime_root,
        source_commit=source_commit,
    )
    if archive_path.exists() or archive_path.is_symlink():
        raise ETTRClaimRuntimeError("claim runtime archive already exists")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(archive_path, flags, 0o600)
    try:
        with (
            os.fdopen(descriptor, "wb", closefd=False) as raw,
            tarfile.open(
                fileobj=raw,
                mode="w",
                format=tarfile.PAX_FORMAT,
            ) as archive,
        ):
            payload = inventory.canonical_bytes()
            archive.addfile(
                _tar_info(
                    name=INVENTORY_NAME,
                    mode=0o444,
                    kind=tarfile.REGTYPE,
                    size=len(payload),
                ),
                io.BytesIO(payload),
            )
            archive.addfile(
                _tar_info(
                    name=RUNTIME_PREFIX,
                    mode=0o555,
                    kind=tarfile.DIRTYPE,
                )
            )
            for member in inventory.members:
                name = f"{RUNTIME_PREFIX}/{member.path}"
                if member.kind == "directory":
                    archive.addfile(
                        _tar_info(
                            name=name,
                            mode=member.mode,
                            kind=tarfile.DIRTYPE,
                        )
                    )
                elif member.kind == "symlink":
                    archive.addfile(
                        _tar_info(
                            name=name,
                            mode=member.mode,
                            kind=tarfile.SYMTYPE,
                            linkname=member.link_target or "",
                        )
                    )
                else:
                    source = runtime_root / member.path
                    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                    if hasattr(os, "O_NOFOLLOW"):
                        flags |= os.O_NOFOLLOW
                    source_descriptor = os.open(source, flags)
                    try:
                        metadata = os.fstat(source_descriptor)
                        if (
                            not stat.S_ISREG(metadata.st_mode)
                            or stat.S_IMODE(metadata.st_mode) != member.mode
                            or metadata.st_size != member.size
                        ):
                            raise ETTRClaimRuntimeError(
                                f"runtime source changed before archiving: {member.path}"
                            )
                        with os.fdopen(
                            source_descriptor,
                            "rb",
                            closefd=False,
                        ) as source_handle:
                            archive.addfile(
                                _tar_info(
                                    name=name,
                                    mode=member.mode,
                                    kind=tarfile.REGTYPE,
                                    size=member.size,
                                ),
                                source_handle,
                            )
                    finally:
                        os.close(source_descriptor)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        archive_path.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    archive_path.chmod(0o444)
    validate_archive(archive_path, expected_inventory=inventory)
    return inventory


def _archive_member_name(member: tarfile.TarInfo) -> PurePosixPath:
    path = _validate_relative_path(member.name, label="archive member path")
    if path.parts[0] not in {INVENTORY_NAME, RUNTIME_PREFIX}:
        raise ETTRClaimRuntimeError("archive member is outside runtime layout")
    return path


def validate_archive(
    archive_path: Path,
    *,
    expected_inventory: ETTRClaimRuntimeInventory | None = None,
) -> ETTRClaimRuntimeInventory:
    try:
        archive_metadata = archive_path.lstat()
    except OSError as exc:
        raise ETTRClaimRuntimeError("claim runtime archive is unavailable") from exc
    if (
        not stat.S_ISREG(archive_metadata.st_mode)
        or archive_path.is_symlink()
        or archive_metadata.st_nlink != 1
        or archive_metadata.st_mode & 0o222
    ):
        raise ETTRClaimRuntimeError(
            "claim runtime archive is not immutable single-link"
        )
    seen: set[str] = set()
    inventory_payload: bytes | None = None
    observed: dict[str, ETTRClaimRuntimeMember] = {}
    try:
        archive = tarfile.open(archive_path, mode="r:")
    except (OSError, tarfile.TarError) as exc:
        raise ETTRClaimRuntimeError("claim runtime archive is malformed") from exc
    with archive:
        for member in archive:
            path = _archive_member_name(member)
            name = path.as_posix()
            if name in seen:
                raise ETTRClaimRuntimeError(
                    "claim runtime archive has duplicate members"
                )
            seen.add(name)
            if name == INVENTORY_NAME:
                if not member.isfile() or member.mode != 0o444:
                    raise ETTRClaimRuntimeError(
                        "claim runtime inventory member differs"
                    )
                handle = archive.extractfile(member)
                if handle is None:
                    raise ETTRClaimRuntimeError(
                        "claim runtime inventory cannot be read"
                    )
                inventory_payload = handle.read()
                continue
            if name == RUNTIME_PREFIX:
                if not member.isdir() or member.mode != 0o555:
                    raise ETTRClaimRuntimeError(
                        "claim runtime root archive member differs"
                    )
                continue
            relative = PurePosixPath(*path.parts[1:]).as_posix()
            if member.isdir():
                row = ETTRClaimRuntimeMember(
                    path=relative,
                    kind="directory",
                    mode=member.mode,
                    size=0,
                    sha256=None,
                    link_target=None,
                )
            elif member.isfile() and not member.islnk():
                handle = archive.extractfile(member)
                if handle is None:
                    raise ETTRClaimRuntimeError(
                        "claim runtime file member cannot be read"
                    )
                digest, size = _sha256_stream(handle)
                row = ETTRClaimRuntimeMember(
                    path=relative,
                    kind="file",
                    mode=member.mode,
                    size=size,
                    sha256=digest,
                    link_target=None,
                )
            elif member.issym():
                row = ETTRClaimRuntimeMember(
                    path=relative,
                    kind="symlink",
                    mode=0o777,
                    size=0,
                    sha256=None,
                    link_target=_safe_symlink_target(
                        PurePosixPath(relative),
                        member.linkname,
                    ),
                )
            else:
                raise ETTRClaimRuntimeError(
                    "claim runtime archive contains an unsupported member"
                )
            row.validate()
            observed[row.path] = row
    if inventory_payload is None:
        raise ETTRClaimRuntimeError("claim runtime archive omits inventory")
    inventory = ETTRClaimRuntimeInventory.from_bytes(inventory_payload)
    if tuple(observed[path] for path in sorted(observed)) != inventory.members:
        raise ETTRClaimRuntimeError(
            "claim runtime archive members differ from inventory"
        )
    if expected_inventory is not None and inventory != expected_inventory:
        raise ETTRClaimRuntimeError(
            "claim runtime archive inventory differs from expected"
        )
    return inventory


def verification_receipt(
    archive_path: Path,
    runtime_root: Path,
    inventory: ETTRClaimRuntimeInventory,
) -> ETTRClaimRuntimeVerificationReceipt:
    validate_runtime_tree(runtime_root, inventory)
    archive_sha256, archive_size = _sha256_file(archive_path)
    members = {member.path: member for member in inventory.members}

    def digest(path: str) -> str:
        value = members[path].sha256
        if value is None:
            raise ETTRClaimRuntimeError(
                f"claim runtime required file is not regular: {path}"
            )
        return value

    receipt = ETTRClaimRuntimeVerificationReceipt(
        schema=CLAIM_RUNTIME_RECEIPT_SCHEMA,
        archive_sha256=archive_sha256,
        archive_size=archive_size,
        inventory_sha256=inventory.sha256(),
        source_commit=inventory.source_commit,
        member_count=len(inventory.members),
        python_sha256=digest(PYTHON_RELATIVE_PATH),
        bootstrap_sha256=digest(BOOTSTRAP_RELATIVE_PATH),
        landlock_sha256=digest(LANDLOCK_RELATIVE_PATH),
        verifier_sha256=digest(VERIFIER_RELATIVE_PATH),
    )
    receipt.validate()
    return receipt


def _write_once(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise ETTRClaimRuntimeError(
                    "claim runtime output write made no progress"
                )
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    path.chmod(mode)


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--runtime-root", type=Path, required=True)
    build.add_argument("--source-commit", required=True)
    build.add_argument("--archive", type=Path, required=True)
    verify_archive_parser = subparsers.add_parser("verify-archive")
    verify_archive_parser.add_argument("--archive", type=Path, required=True)
    verify_tree_parser = subparsers.add_parser("verify-tree")
    verify_tree_parser.add_argument("--archive", type=Path, required=True)
    verify_tree_parser.add_argument("--runtime-root", type=Path, required=True)
    verify_tree_parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "build":
        inventory = build_archive(
            arguments.runtime_root,
            arguments.archive,
            source_commit=arguments.source_commit,
        )
        digest, size = _sha256_file(arguments.archive)
        _write_once(
            Path(f"{arguments.archive}.sha256"),
            f"{digest}  {arguments.archive.name}\n".encode("ascii"),
        )
        _write_once(
            Path(f"{arguments.archive}.inventory.json"),
            inventory.canonical_bytes(),
        )
        print(
            _canonical_json_bytes(
                {
                    "archive_sha256": digest,
                    "archive_size": size,
                    "inventory_sha256": inventory.sha256(),
                    "member_count": len(inventory.members),
                    "source_commit": inventory.source_commit,
                }
            ).decode("ascii"),
            end="",
        )
        return 0
    inventory = validate_archive(arguments.archive)
    if arguments.command == "verify-archive":
        print(inventory.sha256())
        return 0
    receipt = verification_receipt(
        arguments.archive,
        arguments.runtime_root,
        inventory,
    )
    _write_once(arguments.receipt, receipt.canonical_bytes())
    print(receipt.sha256())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "BOOTSTRAP_RELATIVE_PATH",
    "CANDIDATE_SOURCE_PATHS",
    "CANDIDATE_SOURCE_FILES",
    "CANDIDATE_SOURCE_RELATIVE_ROOT",
    "CLAIM_RUNTIME_RECEIPT_SCHEMA",
    "CLAIM_RUNTIME_SCHEMA",
    "DEPLOYMENT_CONTRACT_RELATIVE_PATH",
    "ETTRClaimRuntimeError",
    "ETTRClaimRuntimeInventory",
    "ETTRClaimRuntimeMember",
    "ETTRClaimRuntimeVerificationReceipt",
    "INVENTORY_NAME",
    "LANDLOCK_RELATIVE_PATH",
    "COMMON_CANDIDATE_SOURCE_FILES",
    "PYTHON_RELATIVE_PATH",
    "STAGE_RUNNERS",
    "TOOL_FILES",
    "TOOLS_RELATIVE_ROOT",
    "VERIFIER_RELATIVE_PATH",
    "build_archive",
    "build_inventory",
    "validate_archive",
    "validate_runtime_tree",
    "verification_receipt",
]
