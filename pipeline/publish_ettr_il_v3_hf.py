#!/usr/bin/env python3
"""Publish a frozen ETTR-IL-v3 corpus to a Hugging Face dataset repository.

The publisher is intentionally narrow:

* credentials are read only from ``HF_TOKEN`` for non-dry-run publication;
* the local root must contain exactly the manifest, card, and declared shards;
* all local files must be regular, single-link files beneath the dataset root;
* the card and every shard are verified against the manifest before any upload;
* shards and release metadata use content-addressed remote paths;
* an existing remote path is skipped only when its SHA-256 can be proven equal;
* changing content requires a separately named revision; paths are never replaced.

This module performs no model, training, or checkpoint operations.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
from typing import Any, BinaryIO, Iterator, Mapping, Protocol, Sequence


DEFAULT_REPO_ID = "Godlydonuts/shohin-ettr-il-v3"
MANIFEST_SCHEMA = "r12-ettr-il-v3-hf-publication-manifest-v1"
RECEIPT_SCHEMA = "r12-ettr-il-v3-hf-publication-receipt-v1"
DATASET_PROTOCOL = "R12-ETTR-IL-v3-initializer"
TOKEN_ENVIRONMENT_VARIABLE = "HF_TOKEN"
REPO_TYPE = "dataset"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REPO_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
_SAFE_SHARD_SUFFIXES = (
    ".arrow",
    ".jsonl",
    ".jsonl.gz",
    ".jsonl.zst",
    ".parquet",
)
_SAFE_SPLITS = frozenset(
    {
        "train",
        "development",
        "train_reserve",
        "development_reserve",
    }
)
_READ_CHUNK_BYTES = 8 * 1024 * 1024


class PublicationError(RuntimeError):
    """Raised when publication cannot be proven safe and immutable."""


class HubGateway(Protocol):
    """Minimal remote interface used by the fail-closed publisher."""

    def ensure_dataset_repo(self, repo_id: str, *, private: bool) -> None:
        """Create or validate the dataset repository and its visibility."""

    def revision_exists(self, repo_id: str, revision: str) -> bool:
        """Return whether the named branch exists."""

    def create_revision(self, repo_id: str, revision: str) -> None:
        """Create a new branch for an immutable publication."""

    def list_files(self, repo_id: str, revision: str) -> set[str]:
        """Return every file path in the named revision."""

    def remote_sha256(
        self,
        repo_id: str,
        revision: str,
        remote_path: str,
    ) -> str | None:
        """Return a verified remote SHA-256, or None when unavailable."""

    def upload_file(
        self,
        repo_id: str,
        revision: str,
        remote_path: str,
        fileobj: BinaryIO,
    ) -> None:
        """Upload one file without exposing credentials to the caller."""


@dataclass(frozen=True)
class VerifiedFile:
    role: str
    relative_path: str
    local_path: Path
    sha256: str
    size_bytes: int
    remote_path: str


@dataclass(frozen=True)
class PublicationPlan:
    dataset_root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    release_prefix: str
    files: tuple[VerifiedFile, ...]


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PublicationError(f"duplicate JSON key rejected: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise PublicationError(f"non-finite JSON value rejected: {value}")


def _load_json_bytes(payload: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except PublicationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicationError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise PublicationError(f"{label} must contain one JSON object")
    return value


def _validate_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PublicationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_size(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PublicationError(f"{label} must be a non-negative integer")
    return value


def _validate_relative_path(
    value: object,
    *,
    label: str,
    required_prefix: str | None = None,
) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise PublicationError(f"{label} must be a non-empty POSIX path")
    if "\x00" in value or value != value.strip():
        raise PublicationError(f"{label} contains unsafe characters")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value:
        raise PublicationError(f"{label} must be a normalized relative path")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise PublicationError(f"{label} contains an unsafe path component")
    if any(part.startswith(".") for part in path.parts):
        raise PublicationError(f"{label} may not contain hidden components")
    if required_prefix is not None and (
        len(path.parts) < 2 or path.parts[0] != required_prefix
    ):
        raise PublicationError(f"{label} must be beneath {required_prefix}/")
    return value


def _validate_repo_id(repo_id: str) -> str:
    parts = repo_id.split("/")
    if (
        len(parts) != 2
        or any(_REPO_COMPONENT_RE.fullmatch(part) is None for part in parts)
    ):
        raise PublicationError("repo id must have the form owner/dataset")
    return repo_id


def _validate_revision(revision: str) -> str:
    if (
        _REVISION_RE.fullmatch(revision) is None
        or ".." in revision
        or "//" in revision
        or revision.endswith(("/", "."))
        or revision.lower() in {"main", "master"}
    ):
        raise PublicationError(
            "revision must be an explicit non-default, normalized branch name"
        )
    return revision


def _assert_root_directory(root: Path) -> tuple[int, int]:
    try:
        root_stat = root.lstat()
    except FileNotFoundError as exc:
        raise PublicationError(f"dataset root is missing: {root}") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise PublicationError("dataset root must be a real directory, not a symlink")
    return root_stat.st_dev, root_stat.st_ino


def _relative_to_root(root: Path, path: Path, *, label: str) -> str:
    absolute = path if path.is_absolute() else root / path
    try:
        relative = absolute.absolute().relative_to(root.absolute())
    except ValueError as exc:
        raise PublicationError(f"{label} must be beneath the dataset root") from exc
    return _validate_relative_path(relative.as_posix(), label=label)


def _inventory_root(root: Path) -> set[str]:
    """Inventory without following symlinks or accepting special files."""

    inventory: set[str] = set()
    stack: list[tuple[Path, PurePosixPath]] = [(root, PurePosixPath("."))]
    while stack:
        directory, relative_directory = stack.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise PublicationError("unable to inventory dataset root") from exc
        for entry in entries:
            relative = (
                PurePosixPath(entry.name)
                if str(relative_directory) == "."
                else relative_directory / entry.name
            )
            relative_value = _validate_relative_path(
                relative.as_posix(),
                label="dataset inventory path",
            )
            try:
                if entry.is_symlink():
                    raise PublicationError(
                        f"symlink rejected in dataset root: {relative_value}"
                    )
                if entry.is_dir(follow_symlinks=False):
                    stack.append((Path(entry.path), relative))
                elif entry.is_file(follow_symlinks=False):
                    inventory.add(relative_value)
                else:
                    raise PublicationError(
                        f"special filesystem entry rejected: {relative_value}"
                    )
            except OSError as exc:
                raise PublicationError(
                    f"unable to inspect dataset path: {relative_value}"
                ) from exc
    return inventory


@contextmanager
def _open_rooted_stable_file(
    root: Path,
    relative_path: str,
) -> Iterator[BinaryIO]:
    """Open a root-relative file without following any path-component symlink."""

    relative_path = _validate_relative_path(
        relative_path,
        label="rooted file path",
    )
    display_path = root / PurePosixPath(relative_path)
    directory_flags = os.O_RDONLY
    directory_flags |= getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    file_flags = os.O_RDONLY
    file_flags |= getattr(os, "O_CLOEXEC", 0)
    file_flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = []
    fileobj: BinaryIO | None = None
    try:
        root_identity = _assert_root_directory(root)
        root_descriptor = os.open(root, directory_flags)
        descriptors.append(root_descriptor)
        if (
            os.fstat(root_descriptor).st_dev,
            os.fstat(root_descriptor).st_ino,
        ) != root_identity:
            raise PublicationError("dataset root changed during secure open")

        parts = PurePosixPath(relative_path).parts
        parent_descriptor = root_descriptor
        for component in parts[:-1]:
            child_descriptor = os.open(
                component,
                directory_flags,
                dir_fd=parent_descriptor,
            )
            descriptors.append(child_descriptor)
            parent_descriptor = child_descriptor

        try:
            before = os.stat(
                parts[-1],
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError as exc:
            raise PublicationError(
                f"required file is missing: {display_path}"
            ) from exc
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise PublicationError(
                f"required path is not a regular file: {display_path}"
            )
        if before.st_nlink != 1:
            raise PublicationError(f"hard-linked file rejected: {display_path}")
        descriptor = os.open(
            parts[-1],
            file_flags,
            dir_fd=parent_descriptor,
        )
        descriptors.append(descriptor)
        opened = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != identity
        ):
            raise PublicationError(
                f"file identity changed during open: {display_path}"
            )
        fileobj = os.fdopen(descriptor, "rb", closefd=True)
        descriptors.pop()
        yield fileobj
        after = os.stat(
            parts[-1],
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            stat.S_ISLNK(after.st_mode)
            or (after.st_dev, after.st_ino) != identity
            or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
        ):
            raise PublicationError(
                f"file changed during publication: {display_path}"
            )
    except PublicationError:
        raise
    except OSError as exc:
        raise PublicationError(
            f"unable to open required file safely: {display_path}"
        ) from exc
    finally:
        if fileobj is not None:
            fileobj.close()
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _hash_fileobj(fileobj: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    while True:
        chunk = fileobj.read(_READ_CHUNK_BYTES)
        if not chunk:
            break
        digest.update(chunk)
        total += len(chunk)
    return digest.hexdigest(), total


def _verify_local_file(
    root: Path,
    relative_path: str,
    *,
    expected_sha256: str,
    expected_size: int,
) -> None:
    with _open_rooted_stable_file(root, relative_path) as fileobj:
        observed_sha256, observed_size = _hash_fileobj(fileobj)
    if observed_size != expected_size:
        raise PublicationError(f"local size mismatch for {relative_path}")
    if observed_sha256 != expected_sha256:
        raise PublicationError(f"local SHA-256 mismatch for {relative_path}")


def _read_small_verified_file(
    root: Path,
    relative_path: str,
    *,
    maximum_bytes: int,
) -> tuple[bytes, str]:
    with _open_rooted_stable_file(root, relative_path) as fileobj:
        payload = fileobj.read(maximum_bytes + 1)
        if len(payload) > maximum_bytes:
            raise PublicationError(
                f"metadata file exceeds safe size limit: {relative_path}"
            )
    return payload, hashlib.sha256(payload).hexdigest()


def _content_addressed_shard_path(sha256: str, local_name: str) -> str:
    return f"shards/sha256/{sha256[:2]}/{sha256}/{local_name}"


def build_publication_plan(
    dataset_root: Path,
    manifest_path: Path,
) -> PublicationPlan:
    """Validate the complete local publication and return a frozen upload plan."""

    root = dataset_root.absolute()
    root_identity = _assert_root_directory(root)
    manifest_relative = _relative_to_root(root, manifest_path, label="manifest path")
    manifest_local = root / PurePosixPath(manifest_relative)
    manifest_bytes, manifest_sha256 = _read_small_verified_file(
        root,
        manifest_relative,
        maximum_bytes=16 * 1024 * 1024,
    )
    manifest = _load_json_bytes(manifest_bytes, label="publication manifest")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise PublicationError(f"manifest schema must be {MANIFEST_SCHEMA!r}")
    if manifest.get("dataset_protocol") != DATASET_PROTOCOL:
        raise PublicationError(f"dataset protocol must be {DATASET_PROTOCOL!r}")
    expected_manifest_keys = {"card", "dataset_protocol", "schema", "shards"}
    if set(manifest) != expected_manifest_keys:
        raise PublicationError(
            "publication manifest contains unexpected or missing top-level keys"
        )

    card = manifest.get("card")
    shards = manifest.get("shards")
    if not isinstance(card, dict):
        raise PublicationError("manifest card must be an object")
    if not isinstance(shards, list) or not shards:
        raise PublicationError("manifest shards must be a non-empty list")
    if set(card) != {"path", "sha256", "size_bytes"}:
        raise PublicationError("manifest card contains unexpected or missing keys")

    card_path = _validate_relative_path(
        card.get("path"),
        label="card path",
    )
    if card_path != "README.md":
        raise PublicationError("dataset card path must be README.md")
    card_sha256 = _validate_sha256(
        card.get("sha256"),
        label="card SHA-256",
    )
    card_size = _validate_size(card.get("size_bytes"), label="card size")

    release_prefix = f"releases/sha256/{manifest_sha256}"
    files: list[VerifiedFile] = [
        VerifiedFile(
            role="manifest",
            relative_path=manifest_relative,
            local_path=manifest_local,
            sha256=manifest_sha256,
            size_bytes=len(manifest_bytes),
            remote_path=f"{release_prefix}/manifest.json",
        ),
        VerifiedFile(
            role="card",
            relative_path=card_path,
            local_path=root / PurePosixPath(card_path),
            sha256=card_sha256,
            size_bytes=card_size,
            remote_path=f"{release_prefix}/README.md",
        ),
    ]

    relative_paths = {manifest_relative, card_path}
    casefold_paths = {manifest_relative.casefold(), card_path.casefold()}
    remote_paths = {files[0].remote_path, files[1].remote_path}
    for index, entry in enumerate(shards):
        if not isinstance(entry, dict):
            raise PublicationError(f"shard entry {index} must be an object")
        if set(entry) != {"path", "sha256", "size_bytes", "split"}:
            raise PublicationError(
                f"shard entry {index} contains unexpected or missing keys"
            )
        path_value = _validate_relative_path(
            entry.get("path"),
            label=f"shard {index} path",
            required_prefix="shards",
        )
        pure_path = PurePosixPath(path_value)
        if len(pure_path.parts) < 3:
            raise PublicationError(
                f"shard {index} path must be shards/<split>/<file>"
            )
        split = entry.get("split")
        if (
            not isinstance(split, str)
            or split not in _SAFE_SPLITS
            or pure_path.parts[1] != split
        ):
            raise PublicationError(f"shard {index} has an invalid split path")
        if not path_value.lower().endswith(_SAFE_SHARD_SUFFIXES):
            raise PublicationError(f"shard {index} has an unsupported file suffix")
        shard_sha256 = _validate_sha256(
            entry.get("sha256"),
            label=f"shard {index} SHA-256",
        )
        shard_size = _validate_size(
            entry.get("size_bytes"),
            label=f"shard {index} size",
        )
        if path_value in relative_paths or path_value.casefold() in casefold_paths:
            raise PublicationError(f"duplicate shard path rejected: {path_value}")
        relative_paths.add(path_value)
        casefold_paths.add(path_value.casefold())
        remote_path = _content_addressed_shard_path(
            shard_sha256,
            pure_path.name,
        )
        if remote_path in remote_paths:
            raise PublicationError(
                f"duplicate content-addressed shard path rejected: {remote_path}"
            )
        remote_paths.add(remote_path)
        files.append(
            VerifiedFile(
                role="shard",
                relative_path=path_value,
                local_path=root / pure_path,
                sha256=shard_sha256,
                size_bytes=shard_size,
                remote_path=remote_path,
            )
        )

    inventory = _inventory_root(root)
    missing = sorted(relative_paths - inventory)
    unexpected = sorted(inventory - relative_paths)
    if missing:
        raise PublicationError(f"manifest-declared files are missing: {missing}")
    if unexpected:
        raise PublicationError(f"unexpected files in dataset root: {unexpected}")
    if _assert_root_directory(root) != root_identity:
        raise PublicationError("dataset root identity changed during validation")

    for file in files:
        _verify_local_file(
            root,
            file.relative_path,
            expected_sha256=file.sha256,
            expected_size=file.size_bytes,
        )

    return PublicationPlan(
        dataset_root=root,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        release_prefix=release_prefix,
        files=tuple(files),
    )


def _safe_remote_digest(value: str | None) -> str | None:
    if value is None:
        return None
    return _validate_sha256(value, label="remote SHA-256")


def _upload_verified_file(
    gateway: HubGateway,
    *,
    dataset_root: Path,
    repo_id: str,
    revision: str,
    file: VerifiedFile,
) -> None:
    with _open_rooted_stable_file(dataset_root, file.relative_path) as fileobj:
        observed_sha256, observed_size = _hash_fileobj(fileobj)
        if (
            observed_sha256 != file.sha256
            or observed_size != file.size_bytes
        ):
            raise PublicationError(
                f"local file changed after planning: {file.relative_path}"
            )
        fileobj.seek(0)
        gateway.upload_file(
            repo_id,
            revision,
            file.remote_path,
            fileobj,
        )


def _write_receipt_no_replace(receipt_path: Path, receipt: Mapping[str, Any]) -> None:
    path = receipt_path.absolute()
    parent = path.parent
    try:
        parent_stat = parent.lstat()
    except FileNotFoundError as exc:
        raise PublicationError("receipt parent directory must already exist") from exc
    if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
        raise PublicationError("receipt parent must be a real directory")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    payload = _canonical_json_bytes(receipt)
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError as exc:
        raise PublicationError("refusing to overwrite publication receipt") from exc
    except OSError as exc:
        raise PublicationError("unable to create publication receipt safely") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as fileobj:
            descriptor = -1
            fileobj.write(payload)
            fileobj.flush()
            os.fsync(fileobj.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def publish_plan(
    plan: PublicationPlan,
    *,
    repo_id: str,
    revision: str,
    receipt_path: Path,
    private: bool = True,
    dry_run: bool = False,
    new_revision: bool = False,
    gateway: HubGateway | None = None,
) -> Mapping[str, Any]:
    """Publish a validated plan and write an immutable, credential-free receipt."""

    repo_id = _validate_repo_id(repo_id)
    revision = _validate_revision(revision)
    try:
        receipt_path.absolute().relative_to(plan.dataset_root)
    except ValueError:
        pass
    else:
        raise PublicationError("publication receipt must be outside the dataset root")

    operations: list[dict[str, Any]] = []
    ordered_files = tuple(
        sorted(
            plan.files,
            key=lambda file: (
                {"shard": 0, "card": 1, "manifest": 2}[file.role],
                file.remote_path,
            ),
        )
    )
    if dry_run:
        operations = [
            {
                "action": "would_upload",
                "remote_path": file.remote_path,
                "role": file.role,
                "sha256": file.sha256,
                "size_bytes": file.size_bytes,
            }
            for file in ordered_files
        ]
    else:
        if gateway is None:
            raise PublicationError("a Hugging Face gateway is required for publication")
        gateway.ensure_dataset_repo(repo_id, private=private)
        exists = gateway.revision_exists(repo_id, revision)
        if new_revision:
            if exists:
                raise PublicationError(
                    "explicit new revision already exists; choose another revision"
                )
            gateway.create_revision(repo_id, revision)
        elif not exists:
            raise PublicationError(
                "revision does not exist; pass --new-revision to create it explicitly"
            )

        remote_files = gateway.list_files(repo_id, revision)
        expected_release_files = {
            file.remote_path
            for file in ordered_files
            if file.remote_path.startswith(plan.release_prefix + "/")
        }
        unexpected_release_files = sorted(
            path
            for path in remote_files
            if path.startswith(plan.release_prefix + "/")
            and path not in expected_release_files
        )
        if unexpected_release_files:
            raise PublicationError(
                "unexpected files already occupy the immutable release namespace"
            )

        remote_digests: dict[str, str | None] = {}
        for file in ordered_files:
            if file.remote_path not in remote_files:
                remote_digests[file.remote_path] = None
                continue
            remote_sha256 = _safe_remote_digest(
                gateway.remote_sha256(repo_id, revision, file.remote_path)
            )
            if remote_sha256 is None:
                raise PublicationError(
                    "remote file exists but its SHA-256 cannot be verified; "
                    "refusing mutable overwrite"
                )
            if remote_sha256 != file.sha256:
                raise PublicationError(
                    "remote path contains different content; publish under a "
                    "new explicit revision"
                )
            remote_digests[file.remote_path] = remote_sha256

        for file in ordered_files:
            if remote_digests[file.remote_path] == file.sha256:
                action = "skipped_matching_remote"
            else:
                _upload_verified_file(
                    gateway,
                    dataset_root=plan.dataset_root,
                    repo_id=repo_id,
                    revision=revision,
                    file=file,
                )
                remote_sha256 = _safe_remote_digest(
                    gateway.remote_sha256(
                        repo_id,
                        revision,
                        file.remote_path,
                    )
                )
                if remote_sha256 != file.sha256:
                    raise PublicationError(
                        "remote SHA-256 verification failed after upload"
                    )
                action = "uploaded_and_verified"
            operations.append(
                {
                    "action": action,
                    "remote_path": file.remote_path,
                    "role": file.role,
                    "sha256": file.sha256,
                    "size_bytes": file.size_bytes,
                }
            )

    receipt: Mapping[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "dataset_protocol": DATASET_PROTOCOL,
        "dry_run": dry_run,
        "manifest_sha256": plan.manifest_sha256,
        "new_revision_requested": new_revision,
        "operations": operations,
        "private": private,
        "repo_id": repo_id,
        "repo_type": REPO_TYPE,
        "revision": revision,
        "schema": RECEIPT_SCHEMA,
        "status": "dry_run_validated" if dry_run else "published_and_verified",
    }
    _write_receipt_no_replace(receipt_path, receipt)
    return receipt


class HuggingFaceHubGateway:
    """Credential-holding Hugging Face adapter with sanitized failures."""

    def __init__(self, token: str) -> None:
        if not token:
            raise PublicationError(f"{TOKEN_ENVIRONMENT_VARIABLE} is empty")
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError as exc:
            raise PublicationError(
                "huggingface_hub is required for non-dry-run publication"
            ) from exc
        self._token = token
        self._api = HfApi(token=token)
        self._download = hf_hub_download

    def ensure_dataset_repo(self, repo_id: str, *, private: bool) -> None:
        try:
            self._api.create_repo(
                repo_id=repo_id,
                repo_type=REPO_TYPE,
                private=private,
                exist_ok=True,
            )
            info = self._api.repo_info(repo_id=repo_id, repo_type=REPO_TYPE)
        except Exception:
            raise PublicationError("Hugging Face repository validation failed") from None
        observed_private = getattr(info, "private", None)
        if observed_private is not private:
            raise PublicationError(
                "existing repository visibility does not match the explicit request"
            )

    def revision_exists(self, repo_id: str, revision: str) -> bool:
        try:
            refs = self._api.list_repo_refs(repo_id=repo_id, repo_type=REPO_TYPE)
            branches = getattr(refs, "branches", ())
            return any(getattr(branch, "name", None) == revision for branch in branches)
        except Exception:
            raise PublicationError("Hugging Face revision lookup failed") from None

    def create_revision(self, repo_id: str, revision: str) -> None:
        try:
            self._api.create_branch(
                repo_id=repo_id,
                branch=revision,
                repo_type=REPO_TYPE,
                exist_ok=False,
            )
        except Exception:
            raise PublicationError("Hugging Face revision creation failed") from None

    def list_files(self, repo_id: str, revision: str) -> set[str]:
        try:
            return set(
                self._api.list_repo_files(
                    repo_id=repo_id,
                    repo_type=REPO_TYPE,
                    revision=revision,
                )
            )
        except Exception:
            raise PublicationError("Hugging Face file listing failed") from None

    def remote_sha256(
        self,
        repo_id: str,
        revision: str,
        remote_path: str,
    ) -> str | None:
        try:
            infos = self._api.get_paths_info(
                repo_id=repo_id,
                paths=[remote_path],
                repo_type=REPO_TYPE,
                revision=revision,
            )
            if not infos:
                return None
            lfs = getattr(infos[0], "lfs", None)
            if isinstance(lfs, dict):
                lfs_sha256 = lfs.get("sha256")
            else:
                lfs_sha256 = getattr(lfs, "sha256", None)
            if isinstance(lfs_sha256, str) and _SHA256_RE.fullmatch(lfs_sha256):
                return lfs_sha256
            with tempfile.TemporaryDirectory(prefix="ettr-hf-verify-") as directory:
                downloaded = self._download(
                    repo_id=repo_id,
                    filename=remote_path,
                    repo_type=REPO_TYPE,
                    revision=revision,
                    token=self._token,
                    local_dir=directory,
                )
                with open(downloaded, "rb") as fileobj:
                    return _hash_fileobj(fileobj)[0]
        except Exception:
            raise PublicationError(
                "Hugging Face remote SHA-256 verification failed"
            ) from None

    def upload_file(
        self,
        repo_id: str,
        revision: str,
        remote_path: str,
        fileobj: BinaryIO,
    ) -> None:
        try:
            self._api.upload_file(
                path_or_fileobj=fileobj,
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type=REPO_TYPE,
                revision=revision,
                commit_message=f"Publish immutable ETTR v3 object {remote_path}",
            )
        except Exception:
            raise PublicationError("Hugging Face upload failed") from None


def _token_from_environment(environment: Mapping[str, str] | None = None) -> str:
    values = os.environ if environment is None else environment
    token = values.get(TOKEN_ENVIRONMENT_VARIABLE, "")
    if not token:
        raise PublicationError(
            f"{TOKEN_ENVIRONMENT_VARIABLE} must be set for publication"
        )
    return token


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Securely publish a frozen ETTR-IL-v3 dataset release.",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("publication_manifest.json"),
        help="Manifest path beneath --dataset-root.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--new-revision",
        action="store_true",
        help="Create --revision; fails if that revision already exists.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--public",
        action="store_true",
        help="Explicitly request a public repository; private is the default.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    try:
        manifest_path = arguments.manifest
        if not manifest_path.is_absolute():
            manifest_path = arguments.dataset_root / manifest_path
        plan = build_publication_plan(arguments.dataset_root, manifest_path)
        gateway: HubGateway | None = None
        if not arguments.dry_run:
            gateway = HuggingFaceHubGateway(_token_from_environment())
        receipt = publish_plan(
            plan,
            repo_id=arguments.repo_id,
            revision=arguments.revision,
            receipt_path=arguments.receipt,
            private=not arguments.public,
            dry_run=arguments.dry_run,
            new_revision=arguments.new_revision,
            gateway=gateway,
        )
    except PublicationError as exc:
        print(f"publication refused: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "publication_receipt": str(arguments.receipt.absolute()),
                "status": receipt["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
