"""Independent verification of a sealed Q36 durable-evidence snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import stat
from typing import Any, Mapping

NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class Q36MTREvidenceVerificationError(RuntimeError):
    """The durable Q36 mirror no longer matches its immutable manifest."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_evidence_snapshot(
    manifest_path: Path, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Replay exact tree membership and every mirrored byte from the manifest."""

    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise Q36MTREvidenceVerificationError("Q36 evidence manifest is absent")
    root = manifest_path.parent.resolve(strict=True)
    artifact_root = root / "artifacts"
    if (
        manifest_path.resolve(strict=True) != root / "manifest.json"
        or root.is_symlink()
        or artifact_root.is_symlink()
        or not artifact_root.is_dir()
        or stat.S_IMODE(root.stat().st_mode) & 0o222
        or stat.S_IMODE(artifact_root.stat().st_mode) & 0o222
    ):
        raise Q36MTREvidenceVerificationError("Q36 evidence root geometry differs")
    records = payload.get("records")
    hashes = payload.get("artifact_sha256s")
    if (
        not isinstance(records, list)
        or not isinstance(hashes, dict)
        or payload.get("artifact_count") != len(records)
        or len(records) != len(hashes)
        or payload.get("primary_mirror_hashes_exact") is not True
        or payload.get("write_once_snapshot") is not True
    ):
        raise Q36MTREvidenceVerificationError("Q36 evidence manifest geometry differs")
    observed_names: set[str] = set()
    observed_files: set[Path] = set()
    rows: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "name",
            "primary",
            "mirror",
            "sha256",
            "bytes",
        }:
            raise Q36MTREvidenceVerificationError("Q36 evidence record differs")
        name = record.get("name")
        mirror = Path(str(record.get("mirror", "")))
        digest = record.get("sha256")
        byte_count = record.get("bytes")
        if (
            not isinstance(name, str)
            or not NAME_PATTERN.fullmatch(name)
            or name in observed_names
            or not mirror.is_absolute()
            or mirror.is_symlink()
            or not mirror.is_file()
            or mirror.resolve(strict=True).parent != artifact_root
            or not (mirror.name == name or mirror.name.startswith(f"{name}."))
            or not isinstance(digest, str)
            or len(digest) != 64
            or digest != digest.lower()
            or any(character not in "0123456789abcdef" for character in digest)
            or hashes.get(name) != digest
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
            or mirror.stat().st_size != byte_count
            or stat.S_IMODE(mirror.stat().st_mode) & 0o222
            or sha256_file(mirror) != digest
        ):
            raise Q36MTREvidenceVerificationError(
                f"Q36 durable evidence differs: {name!r}"
            )
        observed_names.add(name)
        observed_files.add(mirror.resolve(strict=True))
        rows.append({"name": name, "sha256": digest, "bytes": byte_count})
    actual_root = {path.resolve(strict=True) for path in root.iterdir()}
    if actual_root != {manifest_path.resolve(strict=True), artifact_root}:
        raise Q36MTREvidenceVerificationError("Q36 evidence root membership differs")
    actual_files = {path.resolve(strict=True) for path in artifact_root.iterdir()}
    if actual_files != observed_files or any(
        not stat.S_ISREG(path.stat().st_mode) for path in artifact_root.iterdir()
    ):
        raise Q36MTREvidenceVerificationError("Q36 artifact membership differs")
    encoded = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in sorted(rows, key=lambda value: value["name"])
    )
    return {
        "artifact_count": len(rows),
        "artifact_tree_sha256": hashlib.sha256(encoded).hexdigest(),
        "exact_membership": True,
        "all_hashes_verified": True,
        "all_members_nonwritable": True,
    }
