"""Freeze the reviewed ETTR-IL-v3 protocol and source inventory.

The output is a no-replace canonical JSON receipt. Generation jobs must bind
this receipt and independently verify every listed source before emitting a
candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Sequence

from ettr_il_v3_protocol import canonical_json_bytes, protocol_receipt


SCHEMA = "r12-ettr-il-v3-protocol-freeze-v1"


class FreezeError(ValueError):
    """The protocol source inventory cannot be frozen."""


def sha256_file(path: Path) -> tuple[int, str]:
    if path.is_symlink():
        raise FreezeError(f"source is a symlink: {path}")
    status = path.stat()
    if not stat.S_ISREG(status.st_mode):
        raise FreezeError(f"source is not regular: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return status.st_size, digest.hexdigest()


def build_freeze(
    root: Path,
    relative_sources: Sequence[str],
    *,
    source_commit: str,
) -> dict[str, object]:
    resolved_root = root.resolve(strict=True)
    if (
        len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise FreezeError("source commit must be lowercase 40-hex")
    ordered = tuple(sorted(relative_sources))
    if not ordered or len(set(ordered)) != len(ordered):
        raise FreezeError("source inventory is empty or duplicated")
    inventory: list[dict[str, object]] = []
    for relative in ordered:
        candidate = Path(relative)
        if (
            candidate.is_absolute()
            or not candidate.parts
            or ".." in candidate.parts
            or str(candidate).startswith(".git/")
        ):
            raise FreezeError(f"source path is unsafe: {relative}")
        unresolved = resolved_root / candidate
        if unresolved.is_symlink():
            raise FreezeError(f"source is a symlink: {relative}")
        path = unresolved.resolve(strict=True)
        try:
            path.relative_to(resolved_root)
        except ValueError as error:
            raise FreezeError(f"source leaves root: {relative}") from error
        size, digest = sha256_file(path)
        inventory.append(
            {
                "path": candidate.as_posix(),
                "bytes": size,
                "sha256": digest,
            }
        )
    inventory_sha256 = hashlib.sha256(
        canonical_json_bytes(inventory)
    ).hexdigest()
    receipt: dict[str, object] = {
        "schema": SCHEMA,
        "source_commit": source_commit,
        "protocol_receipt": protocol_receipt(),
        "source_count": len(inventory),
        "source_inventory": inventory,
        "source_inventory_sha256": inventory_sha256,
    }
    receipt["freeze_sha256"] = hashlib.sha256(
        canonical_json_bytes(receipt)
    ).hexdigest()
    return receipt


def load_and_verify_freeze(
    root: Path,
    receipt_path: Path,
    *,
    source_commit: str,
) -> dict[str, object]:
    """Reload one canonical receipt and recompute its complete source tree."""

    if receipt_path.is_symlink():
        raise FreezeError("freeze receipt is a symlink")
    status = receipt_path.stat()
    if not stat.S_ISREG(status.st_mode) or status.st_size > 16 * 1024 * 1024:
        raise FreezeError("freeze receipt is not a bounded regular file")
    payload = receipt_path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FreezeError("freeze receipt is not canonical JSON") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise FreezeError("freeze receipt is not canonical JSON")
    expected_keys = {
        "freeze_sha256",
        "protocol_receipt",
        "schema",
        "source_commit",
        "source_count",
        "source_inventory",
        "source_inventory_sha256",
    }
    if set(value) != expected_keys or value.get("schema") != SCHEMA:
        raise FreezeError("freeze receipt schema differs")
    if value.get("source_commit") != source_commit:
        raise FreezeError("freeze receipt source commit differs")
    inventory = value.get("source_inventory")
    if (
        not isinstance(inventory, list)
        or not inventory
        or value.get("source_count") != len(inventory)
    ):
        raise FreezeError("freeze receipt source inventory differs")
    sources: list[str] = []
    for entry in inventory:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"bytes", "path", "sha256"}
            or not isinstance(entry.get("path"), str)
        ):
            raise FreezeError("freeze receipt source entry differs")
        sources.append(entry["path"])
    rebuilt = build_freeze(
        root,
        sources,
        source_commit=source_commit,
    )
    if rebuilt != value:
        raise FreezeError("freeze receipt no longer matches source tree")
    return rebuilt


def write_no_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = build_freeze(
        args.root,
        args.source,
        source_commit=args.source_commit,
    )
    write_no_replace(args.out, canonical_json_bytes(receipt))
    print(
        json.dumps(
            {
                "freeze_sha256": receipt["freeze_sha256"],
                "out": str(args.out),
                "source_count": receipt["source_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
