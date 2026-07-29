#!/usr/bin/env python3
"""Validate a pinned Hugging Face file-selection manifest.

The validator performs no network access. It fails closed on path traversal,
duplicate paths, malformed hashes, byte-ledger drift, or source identity
drift. TSV output is intended for a shell downloader that independently
verifies every completed file against its published LFS SHA-256.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re


SCHEMA = "shohin-pinned-hf-file-selection-v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SAFE_PATH_PATTERN = re.compile(r"^[A-Za-z0-9._/-]+$")


class PinnedManifestError(ValueError):
    """The pinned source-file manifest is malformed or inconsistent."""


def validate_manifest(
    payload: object,
    *,
    expected_dataset: str,
    expected_revision: str,
) -> tuple[dict[str, object], ...]:
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise PinnedManifestError("manifest schema differs")
    if payload.get("dataset") != expected_dataset:
        raise PinnedManifestError("manifest dataset differs")
    if payload.get("revision") != expected_revision:
        raise PinnedManifestError("manifest revision differs")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise PinnedManifestError("manifest files differ")

    validated = []
    paths: set[str] = set()
    basenames: set[str] = set()
    total_bytes = 0
    for record in files:
        if not isinstance(record, dict):
            raise PinnedManifestError("manifest file record differs")
        path = record.get("path")
        size = record.get("size")
        digest = record.get("sha256")
        if not isinstance(path, str):
            raise PinnedManifestError("manifest file path differs")
        parsed = PurePosixPath(path)
        if (
            parsed.is_absolute()
            or not parsed.parts
            or any(part in {"", ".", ".."} for part in parsed.parts)
            or str(parsed) != path
            or SAFE_PATH_PATTERN.fullmatch(path) is None
        ):
            raise PinnedManifestError("manifest file path is unsafe")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise PinnedManifestError("manifest file size differs")
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise PinnedManifestError("manifest file SHA-256 differs")
        if path in paths or parsed.name in basenames:
            raise PinnedManifestError("manifest file path is duplicated")
        paths.add(path)
        basenames.add(parsed.name)
        total_bytes += size
        validated.append(
            {
                "path": path,
                "size": size,
                "sha256": digest,
            }
        )

    if payload.get("selected_bytes") != total_bytes:
        raise PinnedManifestError("manifest selected-byte ledger differs")
    selection = payload.get("selection")
    if (
        not isinstance(selection, dict)
        or selection.get("selected_count") != len(validated)
    ):
        raise PinnedManifestError("manifest selected-count ledger differs")
    upstream_count = payload.get("upstream_file_count_considered")
    upstream_bytes = payload.get("upstream_bytes_considered")
    if (
        not isinstance(upstream_count, int)
        or upstream_count < len(validated)
        or not isinstance(upstream_bytes, int)
        or upstream_bytes < total_bytes
    ):
        raise PinnedManifestError("manifest upstream ledger differs")
    return tuple(validated)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--emit-tsv", action="store_true")
    arguments = parser.parse_args()

    payload = json.loads(arguments.manifest.read_text())
    records = validate_manifest(
        payload,
        expected_dataset=arguments.dataset,
        expected_revision=arguments.revision,
    )
    if arguments.emit_tsv:
        for record in records:
            print(record["path"], record["size"], record["sha256"], sep="\t")
    else:
        print(
            json.dumps(
                {
                    "dataset": arguments.dataset,
                    "files": len(records),
                    "revision": arguments.revision,
                    "selected_bytes": sum(record["size"] for record in records),
                    "status": "valid",
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
