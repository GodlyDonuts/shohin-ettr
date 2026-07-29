#!/usr/bin/env python3
"""Build a deterministic manifest for pinned Hugging Face LFS files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
from typing import Iterable

from huggingface_hub import HfApi

from pipeline.verify_pinned_hf_manifest import (
    SCHEMA,
    SHA256_PATTERN,
    PinnedManifestError,
    validate_manifest,
)


REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _safe_prefix(prefix: str) -> str:
    parsed = PurePosixPath(prefix)
    if (
        parsed.is_absolute()
        or not parsed.parts
        or any(part in {"", ".", ".."} for part in parsed.parts)
        or str(parsed) != prefix
    ):
        raise PinnedManifestError("source prefix is unsafe")
    return prefix


def build_manifest(
    *,
    dataset: str,
    revision: str,
    prefix: str,
    suffix: str,
    entries: Iterable[object],
    expected_count: int | None = None,
) -> dict[str, object]:
    if not dataset or "/" not in dataset:
        raise PinnedManifestError("dataset identity differs")
    if not REVISION_PATTERN.fullmatch(revision):
        raise PinnedManifestError("revision must be a full commit SHA")
    prefix = _safe_prefix(prefix)
    if not suffix.startswith(".") or "/" in suffix:
        raise PinnedManifestError("source suffix differs")
    if expected_count is not None and expected_count <= 0:
        raise PinnedManifestError("expected file count differs")

    selected: list[dict[str, object]] = []
    upstream_count = 0
    upstream_bytes = 0
    for entry in entries:
        entry_type = type(entry).__name__
        if entry_type == "RepoFolder":
            continue
        path = getattr(entry, "path", None)
        size = getattr(entry, "size", None)
        if not isinstance(path, str) or not isinstance(size, int) or size <= 0:
            raise PinnedManifestError("upstream file metadata differs")
        parsed = PurePosixPath(path)
        if (
            parsed.is_absolute()
            or any(part in {"", ".", ".."} for part in parsed.parts)
            or str(parsed) != path
            or parsed.parts[: len(PurePosixPath(prefix).parts)]
            != PurePosixPath(prefix).parts
        ):
            raise PinnedManifestError("upstream file path differs")
        upstream_count += 1
        upstream_bytes += size
        if not path.endswith(suffix):
            continue

        lfs = getattr(entry, "lfs", None)
        lfs_size = getattr(lfs, "size", None)
        digest = getattr(lfs, "sha256", None)
        if (
            not isinstance(lfs_size, int)
            or lfs_size != size
            or not isinstance(digest, str)
            or not SHA256_PATTERN.fullmatch(digest)
        ):
            raise PinnedManifestError("selected LFS metadata differs")
        selected.append(
            {
                "path": path,
                "size": size,
                "sha256": digest,
            }
        )

    selected.sort(key=lambda record: str(record["path"]))
    if not selected:
        raise PinnedManifestError("selection is empty")
    if expected_count is not None and len(selected) != expected_count:
        raise PinnedManifestError("selected file count differs")
    if len({record["path"] for record in selected}) != len(selected):
        raise PinnedManifestError("selected file path is duplicated")
    if len(
        {PurePosixPath(str(record["path"])).name for record in selected}
    ) != len(selected):
        raise PinnedManifestError("selected file basename is duplicated")

    payload: dict[str, object] = {
        "schema": SCHEMA,
        "dataset": dataset,
        "revision": revision,
        "upstream_file_count_considered": upstream_count,
        "upstream_bytes_considered": upstream_bytes,
        "selection": {
            "method": "complete_prefix_sorted_tree",
            "prefix": prefix,
            "suffix": suffix,
            "selected_count": len(selected),
        },
        "files": selected,
        "selected_bytes": sum(int(record["size"]) for record in selected),
    }
    validate_manifest(
        payload,
        expected_dataset=dataset,
        expected_revision=revision,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--suffix", default=".parquet")
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    entries = HfApi().list_repo_tree(
        arguments.dataset,
        path_in_repo=arguments.prefix,
        repo_type="dataset",
        revision=arguments.revision,
        recursive=True,
        expand=True,
    )
    payload = build_manifest(
        dataset=arguments.dataset,
        revision=arguments.revision,
        prefix=arguments.prefix,
        suffix=arguments.suffix,
        entries=entries,
        expected_count=arguments.expected_count,
    )
    material = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    with arguments.out.open("x", encoding="ascii") as destination:
        destination.write(material)
    print(
        json.dumps(
            {
                "dataset": arguments.dataset,
                "files": len(payload["files"]),
                "revision": arguments.revision,
                "selected_bytes": payload["selected_bytes"],
                "status": "written",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
