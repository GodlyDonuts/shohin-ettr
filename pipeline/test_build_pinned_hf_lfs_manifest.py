#!/usr/bin/env python3
"""Tests for deterministic Hugging Face LFS source selection."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from pipeline.build_pinned_hf_lfs_manifest import build_manifest
from pipeline.verify_pinned_hf_manifest import PinnedManifestError


def repo_file(path, size, digest):
    return SimpleNamespace(
        path=path,
        size=size,
        lfs=SimpleNamespace(size=size, sha256=digest),
    )


def test_build_manifest_selects_complete_sorted_suffix():
    entries = [
        repo_file("sample/100BT/part-b.parquet", 20, "b" * 64),
        repo_file("sample/100BT/README.txt", 5, "c" * 64),
        repo_file("sample/100BT/part-a.parquet", 10, "a" * 64),
    ]
    payload = build_manifest(
        dataset="owner/dataset",
        revision="d" * 40,
        prefix="sample/100BT",
        suffix=".parquet",
        entries=entries,
        expected_count=2,
    )
    assert [record["path"] for record in payload["files"]] == [
        "sample/100BT/part-a.parquet",
        "sample/100BT/part-b.parquet",
    ]
    assert payload["selected_bytes"] == 30
    assert payload["upstream_file_count_considered"] == 3
    assert payload["upstream_bytes_considered"] == 35
    assert payload["selection"] == {
        "method": "complete_prefix_sorted_tree",
        "prefix": "sample/100BT",
        "suffix": ".parquet",
        "selected_count": 2,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda values: values[0].lfs.__setattr__("sha256", "bad"),
            "LFS metadata",
        ),
        (
            lambda values: values[0].lfs.__setattr__("size", 9),
            "LFS metadata",
        ),
        (
            lambda values: values[0].__setattr__(
                "path",
                "other/part-a.parquet",
            ),
            "path differs",
        ),
    ],
)
def test_build_manifest_rejects_source_identity_drift(mutation, message):
    entries = [repo_file("sample/100BT/part-a.parquet", 10, "a" * 64)]
    mutation(entries)
    with pytest.raises(PinnedManifestError, match=message):
        build_manifest(
            dataset="owner/dataset",
            revision="d" * 40,
            prefix="sample/100BT",
            suffix=".parquet",
            entries=entries,
        )


def test_build_manifest_rejects_count_and_duplicate_basename():
    entries = [
        repo_file("sample/100BT/a/part.parquet", 10, "a" * 64),
        repo_file("sample/100BT/b/part.parquet", 20, "b" * 64),
    ]
    with pytest.raises(PinnedManifestError, match="basename"):
        build_manifest(
            dataset="owner/dataset",
            revision="d" * 40,
            prefix="sample/100BT",
            suffix=".parquet",
            entries=entries,
        )

    one = deepcopy(entries[:1])
    with pytest.raises(PinnedManifestError, match="count"):
        build_manifest(
            dataset="owner/dataset",
            revision="d" * 40,
            prefix="sample/100BT",
            suffix=".parquet",
            entries=one,
            expected_count=2,
        )
