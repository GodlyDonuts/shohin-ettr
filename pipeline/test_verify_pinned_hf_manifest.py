#!/usr/bin/env python3
"""Tests for the direct pinned-file source manifest."""

from copy import deepcopy

import pytest

from pipeline.verify_pinned_hf_manifest import (
    PinnedManifestError,
    validate_manifest,
)


def valid_payload():
    return {
        "schema": "shohin-pinned-hf-file-selection-v1",
        "dataset": "owner/dataset",
        "revision": "a" * 40,
        "upstream_file_count_considered": 2,
        "upstream_bytes_considered": 30,
        "selection": {"selected_count": 2},
        "files": [
            {"path": "data/first.json.gz", "size": 10, "sha256": "b" * 64},
            {"path": "data/second.json.gz", "size": 20, "sha256": "c" * 64},
        ],
        "selected_bytes": 30,
    }


def test_valid_manifest_returns_ordered_records():
    records = validate_manifest(
        valid_payload(),
        expected_dataset="owner/dataset",
        expected_revision="a" * 40,
    )
    assert [record["path"] for record in records] == [
        "data/first.json.gz",
        "data/second.json.gz",
    ]


def test_partitioned_hugging_face_path_is_safe():
    payload = valid_payload()
    payload["files"][0]["path"] = (
        "data/crawl=CC-MAIN-2024-38/train-00000-of-01000.parquet"
    )
    records = validate_manifest(
        payload,
        expected_dataset="owner/dataset",
        expected_revision="a" * 40,
    )
    assert records[0]["path"].startswith("data/crawl=CC-MAIN-2024-38/")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(dataset="other/dataset"), "dataset"),
        (lambda value: value.update(revision="d" * 40), "revision"),
        (
            lambda value: value["files"][0].update(path="../escape.json.gz"),
            "unsafe",
        ),
        (
            lambda value: value["files"][0].update(
                path="data/ambiguous\tpath.json.gz"
            ),
            "unsafe",
        ),
        (
            lambda value: value["files"][1].update(path="other/first.json.gz"),
            "duplicated",
        ),
        (
            lambda value: value["files"][0].update(sha256="not-a-hash"),
            "SHA-256",
        ),
        (lambda value: value.update(selected_bytes=31), "byte ledger"),
        (
            lambda value: value["selection"].update(selected_count=1),
            "count ledger",
        ),
    ],
)
def test_manifest_rejects_identity_and_ledger_drift(mutation, message):
    payload = deepcopy(valid_payload())
    mutation(payload)
    with pytest.raises(PinnedManifestError, match=message):
        validate_manifest(
            payload,
            expected_dataset="owner/dataset",
            expected_revision="a" * 40,
        )
