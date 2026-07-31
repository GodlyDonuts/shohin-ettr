from __future__ import annotations

from pathlib import Path

import pytest

from train_ettr_joint_stream_canary import (
    ETTRJointCanaryError,
    _legacy_general_resolution,
)


def _legacy_shards(root: Path, name: str) -> Path:
    directory = root / name
    directory.mkdir()
    (directory / "shard_00000.u16.zst").write_bytes(b"one")
    (directory / "shard_00001.u16.zst").write_bytes(b"two")
    (directory / "manifest.json").write_text("{}\n")
    return directory


def test_legacy_general_resolution_is_deterministic_and_weighted(
    tmp_path: Path,
) -> None:
    first = _legacy_shards(tmp_path, "first")
    second = _legacy_shards(tmp_path, "second")
    resolution = _legacy_general_resolution(
        (first, second),
        (3.0, 1.0),
        tokenizer_sha256="a" * 64,
    )
    repeated = _legacy_general_resolution(
        (first, second),
        (3.0, 1.0),
        tokenizer_sha256="a" * 64,
    )
    assert repeated == resolution
    assert resolution["domain_weights"] == [0.75, 0.25]
    assert resolution["legacy_scientific_control"] is True
    assert len(resolution["corpora"]) == 2


def test_legacy_general_resolution_detects_inventory_change(
    tmp_path: Path,
) -> None:
    directory = _legacy_shards(tmp_path, "general")
    before = _legacy_general_resolution(
        (directory,),
        (1.0,),
        tokenizer_sha256="b" * 64,
    )
    (directory / "shard_00002.u16.zst").write_bytes(b"three")
    after = _legacy_general_resolution(
        (directory,),
        (1.0,),
        tokenizer_sha256="b" * 64,
    )
    assert before["inventory_sha256"] != after["inventory_sha256"]


def test_legacy_general_resolution_rejects_empty_directory(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ETTRJointCanaryError, match="empty"):
        _legacy_general_resolution(
            (empty,),
            (1.0,),
            tokenizer_sha256="c" * 64,
        )
