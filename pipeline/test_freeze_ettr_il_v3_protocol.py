"""Tests for the ETTR-IL-v3 protocol freezer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ettr_il_v3_protocol import canonical_json_bytes
from freeze_ettr_il_v3_protocol import (
    FreezeError,
    build_freeze,
    write_no_replace,
)


COMMIT = "a" * 40


def test_build_freeze_is_order_independent_and_self_bound(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.txt").write_text("a\n", encoding="ascii")
    (tmp_path / "b.txt").write_text("b\n", encoding="ascii")
    first = build_freeze(
        tmp_path,
        ("b.txt", "a.txt"),
        source_commit=COMMIT,
    )
    second = build_freeze(
        tmp_path,
        ("a.txt", "b.txt"),
        source_commit=COMMIT,
    )
    assert first == second
    assert first["source_count"] == 2
    assert len(first["freeze_sha256"]) == 64
    assert [
        item["path"] for item in first["source_inventory"]
    ] == ["a.txt", "b.txt"]
    json.loads(canonical_json_bytes(first))


def test_freeze_rejects_unsafe_or_duplicated_sources(
    tmp_path: Path,
) -> None:
    (tmp_path / "a").write_text("x", encoding="ascii")
    with pytest.raises(FreezeError):
        build_freeze(tmp_path, ("a", "a"), source_commit=COMMIT)
    with pytest.raises(FreezeError):
        build_freeze(tmp_path, ("../a",), source_commit=COMMIT)
    with pytest.raises(FreezeError):
        build_freeze(tmp_path, ("a",), source_commit="bad")


def test_freeze_rejects_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("x", encoding="ascii")
    (tmp_path / "link").symlink_to(target)
    with pytest.raises(FreezeError):
        build_freeze(tmp_path, ("link",), source_commit=COMMIT)


def test_write_no_replace_is_atomic(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "receipt.json"
    write_no_replace(out, b"first\n")
    assert out.read_bytes() == b"first\n"
    with pytest.raises(FileExistsError):
        write_no_replace(out, b"second\n")
    assert out.read_bytes() == b"first\n"
