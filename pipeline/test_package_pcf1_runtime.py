"""Tests for deterministic, clean-commit PCF1 runtime packaging."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from package_pcf1_runtime import (
    PCF1RuntimeError,
    QUALIFIED_SHARED,
    package,
)


def _repository(root: Path) -> Path:
    source = root / "source"
    source.mkdir()
    entries = sorted((*QUALIFIED_SHARED, "pipeline/pcf1_runtime_allowlist.txt"))
    for entry in entries:
        path = source / entry
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {entry}\n")
    allowlist = source / "pipeline/pcf1_runtime_allowlist.txt"
    allowlist.write_text("\n".join(entries) + "\n")
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=PCF1 Test",
            "-c",
            "user.email=pcf1@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=source,
        check=True,
    )
    return source


def test_package_is_deterministic_and_exact_membership(tmp_path: Path) -> None:
    source = _repository(tmp_path)
    allowlist = source / "pipeline/pcf1_runtime_allowlist.txt"
    first = tmp_path / "runtime-a"
    second = tmp_path / "runtime-b"
    first_digest = package(source, allowlist, first)
    second_digest = package(source, allowlist, second)
    assert first_digest == second_digest
    assert (first / "SHA256SUMS").read_bytes() == (second / "SHA256SUMS").read_bytes()
    runtime = json.loads((first / "runtime.json").read_text())
    assert (
        runtime["allowlist_sha256"]
        == hashlib.sha256(allowlist.read_bytes()).hexdigest()
    )
    manifest_entries = {
        line.split("  ", 1)[1]
        for line in (first / "SHA256SUMS").read_text().splitlines()
    }
    actual = {
        path.relative_to(first).as_posix()
        for path in first.rglob("*")
        if path.is_file()
    }
    assert actual == manifest_entries | {"SHA256SUMS"}
    assert "train/hf_product_reasoning_rollouts.py" not in manifest_entries


def test_package_rejects_dirty_source(tmp_path: Path) -> None:
    source = _repository(tmp_path)
    allowlist = source / "pipeline/pcf1_runtime_allowlist.txt"
    (source / next(iter(QUALIFIED_SHARED))).write_text("dirty\n")
    with pytest.raises(PCF1RuntimeError, match="dirty"):
        package(source, allowlist, tmp_path / "runtime")


def test_package_rejects_protected_output_before_writing(tmp_path: Path) -> None:
    source = _repository(tmp_path)
    allowlist = source / "pipeline/pcf1_runtime_allowlist.txt"
    output = tmp_path / "public" / "runtime"
    with pytest.raises(PCF1RuntimeError, match="protected"):
        package(source, allowlist, output)
    assert not output.parent.exists()
