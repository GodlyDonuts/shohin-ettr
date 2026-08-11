"""Deterministic full-tree receipt tests for the PCF1 environment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import capture_pcf1_environment as environment
from capture_pcf1_environment import PCF1EnvironmentError


def _digest(rows: list[dict[str, object]]) -> str:
    encoded = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_environment_tree_uses_canonical_directory_rows_and_exact_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "environment"
    directory = root / "a"
    directory.mkdir(parents=True)
    payload = b"fixture"
    (directory / "file.bin").write_bytes(payload)
    (root / "link").symlink_to("a/file.bin")
    rows = [
        {"path": "a", "type": "directory"},
        {
            "path": "a/file.bin",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
            "type": "file",
        },
        {"path": "link", "target": "a/file.bin", "type": "symlink"},
    ]
    monkeypatch.setattr(environment, "ENVIRONMENT_TREE_SHA256", _digest(rows))
    monkeypatch.setattr(environment, "ENVIRONMENT_TREE_ENTRIES", 3)
    monkeypatch.setattr(environment, "ENVIRONMENT_TREE_FILES", 1)
    monkeypatch.setattr(environment, "ENVIRONMENT_TREE_DIRECTORIES", 1)
    monkeypatch.setattr(environment, "ENVIRONMENT_TREE_SYMLINKS", 1)
    monkeypatch.setattr(environment, "ENVIRONMENT_TREE_BYTES", len(payload))
    assert environment._environment_tree(root) == {
        "sha256": _digest(rows),
        "entries": 3,
        "files": 1,
        "directories": 1,
        "symlinks": 1,
        "file_bytes": len(payload),
    }

    (root / "extra-empty-directory").mkdir()
    with pytest.raises(PCF1EnvironmentError, match="full environment tree"):
        environment._environment_tree(root)


def test_environment_package_versions_are_exactly_pinned() -> None:
    assert environment.PACKAGE_VERSIONS == {
        "accelerate": "1.14.0",
        "huggingface-hub": "1.22.0",
        "peft": "0.20.0",
        "safetensors": "0.8.0",
        "sentencepiece": "0.2.2",
        "tokenizers": "0.22.2",
        "torch": "2.6.0+cu124",
        "transformers": "5.15.0.dev0",
        "triton": "3.2.0",
    }
    assert environment.PYTHON_ENTRYPOINT == (
        environment.ENVIRONMENT_ROOT / "bin/python"
    )
    assert environment.PYTHON_SITE_PACKAGES == (
        environment.ENVIRONMENT_ROOT / "lib/python3.13/site-packages"
    )
    assert set(environment.PACKAGE_IMPORTS) == set(environment.PACKAGE_VERSIONS)
    assert set(environment.PACKAGE_ORIGINS) == set(environment.PACKAGE_VERSIONS)
    assert environment.PACKAGE_ORIGINS["torch"] == (
        environment.PYTHON_BASE_PREFIX
        / "lib/python3.13/site-packages/torch/__init__.py"
    )
    assert environment.PACKAGE_ORIGINS["transformers"] == (
        environment.PYTHON_SITE_PACKAGES / "transformers/__init__.py"
    )
