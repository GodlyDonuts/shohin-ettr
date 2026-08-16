"""CPU tests for the score-free Nemotron Super mechanics boundary."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from hf_nemotron_super_mechanics import (
    NemotronSuperMechanicsError,
    _atomic_json,
    _state_sha256,
    verify_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_verification_binds_order_bytes_and_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "a").write_bytes(b"alpha")
    (root / "b").write_bytes(b"beta")
    manifest = root / "SHA256SUMS"
    manifest.write_text(f"{_sha256(root / 'a')}  a\n{_sha256(root / 'b')}  b\n")
    receipt = verify_manifest(root, manifest, _sha256(manifest))
    assert receipt == {
        "manifest_sha256": _sha256(manifest),
        "manifest_entries": 2,
        "covered_bytes": 9,
    }
    (root / "b").write_bytes(b"changed")
    with pytest.raises(NemotronSuperMechanicsError):
        verify_manifest(root, manifest, _sha256(manifest))


def test_manifest_rejects_escape_but_accepts_hash_bound_install_order(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    member = root / "a"
    member.write_text("a")
    manifest = root / "SHA256SUMS"
    manifest.write_text(f"{_sha256(member)}  ../a\n")
    with pytest.raises(NemotronSuperMechanicsError):
        verify_manifest(root, manifest, _sha256(manifest))
    second = root / "b"
    second.write_text("b")
    manifest.write_text(f"{_sha256(second)}  b\n{_sha256(member)}  a\n")
    assert verify_manifest(root, manifest, _sha256(manifest)) == {
        "manifest_sha256": _sha256(manifest),
        "manifest_entries": 2,
        "covered_bytes": 2,
    }


def test_state_digest_is_order_independent_and_value_sensitive() -> None:
    first = {
        "b": torch.tensor([2.0], dtype=torch.float32),
        "a": torch.tensor([1.0], dtype=torch.float32),
    }
    second = {"a": first["a"].clone(), "b": first["b"].clone()}
    assert _state_sha256(first) == _state_sha256(second)
    second["b"].add_(1.0)
    assert _state_sha256(first) != _state_sha256(second)


def test_atomic_report_is_write_once(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    _atomic_json(output, {"status": "pass"})
    assert output.read_text() == '{\n  "status": "pass"\n}\n'
    with pytest.raises(NemotronSuperMechanicsError):
        _atomic_json(output, {"status": "changed"})
