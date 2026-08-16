"""CPU tests for the score-free Nemotron Ultra mechanics boundary."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from hf_nemotron_ultra_mechanics import (
    NemotronUltraMechanicsError,
    _atomic_json,
    _state_sha256,
    verify_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_model_manifest_rejects_extra_or_changed_members(tmp_path: Path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text("config")
    manifest = root / "SHA256SUMS"
    manifest.write_text(f"{_sha256(root / 'config.json')}  config.json\n")
    expected = _sha256(manifest)
    assert verify_manifest(root, manifest, expected, exact_membership=True) == {
        "manifest_sha256": expected,
        "manifest_entries": 1,
        "covered_bytes": 6,
        "exact_membership": True,
    }
    extra = root / "extra"
    extra.write_text("unexpected")
    with pytest.raises(NemotronUltraMechanicsError, match="membership"):
        verify_manifest(root, manifest, expected, exact_membership=True)
    extra.unlink()
    (root / "config.json").write_text("changed")
    with pytest.raises(NemotronUltraMechanicsError, match="member differs"):
        verify_manifest(root, manifest, expected, exact_membership=True)


def test_ultra_manifest_rejects_escape_and_symbolic_member(tmp_path: Path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("outside")
    manifest = root / "SHA256SUMS"
    manifest.write_text(f"{_sha256(outside)}  ../outside\n")
    with pytest.raises(NemotronUltraMechanicsError, match="row differs"):
        verify_manifest(root, manifest, _sha256(manifest), exact_membership=True)
    link = root / "member"
    link.symlink_to(outside)
    manifest.write_text(f"{_sha256(outside)}  member\n")
    with pytest.raises(NemotronUltraMechanicsError, match="member differs"):
        verify_manifest(root, manifest, _sha256(manifest), exact_membership=True)


def test_ultra_state_digest_is_order_independent_and_sensitive() -> None:
    first = {
        "b": torch.tensor([2.0], dtype=torch.float32),
        "a": torch.tensor([1.0], dtype=torch.float32),
    }
    second = {"a": first["a"].clone(), "b": first["b"].clone()}
    assert _state_sha256(first) == _state_sha256(second)
    second["b"].add_(1.0)
    assert _state_sha256(first) != _state_sha256(second)


def test_ultra_atomic_report_is_write_once(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    _atomic_json(output, {"status": "pass"})
    assert output.read_text() == '{\n  "status": "pass"\n}\n'
    with pytest.raises(NemotronUltraMechanicsError, match="existing"):
        _atomic_json(output, {"status": "changed"})
