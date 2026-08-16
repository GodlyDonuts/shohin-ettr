from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import derive_q36_mtr_environment_receipt as module


def _base(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": module.SCHEMA,
                "status": "pass",
                "scientific_rows_read": 0,
                "offline_required": True,
                "bytecode_writes_permitted": False,
                "runtime_root": "/old",
                "runtime_manifest_sha256": "0" * 64,
            }
        )
        + "\n"
    )
    return path


def test_derive_rebinds_only_runtime_custody(tmp_path: Path) -> None:
    base = _base(tmp_path / "base.json")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    manifest = runtime / "SHA256SUMS"
    manifest.write_text("a" * 64 + "  member\n")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    output = tmp_path / "derived.json"
    payload = module.derive(base, runtime, digest, output)
    assert payload["runtime_root"] == str(runtime.resolve())
    assert payload["runtime_manifest_sha256"] == digest
    assert payload["scientific_rows_read"] == 0
    assert payload["derivation"] == module.DERIVATION
    assert payload["derived_from_receipt_sha256"] == module.sha256_file(base)


def test_derive_rejects_manifest_mismatch(tmp_path: Path) -> None:
    base = _base(tmp_path / "base.json")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "SHA256SUMS").write_text("manifest\n")
    with pytest.raises(module.Q36MTREnvironmentDerivationError):
        module.derive(base, runtime, "0" * 64, tmp_path / "derived.json")
