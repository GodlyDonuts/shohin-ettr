from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from fetch_gpt_oss_120b import (
    MODEL_BYTES,
    MODEL_FILES,
    MODEL_REVISION,
    GptOssAcquisitionError,
    manifest_text,
    verify_projection,
)


def _tiny_projection(root: Path) -> dict[str, tuple[int, str]]:
    payloads = {"config.json": b"{}\n", "weights.safetensors": b"weights"}
    for relative, payload in payloads.items():
        (root / relative).write_bytes(payload)
    return {
        relative: (len(payload), hashlib.sha256(payload).hexdigest())
        for relative, payload in payloads.items()
    }


def test_exact_transformers_projection_excludes_duplicate_deployments() -> None:
    assert len(MODEL_FILES) == 26
    assert MODEL_BYTES == 65_276_859_410
    assert not any(
        relative.startswith(("metal/", "original/")) for relative in MODEL_FILES
    )
    assert len([name for name in MODEL_FILES if name.endswith(".safetensors")]) == 15


def test_projection_verifier_binds_membership_bytes_and_hashes(tmp_path: Path) -> None:
    files = _tiny_projection(tmp_path)
    receipt = verify_projection(tmp_path, files)
    assert receipt == {
        "files": 2,
        "covered_bytes": 10,
        "exact_membership": True,
    }
    (tmp_path / "weights.safetensors").write_bytes(b"changed")
    with pytest.raises(GptOssAcquisitionError, match="file differs"):
        verify_projection(tmp_path, files)


def test_projection_verifier_rejects_extra_or_symbolic_members(tmp_path: Path) -> None:
    files = _tiny_projection(tmp_path)
    (tmp_path / "extra").write_text("no", encoding="utf-8")
    with pytest.raises(GptOssAcquisitionError, match="membership"):
        verify_projection(tmp_path, files)
    (tmp_path / "extra").unlink()
    (tmp_path / "alias").symlink_to("config.json")
    with pytest.raises(GptOssAcquisitionError, match="linked or special"):
        verify_projection(tmp_path, files)


def test_manifest_is_deterministic_and_binds_source_revision() -> None:
    files = {"z": (1, "b" * 64), "a": (2, "a" * 64)}
    observed = manifest_text(files)
    source = hashlib.sha256(f"{MODEL_REVISION}\n".encode()).hexdigest()
    assert observed.splitlines() == [
        f"{source}  SOURCE_REVISION",
        f"{'a' * 64}  a",
        f"{'b' * 64}  z",
    ]
