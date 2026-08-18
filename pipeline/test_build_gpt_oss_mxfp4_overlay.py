from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from build_gpt_oss_mxfp4_overlay import (
    KERNEL_REVISION,
    PACKAGES,
    GptOssOverlayError,
    manifest_tree,
    pip_command,
)


def test_overlay_pins_exact_packages_and_kernel_revision(tmp_path: Path) -> None:
    command = pip_command(Path("/pinned/python"), tmp_path)
    assert command[-3:] == [
        "kernels==0.16.0",
        "kernels-data==0.16.0",
        "triton==3.4.0",
    ]
    assert PACKAGES == {
        "kernels": "0.16.0",
        "kernels-data": "0.16.0",
        "triton": "3.4.0",
    }
    assert KERNEL_REVISION == "9655fcf7d0f638bec4a82f6f1a70014f0aa8cfb0"


def test_manifest_binds_exact_regular_tree(tmp_path: Path) -> None:
    (tmp_path / "a").write_bytes(b"a")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b").write_bytes(b"bb")
    text, receipt = manifest_tree(tmp_path)
    assert text.splitlines() == [
        f"{hashlib.sha256(b'a').hexdigest()}  a",
        f"{hashlib.sha256(b'bb').hexdigest()}  nested/b",
    ]
    assert receipt == {
        "manifest_entries": 2,
        "covered_bytes": 3,
        "exact_regular_files": True,
    }


def test_manifest_rejects_symbolic_member(tmp_path: Path) -> None:
    (tmp_path / "a").write_text("a", encoding="utf-8")
    (tmp_path / "alias").symlink_to("a")
    with pytest.raises(GptOssOverlayError, match="link or special"):
        manifest_tree(tmp_path)
