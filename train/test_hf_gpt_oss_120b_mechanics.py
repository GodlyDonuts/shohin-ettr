from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from hf_gpt_oss_120b_mechanics import (
    GptOssMechanicsError,
    _gradient_receipt,
    verify_manifest,
)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter_a = nn.Linear(1, 1, bias=False)
        self.adapter_b = nn.Linear(1, 1, bias=False)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(16)])


def test_gradient_receipt_requires_every_post_mxfp4_residual_path() -> None:
    model = _Model()
    for block in model.blocks:
        block.adapter_a.weight.grad = torch.zeros_like(block.adapter_a.weight)
        block.adapter_b.weight.grad = torch.ones_like(block.adapter_b.weight)
    receipt = _gradient_receipt(model)
    assert receipt["parameters"] == 32
    assert receipt["adapter_b_nonzero_gradients"] == 16
    assert receipt["earliest_controlled_layer_nonzero"] is True
    assert receipt["latest_controlled_layer_nonzero"] is True
    model.blocks[0].adapter_b.weight.grad.zero_()
    with pytest.raises(GptOssMechanicsError, match="gradient receipt"):
        _gradient_receipt(model)


def test_manifest_verifier_binds_hash_and_exact_membership(tmp_path: Path) -> None:
    (tmp_path / "payload").write_bytes(b"payload")
    digest = hashlib.sha256(b"payload").hexdigest()
    text = f"{digest}  payload\n"
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text(text, encoding="utf-8")
    receipt = verify_manifest(
        tmp_path, manifest, hashlib.sha256(text.encode()).hexdigest()
    )
    assert receipt["manifest_entries"] == 1
    assert receipt["covered_bytes"] == 7
    (tmp_path / "extra").write_text("no", encoding="utf-8")
    with pytest.raises(GptOssMechanicsError, match="membership"):
        verify_manifest(tmp_path, manifest, hashlib.sha256(text.encode()).hexdigest())


def test_manifest_verifier_rejects_symbolic_member(tmp_path: Path) -> None:
    (tmp_path / "payload").write_text("x", encoding="utf-8")
    (tmp_path / "alias").symlink_to("payload")
    digest = hashlib.sha256(b"x").hexdigest()
    text = f"{digest}  alias\n{digest}  payload\n"
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text(text, encoding="utf-8")
    with pytest.raises(GptOssMechanicsError, match="member"):
        verify_manifest(tmp_path, manifest, hashlib.sha256(text.encode()).hexdigest())
