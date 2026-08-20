from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from hf_gpt_oss_120b_mechanics import (
    EXPECTED_H100_MAX_SHARED_MEMORY,
    GptOssMechanicsError,
    _cuda_residency_receipt,
    _gradient_receipt,
    _kernel_compatibility_receipt,
    _native_mxfp4_load_receipt,
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


class Mxfp4Config:
    def __init__(self, dequantize: bool = False) -> None:
        self.dequantize = dequantize


class Mxfp4HfQuantizer:
    def __init__(self, dequantize: bool = False) -> None:
        self.quantization_config = Mxfp4Config(dequantize)


class _Projection:
    pass


class _Backbone(nn.Module):
    def __init__(
        self, device: object | None, dequantize: bool = False, resident: bool = True
    ) -> None:
        super().__init__()
        if device is not None:
            self.hf_device_map = {"": device}
        self.hf_quantizer = Mxfp4HfQuantizer(dequantize)
        tensor_device = "meta" if not resident else "cpu"
        self.anchor = nn.Parameter(torch.ones(1, device=tensor_device))
        self.register_buffer("buffer", torch.ones(1, device=tensor_device))
        layers = []
        for _ in range(36):
            experts = nn.Module()
            for name in ("gate_up_proj", "down_proj"):
                projection = _Projection()
                projection.storage = _Projection()
                projection.storage.data = torch.ones(1, device=tensor_device)
                setattr(experts, name, projection)
            mlp = nn.Module()
            mlp.experts = experts
            layer = nn.Module()
            layer.mlp = mlp
            layers.append(layer)
        self.model = nn.Module()
        self.model.layers = layers

    def named_parameters(self):
        return [("anchor", self.anchor)]

    def named_buffers(self):
        return [("buffer", self.buffer)]


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


@pytest.mark.parametrize("device", [0, "cuda", "cuda:0", torch.device("cuda:0")])
def test_native_mxfp4_receipt_normalizes_cuda_zero(
    monkeypatch: pytest.MonkeyPatch, device: object
) -> None:
    monkeypatch.setattr(
        "hf_gpt_oss_120b_mechanics._cuda_residency_receipt",
        lambda _: {"all_parameters_cuda_zero": True},
    )
    receipt = _native_mxfp4_load_receipt(_Backbone(device))
    assert receipt["device_map_mode"] == "explicit_cuda_zero"
    assert receipt["dequantize"] is False


def test_native_mxfp4_receipt_accepts_absent_map_with_residency_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "hf_gpt_oss_120b_mechanics._cuda_residency_receipt",
        lambda _: {"all_parameters_cuda_zero": True},
    )
    receipt = _native_mxfp4_load_receipt(_Backbone(None))
    assert receipt["device_map_mode"] == "absent_single_device_load"


@pytest.mark.parametrize(
    ("device", "dequantize"),
    [("cpu", False), ("disk", False), (1, False), ("cuda:0", True)],
)
def test_native_mxfp4_receipt_rejects_offload_or_dequantize(
    monkeypatch: pytest.MonkeyPatch, device: object, dequantize: bool
) -> None:
    monkeypatch.setattr(
        "hf_gpt_oss_120b_mechanics._cuda_residency_receipt",
        lambda _: {"all_parameters_cuda_zero": True},
    )
    with pytest.raises(GptOssMechanicsError, match="native MXFP4 load differs"):
        _native_mxfp4_load_receipt(_Backbone(device, dequantize))


def test_cuda_residency_rejects_non_cuda_model() -> None:
    with pytest.raises(GptOssMechanicsError, match="CUDA residency differs"):
        _cuda_residency_receipt(_Backbone(None))


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


def test_kernel_compatibility_receipt_binds_patch_and_h100_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative = Path(
        "kernel-repo/build/torch-cuda/matmul_ogs_details/opt_flags_details/"
        "opt_flags_nvidia.py"
    )
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(b"patched kernel")
    digest = hashlib.sha256(b"patched kernel").hexdigest()
    monkeypatch.setattr(
        "hf_gpt_oss_120b_mechanics.KERNEL_COMPATIBILITY_PATCHED_SHA256",
        digest,
    )
    receipt = _kernel_compatibility_receipt(
        tmp_path,
        EXPECTED_H100_MAX_SHARED_MEMORY,
        torch_property_present=False,
    )
    assert receipt["patched_sha256"] == digest
    assert receipt["max_shared_memory_bytes"] == 232448


@pytest.mark.parametrize(
    ("max_shared_memory", "torch_property_present"),
    [(0, False), (232448, True)],
)
def test_kernel_compatibility_receipt_rejects_interface_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    max_shared_memory: int,
    torch_property_present: bool,
) -> None:
    relative = Path(
        "kernel-repo/build/torch-cuda/matmul_ogs_details/opt_flags_details/"
        "opt_flags_nvidia.py"
    )
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(b"patched kernel")
    monkeypatch.setattr(
        "hf_gpt_oss_120b_mechanics.KERNEL_COMPATIBILITY_PATCHED_SHA256",
        hashlib.sha256(b"patched kernel").hexdigest(),
    )
    with pytest.raises(GptOssMechanicsError, match="compatibility receipt"):
        _kernel_compatibility_receipt(
            tmp_path,
            max_shared_memory,
            torch_property_present=torch_property_present,
        )
