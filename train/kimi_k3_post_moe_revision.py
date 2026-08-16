"""Frozen-eval post-MoE Shohin trajectory for the pinned Kimi K3 host."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch
import torch.nn as nn

from q36_upward_moe_kimi_k3_host import (
    ALPHA,
    ATTACHMENT_SURFACE,
    CONTROLLED_LAYER_INDICES,
    HIDDEN_SIZE,
    NATIVE_EXECUTION_MODE,
    Q36UpwardMoEKimiK3HostError,
    RANK,
    TRAINABLE_PARAMETERS_PER_ROLE,
    validate_loaded_surface,
)

ARCHITECTURE = "shohin-kimi-k3-frozen-post-moe-v1"


class KimiK3RevisionError(RuntimeError):
    """The pinned Kimi K3 residual surface or eval-only invariant differs."""


class FrozenEvalPostMoEResidual(nn.Module):
    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.base.eval()
        try:
            device = next(base.parameters()).device
        except StopIteration as error:
            raise KimiK3RevisionError("Kimi K3 native MoE has no parameters") from error
        self.adapter_a = nn.Linear(HIDDEN_SIZE, RANK, bias=False).to(
            device=device, dtype=torch.float32
        )
        self.adapter_b = nn.Linear(RANK, HIDDEN_SIZE, bias=False).to(
            device=device, dtype=torch.float32
        )
        nn.init.kaiming_uniform_(self.adapter_a.weight, a=5**0.5)
        nn.init.zeros_(self.adapter_b.weight)
        self.scale = ALPHA / RANK
        self.reset_receipt()

    def train(self, mode: bool = True) -> "FrozenEvalPostMoEResidual":
        self.training = mode
        self.base.eval()
        self.adapter_a.train(mode)
        self.adapter_b.train(mode)
        return self

    def reset_receipt(self) -> None:
        self._tokens = 0
        self._residual_norm = 0.0
        self._native_norm = 0.0

    def receipt(self) -> dict[str, float | int | str]:
        if self._tokens == 0:
            return {"tokens": 0, "native_execution_mode": NATIVE_EXECUTION_MODE}
        return {
            "tokens": self._tokens,
            "mean_residual_norm": self._residual_norm / self._tokens,
            "mean_native_output_norm": self._native_norm / self._tokens,
            "native_execution_mode": NATIVE_EXECUTION_MODE,
        }

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.base.training:
            raise KimiK3RevisionError("Kimi K3 native MoE entered training mode")
        native = self.base(hidden_states)
        if not isinstance(native, torch.Tensor) or native.shape != hidden_states.shape:
            raise KimiK3RevisionError("Kimi K3 native MoE output geometry differs")
        if hidden_states.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                residual = self.adapter_b(self.adapter_a(hidden_states)) * self.scale
        else:
            residual = (
                self.adapter_b(self.adapter_a(hidden_states.float())) * self.scale
            )
        with torch.no_grad():
            tokens = int(native.numel() // native.shape[-1])
            self._tokens += tokens
            self._residual_norm += float(residual.float().norm(dim=-1).sum().cpu())
            self._native_norm += float(native.float().norm(dim=-1).sum().cpu())
        return native + residual.to(native.dtype)


class KimiK3RevisionModel(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        try:
            surface = validate_loaded_surface(backbone)
        except Q36UpwardMoEKimiK3HostError as error:
            raise KimiK3RevisionError(str(error)) from error
        if surface.get("attachment_surface") != ATTACHMENT_SURFACE:
            raise KimiK3RevisionError("Kimi K3 attachment surface differs")
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        layers = self.backbone.language_model.model.layers
        blocks: list[FrozenEvalPostMoEResidual] = []
        for index in CONTROLLED_LAYER_INDICES:
            layer = layers[index]
            if not hasattr(layer, "block_sparse_moe"):
                raise KimiK3RevisionError("Kimi K3 controlled layer is not sparse MoE")
            block = FrozenEvalPostMoEResidual(layer.block_sparse_moe)
            layer.block_sparse_moe = block
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        if self.trainable_parameter_count() != TRAINABLE_PARAMETERS_PER_ROLE:
            raise KimiK3RevisionError("Kimi K3 trainable parameter count differs")
        if any(
            parameter.requires_grad
            for block in blocks
            for parameter in block.base.parameters()
        ):
            raise KimiK3RevisionError("Kimi K3 native router or expert is trainable")

    def train(self, mode: bool = True) -> "KimiK3RevisionModel":
        self.training = mode
        self.backbone.eval()
        for block in self.blocks:
            block.train(mode)
        return self

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.backbone(*args, **kwargs)

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def trainable_parameter_name_sha256(self) -> str:
        names = sorted(
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        )
        if len(names) != len(CONTROLLED_LAYER_INDICES) * 2:
            raise KimiK3RevisionError("Kimi K3 trainable parameter names differ")
        return hashlib.sha256("\n".join(names).encode()).hexdigest()

    def trainable_state(self) -> dict[str, torch.Tensor]:
        state = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        if (
            sum(tensor.numel() for tensor in state.values())
            != TRAINABLE_PARAMETERS_PER_ROLE
        ):
            raise KimiK3RevisionError("Kimi K3 trainable state geometry differs")
        return state

    def trainable_state_sha256(self) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(self.trainable_state().items()):
            value = tensor.contiguous()
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def reset_receipt(self) -> None:
        for block in self.blocks:
            block.reset_receipt()

    def receipt(self) -> dict[str, Any]:
        return {
            "architecture": ARCHITECTURE,
            "attachment_surface": ATTACHMENT_SURFACE,
            "controlled_layer_indices": list(CONTROLLED_LAYER_INDICES),
            "trainable_parameters": self.trainable_parameter_count(),
            "native_execution_mode": NATIVE_EXECUTION_MODE,
            "native_router_expert_trainables": 0,
            "layers": [block.receipt() for block in self.blocks],
        }
