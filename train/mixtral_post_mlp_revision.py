"""Shohin role-owned residuals on the pinned Mixtral-8x22B MoE surface."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch
import torch.nn as nn

from q36_upward_moe_mixtral_host import (
    ALPHA,
    ATTACHMENT_SURFACE,
    CONTROLLED_LAYER_INDICES,
    HIDDEN_SIZE,
    RANK,
    TRAINABLE_PARAMETERS_PER_ROLE,
    Q36UpwardMoEMixtralHostError,
    validate_loaded_surface,
)
from shared_post_mlp_revision import SharedPostMLPConfig, SharedPostMLPResidual

ARCHITECTURE = "shohin-mixtral-8x22b-shared-post-mlp-v1"


class MixtralRevisionError(RuntimeError):
    """The pinned Mixtral revision surface differs."""


class MixtralRevisionModel(nn.Module):
    """Freeze Mixtral and attach residuals after its final sixteen MoE blocks."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        try:
            receipt = validate_loaded_surface(backbone)
        except Q36UpwardMoEMixtralHostError as error:
            raise MixtralRevisionError(str(error)) from error
        if receipt.get("attachment_surface") != ATTACHMENT_SURFACE:
            raise MixtralRevisionError("Mixtral attachment surface differs")
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        config = SharedPostMLPConfig(
            hidden_size=HIDDEN_SIZE,
            controlled_layers=len(CONTROLLED_LAYER_INDICES),
            rank=RANK,
            alpha=ALPHA,
        )
        blocks: list[SharedPostMLPResidual] = []
        for index in CONTROLLED_LAYER_INDICES:
            layer = self.backbone.model.layers[index]
            block = SharedPostMLPResidual(layer.mlp, config)
            layer.mlp = block
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        if self.trainable_parameter_count() != TRAINABLE_PARAMETERS_PER_ROLE:
            raise MixtralRevisionError("Mixtral trainable parameter count differs")
        if any(
            parameter.requires_grad
            for block in self.blocks
            for parameter in block.base.parameters()
        ):
            raise MixtralRevisionError("Mixtral native router or expert is trainable")

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
            raise MixtralRevisionError("Mixtral trainable parameter names differ")
        return hashlib.sha256("\n".join(names).encode()).hexdigest()

    def trainable_state(self) -> dict[str, torch.Tensor]:
        state = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        if (
            sum(value.numel() for value in state.values())
            != TRAINABLE_PARAMETERS_PER_ROLE
        ):
            raise MixtralRevisionError("Mixtral trainable state geometry differs")
        return state

    def trainable_state_sha256(self) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(self.trainable_state().items()):
            value = tensor.contiguous()
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
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
            "native_router_expert_trainables": 0,
            "layers": [block.receipt() for block in self.blocks],
        }
