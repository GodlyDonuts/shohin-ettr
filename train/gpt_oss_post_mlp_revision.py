"""Shohin residuals on the pinned GPT-OSS-120B MXFP4 MoE surface."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch
import torch.nn as nn

from q36_upward_moe_gpt_oss_host import (
    ALPHA,
    ATTACHMENT_SURFACE,
    CONTROLLED_LAYER_INDICES,
    HIDDEN_SIZE,
    RANK,
    TRAINABLE_PARAMETERS_PER_ROLE,
    Q36UpwardMoEGptOssHostError,
    validate_loaded_surface,
)

ARCHITECTURE = "shohin-gpt-oss-120b-shared-post-mlp-v1"


class GptOssRevisionError(RuntimeError):
    """The pinned GPT-OSS revision surface differs."""


class GptOssPostMLPResidual(nn.Module):
    """Frozen tuple-returning GPT-OSS MLP followed by a tokenwise residual."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        try:
            device = next(base.parameters()).device
        except StopIteration as error:
            raise GptOssRevisionError("GPT-OSS MLP has no device anchor") from error
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

    def reset_receipt(self) -> None:
        self._tokens = 0
        self._residual_norm = 0.0
        self._native_norm = 0.0

    def receipt(self) -> dict[str, float | int]:
        if not self._tokens:
            return {"tokens": 0}
        return {
            "tokens": self._tokens,
            "mean_residual_norm": self._residual_norm / self._tokens,
            "mean_native_output_norm": self._native_norm / self._tokens,
        }

    def forward(
        self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, Any]:
        native = self.base(hidden_states, *args, **kwargs)
        if (
            not isinstance(native, tuple)
            or len(native) != 2
            or not isinstance(native[0], torch.Tensor)
            or native[0].shape != hidden_states.shape
        ):
            raise GptOssRevisionError("base GPT-OSS MLP output geometry differs")
        native_hidden, router_scores = native
        if hidden_states.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                residual = self.adapter_b(self.adapter_a(hidden_states)) * self.scale
        else:
            residual = (
                self.adapter_b(self.adapter_a(hidden_states.to(torch.float32)))
                * self.scale
            )
        with torch.no_grad():
            tokens = int(native_hidden.numel() // native_hidden.shape[-1])
            self._tokens += tokens
            self._residual_norm += float(
                residual.float().norm(dim=-1).sum().detach().cpu()
            )
            self._native_norm += float(
                native_hidden.float().norm(dim=-1).sum().detach().cpu()
            )
        return native_hidden + residual.to(native_hidden.dtype), router_scores


class GptOssRevisionModel(nn.Module):
    """Freeze GPT-OSS and attach residuals after its final sixteen MoE blocks."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        try:
            receipt = validate_loaded_surface(backbone)
        except Q36UpwardMoEGptOssHostError as error:
            raise GptOssRevisionError(str(error)) from error
        if receipt.get("attachment_surface") != ATTACHMENT_SURFACE:
            raise GptOssRevisionError("GPT-OSS attachment surface differs")
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        blocks: list[GptOssPostMLPResidual] = []
        for index in CONTROLLED_LAYER_INDICES:
            layer = self.backbone.model.layers[index]
            block = GptOssPostMLPResidual(layer.mlp)
            layer.mlp = block
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        if self.trainable_parameter_count() != TRAINABLE_PARAMETERS_PER_ROLE:
            raise GptOssRevisionError("GPT-OSS trainable parameter count differs")
        if any(
            parameter.requires_grad
            for block in self.blocks
            for parameter in block.base.parameters()
        ):
            raise GptOssRevisionError("GPT-OSS native router or expert is trainable")

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
            raise GptOssRevisionError("GPT-OSS trainable parameter names differ")
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
            raise GptOssRevisionError("GPT-OSS trainable state geometry differs")
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
