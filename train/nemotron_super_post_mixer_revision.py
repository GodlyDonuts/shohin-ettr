"""Role-owned Shohin residuals on the pinned Nemotron Super MoE surface."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch
import torch.nn as nn

from q36_upward_moe_host import (
    ALPHA,
    ATTACHMENT_SURFACE,
    CONTROLLED_LAYER_INDICES,
    HIDDEN_SIZE,
    RANK,
    TRAINABLE_PARAMETERS_PER_ROLE,
    Q36UpwardMoEHostError,
    validate_loaded_surface,
)

ARCHITECTURE = "shohin-nemotron-super-shared-post-mixer-v1"


class NemotronSuperRevisionError(RuntimeError):
    """The pinned upward-MoE residual surface differs."""


class PostMixerResidual(nn.Module):
    """Frozen native MoE mixer followed by one low-rank tokenwise residual."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        try:
            device = next(base.parameters()).device
        except StopIteration as error:
            raise NemotronSuperRevisionError(
                "Nemotron Super native mixer has no parameters"
            ) from error
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
        if self._tokens == 0:
            return {"tokens": 0}
        return {
            "tokens": self._tokens,
            "mean_residual_norm": self._residual_norm / self._tokens,
            "mean_native_output_norm": self._native_norm / self._tokens,
        }

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        native = self.base(hidden_states)
        if not isinstance(native, torch.Tensor) or native.shape != hidden_states.shape:
            raise NemotronSuperRevisionError(
                "Nemotron Super native mixer output geometry differs"
            )
        if hidden_states.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                residual = self.adapter_b(self.adapter_a(hidden_states)) * self.scale
        else:
            residual = (
                self.adapter_b(self.adapter_a(hidden_states.to(torch.float32)))
                * self.scale
            )
        with torch.no_grad():
            tokens = int(native.numel() // native.shape[-1])
            self._tokens += tokens
            self._residual_norm += float(residual.float().norm(dim=-1).sum().cpu())
            self._native_norm += float(native.float().norm(dim=-1).sum().cpu())
        return native + residual.to(native.dtype)


class NemotronSuperRevisionModel(nn.Module):
    """Freeze the 120B host and attach one residual to each final-16 MoE mixer."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        try:
            native_surface = validate_loaded_surface(backbone)
        except Q36UpwardMoEHostError as error:
            raise NemotronSuperRevisionError(str(error)) from error
        if native_surface.get("attachment_surface") != ATTACHMENT_SURFACE:
            raise NemotronSuperRevisionError(
                "Nemotron Super attachment surface differs"
            )
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        layers = self.backbone.model.layers
        blocks: list[PostMixerResidual] = []
        for index in CONTROLLED_LAYER_INDICES:
            layer = layers[index]
            if getattr(layer, "block_type", None) != "moe":
                raise NemotronSuperRevisionError(
                    "Nemotron Super controlled layer is not MoE"
                )
            block = PostMixerResidual(layer.mixer)
            layer.mixer = block
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        if self.trainable_parameter_count() != TRAINABLE_PARAMETERS_PER_ROLE:
            raise NemotronSuperRevisionError(
                "Nemotron Super trainable parameter count differs"
            )
        if any(
            parameter.requires_grad
            for block in self.blocks
            for parameter in block.base.parameters()
        ):
            raise NemotronSuperRevisionError(
                "Nemotron Super native router or expert is trainable"
            )

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
            raise NemotronSuperRevisionError(
                "Nemotron Super trainable parameter names differ"
            )
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
            raise NemotronSuperRevisionError(
                "Nemotron Super trainable state geometry differs"
            )
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
            "native_router_expert_trainables": 0,
            "layers": [block.receipt() for block in self.blocks],
        }
