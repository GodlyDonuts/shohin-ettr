"""Exact temporal causal-gate models for the upward MoE transfer hosts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Mapping

import torch
import torch.nn as nn

from nemotron_super_post_mixer_revision import (
    ATTACHMENT_SURFACE as NEMOTRON_ATTACHMENT_SURFACE,
)
from q36_upward_moe_host import (
    ALPHA as NEMOTRON_ALPHA,
    CONTROLLED_LAYER_INDICES as NEMOTRON_CONTROLLED_LAYER_INDICES,
    HIDDEN_SIZE as NEMOTRON_HIDDEN_SIZE,
    RANK as NEMOTRON_RANK,
    validate_loaded_surface as validate_nemotron_surface,
)
from q36_upward_moe_mixtral_host import (
    ALPHA as MIXTRAL_ALPHA,
    ATTACHMENT_SURFACE as MIXTRAL_ATTACHMENT_SURFACE,
    CONTROLLED_LAYER_INDICES as MIXTRAL_CONTROLLED_LAYER_INDICES,
    HIDDEN_SIZE as MIXTRAL_HIDDEN_SIZE,
    RANK as MIXTRAL_RANK,
    validate_loaded_surface as validate_mixtral_surface,
)
from temporal_residual_gate import (
    TemporalResidualGateConfig,
    TemporalResidualGateError,
    install_temporal_residual_gates,
)

NEMOTRON_ARCHITECTURE = "shohin-nemotron-super-temporal-causal-gate-v1"
MIXTRAL_ARCHITECTURE = "shohin-mixtral-8x22b-temporal-causal-gate-v1"


class UpwardMoETemporalGateError(RuntimeError):
    """The upward MoE temporal-gate host, lineage, or state differs."""


@dataclass(frozen=True)
class UpwardMoETemporalGateSpec:
    host: str
    architecture: str
    attachment_surface: str
    module_attribute: str
    hidden_size: int
    rank: int
    alpha: float
    controlled_layer_indices: tuple[int, ...]
    require_final_contiguous: bool

    @property
    def gate_trainable_parameters(self) -> int:
        return len(self.controlled_layer_indices) * (self.hidden_size + 1)

    def receipt(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "architecture": self.architecture,
            "attachment_surface": self.attachment_surface,
            "module_attribute": self.module_attribute,
            "hidden_size": self.hidden_size,
            "rank": self.rank,
            "alpha": self.alpha,
            "controlled_layer_indices": list(self.controlled_layer_indices),
            "require_final_contiguous": self.require_final_contiguous,
            "gate_trainable_parameters": self.gate_trainable_parameters,
            "native_router_expert_trainables": 0,
        }


NEMOTRON_SPEC = UpwardMoETemporalGateSpec(
    host="Nemotron-Super-120B-A12B",
    architecture=NEMOTRON_ARCHITECTURE,
    attachment_surface=NEMOTRON_ATTACHMENT_SURFACE,
    module_attribute="mixer",
    hidden_size=NEMOTRON_HIDDEN_SIZE,
    rank=NEMOTRON_RANK,
    alpha=NEMOTRON_ALPHA,
    controlled_layer_indices=tuple(NEMOTRON_CONTROLLED_LAYER_INDICES),
    require_final_contiguous=False,
)

MIXTRAL_SPEC = UpwardMoETemporalGateSpec(
    host="Mixtral-8x22B-141B-A39B",
    architecture=MIXTRAL_ARCHITECTURE,
    attachment_surface=MIXTRAL_ATTACHMENT_SURFACE,
    module_attribute="mlp",
    hidden_size=MIXTRAL_HIDDEN_SIZE,
    rank=MIXTRAL_RANK,
    alpha=MIXTRAL_ALPHA,
    controlled_layer_indices=tuple(MIXTRAL_CONTROLLED_LAYER_INDICES),
    require_final_contiguous=True,
)


def static_transfer_contract() -> dict[str, Any]:
    return {
        "source_architecture": "q36-tokenwise-temporal-residual-gate-v1",
        "causal_loss_weight": 1.0,
        "routing_supervision_weight": 0.0,
        "frozen_trajectories": ["owner", "aligned_revision"],
        "hosts": [NEMOTRON_SPEC.receipt(), MIXTRAL_SPEC.receipt()],
    }


class _UpwardMoETemporalGateModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        owner_state: Mapping[str, torch.Tensor],
        revision_state: Mapping[str, torch.Tensor],
        *,
        spec: UpwardMoETemporalGateSpec,
        validate_surface: Callable[[Any], dict[str, Any]],
    ) -> None:
        super().__init__()
        try:
            host_receipt = validate_surface(backbone)
        except Exception as error:
            raise UpwardMoETemporalGateError(
                f"{spec.host} temporal host differs"
            ) from error
        if host_receipt.get("attachment_surface") != spec.attachment_surface:
            raise UpwardMoETemporalGateError(
                f"{spec.host} temporal attachment surface differs"
            )
        text_model = getattr(backbone, "model", None)
        if not isinstance(backbone, nn.Module) or not isinstance(text_model, nn.Module):
            raise UpwardMoETemporalGateError(
                f"{spec.host} temporal model surface differs"
            )
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.spec = spec
        config = TemporalResidualGateConfig(
            hidden_size=spec.hidden_size,
            rank=spec.rank,
            alpha=spec.alpha,
            initial_revision_weight=0.1,
        )
        try:
            installed = install_temporal_residual_gates(
                text_model,
                owner_state,
                revision_state,
                config,
                spec.controlled_layer_indices,
                module_attribute=spec.module_attribute,
                require_final_contiguous=spec.require_final_contiguous,
            )
        except TemporalResidualGateError as error:
            raise UpwardMoETemporalGateError(str(error)) from error
        self.blocks = nn.ModuleList(installed)
        if self.trainable_parameter_count() != spec.gate_trainable_parameters:
            raise UpwardMoETemporalGateError(
                f"{spec.host} temporal gate parameter count differs"
            )
        if any(
            parameter.requires_grad
            for block in self.blocks
            for parameter in block.base.parameters()
        ):
            raise UpwardMoETemporalGateError(
                f"{spec.host} native router or expert is trainable"
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
        if len(names) != len(self.spec.controlled_layer_indices) * 2:
            raise UpwardMoETemporalGateError(
                f"{self.spec.host} temporal gate names differ"
            )
        return hashlib.sha256("\n".join(names).encode()).hexdigest()

    def trainable_state(self) -> dict[str, torch.Tensor]:
        state = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        if sum(value.numel() for value in state.values()) != (
            self.spec.gate_trainable_parameters
        ):
            raise UpwardMoETemporalGateError(
                f"{self.spec.host} temporal gate state differs"
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
            **self.spec.receipt(),
            "trainable_parameter_name_sha256": self.trainable_parameter_name_sha256(),
            "native_router_expert_trainables": 0,
            "layers": [block.receipt() for block in self.blocks],
        }


class NemotronSuperTemporalGateModel(_UpwardMoETemporalGateModel):
    def __init__(
        self,
        backbone: nn.Module,
        owner_state: Mapping[str, torch.Tensor],
        revision_state: Mapping[str, torch.Tensor],
    ) -> None:
        super().__init__(
            backbone,
            owner_state,
            revision_state,
            spec=NEMOTRON_SPEC,
            validate_surface=validate_nemotron_surface,
        )


class MixtralTemporalGateModel(_UpwardMoETemporalGateModel):
    def __init__(
        self,
        backbone: nn.Module,
        owner_state: Mapping[str, torch.Tensor],
        revision_state: Mapping[str, torch.Tensor],
    ) -> None:
        super().__init__(
            backbone,
            owner_state,
            revision_state,
            spec=MIXTRAL_SPEC,
            validate_surface=validate_mixtral_surface,
        )


__all__ = [
    "MIXTRAL_ARCHITECTURE",
    "MIXTRAL_SPEC",
    "NEMOTRON_ARCHITECTURE",
    "NEMOTRON_SPEC",
    "MixtralTemporalGateModel",
    "NemotronSuperTemporalGateModel",
    "UpwardMoETemporalGateError",
    "static_transfer_contract",
]
