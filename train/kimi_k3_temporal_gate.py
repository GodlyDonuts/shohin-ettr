"""Frozen-eval temporal causal gate for the pinned Kimi K3 post-MoE surface."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import torch
import torch.nn as nn

from q36_upward_moe_kimi_k3_host import (
    ALPHA,
    ATTACHMENT_SURFACE,
    CONTROLLED_LAYER_INDICES,
    HIDDEN_SIZE,
    MODEL_CONFIG_SHA256,
    MODEL_REVISION,
    NATIVE_EXECUTION_MODE,
    Q36UpwardMoEKimiK3HostError,
    RANK,
    TEMPORAL_GATE_PARAMETERS,
    validate_loaded_surface,
)
from temporal_residual_gate import (
    TemporalResidualGateConfig,
    TemporalResidualGateError,
    install_temporal_residual_gates,
)

ARCHITECTURE = "shohin-kimi-k3-temporal-causal-gate-v1"
ROLE_STATE_PREFIX = "backbone.language_model.model.layers"
GENERIC_STATE_PREFIX = "backbone.model.layers"
MODULE_ATTRIBUTE = "block_sparse_moe"


class KimiK3TemporalGateError(RuntimeError):
    """The Kimi K3 temporal lineage or frozen native execution differs."""


def _translate_role_state(
    state: Mapping[str, torch.Tensor], *, role: str
) -> dict[str, torch.Tensor]:
    expected = {
        f"{ROLE_STATE_PREFIX}.{index}.{MODULE_ATTRIBUTE}.adapter_{factor}.weight"
        for index in CONTROLLED_LAYER_INDICES
        for factor in ("a", "b")
    }
    if set(state) != expected:
        raise KimiK3TemporalGateError(f"Kimi K3 {role} role state names differ")
    translated = {}
    for name, tensor in state.items():
        translated[name.replace(ROLE_STATE_PREFIX, GENERIC_STATE_PREFIX, 1)] = tensor
    return translated


class KimiK3TemporalGateModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        owner_state: Mapping[str, torch.Tensor],
        revision_state: Mapping[str, torch.Tensor],
    ) -> None:
        super().__init__()
        try:
            surface = validate_loaded_surface(backbone)
        except Q36UpwardMoEKimiK3HostError as error:
            raise KimiK3TemporalGateError(str(error)) from error
        if (
            surface.get("attachment_surface") != ATTACHMENT_SURFACE
            or surface.get("native_execution_mode") != NATIVE_EXECUTION_MODE
        ):
            raise KimiK3TemporalGateError("Kimi K3 temporal host surface differs")
        language_model = getattr(backbone, "language_model", None)
        text_model = getattr(language_model, "model", None)
        if not isinstance(backbone, nn.Module) or not isinstance(text_model, nn.Module):
            raise KimiK3TemporalGateError("Kimi K3 temporal text model differs")
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        config = TemporalResidualGateConfig(
            hidden_size=HIDDEN_SIZE,
            rank=RANK,
            alpha=ALPHA,
            initial_revision_weight=0.1,
        )
        try:
            installed = install_temporal_residual_gates(
                text_model,
                _translate_role_state(owner_state, role="owner"),
                _translate_role_state(revision_state, role="revision"),
                config,
                CONTROLLED_LAYER_INDICES,
                module_attribute=MODULE_ATTRIBUTE,
                require_final_contiguous=True,
            )
        except TemporalResidualGateError as error:
            raise KimiK3TemporalGateError(str(error)) from error
        self.blocks = nn.ModuleList(installed)
        self.train(False)
        if self.trainable_parameter_count() != TEMPORAL_GATE_PARAMETERS:
            raise KimiK3TemporalGateError("Kimi K3 temporal parameter count differs")
        if any(
            parameter.requires_grad
            for block in self.blocks
            for parameter in block.base.parameters()
        ):
            raise KimiK3TemporalGateError(
                "Kimi K3 native router or expert is trainable"
            )

    def train(self, mode: bool = True) -> "KimiK3TemporalGateModel":
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        self.backbone.eval()
        for block in self.blocks:
            block.training = mode
            block.base.eval()
        return self

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if any(block.base.training for block in self.blocks):
            raise KimiK3TemporalGateError("Kimi K3 native MoE entered training mode")
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
            raise KimiK3TemporalGateError("Kimi K3 temporal gate names differ")
        return hashlib.sha256("\n".join(names).encode()).hexdigest()

    def trainable_state(self) -> dict[str, torch.Tensor]:
        state = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        if sum(value.numel() for value in state.values()) != TEMPORAL_GATE_PARAMETERS:
            raise KimiK3TemporalGateError("Kimi K3 temporal gate state differs")
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
            "model_revision": MODEL_REVISION,
            "model_config_sha256": MODEL_CONFIG_SHA256,
            "attachment_surface": ATTACHMENT_SURFACE,
            "controlled_layer_indices": list(CONTROLLED_LAYER_INDICES),
            "initial_revision_weight": 0.1,
            "causal_loss_weight": 1.0,
            "routing_supervision_weight": 0.0,
            "frozen_trajectories": ["owner", "aligned_revision"],
            "trainable_parameters": self.trainable_parameter_count(),
            "native_execution_mode": NATIVE_EXECUTION_MODE,
            "native_router_expert_trainables": 0,
            "layers": [block.receipt() for block in self.blocks],
        }
