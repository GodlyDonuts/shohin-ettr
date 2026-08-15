"""Tokenwise gating between frozen owner and revision residual trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalResidualGateError(RuntimeError):
    """The temporal residual gate geometry or execution differs."""


@dataclass(frozen=True)
class TemporalResidualGateConfig:
    hidden_size: int
    rank: int
    alpha: float
    initial_revision_weight: float = 0.1

    def validate(self) -> None:
        if (
            self.hidden_size <= 0
            or self.rank <= 0
            or not math.isfinite(self.alpha)
            or self.alpha <= 0
            or not math.isfinite(self.initial_revision_weight)
            or not 0.0 < self.initial_revision_weight < 1.0
        ):
            raise TemporalResidualGateError("temporal residual gate config differs")


class TemporalResidualGate(nn.Module):
    """Frozen MLP plus frozen owner/reviser residuals and one learned token gate.

    The gate is initialized to a constant output-interpolation weight. Only
    the hidden-to-scalar gate is trainable; the native block and both
    trajectory residual branches remain frozen. This is deliberately not
    claimed to be numerically identical to factor-weight checkpoint
    interpolation: mixing both low-rank factors introduces cross terms. The
    new surface instead gives a clean owner/reviser output continuum while
    permitting token- and layer-specific revision strength after training.
    """

    def __init__(
        self,
        base: nn.Module,
        config: TemporalResidualGateConfig,
        *,
        owner_a: torch.Tensor,
        owner_b: torch.Tensor,
        revision_a: torch.Tensor,
        revision_b: torch.Tensor,
    ) -> None:
        super().__init__()
        config.validate()
        expected = {
            "owner_a": (config.rank, config.hidden_size),
            "owner_b": (config.hidden_size, config.rank),
            "revision_a": (config.rank, config.hidden_size),
            "revision_b": (config.hidden_size, config.rank),
        }
        values = {
            "owner_a": owner_a,
            "owner_b": owner_b,
            "revision_a": revision_a,
            "revision_b": revision_b,
        }
        if any(
            not isinstance(values[name], torch.Tensor)
            or tuple(values[name].shape) != shape
            or not values[name].dtype.is_floating_point
            for name, shape in expected.items()
        ):
            raise TemporalResidualGateError("temporal residual branch geometry differs")

        self.base = base
        self.base.requires_grad_(False)
        self.config = config
        try:
            device = next(base.parameters()).device
        except StopIteration:
            device = owner_a.device
        for name, value in values.items():
            self.register_buffer(
                name,
                value.detach().to(device=device, dtype=torch.float32).clone(),
                persistent=True,
            )
        self.gate_weight = nn.Parameter(
            torch.zeros((1, config.hidden_size), device=device, dtype=torch.float32)
        )
        initial_logit = math.log(
            config.initial_revision_weight / (1.0 - config.initial_revision_weight)
        )
        self.gate_bias = nn.Parameter(
            torch.tensor([initial_logit], device=device, dtype=torch.float32)
        )
        self.scale = config.alpha / config.rank
        self.reset_receipt()

    def reset_receipt(self) -> None:
        self._tokens = 0
        self._gate_sum = 0.0
        self._owner_norm = 0.0
        self._revision_norm = 0.0
        self._last_gate: torch.Tensor | None = None

    def receipt(self) -> dict[str, float | int]:
        if self._tokens == 0:
            return {"tokens": 0}
        return {
            "tokens": self._tokens,
            "mean_revision_weight": self._gate_sum / self._tokens,
            "mean_owner_residual_norm": self._owner_norm / self._tokens,
            "mean_revision_residual_norm": self._revision_norm / self._tokens,
        }

    def trainable_parameter_count(self) -> int:
        return self.gate_weight.numel() + self.gate_bias.numel()

    def _residual(
        self, hidden_states: torch.Tensor, a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        if hidden_states.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return F.linear(F.linear(hidden_states, a), b) * self.scale
        return F.linear(F.linear(hidden_states.float(), a), b) * self.scale

    def _gate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return torch.sigmoid(
                    F.linear(hidden_states, self.gate_weight, self.gate_bias)
                )
        return torch.sigmoid(
            F.linear(hidden_states.float(), self.gate_weight, self.gate_bias)
        )

    def routing_supervision_loss(
        self, target: float, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        gate = self._last_gate
        if (
            gate is None
            or not math.isfinite(target)
            or not 0.0 <= target <= 1.0
            or attention_mask.shape != gate.shape[:-1]
            or attention_mask.device != gate.device
            or not attention_mask.bool().any()
        ):
            raise TemporalResidualGateError("temporal routing supervision differs")
        losses = F.binary_cross_entropy(
            gate.float(), torch.full_like(gate.float(), target), reduction="none"
        ).squeeze(-1)
        return losses.masked_select(attention_mask.bool()).mean()

    def forward(
        self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        native = self.base(hidden_states, *args, **kwargs)
        if not isinstance(native, torch.Tensor) or native.shape != hidden_states.shape:
            raise TemporalResidualGateError("native temporal block geometry differs")
        owner = self._residual(hidden_states, self.owner_a, self.owner_b)
        revision = self._residual(hidden_states, self.revision_a, self.revision_b)
        gate = self._gate(hidden_states)
        self._last_gate = gate
        mixed = owner + gate * (revision - owner)
        with torch.no_grad():
            tokens = int(native.numel() // native.shape[-1])
            self._tokens += tokens
            self._gate_sum += float(gate.float().sum().cpu())
            self._owner_norm += float(owner.float().norm(dim=-1).sum().cpu())
            self._revision_norm += float(revision.float().norm(dim=-1).sum().cpu())
        return native + mixed.to(native.dtype)


_STATE_NAME = re.compile(
    r"^backbone\.model\.layers\.(?P<layer>[0-9]+)\.mlp\.adapter_(?P<factor>[ab])\.weight$"
)


def temporal_branch_layers(
    owner_state: Mapping[str, torch.Tensor],
    revision_state: Mapping[str, torch.Tensor],
    config: TemporalResidualGateConfig,
    controlled_layer_indices: Sequence[int],
) -> dict[int, dict[str, torch.Tensor]]:
    """Validate and align two real Q36 role states by decoder layer."""

    config.validate()
    indices = tuple(controlled_layer_indices)
    if (
        not indices
        or len(indices) != len(set(indices))
        or tuple(sorted(indices)) != indices
        or any(
            not isinstance(index, int) or isinstance(index, bool) for index in indices
        )
        or set(owner_state) != set(revision_state)
    ):
        raise TemporalResidualGateError("temporal role state identity differs")
    expected_names = {
        f"backbone.model.layers.{index}.mlp.adapter_{factor}.weight"
        for index in indices
        for factor in ("a", "b")
    }
    if set(owner_state) != expected_names:
        raise TemporalResidualGateError("temporal role state names differ")
    layers: dict[int, dict[str, torch.Tensor]] = {index: {} for index in indices}
    for name in sorted(expected_names):
        match = _STATE_NAME.fullmatch(name)
        if match is None:
            raise TemporalResidualGateError("temporal role state name differs")
        index = int(match.group("layer"))
        factor = match.group("factor")
        expected_shape = (
            (config.rank, config.hidden_size)
            if factor == "a"
            else (config.hidden_size, config.rank)
        )
        for role, state in (("owner", owner_state), ("revision", revision_state)):
            tensor = state[name]
            if (
                not isinstance(tensor, torch.Tensor)
                or tuple(tensor.shape) != expected_shape
                or tensor.dtype != torch.float32
                or not torch.isfinite(tensor).all()
            ):
                raise TemporalResidualGateError(f"temporal {role} role tensor differs")
            layers[index][f"{role}_{factor}"] = tensor
    return layers


def install_temporal_residual_gates(
    text_model: nn.Module,
    owner_state: Mapping[str, torch.Tensor],
    revision_state: Mapping[str, torch.Tensor],
    config: TemporalResidualGateConfig,
    controlled_layer_indices: Sequence[int],
) -> tuple[TemporalResidualGate, ...]:
    """Replace exact decoder MLPs with tokenwise temporal residual gates."""

    model_layers = getattr(text_model, "layers", None)
    if not isinstance(model_layers, nn.ModuleList):
        raise TemporalResidualGateError("temporal decoder layers differ")
    text_model.requires_grad_(False)
    indices = tuple(controlled_layer_indices)
    if indices != tuple(range(len(model_layers) - len(indices), len(model_layers))):
        raise TemporalResidualGateError("temporal controlled layers differ")
    branches = temporal_branch_layers(
        owner_state, revision_state, config, controlled_layer_indices
    )
    installed = []
    for index in indices:
        layer = model_layers[index]
        mlp = getattr(layer, "mlp", None)
        if not isinstance(mlp, nn.Module) or isinstance(mlp, TemporalResidualGate):
            raise TemporalResidualGateError("temporal native MLP differs")
        block = TemporalResidualGate(mlp, config, **branches[index])
        layer.mlp = block
        installed.append(block)
    expected_trainables = len(indices) * (config.hidden_size + 1)
    if (
        sum(block.trainable_parameter_count() for block in installed)
        != expected_trainables
    ):
        raise TemporalResidualGateError("temporal gate parameter count differs")
    return tuple(installed)


class TemporalGatedProductModel(nn.Module):
    """Drop-in Q36 training/generation surface with installed temporal gates."""

    def __init__(
        self,
        backbone: nn.Module,
        text_model: nn.Module,
        lm_head: nn.Module,
        config: TemporalResidualGateConfig,
        *,
        owner_state: Mapping[str, torch.Tensor],
        revision_state: Mapping[str, torch.Tensor],
        controlled_layer_indices: Sequence[int],
    ) -> None:
        super().__init__()
        if not all(
            isinstance(module, nn.Module) for module in (backbone, text_model, lm_head)
        ):
            raise TemporalResidualGateError("temporal product modules differ")
        backbone.requires_grad_(False)
        lm_head.requires_grad_(False)
        self.backbone = backbone
        self.text_model = text_model
        self.lm_head = lm_head
        self.config = config
        self.controlled_layer_indices = tuple(controlled_layer_indices)
        self.blocks = nn.ModuleList(
            install_temporal_residual_gates(
                text_model,
                owner_state,
                revision_state,
                config,
                self.controlled_layer_indices,
            )
        )
        self._generation_prompt_attention: torch.Tensor | None = None
        self._generation_position_ids: torch.Tensor | None = None
        self._generation_prompt_ids: torch.Tensor | None = None

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
        if not names:
            raise TemporalResidualGateError("temporal gate trainable names are absent")
        return hashlib.sha256("\n".join(names).encode()).hexdigest()

    def reset_routing_receipt(self) -> None:
        for block in self.blocks:
            block.reset_receipt()

    def routing_receipt(self) -> dict[str, Any]:
        return {
            "controlled_layer_indices": list(self.controlled_layer_indices),
            "layers": [block.receipt() for block in self.blocks],
        }

    def routing_supervision_loss(
        self, target: float, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if not self.blocks:
            raise TemporalResidualGateError("temporal routing blocks are absent")
        return torch.stack(
            [
                block.routing_supervision_loss(target, attention_mask)
                for block in self.blocks
            ]
        ).mean()

    def prepare_generation_draft_attention(
        self,
        tokenizer: Any,
        rendered: list[str],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        if (
            input_ids.ndim != 2
            or attention_mask.shape != input_ids.shape
            or input_ids.device != attention_mask.device
            or len(rendered) != input_ids.shape[0]
        ):
            raise TemporalResidualGateError("temporal generation prompt differs")
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(~attention_mask.bool(), 0)
        self._generation_position_ids = position_ids
        self._generation_prompt_ids = input_ids.detach().clone()
        self._generation_prompt_attention = attention_mask.detach().clone()

    def generation_position_ids(self) -> torch.Tensor:
        if self._generation_position_ids is None:
            raise TemporalResidualGateError("temporal generation positions are absent")
        return self._generation_position_ids

    def generation_embeddings(
        self, prompt_ids: torch.Tensor, prompt_attention: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention = self._generation_prompt_attention
        prompt_ids_receipt = self._generation_prompt_ids
        embed_tokens = getattr(self.text_model, "embed_tokens", None)
        if (
            attention is None
            or attention.shape != prompt_attention.shape
            or prompt_ids_receipt is None
            or prompt_ids_receipt.shape != prompt_ids.shape
            or not torch.equal(prompt_ids_receipt, prompt_ids)
            or not isinstance(embed_tokens, nn.Module)
        ):
            raise TemporalResidualGateError("temporal generation state is absent")
        return embed_tokens(prompt_ids), attention


__all__ = [
    "TemporalResidualGate",
    "TemporalResidualGateConfig",
    "TemporalResidualGateError",
    "TemporalGatedProductModel",
    "install_temporal_residual_gates",
    "temporal_branch_layers",
]
