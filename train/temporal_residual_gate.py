"""Tokenwise gating between frozen owner and revision residual trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

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

    def forward(
        self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        native = self.base(hidden_states, *args, **kwargs)
        if not isinstance(native, torch.Tensor) or native.shape != hidden_states.shape:
            raise TemporalResidualGateError("native temporal block geometry differs")
        owner = self._residual(hidden_states, self.owner_a, self.owner_b)
        revision = self._residual(hidden_states, self.revision_a, self.revision_b)
        gate = self._gate(hidden_states)
        mixed = owner + gate * (revision - owner)
        with torch.no_grad():
            tokens = int(native.numel() // native.shape[-1])
            self._tokens += tokens
            self._gate_sum += float(gate.float().sum().cpu())
            self._owner_norm += float(owner.float().norm(dim=-1).sum().cpu())
            self._revision_norm += float(revision.float().norm(dim=-1).sum().cpu())
        return native + mixed.to(native.dtype)


__all__ = [
    "TemporalResidualGate",
    "TemporalResidualGateConfig",
    "TemporalResidualGateError",
]
