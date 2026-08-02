"""Exact-tensor interface sufficiency preflight for the capability floor.

The historical symbolic oracle consumed resolved parser/state factors that the
neural arm did not receive in equivalent form.  This module makes that
comparison explicit.  The probe below accepts only the projected token
features, public source-span masks, and typed state tensors admitted by the
unified trajectory.  Receipts bind every input tensor and refuse promotion
when symbolic, tensor, renderer-orbit, or negative-control gates disagree.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping

import torch
import torch.nn as nn

from capability_floor_trajectory import (
    UnifiedStateEncoder,
    UnifiedTrajectoryConfig,
    UnifiedTypedState,
    validate_unified_state,
)


SUFFICIENCY_SCHEMA = "shohin-ettr-exact-tensor-sufficiency-v1"
FAMILY_NAMES = ("NONE", "WRITE", "LINK")


class TensorSufficiencyError(ValueError):
    """The tensor probe or its evidence receipt is invalid."""


@dataclass(frozen=True, slots=True)
class SufficiencyScores:
    symbolic_reference_accuracy: float
    tensor_probe_accuracy: float
    renderer_orbit_accuracy: float
    renderer_orbit_prediction_agreement: float
    binding_deranged_accuracy: float
    state_value_permuted_accuracy: float
    empirical_chance_accuracy: float

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, float) or not 0.0 <= value <= 1.0:
                raise TensorSufficiencyError(f"{name} must be a unit-interval float")


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash dtype, shape, and exact contiguous CPU bytes."""

    if not isinstance(tensor, torch.Tensor):
        raise TensorSufficiencyError("tensor receipt member differs")
    value = tensor.detach().contiguous().cpu()
    header = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    raw = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(header + b"\n" + raw).hexdigest()


def state_tensor_hashes(state: UnifiedTypedState) -> dict[str, str]:
    return {
        name: tensor_sha256(getattr(state, name))
        for name in (
            "value_probabilities",
            "type_probabilities",
            "relations",
            "active",
            "root",
            "committed",
        )
    }


class OperationFamilyTensorProbe(nn.Module):
    """Probe operation family from exactly the tensors exposed to ETTR.

    ``role_masks`` are public source-span memberships derived from the shared
    canonical renderer.  They are not assessor labels.  Every semantic role
    reads projected token features, attends the allowed typed state, and uses
    a multiplicative role/state binding before the three-way family head.
    """

    def __init__(
        self,
        config: UnifiedTrajectoryConfig,
        *,
        max_roles: int = 4,
    ):
        super().__init__()
        config.validate()
        if max_roles <= 0:
            raise TensorSufficiencyError("max_roles must be positive")
        self.config = config
        self.max_roles = int(max_roles)
        width = config.state_width
        self.source_projection = nn.Linear(config.input_width, width, bias=False)
        self.role_embedding = nn.Parameter(torch.empty(max_roles, width))
        self.state_encoder = UnifiedStateEncoder(config)
        self.role_query = nn.Linear(width, width, bias=False)
        self.state_key = nn.Linear(width, width, bias=False)
        self.state_value = nn.Linear(width, width, bias=False)
        self.fusion = nn.Sequential(
            nn.Linear(3 * width, width),
            nn.GELU(),
            nn.RMSNorm(width),
        )
        self.family_head = nn.Linear(width, len(FAMILY_NAMES))
        nn.init.normal_(self.role_embedding, std=0.02)

    def forward(
        self,
        source_features: torch.Tensor,
        source_mask: torch.Tensor,
        role_masks: torch.Tensor,
        state: UnifiedTypedState,
    ) -> torch.Tensor:
        validate_unified_state(state, self.config)
        if source_features.ndim != 3:
            raise TensorSufficiencyError("source feature rank differs")
        batch, tokens, width = source_features.shape
        if (
            width != self.config.input_width
            or source_mask.shape != (batch, tokens)
            or source_mask.dtype != torch.bool
            or role_masks.ndim != 3
            or role_masks.shape[0] != batch
            or role_masks.shape[2] != tokens
            or role_masks.shape[1] > self.max_roles
            or role_masks.dtype != torch.bool
            or state.active.shape[0] != batch
        ):
            raise TensorSufficiencyError("exact tensor interface geometry differs")
        if (role_masks & ~source_mask[:, None, :]).any():
            raise TensorSufficiencyError("role mask selects padded source tokens")
        role_present = role_masks.any(-1)
        weights = role_masks.to(source_features.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1.0)
        source = self.source_projection(source_features)
        roles = torch.einsum("brt,btw->brw", weights, source)
        roles = roles + self.role_embedding[: role_masks.shape[1]].unsqueeze(0)

        state_slots = self.state_encoder(state)
        queries = self.role_query(roles)
        keys = self.state_key(state_slots)
        attention = torch.einsum("brw,bsw->brs", queries, keys)
        attention = attention / self.config.state_width**0.5
        active = state.active.ge(0.5)
        fallback = ~active.any(-1)
        active = active.clone()
        if fallback.any():
            active[fallback, 0] = True
        attention = attention.masked_fill(
            ~active[:, None, :],
            torch.finfo(attention.dtype).min,
        ).softmax(-1)
        state_read = torch.einsum(
            "brs,bsw->brw",
            attention,
            self.state_value(state_slots),
        )
        hidden = self.fusion(torch.cat((roles, state_read, roles * state_read), -1))
        logits = self.family_head(hidden)
        return logits.masked_fill(~role_present.unsqueeze(-1), 0.0)


def sufficiency_decision(
    scores: SufficiencyScores,
    *,
    strict_threshold: float = 0.95,
    negative_slack: float = 0.02,
) -> str:
    scores.validate()
    if not 0.0 < strict_threshold <= 1.0 or not 0.0 <= negative_slack < 1.0:
        raise TensorSufficiencyError("sufficiency thresholds differ")
    if scores.symbolic_reference_accuracy < strict_threshold:
        return "reject-symbolic-reference-or-corpus"
    if scores.tensor_probe_accuracy < strict_threshold:
        return "redesign-neural-interface"
    if (
        scores.renderer_orbit_accuracy < strict_threshold
        or scores.renderer_orbit_prediction_agreement < strict_threshold
    ):
        return "reject-renderer-instability"
    negative_ceiling = scores.empirical_chance_accuracy + negative_slack
    if (
        scores.binding_deranged_accuracy > negative_ceiling
        or scores.state_value_permuted_accuracy > negative_ceiling
    ):
        return "reject-binding-control-leakage"
    return "pass-interface-sufficiency"


def build_sufficiency_receipt(
    *,
    candidate: str,
    component: str,
    split_sha256: str,
    source_features: torch.Tensor,
    source_mask: torch.Tensor,
    role_masks: torch.Tensor,
    state: UnifiedTypedState,
    labels: torch.Tensor,
    scores: SufficiencyScores,
) -> dict[str, object]:
    if not candidate or not component:
        raise TensorSufficiencyError("candidate and component are required")
    if len(split_sha256) != 64:
        raise TensorSufficiencyError("split digest differs")
    scores.validate()
    decision = sufficiency_decision(scores)
    return {
        "assessor_features_available_at_inference": False,
        "candidate": candidate,
        "component": component,
        "decision": decision,
        "input_hashes": {
            "labels": tensor_sha256(labels),
            "role_masks": tensor_sha256(role_masks),
            "source_features": tensor_sha256(source_features),
            "source_mask": tensor_sha256(source_mask),
            "state": state_tensor_hashes(state),
        },
        "negative_control_slack": 0.02,
        "schema": SUFFICIENCY_SCHEMA,
        "scores": asdict(scores),
        "split_sha256": split_sha256,
        "strict_threshold": 0.95,
    }


def validate_sufficiency_receipt(receipt: Mapping[str, object]) -> None:
    if (
        receipt.get("schema") != SUFFICIENCY_SCHEMA
        or receipt.get("assessor_features_available_at_inference") is not False
        or receipt.get("strict_threshold") != 0.95
        or receipt.get("negative_control_slack") != 0.02
    ):
        raise TensorSufficiencyError("sufficiency custody differs")
    scores_payload = receipt.get("scores")
    if not isinstance(scores_payload, Mapping):
        raise TensorSufficiencyError("sufficiency scores differ")
    try:
        scores = SufficiencyScores(**dict(scores_payload))
    except TypeError as error:
        raise TensorSufficiencyError("sufficiency scores differ") from error
    if receipt.get("decision") != sufficiency_decision(scores):
        raise TensorSufficiencyError("sufficiency decision differs")
    input_hashes = receipt.get("input_hashes")
    if not isinstance(input_hashes, Mapping) or set(input_hashes) != {
        "labels",
        "role_masks",
        "source_features",
        "source_mask",
        "state",
    }:
        raise TensorSufficiencyError("sufficiency input receipt differs")
