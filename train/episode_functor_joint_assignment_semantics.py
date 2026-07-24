"""Parameter-free machine-to-physical assignment compatibility.

The existing physical-key nerve constructs behavior signatures for every
source key using the current soft assignment. This module closes the missing
edge in the architecture: it constructs the same signatures directly from
the provisional categorical machine and compares semantic slots with physical
keys. The result can revise binding before the next machine update.

No learned coordinate embeddings, source grammar oracle, query input, host
solver, or runtime autograd are used here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from episode_functor_constrained_transport import (
    PRIMARY_ACTIONS,
    PRIMARY_ANSWERS,
    PRIMARY_KEYS,
    PRIMARY_OBSERVERS,
    PRIMARY_STATES,
)
from episode_functor_physical_key_nerve import PhysicalKeyNerveResult


JOINT_COMPATIBILITY_MODES = frozenset(
    {
        "causal",
        "machine-to-assignment-cut",
        "sign-reversed",
        "one-step-only",
    }
)
UNAVAILABLE_COMPATIBILITY = -2.0


class JointAssignmentSemanticsError(ValueError):
    """Joint assignment-semantics geometry or values failed closed."""


@dataclass(frozen=True, slots=True)
class JointSemanticCompatibility:
    """Machine signatures and their compatibility with physical keys."""

    machine_state_signature: torch.Tensor
    machine_action_signature: torch.Tensor
    machine_observer_signature: torch.Tensor
    state_compatibility: torch.Tensor
    action_compatibility: torch.Tensor
    observer_compatibility: torch.Tensor
    assignment_compatibility: torch.Tensor
    mode: str

    def __post_init__(self) -> None:
        batch = int(self.machine_state_signature.shape[0])
        unique = int(self.state_compatibility.shape[-1])
        if (
            self.machine_state_signature.shape
            != (batch, PRIMARY_STATES, 104)
            or self.machine_action_signature.shape
            != (batch, PRIMARY_ACTIONS, 512)
            or self.machine_observer_signature.shape
            != (batch, PRIMARY_OBSERVERS, 32)
            or self.state_compatibility.shape
            != (batch, PRIMARY_STATES, unique)
            or self.action_compatibility.shape
            != (batch, PRIMARY_ACTIONS, unique)
            or self.observer_compatibility.shape
            != (batch, PRIMARY_OBSERVERS, unique)
            or self.assignment_compatibility.shape
            != (
                batch,
                PRIMARY_STATES + PRIMARY_ACTIONS + PRIMARY_OBSERVERS,
                unique,
            )
            or self.mode not in JOINT_COMPATIBILITY_MODES
        ):
            raise JointAssignmentSemanticsError(
                "joint semantic compatibility geometry differs"
            )
        values = (
            self.machine_state_signature,
            self.machine_action_signature,
            self.machine_observer_signature,
            self.state_compatibility,
            self.action_compatibility,
            self.observer_compatibility,
            self.assignment_compatibility,
        )
        if any(
            not value.is_floating_point()
            or not bool(torch.isfinite(value).all())
            for value in values
        ):
            raise JointAssignmentSemanticsError(
                "joint semantic compatibility values differ"
            )


def _normalize_probabilities(
    values: torch.Tensor,
    *,
    label: str,
) -> torch.Tensor:
    if (
        not values.is_floating_point()
        or not bool(torch.isfinite(values).all())
        or bool(values.lt(0).any())
    ):
        raise JointAssignmentSemanticsError(
            f"{label} probabilities differ"
        )
    total = values.sum(-1, keepdim=True)
    if bool(total.le(0).any()):
        raise JointAssignmentSemanticsError(
            f"{label} probability rows are empty"
        )
    return values / total


def machine_behavior_signatures(
    transition: torch.Tensor,
    observer: torch.Tensor,
    *,
    one_step_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct state/action/observer signatures from one soft machine."""

    if (
        transition.ndim != 4
        or transition.shape[1:]
        != (PRIMARY_ACTIONS, PRIMARY_STATES, PRIMARY_STATES)
        or observer.shape
        != (
            transition.shape[0],
            PRIMARY_OBSERVERS,
            PRIMARY_STATES,
            PRIMARY_ANSWERS,
        )
        or observer.device != transition.device
    ):
        raise JointAssignmentSemanticsError(
            "machine probability geometry differs"
        )
    transition = _normalize_probabilities(
        transition.float(),
        label="transition",
    )
    observer = _normalize_probabilities(
        observer.float(),
        label="observer",
    )

    transition_observer = torch.einsum(
        "baij,bojy->biaoy",
        transition,
        observer,
    )
    two_step = torch.einsum(
        "baij,bcjk->biack",
        transition,
        transition,
    )
    ordered_path = torch.einsum(
        "baij,bcjk->bacik",
        transition,
        transition,
    )
    if one_step_only:
        two_step = two_step * 0.0
        ordered_path = ordered_path * 0.0

    state_signature = torch.cat(
        (
            observer.permute(0, 2, 1, 3).flatten(2),
            transition_observer.flatten(2),
            two_step.flatten(2),
        ),
        dim=-1,
    )
    action_signature = torch.cat(
        (
            transition.flatten(2),
            transition_observer.permute(0, 2, 1, 3, 4).flatten(2),
            ordered_path.flatten(2),
            ordered_path.transpose(1, 2).flatten(2),
        ),
        dim=-1,
    )
    observer_signature = observer.flatten(2)
    if (
        state_signature.shape[2] != 104
        or action_signature.shape[2] != 512
        or observer_signature.shape[2] != 32
    ):
        raise JointAssignmentSemanticsError(
            "machine behavior signature geometry differs"
        )
    return state_signature, action_signature, observer_signature


def _negative_jsd(
    semantic: torch.Tensor,
    physical: torch.Tensor,
) -> torch.Tensor:
    """Compare nonnegative signatures with a zero-safe Jensen-Shannon score."""

    if (
        semantic.ndim != 3
        or physical.ndim != 3
        or semantic.shape[0] != physical.shape[0]
        or semantic.shape[-1] != physical.shape[-1]
        or not semantic.is_floating_point()
        or not physical.is_floating_point()
        or not bool(torch.isfinite(semantic).all())
        or not bool(torch.isfinite(physical).all())
        or bool(semantic.lt(0).any())
        or bool(physical.lt(0).any())
    ):
        raise JointAssignmentSemanticsError(
            "semantic and physical signatures differ"
        )
    left_total = semantic.sum(-1, keepdim=True)
    right_total = physical.sum(-1, keepdim=True)
    left = semantic / left_total.clamp_min(
        torch.finfo(semantic.dtype).tiny
    )
    right = physical / right_total.clamp_min(
        torch.finfo(physical.dtype).tiny
    )
    left = torch.where(left_total.gt(0), left, torch.zeros_like(left))
    right = torch.where(right_total.gt(0), right, torch.zeros_like(right))
    left = left[:, :, None]
    right = right[:, None]
    midpoint = 0.5 * (left + right)
    tiny = torch.finfo(midpoint.dtype).tiny
    divergence = 0.5 * (
        left
        * (
            left.clamp_min(tiny).log()
            - midpoint.clamp_min(tiny).log()
        )
        + right
        * (
            right.clamp_min(tiny).log()
            - midpoint.clamp_min(tiny).log()
        )
    ).sum(-1)
    both_empty = left_total[:, :, None, 0].eq(0) & right_total[
        :, None, :, 0
    ].eq(0)
    return torch.where(
        both_empty,
        torch.zeros_like(divergence),
        -divergence,
    )


def _grouped_negative_jsd(
    semantic: torch.Tensor,
    physical: torch.Tensor,
    groups: tuple[int, ...],
) -> torch.Tensor:
    if sum(groups) != semantic.shape[-1] or any(
        width < 1 for width in groups
    ):
        raise JointAssignmentSemanticsError(
            "semantic signature groups differ"
        )
    scores: list[torch.Tensor] = []
    start = 0
    for width in groups:
        end = start + width
        scores.append(
            _negative_jsd(
                semantic[..., start:end],
                physical[..., start:end],
            )
        )
        start = end
    return torch.stack(scores, dim=-1).mean(-1)


def _evidence_mask(
    physical: torch.Tensor,
    key_valid: torch.Tensor,
) -> torch.Tensor:
    return key_valid & physical.sum(-1).gt(0)


def joint_semantic_compatibility(
    nerve: PhysicalKeyNerveResult,
    transition: torch.Tensor,
    observer: torch.Tensor,
    key_valid: torch.Tensor,
    *,
    mode: str = "causal",
) -> JointSemanticCompatibility:
    """Compare a provisional machine with physical-key path behavior."""

    if mode not in JOINT_COMPATIBILITY_MODES:
        raise JointAssignmentSemanticsError(
            f"unknown joint compatibility mode: {mode}"
        )
    batch, unique = nerve.state_signature.shape[:2]
    if (
        key_valid.shape != (batch, unique)
        or key_valid.dtype != torch.bool
        or key_valid.device != nerve.state_signature.device
        or transition.device != key_valid.device
        or observer.device != key_valid.device
        or bool(key_valid.sum(-1).ne(PRIMARY_KEYS).any())
    ):
        raise JointAssignmentSemanticsError(
            "joint compatibility key geometry differs"
        )
    one_step_only = mode == "one-step-only"
    machine_state, machine_action, machine_observer = (
        machine_behavior_signatures(
            transition,
            observer,
            one_step_only=one_step_only,
        )
    )
    physical_state = nerve.state_signature.float()
    physical_action = nerve.action_signature.float()
    physical_observer = nerve.observer_signature.float()
    if one_step_only:
        physical_state = physical_state.clone()
        physical_state[..., 32:] = 0.0
        physical_action = physical_action.clone()
        physical_action[..., 128:] = 0.0
    state_evidence = _evidence_mask(physical_state, key_valid)
    action_evidence = _evidence_mask(physical_action, key_valid)
    observer_evidence = _evidence_mask(physical_observer, key_valid)
    if (
        bool(state_evidence.sum(-1).lt(PRIMARY_STATES).any())
        or bool(action_evidence.sum(-1).lt(PRIMARY_ACTIONS).any())
        or bool(
            observer_evidence.sum(-1).lt(PRIMARY_OBSERVERS).any()
        )
    ):
        raise JointAssignmentSemanticsError(
            "joint compatibility lacks physical role evidence"
        )
    state = _grouped_negative_jsd(
        machine_state,
        physical_state,
        (8, 24, 72),
    )
    action = _grouped_negative_jsd(
        machine_action,
        physical_action,
        (64, 64, 192, 192),
    )
    observer_compatibility = _grouped_negative_jsd(
        machine_observer,
        physical_observer,
        (32,),
    )
    if mode == "machine-to-assignment-cut":
        state = state * 0.0
        action = action * 0.0
        observer_compatibility = observer_compatibility * 0.0
        machine_state = machine_state * 0.0
        machine_action = machine_action * 0.0
        machine_observer = machine_observer * 0.0
    elif mode == "sign-reversed":
        state = -state
        action = -action
        observer_compatibility = -observer_compatibility
    state = state.masked_fill(
        ~state_evidence[:, None],
        UNAVAILABLE_COMPATIBILITY,
    )
    action = action.masked_fill(
        ~action_evidence[:, None],
        UNAVAILABLE_COMPATIBILITY,
    )
    observer_compatibility = observer_compatibility.masked_fill(
        ~observer_evidence[:, None],
        UNAVAILABLE_COMPATIBILITY,
    )
    assignment = torch.cat(
        (state, action, observer_compatibility),
        dim=1,
    )
    return JointSemanticCompatibility(
        machine_state_signature=machine_state,
        machine_action_signature=machine_action,
        machine_observer_signature=machine_observer,
        state_compatibility=state,
        action_compatibility=action,
        observer_compatibility=observer_compatibility,
        assignment_compatibility=assignment,
        mode=mode,
    )


__all__ = [
    "JOINT_COMPATIBILITY_MODES",
    "UNAVAILABLE_COMPATIBILITY",
    "JointAssignmentSemanticsError",
    "JointSemanticCompatibility",
    "joint_semantic_compatibility",
    "machine_behavior_signatures",
]
