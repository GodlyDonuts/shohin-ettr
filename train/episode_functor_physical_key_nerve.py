"""Parameter-free physical-key nerve and ordered path signatures.

The source records already share an exact physical-key basis. This module
preserves that basis long enough to form directed one- and two-step path
signatures before anonymous semantic-slot compression. It contains no host
solver, learned coordinate embedding, query input, or runtime autograd.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from episode_functor_constrained_transport import (
    PRIMARY_ACTIONS,
    PRIMARY_ANSWERS,
    PRIMARY_OBSERVERS,
    PRIMARY_STATES,
)
from episode_functor_machine import MAX_ACTIONS, MAX_STATES
from episode_functor_witness_compiler import (
    RECORD_OBSERVATION,
    RECORD_TRANSITION,
    ROLE_ACTION,
    ROLE_OBSERVATION_STATE,
    ROLE_OBSERVER,
    ROLE_TRANSITION_DESTINATION,
    ROLE_TRANSITION_SOURCE,
    WitnessCompilerOutput,
)


NERVE_MODES = frozenset({"causal", "broken-glue", "one-step-only"})


class PhysicalKeyNerveError(ValueError):
    """Physical-key path algebra geometry or value failed closed."""


@dataclass(frozen=True, slots=True)
class PhysicalKeyNerveResult:
    transition_relation: torch.Tensor
    observation_relation: torch.Tensor
    state_signature: torch.Tensor
    action_signature: torch.Tensor
    observer_signature: torch.Tensor
    state_compatibility: torch.Tensor
    action_compatibility: torch.Tensor
    observer_compatibility: torch.Tensor
    action_left_compatibility: torch.Tensor
    action_right_compatibility: torch.Tensor
    action_observer_compatibility: torch.Tensor
    action_commutator_compatibility: torch.Tensor
    path_mass: torch.Tensor
    mode: str

    def __post_init__(self) -> None:
        batch, unique, action_unique, destination = (
            self.transition_relation.shape
        )
        if (
            unique != action_unique
            or unique != destination
            or self.observation_relation.shape
            != (batch, unique, unique, PRIMARY_ANSWERS)
            or self.state_signature.shape[:2] != (batch, unique)
            or self.action_signature.shape[:2] != (batch, unique)
            or self.observer_signature.shape[:2] != (batch, unique)
            or self.state_compatibility.shape
            != (batch, PRIMARY_STATES, unique)
            or self.action_compatibility.shape
            != (batch, PRIMARY_ACTIONS, unique)
            or self.observer_compatibility.shape
            != (batch, PRIMARY_OBSERVERS, unique)
            or self.action_left_compatibility.shape
            != (batch, PRIMARY_ACTIONS, unique)
            or self.action_right_compatibility.shape
            != (batch, PRIMARY_ACTIONS, unique)
            or self.action_observer_compatibility.shape
            != (batch, PRIMARY_ACTIONS, unique)
            or self.action_commutator_compatibility.shape
            != (batch, PRIMARY_ACTIONS, unique)
            or self.path_mass.shape != (batch,)
            or self.mode not in NERVE_MODES
        ):
            raise PhysicalKeyNerveError(
                "physical-key nerve result geometry differs"
            )
        values = (
            self.transition_relation,
            self.observation_relation,
            self.state_signature,
            self.action_signature,
            self.observer_signature,
            self.state_compatibility,
            self.action_compatibility,
            self.observer_compatibility,
            self.action_left_compatibility,
            self.action_right_compatibility,
            self.action_observer_compatibility,
            self.action_commutator_compatibility,
            self.path_mass,
        )
        if any(not bool(torch.isfinite(value).all()) for value in values):
            raise PhysicalKeyNerveError(
                "physical-key nerve result is nonfinite"
            )


def _normalize(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
    total = values.sum(dim, keepdim=True)
    normalized = values / total.clamp_min(
        torch.finfo(values.dtype).tiny
    )
    return torch.where(total.gt(0), normalized, torch.zeros_like(values))


def _compatibility(
    slot_signature: torch.Tensor,
    physical_signature: torch.Tensor,
) -> torch.Tensor:
    left = _normalize(slot_signature.clamp_min(0))
    right = _normalize(physical_signature.clamp_min(0))
    midpoint = 0.5 * (left[:, :, None] + right[:, None])
    tiny = torch.finfo(midpoint.dtype).tiny
    jsd = 0.5 * (
        left[:, :, None]
        * (
            left[:, :, None].clamp_min(tiny).log()
            - midpoint.clamp_min(tiny).log()
        )
        + right[:, None]
        * (
            right[:, None].clamp_min(tiny).log()
            - midpoint.clamp_min(tiny).log()
        )
    ).sum(-1)
    return -jsd


def _signed_compatibility(
    slot_signature: torch.Tensor,
    physical_signature: torch.Tensor,
) -> torch.Tensor:
    left = slot_signature[:, :, None]
    right = physical_signature[:, None]
    scale = (
        1.0
        + left.square().mean(-1, keepdim=True)
        + right.square().mean(-1, keepdim=True)
    )
    return -((left - right).square() / scale).mean(-1)


def _slot_transport(
    witness: WitnessCompilerOutput,
    key_assignment_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = (
        witness.key_assignment_logits
        if key_assignment_logits is None
        else key_assignment_logits
    )
    if (
        selected.shape != witness.key_assignment_logits.shape
        or selected.device != witness.key_assignment_logits.device
        or not selected.is_floating_point()
        or not bool(torch.isfinite(selected).all())
    ):
        raise PhysicalKeyNerveError(
            "physical-key assignment transport differs"
        )
    logits = selected.float()
    valid = witness.unique_key_valid[:, None]
    negative = torch.finfo(logits.dtype).min
    transport = logits.masked_fill(~valid, negative).softmax(-1)
    states = transport[:, :PRIMARY_STATES]
    actions = transport[
        :,
        MAX_STATES : MAX_STATES + PRIMARY_ACTIONS,
    ]
    observers = transport[
        :,
        MAX_STATES
        + MAX_ACTIONS : MAX_STATES
        + MAX_ACTIONS
        + PRIMARY_OBSERVERS,
    ]
    return states, actions, observers


def _off_diagonal_coupling(
    target: torch.Tensor,
    key_valid: torch.Tensor,
    *,
    iterations: int = 256,
) -> torch.Tensor:
    tiny = torch.finfo(target.dtype).tiny
    support = key_valid & target.gt(tiny)
    support_count = support.sum(-1)
    total = target.sum(-1)
    positive = total.gt(tiny)
    impossible = (
        positive
        & (
            support_count.lt(2)
            | target.amax(-1).gt(0.5 * total + 1e-6)
        )
    )
    if bool(impossible.any()):
        raise PhysicalKeyNerveError(
            "broken glue lacks a mass-preserving off-diagonal coupling"
        )
    unique = int(target.shape[-1])
    diagonal = torch.eye(
        unique,
        dtype=torch.bool,
        device=target.device,
    )[None]
    allowed = (
        support[:, :, None]
        & support[:, None, :]
        & ~diagonal
    )
    joint = (
        target[:, :, None]
        * target[:, None, :]
        * allowed.to(target.dtype)
    )
    for _ in range(iterations):
        row_scale = target / joint.sum(-1).clamp_min(tiny)
        joint = joint * row_scale[:, :, None]
        column_scale = target / joint.sum(-2).clamp_min(tiny)
        joint = joint * column_scale[:, None, :]
    joint = joint * allowed.to(joint.dtype)
    if not (
        torch.allclose(
            joint.sum(-1),
            target,
            atol=2e-5,
            rtol=2e-5,
        )
        and torch.allclose(
            joint.sum(-2),
            target,
            atol=2e-5,
            rtol=2e-5,
        )
    ):
        raise PhysicalKeyNerveError(
            "broken glue coupling failed its marginal constraints"
        )
    return joint


def _deranged_contract(
    left: torch.Tensor,
    right: torch.Tensor,
    key_valid: torch.Tensor,
) -> torch.Tensor:
    batch = int(left.shape[0])
    unique = int(left.shape[-1])
    if right.shape[:2] != (batch, unique):
        raise PhysicalKeyNerveError(
            "broken glue intermediate geometry differs"
        )
    left_flat = left.reshape(batch, -1, unique)
    right_flat = right.reshape(batch, unique, -1)
    left_mass = left_flat.sum(1)
    right_mass = right_flat.sum(-1)
    target = left_mass * right_mass
    joint = _off_diagonal_coupling(target, key_valid)
    denominator = left_mass[:, :, None] * right_mass[:, None, :]
    glue = torch.where(
        denominator.gt(0),
        joint / denominator.clamp_min(
            torch.finfo(denominator.dtype).tiny
        ),
        torch.zeros_like(joint),
    )
    contracted = torch.einsum(
        "blm,bmn,bnr->blr",
        left_flat,
        glue,
        right_flat,
    )
    return contracted.reshape(
        left.shape[:-1] + right.shape[2:]
    )


def _physical_paths(
    transition: torch.Tensor,
    observation: torch.Tensor,
    key_valid: torch.Tensor,
    *,
    broken_glue: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    incoming = transition.sum((1, 2))
    outgoing = transition.sum((2, 3))
    degree = 0.5 * (incoming + outgoing)
    inverse_degree = torch.where(
        key_valid & degree.gt(0),
        degree.clamp_min(torch.finfo(degree.dtype).tiny).reciprocal(),
        torch.zeros_like(degree),
    )
    weighted_transition = (
        transition * inverse_degree[:, :, None, None]
    )
    weighted_observation = (
        observation * inverse_degree[:, :, None, None]
    )
    causal_two_step = torch.einsum(
        "buvm,bmxw->buvxw",
        transition,
        weighted_transition,
    )
    causal_transition_observer = torch.einsum(
        "buvm,bmoy->buvoy",
        transition,
        weighted_observation,
    )
    if not broken_glue:
        return causal_two_step, causal_transition_observer
    broken_two_step = _deranged_contract(
        transition,
        weighted_transition,
        key_valid,
    )
    broken_transition_observer = _deranged_contract(
        transition,
        weighted_observation,
        key_valid,
    )
    return broken_two_step, broken_transition_observer


def physical_key_nerve(
    witness: WitnessCompilerOutput,
    record_valid: torch.Tensor,
    *,
    mode: str = "causal",
    key_assignment_logits: torch.Tensor | None = None,
) -> PhysicalKeyNerveResult:
    """Build recoding-equivariant physical relations and path signatures."""

    if mode not in NERVE_MODES:
        raise PhysicalKeyNerveError(
            f"unknown physical-key nerve mode: {mode}"
        )
    role = witness.relation_evidence.record_role_unique.float()
    batch, records, _, unique = role.shape
    if (
        record_valid.shape != (batch, records)
        or record_valid.dtype != torch.bool
        or record_valid.device != role.device
        or witness.record_type_logits.shape[:2] != (batch, records)
        or witness.answer_logits.shape != (
            batch,
            records,
            PRIMARY_ANSWERS,
        )
    ):
        raise PhysicalKeyNerveError(
            "physical-key nerve input geometry differs"
        )
    valid = record_valid.to(role.dtype)
    record_type = witness.record_type_logits.float().softmax(-1)
    answer = witness.answer_logits.float().softmax(-1)
    transition_relation = torch.einsum(
        "br,bru,brv,brw->buvw",
        valid * record_type[:, :, RECORD_TRANSITION],
        role[:, :, ROLE_TRANSITION_SOURCE],
        role[:, :, ROLE_ACTION],
        role[:, :, ROLE_TRANSITION_DESTINATION],
    )
    observation_relation = torch.einsum(
        "br,bru,brv,bry->buvy",
        valid * record_type[:, :, RECORD_OBSERVATION],
        role[:, :, ROLE_OBSERVATION_STATE],
        role[:, :, ROLE_OBSERVER],
        answer,
    )
    key_valid = witness.unique_key_valid
    transition_mask = (
        key_valid[:, :, None, None]
        & key_valid[:, None, :, None]
        & key_valid[:, None, None, :]
    )
    observation_mask = (
        key_valid[:, :, None, None]
        & key_valid[:, None, :, None]
    )
    transition_relation = transition_relation * transition_mask
    observation_relation = observation_relation * observation_mask
    state_transport, action_transport, observer_transport = _slot_transport(
        witness,
        key_assignment_logits,
    )
    physical_two_step, physical_transition_observer = _physical_paths(
        transition_relation,
        observation_relation,
        key_valid,
        broken_glue=mode == "broken-glue",
    )
    machine_path = torch.einsum(
        "biu,bav,bcx,bkw,buvxw->bacik",
        state_transport,
        action_transport,
        action_transport,
        state_transport,
        physical_two_step,
    )
    machine_transition_observer = torch.einsum(
        "biu,bav,bqo,buvoy->baqiy",
        state_transport,
        action_transport,
        observer_transport,
        physical_transition_observer,
    )
    physical_left_path = torch.einsum(
        "biu,bcx,bkw,buvxw->bvcik",
        state_transport,
        action_transport,
        state_transport,
        physical_two_step,
    )
    physical_right_path = torch.einsum(
        "biu,bav,bkw,buvxw->bxaik",
        state_transport,
        action_transport,
        state_transport,
        physical_two_step,
    )
    physical_action_observer = torch.einsum(
        "biu,bqo,buvoy->bvqiy",
        state_transport,
        observer_transport,
        physical_transition_observer,
    )
    physical_commutator = physical_left_path - physical_right_path
    machine_commutator = machine_path - machine_path.transpose(1, 2)
    action_left_compatibility = _compatibility(
        machine_path.flatten(2),
        physical_left_path.flatten(2),
    )
    action_right_compatibility = _compatibility(
        machine_path.transpose(1, 2).flatten(2),
        physical_right_path.flatten(2),
    )
    action_observer_compatibility = _compatibility(
        machine_transition_observer.flatten(2),
        physical_action_observer.flatten(2),
    )
    action_commutator_compatibility = _signed_compatibility(
        machine_commutator.flatten(2),
        physical_commutator.flatten(2),
    )
    path_mass = (
        physical_two_step.sum((1, 2, 3, 4))
        + physical_transition_observer.sum((1, 2, 3, 4))
    )
    state_observation_signature = torch.einsum(
        "bqo,buoy->buqy",
        observer_transport,
        observation_relation,
    )
    state_transition_observer_signature = torch.einsum(
        "bav,bqo,buvoy->buaqy",
        action_transport,
        observer_transport,
        physical_transition_observer,
    )
    state_two_step_signature = torch.einsum(
        "bav,bcx,bkw,buvxw->buack",
        action_transport,
        action_transport,
        state_transport,
        physical_two_step,
    )
    action_transition_signature = torch.einsum(
        "biu,bkw,buvw->bvik",
        state_transport,
        state_transport,
        transition_relation,
    )
    action_observation_signature = torch.einsum(
        "biu,bqo,buvoy->bviqy",
        state_transport,
        observer_transport,
        physical_transition_observer,
    )
    observer_observation_signature = torch.einsum(
        "biu,buoy->boiy",
        state_transport,
        observation_relation,
    )
    if mode == "one-step-only":
        path_mass = physical_transition_observer.sum((1, 2, 3, 4))
        state_two_step_signature = state_two_step_signature * 0.0
        physical_left_path = physical_left_path * 0.0
        physical_right_path = physical_right_path * 0.0
        action_left_compatibility = torch.zeros_like(
            action_left_compatibility
        )
        action_right_compatibility = torch.zeros_like(
            action_right_compatibility
        )
        action_commutator_compatibility = torch.zeros_like(
            action_commutator_compatibility
        )
    state_signature = torch.cat(
        (
            state_observation_signature.flatten(2),
            state_transition_observer_signature.flatten(2),
            state_two_step_signature.flatten(2),
        ),
        dim=-1,
    )
    action_signature = torch.cat(
        (
            action_transition_signature.flatten(2),
            action_observation_signature.flatten(2),
            physical_left_path.flatten(2),
            physical_right_path.flatten(2),
        ),
        dim=-1,
    )
    observer_signature = observer_observation_signature.flatten(2)
    state_slot_signature = torch.einsum(
        "biu,buf->bif",
        state_transport,
        state_signature,
    )
    action_slot_signature = torch.einsum(
        "biu,buf->bif",
        action_transport,
        action_signature,
    )
    observer_slot_signature = torch.einsum(
        "biu,buf->bif",
        observer_transport,
        observer_signature,
    )
    return PhysicalKeyNerveResult(
        transition_relation=transition_relation,
        observation_relation=observation_relation,
        state_signature=state_signature,
        action_signature=action_signature,
        observer_signature=observer_signature,
        state_compatibility=_compatibility(
            state_slot_signature,
            state_signature,
        ),
        action_compatibility=_compatibility(
            action_slot_signature,
            action_signature,
        ),
        observer_compatibility=_compatibility(
            observer_slot_signature,
            observer_signature,
        ),
        action_left_compatibility=action_left_compatibility,
        action_right_compatibility=action_right_compatibility,
        action_observer_compatibility=action_observer_compatibility,
        action_commutator_compatibility=(
            action_commutator_compatibility
        ),
        path_mass=path_mass,
        mode=mode,
    )


__all__ = [
    "NERVE_MODES",
    "PhysicalKeyNerveError",
    "PhysicalKeyNerveResult",
    "physical_key_nerve",
]
