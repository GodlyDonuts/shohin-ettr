"""Unified model-owned trajectory for the ETTR capability-floor campaign.

This module deliberately does not compose the historical compiler, reactor,
and reader checkpoints.  One shared recurrent cell owns both source phases:

    WORLD -> typed state -> COMMAND -> terminal state -> late QUERY

The frozen backbone is outside this module.  It must provide final post-norm
token features projected to ``input_width``.  WORLD and COMMAND features are
consumed by the same cell; QUERY features cannot enter either transition and
are accepted only by :meth:`UnifiedETTRTrajectory.read_query` after COMMAND
has reached its adaptive STOP.  State mutation is an exact, fixed typed-state
algebra rather than a learned host executor.

The implementation is a mechanism candidate, not evidence of reasoning.
Promotion still requires the source-deleted gates in the capability-floor
preregistration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


MECHANISM_SCHEMA = "shohin-ettr-unified-trajectory-v1"
PHASES = ("WORLD", "COMMAND")
OPERATION_NAMES = ("ALLOCATE", "WRITE", "CLEAR", "LINK", "UNLINK", "SET_ROOT")


class UnifiedTrajectoryError(ValueError):
    """The unified trajectory violated an ownership or tensor contract."""


@dataclass(frozen=True, slots=True)
class UnifiedTrajectoryConfig:
    input_width: int = 512
    state_width: int = 512
    num_slots: int = 64
    num_types: int = 8
    num_relations: int = 16
    num_value_codes: int = 256
    num_heads: int = 8
    core_layers: int = 4
    reader_layers: int = 2
    ff_multiplier: int = 4
    max_world_steps: int = 32
    max_command_steps: int = 32
    min_world_steps: int = 1
    min_command_steps: int = 1
    max_edges: int = 256

    def validate(self) -> None:
        positive = (
            self.input_width,
            self.state_width,
            self.num_slots,
            self.num_types,
            self.num_relations,
            self.num_value_codes,
            self.num_heads,
            self.core_layers,
            self.reader_layers,
            self.ff_multiplier,
            self.max_world_steps,
            self.max_command_steps,
            self.max_edges,
        )
        if any(value <= 0 for value in positive):
            raise UnifiedTrajectoryError("all trajectory dimensions must be positive")
        if self.state_width % self.num_heads:
            raise UnifiedTrajectoryError("state width must divide evenly across heads")
        if not 0 <= self.min_world_steps <= self.max_world_steps:
            raise UnifiedTrajectoryError("WORLD minimum steps differ")
        if not 0 <= self.min_command_steps <= self.max_command_steps:
            raise UnifiedTrajectoryError("COMMAND minimum steps differ")
        if self.max_edges > self.num_relations * self.num_slots * self.num_slots:
            raise UnifiedTrajectoryError("max_edges exceeds relation capacity")


@dataclass(frozen=True, slots=True)
class UnifiedTypedState:
    value_probabilities: torch.Tensor
    type_probabilities: torch.Tensor
    relations: torch.Tensor
    active: torch.Tensor
    root: torch.Tensor
    committed: torch.Tensor
    step: int


@dataclass(frozen=True, slots=True)
class UnifiedActionPolicy:
    operation_probabilities: torch.Tensor
    source_probabilities: torch.Tensor
    target_probabilities: torch.Tensor
    relation_probabilities: torch.Tensor
    type_probabilities: torch.Tensor
    value_probabilities: torch.Tensor
    stop_probabilities: torch.Tensor
    applied_operation: torch.Tensor
    applied_source: torch.Tensor
    applied_target: torch.Tensor
    applied_relation: torch.Tensor
    applied_type: torch.Tensor
    applied_value: torch.Tensor
    applied_stop: torch.Tensor


@dataclass(frozen=True, slots=True)
class UnifiedPhaseTrace:
    phase: str
    operation_probabilities: torch.Tensor
    source_probabilities: torch.Tensor
    target_probabilities: torch.Tensor
    relation_probabilities: torch.Tensor
    type_probabilities: torch.Tensor
    value_probabilities: torch.Tensor
    stop_probabilities: torch.Tensor
    applied_operation: torch.Tensor
    applied_source: torch.Tensor
    applied_target: torch.Tensor
    applied_relation: torch.Tensor
    applied_type: torch.Tensor
    applied_value: torch.Tensor
    applied_stop: torch.Tensor
    alive_before_step: torch.Tensor
    stop_step: torch.Tensor


@dataclass(frozen=True, slots=True)
class UnifiedTrajectoryOutput:
    terminal_state: UnifiedTypedState
    world_trace: UnifiedPhaseTrace
    command_trace: UnifiedPhaseTrace
    query_delta: torch.Tensor


def _straight_through_one_hot(probabilities: torch.Tensor) -> torch.Tensor:
    hard = F.one_hot(
        probabilities.argmax(-1),
        num_classes=probabilities.shape[-1],
    ).to(probabilities.dtype)
    return hard + probabilities - probabilities.detach()


def _straight_through_binary(probabilities: torch.Tensor) -> torch.Tensor:
    hard = probabilities.ge(0.5).to(probabilities.dtype)
    return hard + probabilities - probabilities.detach()


def _straight_through_edge_cap(
    probabilities: torch.Tensor,
    maximum: int,
) -> torch.Tensor:
    flat = probabilities.flatten(1)
    count = min(maximum, flat.shape[1])
    indices = flat.topk(k=count, dim=-1).indices
    selected = torch.zeros_like(flat).scatter(1, indices, 1.0)
    hard = selected * flat.ge(0.5).to(flat.dtype)
    hard = hard.view_as(probabilities)
    return hard + probabilities - probabilities.detach()


def _masked_softmax(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if logits.shape != valid.shape:
        raise UnifiedTrajectoryError("masked softmax geometry differs")
    fallback = ~valid.any(-1)
    valid = valid.clone()
    if fallback.any():
        valid[fallback, 0] = True
    floor = torch.finfo(logits.dtype).min
    return logits.masked_fill(~valid, floor).softmax(-1)


def _blend_tensor(
    previous: torch.Tensor,
    candidate: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    while gate.ndim < previous.ndim:
        gate = gate.unsqueeze(-1)
    return previous * (1.0 - gate) + candidate * gate


def validate_unified_state(
    state: UnifiedTypedState,
    config: UnifiedTrajectoryConfig,
) -> None:
    config.validate()
    if not isinstance(state, UnifiedTypedState):
        raise UnifiedTrajectoryError("typed state differs")
    if state.value_probabilities.ndim != 3:
        raise UnifiedTrajectoryError("value state rank differs")
    batch = state.value_probabilities.shape[0]
    expected = {
        "value_probabilities": (batch, config.num_slots, config.num_value_codes),
        "type_probabilities": (batch, config.num_slots, config.num_types),
        "relations": (batch, config.num_relations, config.num_slots, config.num_slots),
        "active": (batch, config.num_slots),
        "root": (batch, config.num_slots),
        "committed": (batch,),
    }
    for name, shape in expected.items():
        value = getattr(state, name)
        if value.shape != shape or not value.is_floating_point():
            raise UnifiedTrajectoryError(f"{name} geometry differs")
        if not torch.isfinite(value).all():
            raise UnifiedTrajectoryError(f"{name} is nonfinite")
    if not isinstance(state.step, int) or state.step < 0:
        raise UnifiedTrajectoryError("state step differs")


def empty_unified_state(
    batch: int,
    config: UnifiedTrajectoryConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> UnifiedTypedState:
    if batch <= 0:
        raise UnifiedTrajectoryError("state batch must be positive")
    state = UnifiedTypedState(
        value_probabilities=torch.zeros(
            batch,
            config.num_slots,
            config.num_value_codes,
            device=device,
            dtype=dtype,
        ),
        type_probabilities=torch.zeros(
            batch,
            config.num_slots,
            config.num_types,
            device=device,
            dtype=dtype,
        ),
        relations=torch.zeros(
            batch,
            config.num_relations,
            config.num_slots,
            config.num_slots,
            device=device,
            dtype=dtype,
        ),
        active=torch.zeros(batch, config.num_slots, device=device, dtype=dtype),
        root=torch.zeros(batch, config.num_slots, device=device, dtype=dtype),
        committed=torch.zeros(batch, device=device, dtype=dtype),
        step=0,
    )
    validate_unified_state(state, config)
    return state


class UnifiedStateEncoder(nn.Module):
    """The one state representation shared by transition and late readout."""

    def __init__(self, config: UnifiedTrajectoryConfig):
        super().__init__()
        config.validate()
        self.config = config
        width = config.state_width
        self.value_embedding = nn.Parameter(torch.empty(config.num_value_codes, width))
        self.type_embedding = nn.Parameter(torch.empty(config.num_types, width))
        self.slot_embedding = nn.Parameter(torch.empty(config.num_slots, width))
        self.active_projection = nn.Linear(1, width, bias=False)
        self.root_projection = nn.Linear(1, width, bias=False)
        self.relation_out = nn.Linear(config.num_relations * width, width, bias=False)
        self.relation_in = nn.Linear(config.num_relations * width, width, bias=False)
        self.norm = nn.RMSNorm(width)
        nn.init.normal_(self.value_embedding, std=0.02)
        nn.init.normal_(self.type_embedding, std=0.02)
        nn.init.normal_(self.slot_embedding, std=0.02)

    def forward(self, state: UnifiedTypedState) -> torch.Tensor:
        validate_unified_state(state, self.config)
        slots = torch.einsum(
            "bsc,cw->bsw",
            state.value_probabilities,
            self.value_embedding,
        )
        slots = slots + torch.einsum(
            "bst,tw->bsw",
            state.type_probabilities,
            self.type_embedding,
        )
        slots = slots + self.slot_embedding.unsqueeze(0)
        slots = slots + self.active_projection(state.active.unsqueeze(-1))
        slots = slots + self.root_projection(state.root.unsqueeze(-1))
        outgoing = torch.einsum("brst,btw->bsrw", state.relations, slots)
        incoming = torch.einsum("brst,bsw->btrw", state.relations, slots)
        slots = slots + self.relation_out(outgoing.flatten(2))
        slots = slots + self.relation_in(incoming.flatten(2))
        return self.norm(slots)


class TiedTrajectoryCell(nn.Module):
    """One parameter-tied policy cell used for every WORLD and COMMAND step."""

    def __init__(
        self,
        config: UnifiedTrajectoryConfig,
        state_encoder: UnifiedStateEncoder,
    ):
        super().__init__()
        config.validate()
        self.config = config
        self.state_encoder = state_encoder
        width = config.state_width
        self.source_projection = nn.Linear(config.input_width, width, bias=False)
        self.source_norm = nn.RMSNorm(width)
        self.control_seed = nn.Parameter(torch.empty(width))
        self.phase_embedding = nn.Parameter(torch.empty(len(PHASES), width))
        self.step_embedding = nn.Parameter(
            torch.empty(config.max_world_steps + config.max_command_steps, width)
        )
        self.source_attention = nn.MultiheadAttention(
            width,
            config.num_heads,
            batch_first=True,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config.num_heads,
            dim_feedforward=width * config.ff_multiplier,
            batch_first=True,
            norm_first=True,
            activation="gelu",
            dropout=0.0,
        )
        self.core = nn.TransformerEncoder(
            layer,
            num_layers=config.core_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.RMSNorm(width)
        self.operation_head = nn.Linear(width, len(OPERATION_NAMES))
        self.source_query = nn.Linear(width, width, bias=False)
        self.target_query = nn.Linear(width, width, bias=False)
        self.slot_key = nn.Linear(width, width, bias=False)
        self.relation_head = nn.Linear(width, config.num_relations)
        self.type_head = nn.Linear(width, config.num_types)
        self.value_head = nn.Linear(width, config.num_value_codes)
        self.stop_head = nn.Linear(width, 1)
        nn.init.normal_(self.control_seed, std=0.02)
        nn.init.normal_(self.phase_embedding, std=0.02)
        nn.init.normal_(self.step_embedding, std=0.02)

    def forward(
        self,
        state: UnifiedTypedState,
        source_features: torch.Tensor,
        source_mask: torch.Tensor,
        *,
        phase: Literal["WORLD", "COMMAND"],
        phase_step: int,
        hard: bool,
        force_stop: bool,
        forbid_stop: bool,
    ) -> UnifiedActionPolicy:
        validate_unified_state(state, self.config)
        if phase not in PHASES:
            raise UnifiedTrajectoryError("trajectory phase differs")
        batch, tokens, width = source_features.shape
        if (
            width != self.config.input_width
            or source_mask.shape != (batch, tokens)
            or source_mask.dtype != torch.bool
            or batch != state.active.shape[0]
            or not source_mask.any(-1).all()
        ):
            raise UnifiedTrajectoryError("source tensor interface differs")
        phase_limit = (
            self.config.max_world_steps if phase == "WORLD" else self.config.max_command_steps
        )
        if not 0 <= phase_step < phase_limit:
            raise UnifiedTrajectoryError("phase step differs")
        slots = self.state_encoder(state)
        active = state.active.unsqueeze(-1)
        pooled = (slots * active).sum(1) / active.sum(1).clamp_min(1.0)
        global_step = phase_step if phase == "WORLD" else self.config.max_world_steps + phase_step
        control = (
            self.control_seed.to(slots.dtype).unsqueeze(0)
            + self.phase_embedding[PHASES.index(phase)].to(slots.dtype)
            + self.step_embedding[global_step].to(slots.dtype)
            + pooled
        )
        source = self.source_norm(self.source_projection(source_features))
        source_read, _ = self.source_attention(
            control[:, None, :],
            source,
            source,
            key_padding_mask=~source_mask,
            need_weights=False,
        )
        encoded = self.core(torch.cat((control[:, None, :] + source_read, slots), dim=1))
        control = self.output_norm(encoded[:, 0])
        encoded_slots = self.output_norm(encoded[:, 1:])

        operation_probabilities = self.operation_head(control).float().softmax(-1)
        operation_for_masks = _straight_through_one_hot(operation_probabilities)
        keys = self.slot_key(encoded_slots)
        source_logits = torch.einsum("bw,bsw->bs", self.source_query(control), keys).float()
        target_logits = torch.einsum("bw,bsw->bs", self.target_query(control), keys).float()
        active_mask = state.active.ge(0.5)
        allocate = operation_for_masks[:, 0:1]
        active_source = operation_for_masks[:, 1:].sum(-1, keepdim=True)
        source_valid = allocate * (~active_mask).float() + active_source * active_mask.float()
        relational = operation_for_masks[:, 3:5].sum(-1, keepdim=True)
        target_valid = relational * active_mask.float() + (1.0 - relational)
        source_probabilities = _masked_softmax(source_logits, source_valid.gt(0.0))
        target_probabilities = _masked_softmax(target_logits, target_valid.gt(0.0))
        relation_probabilities = self.relation_head(control).float().softmax(-1)
        type_probabilities = self.type_head(control).float().softmax(-1)
        value_probabilities = self.value_head(control).float().softmax(-1)
        stop_probabilities = self.stop_head(control).float().sigmoid().squeeze(-1)
        if forbid_stop and force_stop:
            raise UnifiedTrajectoryError("STOP cannot be forced and forbidden")
        if forbid_stop:
            stop_probabilities = stop_probabilities * 0.0
        if force_stop:
            stop_probabilities = stop_probabilities * 0.0 + 1.0

        if hard:
            operation = operation_for_masks
            source_choice = _straight_through_one_hot(source_probabilities)
            target_choice = _straight_through_one_hot(target_probabilities)
            relation = _straight_through_one_hot(relation_probabilities)
            type_choice = _straight_through_one_hot(type_probabilities)
            value = _straight_through_one_hot(value_probabilities)
            stop = _straight_through_binary(stop_probabilities)
        else:
            operation = operation_probabilities
            source_choice = source_probabilities
            target_choice = target_probabilities
            relation = relation_probabilities
            type_choice = type_probabilities
            value = value_probabilities
            stop = stop_probabilities
        dtype = state.active.dtype
        return UnifiedActionPolicy(
            operation_probabilities=operation_probabilities,
            source_probabilities=source_probabilities,
            target_probabilities=target_probabilities,
            relation_probabilities=relation_probabilities,
            type_probabilities=type_probabilities,
            value_probabilities=value_probabilities,
            stop_probabilities=stop_probabilities,
            applied_operation=operation.to(dtype),
            applied_source=source_choice.to(dtype),
            applied_target=target_choice.to(dtype),
            applied_relation=relation.to(dtype),
            applied_type=type_choice.to(dtype),
            applied_value=value.to(dtype),
            applied_stop=stop.to(dtype),
        )


def apply_unified_action(
    state: UnifiedTypedState,
    policy: UnifiedActionPolicy,
    config: UnifiedTrajectoryConfig,
    *,
    hard: bool,
) -> UnifiedTypedState:
    """Apply one policy through the fixed, ontology-neutral state algebra."""

    validate_unified_state(state, config)
    operation = policy.applied_operation
    allocate = operation[:, 0:1] * policy.applied_source
    write = operation[:, 1:2] * policy.applied_source
    clear = operation[:, 2:3] * policy.applied_source
    link = operation[:, 3]
    unlink = operation[:, 4]
    set_root = operation[:, 5:6] * policy.applied_source

    allocated = allocate * (1.0 - state.active)
    cleared = clear * state.active
    active = (state.active + allocated) * (1.0 - cleared)
    type_write = allocated.unsqueeze(-1)
    type_probabilities = (
        state.type_probabilities * (1.0 - type_write)
        + policy.applied_type[:, None, :] * type_write
    ) * (1.0 - cleared.unsqueeze(-1))
    value_write = ((write * state.active) + allocated).clamp(max=1.0).unsqueeze(-1)
    value_probabilities = (
        state.value_probabilities * (1.0 - value_write)
        + policy.applied_value[:, None, :] * value_write
    ) * (1.0 - cleared.unsqueeze(-1))

    edge = (
        policy.applied_relation[:, :, None, None]
        * policy.applied_source[:, None, :, None]
        * policy.applied_target[:, None, None, :]
    )
    relations = state.relations + link[:, None, None, None] * edge * (1.0 - state.relations)
    relations = relations * (1.0 - unlink[:, None, None, None] * edge)
    clear_pair = (cleared[:, None, :, None] + cleared[:, None, None, :]).clamp(max=1.0)
    relations = relations * (1.0 - clear_pair)
    pair_active = active[:, None, :, None] * active[:, None, None, :]
    relations = relations * pair_active

    root = state.root * (1.0 - set_root.sum(-1, keepdim=True)) + set_root
    root = root * active
    root = root / root.sum(-1, keepdim=True).clamp_min(1e-6)
    if hard:
        active = _straight_through_binary(active)
        value_probabilities = _straight_through_one_hot(value_probabilities.clamp_min(0.0)) * active.unsqueeze(-1)
        type_probabilities = _straight_through_one_hot(type_probabilities.clamp_min(0.0)) * active.unsqueeze(-1)
        relations = _straight_through_edge_cap(relations, config.max_edges) * pair_active
        nonempty = active.any(-1)
        hard_root = _straight_through_one_hot(root.clamp_min(0.0)) * active
        root = torch.where(nonempty[:, None], hard_root, torch.zeros_like(hard_root))
    result = UnifiedTypedState(
        value_probabilities=value_probabilities,
        type_probabilities=type_probabilities,
        relations=relations,
        active=active,
        root=root,
        committed=state.committed,
        step=state.step + 1,
    )
    validate_unified_state(result, config)
    return result


def _blend_state(
    previous: UnifiedTypedState,
    candidate: UnifiedTypedState,
    gate: torch.Tensor,
) -> UnifiedTypedState:
    return UnifiedTypedState(
        value_probabilities=_blend_tensor(previous.value_probabilities, candidate.value_probabilities, gate),
        type_probabilities=_blend_tensor(previous.type_probabilities, candidate.type_probabilities, gate),
        relations=_blend_tensor(previous.relations, candidate.relations, gate),
        active=_blend_tensor(previous.active, candidate.active, gate),
        root=_blend_tensor(previous.root, candidate.root, gate),
        committed=_blend_tensor(previous.committed, candidate.committed, gate),
        step=candidate.step,
    )


class LateQueryReader(nn.Module):
    """Read only terminal state and late QUERY features."""

    def __init__(
        self,
        config: UnifiedTrajectoryConfig,
        state_encoder: UnifiedStateEncoder,
    ):
        super().__init__()
        config.validate()
        self.config = config
        self.state_encoder = state_encoder
        width = config.state_width
        self.query_projection = nn.Linear(config.input_width, width, bias=False)
        self.cross_attention = nn.MultiheadAttention(
            width,
            config.num_heads,
            batch_first=True,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config.num_heads,
            dim_feedforward=width * config.ff_multiplier,
            batch_first=True,
            norm_first=True,
            activation="gelu",
            dropout=0.0,
        )
        self.core = nn.TransformerEncoder(
            layer,
            num_layers=config.reader_layers,
            enable_nested_tensor=False,
        )
        self.output = nn.Linear(width, config.input_width, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        terminal_state: UnifiedTypedState,
        query_features: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        validate_unified_state(terminal_state, self.config)
        if not terminal_state.committed.ge(1.0 - 1e-6).all():
            raise UnifiedTrajectoryError("QUERY readout requires COMMAND termination")
        batch, tokens, width = query_features.shape
        if (
            width != self.config.input_width
            or query_mask.shape != (batch, tokens)
            or query_mask.dtype != torch.bool
            or batch != terminal_state.active.shape[0]
            or not query_mask.any(-1).all()
        ):
            raise UnifiedTrajectoryError("QUERY tensor interface differs")
        query = self.query_projection(query_features)
        memory = self.state_encoder(terminal_state)
        state_padding = ~terminal_state.active.ge(0.5)
        empty = state_padding.all(-1)
        if empty.any():
            state_padding = state_padding.clone()
            state_padding[empty, 0] = False
        read, _ = self.cross_attention(
            query,
            memory,
            memory,
            key_padding_mask=state_padding,
            need_weights=False,
        )
        causal_mask = torch.ones(tokens, tokens, dtype=torch.bool, device=query.device).triu(1)
        hidden = self.core(
            query + read,
            mask=causal_mask,
            src_key_padding_mask=~query_mask,
        )
        return torch.tanh(self.gate).to(hidden.dtype) * self.output(hidden)


class UnifiedETTRTrajectory(nn.Module):
    """Single differentiable WORLD/COMMAND trajectory with late QUERY readout."""

    def __init__(self, config: UnifiedTrajectoryConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.state_encoder = UnifiedStateEncoder(config)
        self.cell = TiedTrajectoryCell(config, self.state_encoder)
        self.query_reader = LateQueryReader(config, self.state_encoder)

    def initial_state(
        self,
        source_features: torch.Tensor,
    ) -> UnifiedTypedState:
        if source_features.ndim != 3 or not source_features.is_floating_point():
            raise UnifiedTrajectoryError("source features must be rank-three floating tensor")
        return empty_unified_state(
            source_features.shape[0],
            self.config,
            device=source_features.device,
            dtype=source_features.dtype,
        )

    def run_phase(
        self,
        state: UnifiedTypedState,
        source_features: torch.Tensor,
        source_mask: torch.Tensor,
        *,
        phase: Literal["WORLD", "COMMAND"],
        hard: bool,
    ) -> tuple[UnifiedTypedState, UnifiedPhaseTrace]:
        if phase not in PHASES:
            raise UnifiedTrajectoryError("trajectory phase differs")
        maximum = self.config.max_world_steps if phase == "WORLD" else self.config.max_command_steps
        minimum = self.config.min_world_steps if phase == "WORLD" else self.config.min_command_steps
        alive = torch.ones_like(state.committed)
        policies: list[UnifiedActionPolicy] = []
        alive_history: list[torch.Tensor] = []
        stop_step = torch.full(
            (state.active.shape[0],),
            maximum,
            dtype=torch.long,
            device=state.active.device,
        )
        for phase_step in range(maximum):
            alive_before = alive
            policy = self.cell(
                state,
                source_features,
                source_mask,
                phase=phase,
                phase_step=phase_step,
                hard=hard,
                force_stop=phase_step == maximum - 1,
                forbid_stop=phase_step + 1 < minimum,
            )
            candidate = apply_unified_action(state, policy, self.config, hard=hard)
            state = _blend_state(state, candidate, alive_before)
            effective_stop = alive_before * policy.applied_stop
            newly_stopped = effective_stop.ge(0.5) & stop_step.eq(maximum)
            stop_step = torch.where(
                newly_stopped,
                torch.full_like(stop_step, phase_step + 1),
                stop_step,
            )
            alive = alive_before * (1.0 - policy.applied_stop)
            policies.append(policy)
            alive_history.append(alive_before)
        if hard and not alive.eq(0).all():
            raise UnifiedTrajectoryError("hard trajectory failed to terminate")
        if phase == "COMMAND":
            state = UnifiedTypedState(
                value_probabilities=state.value_probabilities,
                type_probabilities=state.type_probabilities,
                relations=state.relations,
                active=state.active,
                root=state.root,
                committed=torch.ones_like(state.committed),
                step=state.step,
            )
        validate_unified_state(state, self.config)
        trace = UnifiedPhaseTrace(
            phase=phase,
            operation_probabilities=torch.stack([item.operation_probabilities for item in policies], 1),
            source_probabilities=torch.stack([item.source_probabilities for item in policies], 1),
            target_probabilities=torch.stack([item.target_probabilities for item in policies], 1),
            relation_probabilities=torch.stack([item.relation_probabilities for item in policies], 1),
            type_probabilities=torch.stack([item.type_probabilities for item in policies], 1),
            value_probabilities=torch.stack([item.value_probabilities for item in policies], 1),
            stop_probabilities=torch.stack([item.stop_probabilities for item in policies], 1),
            applied_operation=torch.stack([item.applied_operation for item in policies], 1),
            applied_source=torch.stack([item.applied_source for item in policies], 1),
            applied_target=torch.stack([item.applied_target for item in policies], 1),
            applied_relation=torch.stack([item.applied_relation for item in policies], 1),
            applied_type=torch.stack([item.applied_type for item in policies], 1),
            applied_value=torch.stack([item.applied_value for item in policies], 1),
            applied_stop=torch.stack([item.applied_stop for item in policies], 1),
            alive_before_step=torch.stack(alive_history, 1),
            stop_step=stop_step,
        )
        return state, trace

    def read_query(
        self,
        terminal_state: UnifiedTypedState,
        query_features: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.query_reader(terminal_state, query_features, query_mask)

    def forward(
        self,
        world_features: torch.Tensor,
        world_mask: torch.Tensor,
        command_features: torch.Tensor,
        command_mask: torch.Tensor,
        query_features: torch.Tensor,
        query_mask: torch.Tensor,
        *,
        hard: bool,
    ) -> UnifiedTrajectoryOutput:
        state = self.initial_state(world_features)
        state, world_trace = self.run_phase(
            state,
            world_features,
            world_mask,
            phase="WORLD",
            hard=hard,
        )
        state, command_trace = self.run_phase(
            state,
            command_features,
            command_mask,
            phase="COMMAND",
            hard=hard,
        )
        query_delta = self.read_query(state, query_features, query_mask)
        return UnifiedTrajectoryOutput(
            terminal_state=state,
            world_trace=world_trace,
            command_trace=command_trace,
            query_delta=query_delta,
        )

    def architecture_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def mechanism_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def build_mechanism_receipt() -> dict[str, object]:
    config = UnifiedTrajectoryConfig()
    config.validate()
    return {
        "adaptive_stop": {
            "forced_at_phase_limit": True,
            "monotone_per_example": True,
            "post_stop_updates_frozen": True,
        },
        "architecture": "one-shared-cell-world-and-command-fixed-typed-algebra-late-query",
        "config": asdict(config),
        "forbidden_in_transition": [
            "query-features",
            "answer-label",
            "oracle-program",
            "oracle-successor-state",
            "host-semantic-executor",
        ],
        "operation_names": list(OPERATION_NAMES),
        "phase_order": ["WORLD", "COMMAND", "QUERY"],
        "schema": MECHANISM_SCHEMA,
        "source_sha256": mechanism_source_sha256(),
        "state_fields": [
            "value-probabilities",
            "type-probabilities",
            "relations",
            "active",
            "root",
            "committed",
        ],
        "status": "mechanism-source-frozen-preflight-required",
        "tied_objects": ["state-encoder", "trajectory-cell"],
    }


def mechanism_architecture_sha256() -> str:
    payload = json.dumps(
        build_mechanism_receipt(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()
