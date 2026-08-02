"""Favorable dense recurrent control for the ETTR capability-floor matrix.

The control receives the same frozen-backbone tensors, rectangle schedule,
charged positions, optimizer budget, adaptive phase limits, and late QUERY.
It is deliberately not forced through ETTR's discrete transaction algebra:
WORLD and COMMAND have independent recurrent cells and retain a full dense
state.  Learned heads decode that state into the same terminal packet fields
for the unchanged evaluator.  This makes the control favorable rather than a
straw man.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn

from capability_floor_trajectory import (
    UnifiedTrajectoryConfig,
    UnifiedTrajectoryError,
    UnifiedTypedState,
    validate_unified_state,
)


DENSE_CONTROL_SCHEMA = "shohin-ettr-favorable-dense-recurrent-control-v1"


@dataclass(frozen=True, slots=True)
class DenseControlConfig:
    input_width: int
    hidden_width: int
    state_tokens: int
    num_heads: int
    phase_layers: int
    reader_layers: int
    ff_multiplier: int
    max_world_steps: int
    max_command_steps: int
    min_world_steps: int
    min_command_steps: int
    num_slots: int
    num_types: int
    num_relations: int
    num_value_codes: int
    max_edges: int
    capacity_hidden: int = 0
    capacity_tail: int = 0

    @classmethod
    def from_treatment(
        cls,
        treatment: UnifiedTrajectoryConfig,
        *,
        hidden_width: int,
        capacity_hidden: int = 0,
        capacity_tail: int = 0,
    ) -> "DenseControlConfig":
        treatment.validate()
        return cls(
            input_width=treatment.input_width,
            hidden_width=hidden_width,
            state_tokens=treatment.num_slots,
            num_heads=treatment.num_heads,
            phase_layers=treatment.core_layers,
            reader_layers=treatment.reader_layers,
            ff_multiplier=treatment.ff_multiplier,
            max_world_steps=treatment.max_world_steps,
            max_command_steps=treatment.max_command_steps,
            min_world_steps=treatment.min_world_steps,
            min_command_steps=treatment.min_command_steps,
            num_slots=treatment.num_slots,
            num_types=treatment.num_types,
            num_relations=treatment.num_relations,
            num_value_codes=treatment.num_value_codes,
            max_edges=treatment.max_edges,
            capacity_hidden=capacity_hidden,
            capacity_tail=capacity_tail,
        )

    def validate(self) -> None:
        positive = {
            name: value
            for name, value in asdict(self).items()
            if name not in {"capacity_hidden", "capacity_tail"}
        }
        if (
            min(positive.values()) <= 0
            or self.capacity_hidden < 0
            or self.capacity_tail < 0
            or self.capacity_tail >= 2 * self.hidden_width + 1
        ):
            raise UnifiedTrajectoryError("all dense-control dimensions must be positive")
        if self.hidden_width % self.num_heads:
            raise UnifiedTrajectoryError("dense width must divide across heads")
        if self.min_world_steps > self.max_world_steps:
            raise UnifiedTrajectoryError("dense WORLD stop bounds differ")
        if self.min_command_steps > self.max_command_steps:
            raise UnifiedTrajectoryError("dense COMMAND stop bounds differ")


@dataclass(frozen=True, slots=True)
class DensePhaseTrace:
    phase: str
    stop_probabilities: torch.Tensor
    applied_stop: torch.Tensor
    alive_before_step: torch.Tensor
    stop_step: torch.Tensor


@dataclass(frozen=True, slots=True)
class DenseControlOutput:
    terminal_state: UnifiedTypedState
    dense_terminal: torch.Tensor
    world_trace: DensePhaseTrace
    command_trace: DensePhaseTrace
    query_delta: torch.Tensor


def _straight_through_binary(probabilities: torch.Tensor) -> torch.Tensor:
    hard = probabilities.ge(0.5).to(probabilities.dtype)
    return hard + probabilities - probabilities.detach()


def _hard_edge_cap(probabilities: torch.Tensor, maximum: int) -> torch.Tensor:
    flat = probabilities.flatten(1)
    count = min(maximum, flat.shape[1])
    indices = flat.topk(k=count, dim=-1).indices
    selected = torch.zeros_like(flat).scatter(1, indices, 1.0)
    return (selected * flat.ge(0.5).to(flat.dtype)).view_as(probabilities)


class DensePhaseCell(nn.Module):
    def __init__(self, config: DenseControlConfig):
        super().__init__()
        config.validate()
        self.config = config
        width = config.hidden_width
        self.source_projection = nn.Linear(config.input_width, width, bias=False)
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
            num_layers=config.phase_layers,
            enable_nested_tensor=False,
        )
        self.control_seed = nn.Parameter(torch.empty(width))
        self.step_embedding = nn.Parameter(
            torch.empty(max(config.max_world_steps, config.max_command_steps), width)
        )
        self.state_norm = nn.RMSNorm(width)
        self.stop_head = nn.Linear(width, 1)
        nn.init.normal_(self.control_seed, std=0.02)
        nn.init.normal_(self.step_embedding, std=0.02)

    def forward(
        self,
        state: torch.Tensor,
        source_features: torch.Tensor,
        source_mask: torch.Tensor,
        *,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state.ndim != 3 or state.shape[-1] != self.config.hidden_width:
            raise UnifiedTrajectoryError("dense state geometry differs")
        batch, tokens, width = source_features.shape
        if (
            width != self.config.input_width
            or source_mask.shape != (batch, tokens)
            or source_mask.dtype != torch.bool
            or state.shape[0] != batch
            or not source_mask.any(-1).all()
        ):
            raise UnifiedTrajectoryError("dense source interface differs")
        control = (
            self.control_seed.to(state.dtype).unsqueeze(0)
            + self.step_embedding[step].to(state.dtype)
            + self.state_norm(state).mean(1)
        )
        source = self.source_projection(source_features)
        source_read, _ = self.source_attention(
            control[:, None, :],
            source,
            source,
            key_padding_mask=~source_mask,
            need_weights=False,
        )
        encoded = self.core(torch.cat((control[:, None, :] + source_read, state), 1))
        return self.state_norm(encoded[:, 1:]), self.stop_head(encoded[:, 0]).sigmoid().squeeze(-1)


class FavorableDenseRecurrentControl(nn.Module):
    def __init__(self, config: DenseControlConfig):
        super().__init__()
        config.validate()
        self.config = config
        width = config.hidden_width
        self.initial_state = nn.Parameter(torch.empty(config.state_tokens, width))
        if config.capacity_hidden:
            self.capacity_up = nn.Linear(width, config.capacity_hidden)
            self.capacity_down = nn.Linear(config.capacity_hidden, width, bias=False)
        else:
            self.capacity_up = None
            self.capacity_down = None
        self.capacity_tail = nn.Parameter(torch.empty(config.capacity_tail))
        self.world_cell = DensePhaseCell(config)
        self.command_cell = DensePhaseCell(config)
        self.value_head = nn.Linear(width, config.num_value_codes)
        self.type_head = nn.Linear(width, config.num_types)
        self.active_head = nn.Linear(width, 1)
        self.root_head = nn.Linear(width, 1)
        self.relation_left = nn.Linear(width, config.num_relations * width, bias=False)
        self.relation_right = nn.Linear(width, config.num_relations * width, bias=False)
        self.query_projection = nn.Linear(config.input_width, width, bias=False)
        self.query_attention = nn.MultiheadAttention(
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
        self.query_core = nn.TransformerEncoder(
            layer,
            num_layers=config.reader_layers,
            enable_nested_tensor=False,
        )
        self.query_output = nn.Linear(width, config.input_width, bias=False)
        self.query_gate = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.initial_state, std=0.02)
        if self.capacity_tail.numel():
            nn.init.normal_(self.capacity_tail, std=0.02)

    def _capacity_delta(self, state: torch.Tensor) -> torch.Tensor:
        """Use every parameter added solely for exact parameter matching."""

        width = self.config.hidden_width
        summary = state.mean(1)
        delta = torch.zeros_like(summary)
        if self.capacity_up is not None and self.capacity_down is not None:
            delta = delta + self.capacity_down(
                torch.nn.functional.gelu(self.capacity_up(summary))
            )
        if not self.capacity_tail.numel():
            return delta
        indices = torch.arange(
            self.capacity_tail.numel(),
            device=self.capacity_tail.device,
        )
        input_indices = indices.remainder(width)
        output_indices = indices.mul(0x9E3779B1).remainder(width)
        contributions = summary[:, input_indices] * self.capacity_tail.unsqueeze(0)
        tail = summary.new_zeros(summary.shape).scatter_add(
            1,
            output_indices.unsqueeze(0).expand(summary.shape[0], -1),
            contributions,
        )
        return delta + tail

    def _run_phase(
        self,
        state: torch.Tensor,
        source_features: torch.Tensor,
        source_mask: torch.Tensor,
        *,
        phase: str,
        hard: bool,
    ) -> tuple[torch.Tensor, DensePhaseTrace]:
        if phase == "WORLD":
            cell = self.world_cell
            maximum = self.config.max_world_steps
            minimum = self.config.min_world_steps
        elif phase == "COMMAND":
            cell = self.command_cell
            maximum = self.config.max_command_steps
            minimum = self.config.min_command_steps
        else:
            raise UnifiedTrajectoryError("dense phase differs")
        alive = torch.ones(state.shape[0], device=state.device, dtype=state.dtype)
        stops: list[torch.Tensor] = []
        applied: list[torch.Tensor] = []
        alive_history: list[torch.Tensor] = []
        stop_step = torch.full(
            (state.shape[0],),
            maximum,
            dtype=torch.long,
            device=state.device,
        )
        for step in range(maximum):
            alive_before = alive
            candidate, stop_probability = cell(
                state,
                source_features,
                source_mask,
                step=step,
            )
            candidate = candidate + self._capacity_delta(candidate).unsqueeze(1)
            if step + 1 < minimum:
                stop_probability = stop_probability * 0.0
            if step == maximum - 1:
                stop_probability = stop_probability * 0.0 + 1.0
            stop = _straight_through_binary(stop_probability) if hard else stop_probability
            state = state * (1.0 - alive_before[:, None, None]) + candidate * alive_before[:, None, None]
            effective_stop = alive_before * stop
            newly_stopped = effective_stop.ge(0.5) & stop_step.eq(maximum)
            stop_step = torch.where(
                newly_stopped,
                torch.full_like(stop_step, step + 1),
                stop_step,
            )
            alive = alive_before * (1.0 - stop)
            stops.append(stop_probability)
            applied.append(stop)
            alive_history.append(alive_before)
        if hard and not alive.eq(0).all():
            raise UnifiedTrajectoryError("dense hard trajectory failed to terminate")
        return state, DensePhaseTrace(
            phase=phase,
            stop_probabilities=torch.stack(stops, 1),
            applied_stop=torch.stack(applied, 1),
            alive_before_step=torch.stack(alive_history, 1),
            stop_step=stop_step,
        )

    def decode_terminal(self, state: torch.Tensor, *, hard: bool) -> UnifiedTypedState:
        batch = state.shape[0]
        slots = state[:, : self.config.num_slots]
        value = self.value_head(slots).float().softmax(-1)
        type_probabilities = self.type_head(slots).float().softmax(-1)
        active = self.active_head(slots).float().sigmoid().squeeze(-1)
        root = self.root_head(slots).float().squeeze(-1)
        left = self.relation_left(slots).view(
            batch,
            self.config.num_slots,
            self.config.num_relations,
            self.config.hidden_width,
        )
        right = self.relation_right(slots).view_as(left)
        relations = torch.einsum("bsrw,btrw->brst", left, right)
        relations = (relations / self.config.hidden_width**0.5).sigmoid()
        if hard:
            value = torch.nn.functional.one_hot(value.argmax(-1), self.config.num_value_codes).to(value.dtype)
            type_probabilities = torch.nn.functional.one_hot(
                type_probabilities.argmax(-1), self.config.num_types
            ).to(type_probabilities.dtype)
            active = active.ge(0.5).to(active.dtype)
            root = torch.nn.functional.one_hot(root.argmax(-1), self.config.num_slots).to(root.dtype)
            relations = _hard_edge_cap(relations, self.config.max_edges)
        value = value * active.unsqueeze(-1)
        type_probabilities = type_probabilities * active.unsqueeze(-1)
        root = root.softmax(-1) if not hard else root
        root = root * active
        root = root / root.sum(-1, keepdim=True).clamp_min(1e-6)
        relations = relations * (
            active[:, None, :, None] * active[:, None, None, :]
        )
        terminal = UnifiedTypedState(
            value_probabilities=value.to(state.dtype),
            type_probabilities=type_probabilities.to(state.dtype),
            relations=relations.to(state.dtype),
            active=active.to(state.dtype),
            root=root.to(state.dtype),
            committed=torch.ones(batch, device=state.device, dtype=state.dtype),
            step=self.config.max_world_steps + self.config.max_command_steps,
        )
        treatment_config = UnifiedTrajectoryConfig(
            input_width=self.config.input_width,
            state_width=self.config.hidden_width,
            num_slots=self.config.num_slots,
            num_types=self.config.num_types,
            num_relations=self.config.num_relations,
            num_value_codes=self.config.num_value_codes,
            num_heads=self.config.num_heads,
            core_layers=self.config.phase_layers,
            reader_layers=self.config.reader_layers,
            ff_multiplier=self.config.ff_multiplier,
            max_world_steps=self.config.max_world_steps,
            max_command_steps=self.config.max_command_steps,
            min_world_steps=self.config.min_world_steps,
            min_command_steps=self.config.min_command_steps,
            max_edges=self.config.max_edges,
        )
        validate_unified_state(terminal, treatment_config)
        return terminal

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
    ) -> DenseControlOutput:
        batch = world_features.shape[0]
        state = self.initial_state.to(world_features.dtype).unsqueeze(0).expand(batch, -1, -1)
        state, world_trace = self._run_phase(
            state,
            world_features,
            world_mask,
            phase="WORLD",
            hard=hard,
        )
        state, command_trace = self._run_phase(
            state,
            command_features,
            command_mask,
            phase="COMMAND",
            hard=hard,
        )
        terminal = self.decode_terminal(state, hard=hard)
        query = self.query_projection(query_features)
        read, _ = self.query_attention(query, state, state, need_weights=False)
        tokens = query.shape[1]
        causal_mask = torch.ones(tokens, tokens, dtype=torch.bool, device=query.device).triu(1)
        hidden = self.query_core(
            query + read,
            mask=causal_mask,
            src_key_padding_mask=~query_mask,
        )
        query_delta = torch.tanh(self.query_gate).to(hidden.dtype) * self.query_output(hidden)
        return DenseControlOutput(
            terminal_state=terminal,
            dense_terminal=state,
            world_trace=world_trace,
            command_trace=command_trace,
            query_delta=query_delta,
        )

    def architecture_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def find_parameter_matched_dense_width(
    treatment_parameters: int,
    treatment_config: UnifiedTrajectoryConfig,
    *,
    minimum_width: int = 64,
    maximum_width: int = 1024,
    tolerance: float = 0.01,
) -> tuple[DenseControlConfig, int, float]:
    if treatment_parameters <= 0 or not 0.0 < tolerance < 1.0:
        raise UnifiedTrajectoryError("dense parameter-match request differs")
    best_below: tuple[DenseControlConfig, int] | None = None
    best: tuple[DenseControlConfig, int, float] | None = None
    for width in range(minimum_width, maximum_width + 1, treatment_config.num_heads):
        config = DenseControlConfig.from_treatment(treatment_config, hidden_width=width)
        candidate = FavorableDenseRecurrentControl(config)
        parameters = candidate.architecture_parameters()
        relative = abs(parameters - treatment_parameters) / treatment_parameters
        if best is None or relative < best[2]:
            best = (config, parameters, relative)
        if parameters <= treatment_parameters:
            best_below = (config, parameters)
        if relative <= tolerance:
            return config, parameters, relative
        if parameters > treatment_parameters and best_below is not None:
            break
    if best_below is not None:
        config, parameters = best_below
        reserve = treatment_parameters - parameters
        capacity_unit = 2 * config.hidden_width + 1
        capacity_hidden, capacity_tail = divmod(reserve, capacity_unit)
        matched_config = DenseControlConfig.from_treatment(
            treatment_config,
            hidden_width=config.hidden_width,
            capacity_hidden=capacity_hidden,
            capacity_tail=capacity_tail,
        )
        matched_parameters = FavorableDenseRecurrentControl(
            matched_config
        ).architecture_parameters()
        relative = abs(matched_parameters - treatment_parameters) / treatment_parameters
        if relative <= tolerance:
            return matched_config, matched_parameters, relative
    if best is None:
        raise UnifiedTrajectoryError("dense parameter search is empty")
    raise UnifiedTrajectoryError(
        f"no dense width matches parameter tolerance; best relative error {best[2]:.6f}"
    )


def dense_control_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def build_dense_control_descriptor(
    treatment_parameters: int,
    treatment_config: UnifiedTrajectoryConfig,
) -> dict[str, object]:
    config, parameters, relative = find_parameter_matched_dense_width(
        treatment_parameters,
        treatment_config,
        minimum_width=treatment_config.num_heads,
    )
    return {
        "config": asdict(config),
        "flop_receipt": None,
        "parameter_match_relative_error": relative,
        "parameters": parameters,
        "schema": DENSE_CONTROL_SCHEMA,
        "source_sha256": dense_control_source_sha256(),
        "status": "parameter-matched-flop-measurement-required",
        "treatment_parameters": treatment_parameters,
        "untied_world_and_command_cells": True,
        "parameter_match_capacity_is_live_dense_mlp": True,
    }


def dense_control_descriptor_sha256(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()
