#!/usr/bin/env python3
"""Candidate-only joint epistemic trajectory for DIVERGE-JET1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import inspect
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


REGISTER_COUNT = 5
VALUE_COUNT = 128
FIELD_COUNT = 2 * REGISTER_COUNT
ACTION_COUNT = 4
DELTAS = tuple(range(-3, 4))


class JET1ContractError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class JET1Config:
    input_width: int = 1024
    reader_width: int = 256
    reader_heads: int = 8
    reader_layers: int = 1
    ff_multiplier: int = 4
    register_count: int = REGISTER_COUNT
    value_count: int = VALUE_COUNT
    action_count: int = ACTION_COUNT
    maximum_candidates: int = 2
    maximum_program_actions: int = 2

    def validate(self) -> None:
        if min(self.input_width, self.reader_width, self.reader_heads) <= 0:
            raise JET1ContractError("JET1 widths must be positive")
        if self.reader_width % self.reader_heads:
            raise JET1ContractError("JET1 reader width must divide by heads")
        if self.reader_layers <= 0 or self.ff_multiplier <= 0:
            raise JET1ContractError("JET1 reader geometry differs")
        if (
            self.register_count != REGISTER_COUNT
            or self.value_count != VALUE_COUNT
            or self.action_count != ACTION_COUNT
            or self.maximum_candidates != 2
            or self.maximum_program_actions != 2
        ):
            raise JET1ContractError("JET1 typed domain differs")


@dataclass(frozen=True)
class JET1Output:
    evidence_probabilities: torch.Tensor
    choice_logits: torch.Tensor
    choice_probabilities: torch.Tensor
    selected_candidates: torch.Tensor
    terminal_probabilities: torch.Tensor
    answer_probabilities: torch.Tensor
    invalid_mass: torch.Tensor


class _StraightThroughOneHot(torch.autograd.Function):
    """Return an exact one-hot forward value with an identity backward map."""

    @staticmethod
    def forward(ctx: object, probabilities: torch.Tensor) -> torch.Tensor:
        return F.one_hot(
            probabilities.argmax(-1), num_classes=probabilities.shape[-1]
        ).to(probabilities.dtype)

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> torch.Tensor:
        return gradient


def _straight_through_one_hot(probabilities: torch.Tensor) -> torch.Tensor:
    return _StraightThroughOneHot.apply(probabilities)


class _StraightThroughValue(torch.autograd.Function):
    """Use a discrete forward score and the corresponding soft-score gradient."""

    @staticmethod
    def forward(
        ctx: object, soft_value: torch.Tensor, hard_value: torch.Tensor
    ) -> torch.Tensor:
        return hard_value

    @staticmethod
    def backward(
        ctx: object, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        return gradient, None


def _hard_or_straight_through(
    probabilities: torch.Tensor, *, hard_forward: bool
) -> torch.Tensor:
    return _straight_through_one_hot(probabilities) if hard_forward else probabilities


class EvidenceFieldReader(nn.Module):
    """Read one complete typed state pair from one contextual source pass."""

    def __init__(self, config: JET1Config):
        super().__init__()
        config.validate()
        self.config = config
        self.input_projection = nn.Linear(
            config.input_width, config.reader_width, bias=False
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.reader_width,
            nhead=config.reader_heads,
            dim_feedforward=config.reader_width * config.ff_multiplier,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.source_encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.reader_layers,
            enable_nested_tensor=False,
        )
        self.phase_queries = nn.Parameter(torch.empty(2, config.reader_width))
        self.slot_queries = nn.Parameter(
            torch.empty(REGISTER_COUNT, config.reader_width)
        )
        self.query_attention = nn.MultiheadAttention(
            config.reader_width,
            config.reader_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.value_head = nn.Sequential(
            nn.RMSNorm(config.reader_width),
            nn.Linear(config.reader_width, config.reader_width),
            nn.GELU(),
            nn.Linear(config.reader_width, VALUE_COUNT),
        )
        nn.init.normal_(self.phase_queries, std=0.02)
        nn.init.normal_(self.slot_queries, std=0.02)

    def forward(
        self, source_features: torch.Tensor, source_mask: torch.Tensor
    ) -> torch.Tensor:
        if (
            source_features.ndim != 3
            or source_features.shape[-1] != self.config.input_width
            or source_mask.shape != source_features.shape[:2]
            or source_mask.dtype != torch.bool
            or not source_mask.any(-1).all()
        ):
            raise JET1ContractError("JET1 source tensor interface differs")
        source = self.input_projection(source_features)
        source = self.source_encoder(source, src_key_padding_mask=~source_mask)
        queries = (
            self.phase_queries[:, None, :] + self.slot_queries[None, :, :]
        ).reshape(FIELD_COUNT, self.config.reader_width)
        queries = queries.unsqueeze(0).expand(source.shape[0], -1, -1)
        fields, _ = self.query_attention(
            queries,
            source,
            source,
            key_padding_mask=~source_mask,
            need_weights=False,
        )
        return self.value_head(fields).float()


class DifferentiableRegisterExecutor(nn.Module):
    """A tied route-plus-delta operator over categorical register states."""

    def __init__(self, config: JET1Config):
        super().__init__()
        config.validate()
        self.config = config
        self.route_logits = nn.Parameter(
            torch.empty(ACTION_COUNT, REGISTER_COUNT, REGISTER_COUNT)
        )
        self.delta_logits = nn.Parameter(
            torch.empty(ACTION_COUNT, REGISTER_COUNT, len(DELTAS))
        )
        nn.init.normal_(self.route_logits, std=0.02)
        nn.init.normal_(self.delta_logits, std=0.02)

    @staticmethod
    def _shifted(states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = states.shape[0]
        shifted = states.new_zeros(
            batch, REGISTER_COUNT, len(DELTAS), VALUE_COUNT
        )
        invalid = states.new_zeros(batch, REGISTER_COUNT, len(DELTAS))
        for index, delta in enumerate(DELTAS):
            if delta < 0:
                shifted[:, :, index, : VALUE_COUNT + delta] = states[:, :, -delta:]
                invalid[:, :, index] = states[:, :, : -delta].sum(-1)
            elif delta > 0:
                shifted[:, :, index, delta:] = states[:, :, : VALUE_COUNT - delta]
                invalid[:, :, index] = states[:, :, VALUE_COUNT - delta :].sum(-1)
            else:
                shifted[:, :, index] = states
        return shifted, invalid

    def step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        *,
        hard_forward: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            states.ndim != 3
            or states.shape[1:] != (REGISTER_COUNT, VALUE_COUNT)
            or actions.shape != (states.shape[0],)
            or actions.dtype != torch.long
            or actions.numel() and (int(actions.min()) < 0 or int(actions.max()) >= ACTION_COUNT)
        ):
            raise JET1ContractError("JET1 executor tensor interface differs")
        route = self.route_logits[actions].float().softmax(-1)
        delta = self.delta_logits[actions].float().softmax(-1)
        route = _hard_or_straight_through(route, hard_forward=hard_forward)
        delta = _hard_or_straight_through(delta, hard_forward=hard_forward)
        shifted, invalid = self._shifted(states.float())
        output = torch.einsum("boi,bod,bidv->bov", route, delta, shifted)
        invalid_mass = torch.einsum("boi,bod,bid->bo", route, delta, invalid)
        return output, invalid_mass

    def apply_programs(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        hard_forward: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            actions.ndim != 2
            or actions.shape[0] != states.shape[0]
            or action_mask.shape != actions.shape
            or action_mask.dtype != torch.bool
        ):
            raise JET1ContractError("JET1 program tensor interface differs")
        output = states
        invalid = states.new_zeros(states.shape[0], REGISTER_COUNT)
        for step in range(actions.shape[1]):
            candidate, step_invalid = self.step(
                output, actions[:, step], hard_forward=hard_forward
            )
            gate = action_mask[:, step, None, None].to(candidate.dtype)
            output = output * (1.0 - gate) + candidate * gate
            invalid = invalid + step_invalid * action_mask[:, step, None].to(
                step_invalid.dtype
            )
        return output, invalid


class JointEpistemicTrajectory(nn.Module):
    """Joint language evidence, whole-program commit, execution, and readout."""

    def __init__(self, config: JET1Config):
        super().__init__()
        config.validate()
        self.config = config
        self.evidence = EvidenceFieldReader(config)
        self.executor = DifferentiableRegisterExecutor(config)
        self.query_route_logits = nn.Parameter(
            torch.empty(REGISTER_COUNT, REGISTER_COUNT)
        )
        nn.init.normal_(self.query_route_logits, std=0.02)

    def forward(
        self,
        evidence_features: torch.Tensor,
        evidence_mask: torch.Tensor,
        initial_values: torch.Tensor,
        program_actions: torch.Tensor,
        program_action_mask: torch.Tensor,
        prior_logits: torch.Tensor,
        query_slots: torch.Tensor,
        *,
        hard_forward: bool,
    ) -> JET1Output:
        if evidence_features.ndim != 4:
            raise JET1ContractError("JET1 evidence trajectory rank differs")
        batch, depth, tokens, width = evidence_features.shape
        if (
            width != self.config.input_width
            or evidence_mask.shape != (batch, depth, tokens)
            or evidence_mask.dtype != torch.bool
            or initial_values.shape != (batch, REGISTER_COUNT)
            or initial_values.dtype != torch.long
            or program_actions.shape
            != (batch, depth, 2, self.config.maximum_program_actions)
            or program_action_mask.shape != program_actions.shape
            or program_action_mask.dtype != torch.bool
            or prior_logits.shape != (batch, depth, 2)
            or query_slots.shape != (batch,)
            or query_slots.dtype != torch.long
        ):
            raise JET1ContractError("JET1 trajectory tensor interface differs")
        flat_features = evidence_features.reshape(batch * depth, tokens, width)
        flat_mask = evidence_mask.reshape(batch * depth, tokens)
        evidence_logits = self.evidence(flat_features, flat_mask).reshape(
            batch, depth, FIELD_COUNT, VALUE_COUNT
        )
        evidence_soft = evidence_logits.softmax(-1)
        evidence_state = _hard_or_straight_through(
            evidence_soft, hard_forward=hard_forward
        )
        before_soft = evidence_soft[:, :, :REGISTER_COUNT]
        after_soft = evidence_soft[:, :, REGISTER_COUNT:]
        before_state = evidence_state[:, :, :REGISTER_COUNT]
        after_state = evidence_state[:, :, REGISTER_COUNT:]

        state = F.one_hot(initial_values, VALUE_COUNT).float()
        choice_logits = []
        choices = []
        invalid_mass = state.new_zeros(batch, REGISTER_COUNT)
        for step in range(depth):
            actions = program_actions[:, step].reshape(
                batch * 2, self.config.maximum_program_actions
            )
            masks = program_action_mask[:, step].reshape(
                batch * 2, self.config.maximum_program_actions
            )
            probe_before = before_state[:, step, None].expand(-1, 2, -1, -1)
            probe_before = probe_before.reshape(batch * 2, REGISTER_COUNT, VALUE_COUNT)
            predicted_probe, probe_invalid = self.executor.apply_programs(
                probe_before,
                actions,
                masks,
                hard_forward=hard_forward,
            )
            predicted_probe = predicted_probe.reshape(
                batch, 2, REGISTER_COUNT, VALUE_COUNT
            )
            hard_agreement = (
                predicted_probe * after_state[:, step, None]
            ).sum(-1)
            if hard_forward:
                soft_probe, _ = self.executor.apply_programs(
                    before_soft[:, step, None]
                    .expand(-1, 2, -1, -1)
                    .reshape(batch * 2, REGISTER_COUNT, VALUE_COUNT),
                    actions,
                    masks,
                    hard_forward=False,
                )
                soft_agreement = (
                    soft_probe.reshape(batch, 2, REGISTER_COUNT, VALUE_COUNT)
                    * after_soft[:, step, None]
                ).sum(-1)
                agreement = _StraightThroughValue.apply(
                    soft_agreement, hard_agreement
                )
            else:
                agreement = hard_agreement
            logits = prior_logits[:, step].float() + (agreement + 1e-6).log().sum(-1)
            probabilities = logits.softmax(-1)
            selected = _hard_or_straight_through(
                probabilities, hard_forward=hard_forward
            )

            persistent = state[:, None].expand(-1, 2, -1, -1).reshape(
                batch * 2, REGISTER_COUNT, VALUE_COUNT
            )
            candidate_states, candidate_invalid = self.executor.apply_programs(
                persistent,
                actions,
                masks,
                hard_forward=hard_forward,
            )
            candidate_states = candidate_states.reshape(
                batch, 2, REGISTER_COUNT, VALUE_COUNT
            )
            state = torch.einsum("bk,bkrv->brv", selected, candidate_states)
            invalid_mass = invalid_mass + torch.einsum(
                "bk,bkr->br",
                selected,
                (candidate_invalid + probe_invalid).reshape(batch, 2, REGISTER_COUNT),
            )
            choice_logits.append(logits)
            choices.append(probabilities)

        answer = self.read_query(state, query_slots, hard_forward=hard_forward)
        stacked_logits = torch.stack(choice_logits, dim=1)
        stacked_choices = torch.stack(choices, dim=1)
        return JET1Output(
            evidence_probabilities=evidence_soft,
            choice_logits=stacked_logits,
            choice_probabilities=stacked_choices,
            selected_candidates=stacked_logits.argmax(-1),
            terminal_probabilities=state,
            answer_probabilities=answer,
            invalid_mass=invalid_mass,
        )

    def read_query(
        self,
        state: torch.Tensor,
        query_slots: torch.Tensor,
        *,
        hard_forward: bool,
    ) -> torch.Tensor:
        if (
            state.ndim != 3
            or state.shape[1:] != (REGISTER_COUNT, VALUE_COUNT)
            or query_slots.shape != (state.shape[0],)
            or query_slots.dtype != torch.long
        ):
            raise JET1ContractError("JET1 query tensor interface differs")
        route = self.query_route_logits[query_slots].float().softmax(-1)
        route = _hard_or_straight_through(route, hard_forward=hard_forward)
        return torch.einsum("br,brv->bv", route, state)

    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def source_audit() -> dict[str, object]:
    source = inspect.getsource(sys.modules[__name__])
    executable = source[: source.index("def source_audit")]
    forbidden = (
        "diverge_" + "mei1_data",
        "diverge_" + "v0",
        "exact_" + "action",
        "exact_" + "program",
        "PROGRAM" + "S",
        "apply_" + "transaction",
        "tokenizer.decode",
        "import re",
    )
    findings = [needle for needle in forbidden if needle in executable]
    return {"pass": not findings, "forbidden_findings": findings}


def architecture_receipt(model: JointEpistemicTrajectory) -> dict[str, object]:
    return {
        "config": asdict(model.config),
        "trajectory_trainable_parameters": model.trainable_parameters(),
        "whole_program_hard_forward": True,
        "fieldwise_hypothesis_averaging": False,
        "tied_recurrent_executor": True,
        "source_audit": source_audit(),
    }
