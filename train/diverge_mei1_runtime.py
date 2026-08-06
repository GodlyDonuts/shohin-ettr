#!/usr/bin/env python3
"""Candidate-only neural interfaces for the bounded DIVERGE-MEI1 gate.

This module deliberately has no dependency on DIVERGE's exact typed semantic
executor or query reader.  Exact mechanics may supervise or assess this model
from another process, but candidate execution is determined only by learned
parameters and the sealed discrete packet passed through this interface.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


SCHEMA = "shohin-diverge-mei1-model-owned-runtime-v1"
REGISTER_COUNT = 5
VALUE_COUNT = 128
ACTION_NAMES = ("ADD0_3", "SWAP01", "SWAP23", "SWAP34")
DELTAS = tuple(range(-3, 4))
EVIDENCE_FIELDS = REGISTER_COUNT * 2


class MEI1ContractError(ValueError):
    """Raised when candidate execution would become partial or ambiguous."""


@dataclass(frozen=True, slots=True)
class MEI1Config:
    input_width: int = 192
    evidence_width: int = 192
    evidence_heads: int = 4
    evidence_layers: int = 2
    evidence_ff_multiplier: int = 4
    register_count: int = REGISTER_COUNT
    value_count: int = VALUE_COUNT

    def validate(self) -> None:
        if self.input_width <= 0 or self.evidence_width <= 0:
            raise MEI1ContractError("model widths must be positive")
        if self.evidence_width % self.evidence_heads:
            raise MEI1ContractError("evidence width must divide its head count")
        if self.evidence_layers <= 0 or self.evidence_ff_multiplier <= 0:
            raise MEI1ContractError("evidence encoder geometry must be positive")
        if self.register_count != REGISTER_COUNT:
            raise MEI1ContractError("MEI1 v1 fixes five registers")
        if self.value_count != VALUE_COUNT:
            raise MEI1ContractError("MEI1 v1 fixes 128 categorical values")


@dataclass(frozen=True, slots=True)
class ModelState:
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.values) != REGISTER_COUNT:
            raise MEI1ContractError("model state has the wrong register count")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < VALUE_COUNT
            for value in self.values
        ):
            raise MEI1ContractError("model state value is outside the closed domain")

    def record(self) -> dict[str, object]:
        return {"values": list(self.values)}


@dataclass(frozen=True, slots=True)
class ModelChoice:
    record_index: int
    domain_value: int
    mass: int
    actions: tuple[int, ...]
    semantic_key: str
    provenance: str

    def __post_init__(self) -> None:
        if min(self.record_index, self.domain_value) < 0 or self.mass <= 0:
            raise MEI1ContractError("choice indices and mass are invalid")
        if any(not 0 <= action < len(ACTION_NAMES) for action in self.actions):
            raise MEI1ContractError("choice contains an unknown action token")
        if not self.semantic_key or not self.provenance:
            raise MEI1ContractError("choice identity and provenance must be nonempty")

    def record(self) -> dict[str, object]:
        return {
            "record_index": self.record_index,
            "domain_value": self.domain_value,
            "mass": self.mass,
            "actions": list(self.actions),
            "semantic_key": self.semantic_key,
            "provenance": self.provenance,
        }


def action_id(opcode: str, arguments: Sequence[int]) -> int:
    """Tokenize a typed packet action without implementing its semantics."""

    key = (str(opcode), tuple(int(value) for value in arguments))
    vocabulary = {
        ("ADD_VALUE", (0, 3)): 0,
        ("SWAP_VALUE", (0, 1)): 1,
        ("SWAP_VALUE", (2, 3)): 2,
        ("SWAP_VALUE", (3, 4)): 3,
    }
    if key not in vocabulary:
        raise MEI1ContractError(f"unrecognized typed action token {key!r}")
    return vocabulary[key]


class StructuredRegisterExecutor(nn.Module):
    """Learn a tied route-plus-delta transition from successor-state labels."""

    def __init__(self, config: MEI1Config):
        super().__init__()
        config.validate()
        self.config = config
        self.route_logits = nn.Parameter(
            torch.empty(len(ACTION_NAMES), REGISTER_COUNT, REGISTER_COUNT)
        )
        self.delta_logits = nn.Parameter(
            torch.empty(len(ACTION_NAMES), REGISTER_COUNT, len(DELTAS))
        )
        nn.init.normal_(self.route_logits, std=0.02)
        nn.init.normal_(self.delta_logits, std=0.02)

    def forward(
        self,
        state_values: torch.Tensor,
        action_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            state_values.ndim != 2
            or state_values.shape[1] != REGISTER_COUNT
            or state_values.dtype != torch.long
            or action_ids.shape != (state_values.shape[0],)
            or action_ids.dtype != torch.long
        ):
            raise MEI1ContractError("executor tensor interface differs")
        if state_values.numel() and (
            int(state_values.min()) < 0 or int(state_values.max()) >= VALUE_COUNT
        ):
            raise MEI1ContractError("executor state leaves the value domain")
        if action_ids.numel() and (
            int(action_ids.min()) < 0 or int(action_ids.max()) >= len(ACTION_NAMES)
        ):
            raise MEI1ContractError("executor action leaves the vocabulary")

        batch = state_values.shape[0]
        route = self.route_logits[action_ids].float().softmax(-1)
        delta = self.delta_logits[action_ids].float().softmax(-1)
        shifted = torch.zeros(
            batch,
            REGISTER_COUNT,
            len(DELTAS),
            VALUE_COUNT,
            dtype=route.dtype,
            device=state_values.device,
        )
        invalid = torch.zeros(
            batch,
            REGISTER_COUNT,
            len(DELTAS),
            dtype=route.dtype,
            device=state_values.device,
        )
        for delta_index, offset in enumerate(DELTAS):
            targets = state_values + offset
            valid = targets.ge(0) & targets.lt(VALUE_COUNT)
            shifted[:, :, delta_index].scatter_(
                2,
                targets.clamp(0, VALUE_COUNT - 1).unsqueeze(-1),
                valid.to(route.dtype).unsqueeze(-1),
            )
            invalid[:, :, delta_index] = (~valid).to(route.dtype)
        probabilities = torch.einsum(
            "boi,bod,bidv->bov", route, delta, shifted
        )
        invalid_mass = torch.einsum("boi,bod,bid->bo", route, delta, invalid)
        return probabilities, invalid_mass

    @torch.no_grad()
    def hard_step(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Apply the learned discrete operator; any range violation fails closed."""

        if (
            states.ndim != 2
            or states.shape[1] != REGISTER_COUNT
            or states.dtype != torch.long
            or actions.shape != (states.shape[0],)
            or actions.dtype != torch.long
        ):
            raise MEI1ContractError("hard executor tensor interface differs")
        routes = self.route_logits[actions].argmax(-1)
        delta_indices = self.delta_logits[actions].argmax(-1)
        offsets = torch.tensor(DELTAS, device=states.device)[delta_indices]
        selected = states.gather(1, routes)
        output = selected + offsets
        if output.numel() and (int(output.min()) < 0 or int(output.max()) >= VALUE_COUNT):
            raise MEI1ContractError("learned executor produced an out-of-range value")
        return output


class StructuredQueryReader(nn.Module):
    """Learn which complete-state register answers each typed late query."""

    def __init__(self, config: MEI1Config):
        super().__init__()
        config.validate()
        self.config = config
        self.route_logits = nn.Parameter(
            torch.empty(REGISTER_COUNT, REGISTER_COUNT)
        )
        nn.init.normal_(self.route_logits, std=0.02)

    def forward(
        self,
        state_values: torch.Tensor,
        query_slots: torch.Tensor,
    ) -> torch.Tensor:
        if (
            state_values.ndim != 2
            or state_values.shape[1] != REGISTER_COUNT
            or state_values.dtype != torch.long
            or query_slots.shape != (state_values.shape[0],)
            or query_slots.dtype != torch.long
        ):
            raise MEI1ContractError("query reader tensor interface differs")
        if state_values.numel() and (
            int(state_values.min()) < 0 or int(state_values.max()) >= VALUE_COUNT
        ):
            raise MEI1ContractError("query reader state leaves the value domain")
        if query_slots.numel() and (
            int(query_slots.min()) < 0 or int(query_slots.max()) >= REGISTER_COUNT
        ):
            raise MEI1ContractError("query reader slot leaves the query domain")
        route = self.route_logits[query_slots].float().softmax(-1)
        values = F.one_hot(state_values, VALUE_COUNT).to(route.dtype)
        return torch.einsum("bi,biv->bv", route, values)

    @torch.no_grad()
    def hard_read(self, states: torch.Tensor, query_slots: torch.Tensor) -> torch.Tensor:
        routes = self.route_logits[query_slots].argmax(-1)
        return states.gather(1, routes.unsqueeze(-1)).squeeze(-1)


class EvidenceInterpreter(nn.Module):
    """Read ten probe-state fields from delayed natural-language evidence."""

    def __init__(self, config: MEI1Config):
        super().__init__()
        config.validate()
        self.config = config
        width = config.evidence_width
        self.input_projection = nn.Linear(config.input_width, width, bias=False)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config.evidence_heads,
            dim_feedforward=width * config.evidence_ff_multiplier,
            batch_first=True,
            norm_first=True,
            activation="gelu",
            dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.evidence_layers,
            enable_nested_tensor=False,
        )
        self.field_queries = nn.Parameter(torch.empty(EVIDENCE_FIELDS, width))
        self.cross_attention = nn.MultiheadAttention(
            width, config.evidence_heads, batch_first=True
        )
        self.output_norm = nn.LayerNorm(width)
        self.value_head = nn.Linear(width, VALUE_COUNT)
        nn.init.normal_(self.field_queries, std=0.02)

    def forward(
        self,
        word_features: torch.Tensor,
        word_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            word_features.ndim != 3
            or word_features.shape[2] != self.config.input_width
            or word_mask.shape != word_features.shape[:2]
            or word_mask.dtype != torch.bool
            or not word_mask.any(-1).all()
        ):
            raise MEI1ContractError("evidence interpreter tensor interface differs")
        memory = self.input_projection(word_features)
        memory = self.encoder(memory, src_key_padding_mask=~word_mask)
        queries = self.field_queries.unsqueeze(0).expand(word_features.shape[0], -1, -1)
        fields, _ = self.cross_attention(
            queries,
            memory,
            memory,
            key_padding_mask=~word_mask,
            need_weights=False,
        )
        return self.value_head(self.output_norm(fields + queries)).float()

    @torch.no_grad()
    def hard_states(
        self,
        word_features: torch.Tensor,
        word_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = self(word_features, word_mask).argmax(-1)
        return values[:, :REGISTER_COUNT], values[:, REGISTER_COUNT:]


class DIVERGEMEI1(nn.Module):
    def __init__(self, config: MEI1Config):
        super().__init__()
        config.validate()
        self.config = config
        self.evidence = EvidenceInterpreter(config)
        self.executor = StructuredRegisterExecutor(config)
        self.query = StructuredQueryReader(config)

    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@dataclass(frozen=True)
class ExpressionNode:
    kind: str
    parent: int | None = None
    variable: int | None = None
    choice: int | None = None
    mass: int = 1
    children: tuple[int, ...] = ()


class ExpressionArena:
    """Exact hash-consed disjoint lineage expressions; no semantic executor."""

    def __init__(self) -> None:
        self.nodes = [ExpressionNode("base")]
        self._intern = {self.nodes[0]: 0}

    @property
    def base(self) -> int:
        return 0

    def _add(self, node: ExpressionNode) -> int:
        previous = self._intern.get(node)
        if previous is not None:
            return previous
        index = len(self.nodes)
        self.nodes.append(node)
        self._intern[node] = index
        self.assignment_count.cache_clear()
        self.total_mass.cache_clear()
        return index

    def extend(self, parent: int, variable: int, choice: int, mass: int) -> int:
        if not 0 <= parent < len(self.nodes) or min(variable, choice) < 0 or mass <= 0:
            raise MEI1ContractError("invalid lineage extension")
        return self._add(ExpressionNode("extend", parent, variable, choice, mass))

    def union(self, roots: Iterable[int]) -> int:
        children: list[int] = []
        for root in roots:
            if not 0 <= root < len(self.nodes):
                raise MEI1ContractError("lineage union child is absent")
            node = self.nodes[root]
            children.extend(node.children if node.kind == "union" else (root,))
        children = sorted(set(children))
        if not children:
            raise MEI1ContractError("lineage union cannot be empty")
        if len(children) == 1:
            return children[0]
        return self._add(ExpressionNode("union", children=tuple(children)))

    @lru_cache(maxsize=None)
    def assignment_count(self, root: int) -> int:
        node = self.nodes[root]
        if node.kind == "base":
            return 1
        if node.kind == "extend":
            assert node.parent is not None
            return self.assignment_count(node.parent)
        return sum(self.assignment_count(child) for child in node.children)

    @lru_cache(maxsize=None)
    def total_mass(self, root: int) -> int:
        node = self.nodes[root]
        if node.kind == "base":
            return 1
        if node.kind == "extend":
            assert node.parent is not None
            return node.mass * self.total_mass(node.parent)
        return sum(self.total_mass(child) for child in node.children)

    def constrained_mass(
        self,
        root: int,
        allowed: Mapping[int, frozenset[int]],
    ) -> int:
        memo: dict[int, int] = {}

        def visit(index: int) -> int:
            if index in memo:
                return memo[index]
            node = self.nodes[index]
            if node.kind == "base":
                value = 1
            elif node.kind == "extend":
                assert node.parent is not None and node.variable is not None
                assert node.choice is not None
                permit = allowed.get(node.variable)
                value = (
                    0
                    if permit is not None and node.choice not in permit
                    else node.mass * visit(node.parent)
                )
            else:
                value = sum(visit(child) for child in node.children)
            memo[index] = value
            return value

        return visit(root)

    def accepts(self, root: int, assignment: Sequence[int]) -> bool:
        memo: dict[int, bool] = {}

        def visit(index: int) -> bool:
            if index in memo:
                return memo[index]
            node = self.nodes[index]
            if node.kind == "base":
                value = True
            elif node.kind == "extend":
                assert node.parent is not None and node.variable is not None
                assert node.choice is not None
                value = (
                    node.variable < len(assignment)
                    and assignment[node.variable] == node.choice
                    and visit(node.parent)
                )
            else:
                value = any(visit(child) for child in node.children)
            memo[index] = value
            return value

        return visit(root)


@dataclass(frozen=True, slots=True)
class ModelStateGroup:
    state: ModelState
    expression: int


@dataclass(frozen=True, slots=True)
class ModelExecution:
    arena: ExpressionArena
    groups: tuple[ModelStateGroup, ...]
    choices: tuple[tuple[ModelChoice, ...], ...]
    represented_worlds: int
    unique_action_applications: int
    logical_action_applications: int
    peak_groups: int
    overflow: bool = False


@dataclass(frozen=True, slots=True)
class ModelQueryDecision:
    disposition: str
    answer: int | None
    marginal: tuple[tuple[int, int], ...]
    total_mass: int


def _validate_choices(
    choices: Sequence[Sequence[ModelChoice]],
) -> tuple[tuple[ModelChoice, ...], ...]:
    rows = []
    for record_index, raw in enumerate(choices):
        row = tuple(sorted(raw, key=lambda item: item.domain_value))
        if not row or any(item.record_index != record_index for item in row):
            raise MEI1ContractError("model choices are not record aligned")
        if tuple(item.domain_value for item in row) != tuple(range(len(row))):
            raise MEI1ContractError("model choice domains must be contiguous")
        if len({item.semantic_key for item in row}) != len(row):
            raise MEI1ContractError("model choice semantics are duplicated")
        rows.append(row)
    if not rows:
        raise MEI1ContractError("model execution requires at least one record")
    return tuple(rows)


@torch.no_grad()
def apply_action_sequences(
    executor: StructuredRegisterExecutor,
    states: Sequence[ModelState],
    action_sequences: Sequence[Sequence[int]],
) -> tuple[ModelState, ...]:
    if len(states) != len(action_sequences):
        raise MEI1ContractError("state and action sequence batches differ")
    if not states:
        return ()
    device = executor.route_logits.device
    values = torch.tensor([state.values for state in states], dtype=torch.long, device=device)
    maximum = max((len(row) for row in action_sequences), default=0)
    for step in range(maximum):
        active = [index for index, row in enumerate(action_sequences) if step < len(row)]
        if not active:
            continue
        indices = torch.tensor(active, dtype=torch.long, device=device)
        actions = torch.tensor(
            [action_sequences[index][step] for index in active],
            dtype=torch.long,
            device=device,
        )
        updated = executor.hard_step(values.index_select(0, indices), actions)
        values.index_copy_(0, indices, updated)
    return tuple(ModelState(tuple(int(value) for value in row)) for row in values.cpu().tolist())


@torch.no_grad()
def execute_model_mdd(
    initial_state: ModelState,
    choices: Sequence[Sequence[ModelChoice]],
    executor: StructuredRegisterExecutor,
    *,
    max_nodes: int = 1_000_000,
    max_groups: int = 100_000,
) -> ModelExecution:
    choices = _validate_choices(choices)
    arena = ExpressionArena()
    groups = (ModelStateGroup(initial_state, arena.base),)
    unique = 0
    logical = 0
    peak = 1
    represented = 1
    for variable, row in enumerate(choices):
        states = []
        sequences = []
        metadata = []
        for group in groups:
            prefixes = arena.assignment_count(group.expression)
            for choice in row:
                states.append(group.state)
                sequences.append(choice.actions)
                metadata.append((group.expression, choice, prefixes))
                unique += len(choice.actions)
                logical += prefixes * len(choice.actions)
        outputs = apply_action_sequences(executor, states, sequences)
        next_groups: dict[tuple[int, ...], tuple[ModelState, list[int]]] = {}
        for state, (parent, choice, _) in zip(outputs, metadata, strict=True):
            expression = arena.extend(
                parent, variable, choice.domain_value, choice.mass
            )
            if state.values not in next_groups:
                next_groups[state.values] = (state, [])
            next_groups[state.values][1].append(expression)
        groups = tuple(
            ModelStateGroup(state, arena.union(expressions))
            for _, (state, expressions) in sorted(next_groups.items())
        )
        represented *= len(row)
        peak = max(peak, len(groups))
        if len(arena.nodes) > max_nodes or len(groups) > max_groups:
            return ModelExecution(arena, (), choices, 0, unique, logical, peak, True)
    if sum(arena.assignment_count(group.expression) for group in groups) != represented:
        raise AssertionError("model MDD lost or duplicated a complete lineage")
    return ModelExecution(arena, groups, choices, represented, unique, logical, peak)


@torch.no_grad()
def derive_model_allowed(
    choices: Sequence[Sequence[ModelChoice]],
    probe_before: Sequence[ModelState],
    probe_after: Sequence[ModelState],
    executor: StructuredRegisterExecutor,
) -> dict[int, frozenset[int]]:
    if not (len(choices) == len(probe_before) == len(probe_after)):
        raise MEI1ContractError("evidence does not cover every model record")
    allowed = {}
    for record_index, (row, before, after) in enumerate(
        zip(choices, probe_before, probe_after, strict=True)
    ):
        predicted = apply_action_sequences(
            executor,
            [before] * len(row),
            [choice.actions for choice in row],
        )
        permit = frozenset(
            choice.domain_value
            for choice, state in zip(row, predicted, strict=True)
            if state == after
        )
        if not permit:
            raise MEI1ContractError("learned evidence removes the complete domain")
        allowed[record_index] = permit
    return allowed


@torch.no_grad()
def query_model_mdd(
    execution: ModelExecution,
    query_slot: int,
    reader: StructuredQueryReader,
    *,
    allowed: Mapping[int, frozenset[int]] | None = None,
) -> ModelQueryDecision:
    if execution.overflow:
        return ModelQueryDecision("OVERFLOW", None, (), 0)
    if not 0 <= query_slot < REGISTER_COUNT:
        raise MEI1ContractError("model query slot leaves the domain")
    allowed = allowed or {}
    retained = []
    masses = []
    for group in execution.groups:
        mass = execution.arena.constrained_mass(group.expression, allowed)
        if mass:
            retained.append(group.state)
            masses.append(mass)
    if not retained:
        return ModelQueryDecision("REJECT", None, (), 0)
    device = reader.route_logits.device
    states = torch.tensor(
        [state.values for state in retained], dtype=torch.long, device=device
    )
    slots = torch.full((len(retained),), query_slot, dtype=torch.long, device=device)
    answers = reader.hard_read(states, slots).cpu().tolist()
    marginal: dict[int, int] = {}
    for answer, mass in zip(answers, masses, strict=True):
        marginal[int(answer)] = marginal.get(int(answer), 0) + mass
    items = tuple(sorted(marginal.items()))
    total = sum(marginal.values())
    if len(items) == 1:
        return ModelQueryDecision("ANSWER", items[0][0], items, total)
    return ModelQueryDecision("ABSTAIN", None, items, total)


def support_contains(execution: ModelExecution, assignment: Sequence[int]) -> bool:
    return any(
        execution.arena.accepts(group.expression, assignment)
        for group in execution.groups
    )


def source_audit() -> dict[str, object]:
    source = Path(__file__).read_text(encoding="utf-8")
    forbidden = (
        "diverge_v0 import",
        "apply" + "_transaction",
        "read" + "_query",
        "execute" + "_packet",
    )
    # Exclude this audit's own split string literals from the search.
    executable = source[: source.index("def source_audit")]
    hits = [token for token in forbidden if token in executable]
    return {
        "forbidden_hits": hits,
        "pass": not hits,
        "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }


def architecture_receipt(config: MEI1Config) -> dict[str, object]:
    config.validate()
    model = DIVERGEMEI1(config)
    return {
        "schema": SCHEMA,
        "config": asdict(config),
        "action_names": list(ACTION_NAMES),
        "deltas": list(DELTAS),
        "trainable_parameters": model.trainable_parameters(),
        "state_fields": ["five-complete-categorical-register-values"],
        "whole_hypothesis_only": True,
        "candidate_source_audit": source_audit(),
    }


if __name__ == "__main__":
    print(json.dumps(architecture_receipt(MEI1Config()), sort_keys=True, indent=2))
