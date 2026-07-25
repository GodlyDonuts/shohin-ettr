#!/usr/bin/env python3
"""Bounded verifier-shaped search falsifier for the SSQAC primitive VM.

The candidate in this module receives only an immutable, hash-sealed field
matrix and the legal transition semantics of the primitive row machine.  It
does not receive a source string, a workspace, a late query, an expert trace,
an answer, or a verifier callback.  Candidate search uses a source-independent
RREF defect vector as a potential; a separate assessor executes and verifies
the final primitive program only after search has terminated.

This is deliberately a mechanics falsifier, not a reasoning result.  Bounded
beam search over legal row repairs is a conventional search procedure.  Even a
perfect score would establish only that a weak local scorer can be repaired by
explicit internal search on this finite algebraic task.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Iterable, Mapping, Sequence

from episode_functor_algebra_machine import (
    FIELD_MODULUS,
    OP_AXPY,
    OP_HALT,
    OP_INV,
    OP_LOAD,
    OP_NEG,
    OP_SCALE,
    OP_SWAP,
    AlgebraInstruction,
    AlgebraMachineError,
    AlgebraMachineState,
    execute_program,
    verify_reduction_program,
)


CANDIDATE_PACKET_SCHEMA = "ssqac_sealed_algebra_candidate_packet_v1"
SEARCH_RECEIPT_SCHEMA = "ssqac_verifier_guided_search_receipt_v1"
ASSESSMENT_SCHEMA = "ssqac_separate_program_assessment_v1"
BENCHMARK_SCHEMA = "ssqac_verifier_guided_search_falsifier_v1"
STATUS = "mechanics_falsifier_only_not_reasoning"
GUIDANCE_STRUCTURAL = "structural_rref_defect"
GUIDANCE_RANDOM_RELABEL = "hash_random_relabel_control"
STRATEGY_GREEDY = "policy_only_greedy"
STRATEGY_BEAM = "bounded_verifier_shaped_beam"
FORBIDDEN_CANDIDATE_FIELDS = frozenset(
    {
        "answer",
        "completion",
        "expert",
        "label",
        "oracle",
        "query",
        "source",
        "target",
        "verifier",
        "workspace",
    }
)


class SearchFalsifierError(ValueError):
    """A sealed packet, search budget, or receipt failed closed."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise SearchFalsifierError("value is not canonical ASCII JSON data") from error


def _digest(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _plain_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SearchFalsifierError(f"{label} must be a plain integer")
    return value


def _canonical_rows(
    rows: Iterable[Iterable[int]],
) -> tuple[tuple[int, ...], ...]:
    frozen = tuple(
        tuple(
            _plain_int(value, label="matrix coefficient") % FIELD_MODULUS
            for value in row
        )
        for row in rows
    )
    if not frozen or not frozen[0]:
        raise SearchFalsifierError("sealed matrix must be nonempty")
    width = len(frozen[0])
    if any(len(row) != width for row in frozen):
        raise SearchFalsifierError("sealed matrix must be rectangular")
    # Replaying the empty legal program delegates the VM's public geometry
    # limits to the transition semantics without exposing any assessor.
    try:
        execute_program(frozen, ())
    except AlgebraMachineError as error:
        raise SearchFalsifierError("sealed matrix violates VM bounds") from error
    return frozen


@dataclass(frozen=True, slots=True)
class SealedAlgebraPacket:
    """The exact and only candidate-visible payload."""

    schema: str
    field_modulus: int
    register_count: int
    rows: tuple[tuple[int, ...], ...]
    seal_sha256: str

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Iterable[int]],
        *,
        register_count: int = 4,
    ) -> "SealedAlgebraPacket":
        frozen = _canonical_rows(rows)
        registers = _plain_int(register_count, label="register count")
        payload = {
            "field_modulus": FIELD_MODULUS,
            "register_count": registers,
            "rows": [list(row) for row in frozen],
            "schema": CANDIDATE_PACKET_SCHEMA,
        }
        packet = cls(
            schema=CANDIDATE_PACKET_SCHEMA,
            field_modulus=FIELD_MODULUS,
            register_count=registers,
            rows=frozen,
            seal_sha256=_digest(payload),
        )
        packet.validate()
        return packet

    def payload_data(self) -> dict[str, object]:
        return {
            "field_modulus": self.field_modulus,
            "register_count": self.register_count,
            "rows": [list(row) for row in self.rows],
            "schema": self.schema,
        }

    def canonical_data(self) -> dict[str, object]:
        return {**self.payload_data(), "seal_sha256": self.seal_sha256}

    def validate(self) -> None:
        if self.schema != CANDIDATE_PACKET_SCHEMA:
            raise SearchFalsifierError("candidate packet schema differs")
        if self.field_modulus != FIELD_MODULUS:
            raise SearchFalsifierError("candidate packet field differs")
        if (
            not 1
            <= _plain_int(
                self.register_count,
                label="register count",
            )
            <= 16
        ):
            raise SearchFalsifierError("candidate register count is out of range")
        if _canonical_rows(self.rows) != self.rows:
            raise SearchFalsifierError("candidate rows are not canonical")
        if self.seal_sha256 != _digest(self.payload_data()):
            raise SearchFalsifierError("candidate packet seal differs")


def load_candidate_packet(value: Mapping[str, object]) -> SealedAlgebraPacket:
    """Load an exact-key packet and reject any metadata side channel."""

    if not isinstance(value, Mapping):
        raise SearchFalsifierError("candidate packet must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise SearchFalsifierError("candidate packet keys must be strings")
    forbidden = sorted(
        key
        for key in value
        if any(token in key.lower() for token in FORBIDDEN_CANDIDATE_FIELDS)
    )
    if forbidden:
        raise SearchFalsifierError(
            f"candidate packet contains forbidden fields: {forbidden}"
        )
    expected = {
        "field_modulus",
        "register_count",
        "rows",
        "schema",
        "seal_sha256",
    }
    if set(value) != expected:
        raise SearchFalsifierError("candidate packet keys differ")
    raw_rows = value["rows"]
    if isinstance(raw_rows, (str, bytes, bytearray)) or not isinstance(
        raw_rows, Sequence
    ):
        raise SearchFalsifierError("candidate rows must be a sequence")
    rows: list[tuple[int, ...]] = []
    for raw_row in raw_rows:
        if isinstance(raw_row, (str, bytes, bytearray)) or not isinstance(
            raw_row, Sequence
        ):
            raise SearchFalsifierError("candidate row must be a sequence")
        rows.append(
            tuple(
                _plain_int(coefficient, label="matrix coefficient")
                for coefficient in raw_row
            )
        )
    packet = SealedAlgebraPacket(
        schema=value["schema"] if isinstance(value["schema"], str) else "",
        field_modulus=_plain_int(
            value["field_modulus"],
            label="field modulus",
        ),
        register_count=_plain_int(
            value["register_count"],
            label="register count",
        ),
        rows=tuple(rows),
        seal_sha256=value["seal_sha256"]
        if isinstance(value["seal_sha256"], str)
        else "",
    )
    packet.validate()
    return packet


@dataclass(frozen=True, slots=True)
class StructuralPotential:
    """A source-independent defect vector that is zero exactly at RREF."""

    zero_row_order_pairs: int
    pivot_order_pairs: int
    nonunit_pivots: int
    pivot_column_extras: int

    @property
    def solved(self) -> bool:
        return (
            self.zero_row_order_pairs == 0
            and self.pivot_order_pairs == 0
            and self.nonunit_pivots == 0
            and self.pivot_column_extras == 0
        )

    @property
    def weighted_total(self) -> int:
        return (
            9 * self.zero_row_order_pairs
            + 9 * self.pivot_order_pairs
            + 3 * self.nonunit_pivots
            + 2 * self.pivot_column_extras
        )

    def canonical_data(self) -> list[int]:
        return [
            self.zero_row_order_pairs,
            self.pivot_order_pairs,
            self.nonunit_pivots,
            self.pivot_column_extras,
        ]


def structural_potential(
    rows: Sequence[Sequence[int]],
) -> StructuralPotential:
    """Measure only canonical row-form defects, never a target answer."""

    leading: list[int | None] = []
    nonunit = 0
    for row in rows:
        pivot = next((index for index, value in enumerate(row) if value), None)
        leading.append(pivot)
        if pivot is not None and row[pivot] % FIELD_MODULUS != 1:
            nonunit += 1
    zero_order = sum(
        1
        for left in range(len(rows))
        for right in range(left + 1, len(rows))
        if leading[left] is None and leading[right] is not None
    )
    pivot_order = sum(
        1
        for left in range(len(rows))
        for right in range(left + 1, len(rows))
        if leading[left] is not None
        and leading[right] is not None
        and leading[left] >= leading[right]
    )
    extras = 0
    for row_index, pivot in enumerate(leading):
        if pivot is None:
            continue
        extras += sum(
            1
            for other_index, other_row in enumerate(rows)
            if other_index != row_index and other_row[pivot] % FIELD_MODULUS != 0
        )
    return StructuralPotential(
        zero_row_order_pairs=zero_order,
        pivot_order_pairs=pivot_order,
        nonunit_pivots=nonunit,
        pivot_column_extras=extras,
    )


@dataclass(frozen=True, slots=True)
class RepairAction:
    """One search edge, expanded into public primitive VM instructions."""

    kind: str
    instructions: tuple[AlgebraInstruction, ...]
    pivot_row: int
    target_row: int
    pivot_column: int

    def canonical_data(self) -> list[object]:
        return [
            self.kind,
            self.pivot_row,
            self.target_row,
            self.pivot_column,
            [instruction.canonical_data() for instruction in self.instructions],
        ]

    @property
    def action_sha256(self) -> str:
        return _digest(self.canonical_data())


def enumerate_legal_repair_actions(
    state: AlgebraMachineState,
) -> tuple[RepairAction, ...]:
    """Enumerate source-blind row repairs whose legality is VM-replayed."""

    rows = state.rows
    leading = tuple(
        next((column for column, value in enumerate(row) if value), None)
        for row in rows
    )
    actions: list[RepairAction] = []
    for row_index, pivot in enumerate(leading):
        if pivot is None:
            continue
        value = rows[row_index][pivot]
        if value != 1:
            actions.append(
                RepairAction(
                    kind="NORMALIZE",
                    instructions=(
                        AlgebraInstruction(OP_LOAD, row_index, pivot, 0),
                        AlgebraInstruction(OP_INV, 0, 1),
                        AlgebraInstruction(OP_SCALE, row_index, 1),
                    ),
                    pivot_row=row_index,
                    target_row=row_index,
                    pivot_column=pivot,
                )
            )
        else:
            for target_row, target in enumerate(rows):
                if target_row != row_index and target[pivot] % FIELD_MODULUS != 0:
                    actions.append(
                        RepairAction(
                            kind="ELIMINATE",
                            instructions=(
                                AlgebraInstruction(
                                    OP_LOAD,
                                    target_row,
                                    pivot,
                                    0,
                                ),
                                AlgebraInstruction(OP_NEG, 0, 2),
                                AlgebraInstruction(
                                    OP_AXPY,
                                    target_row,
                                    row_index,
                                    2,
                                ),
                            ),
                            pivot_row=row_index,
                            target_row=target_row,
                            pivot_column=pivot,
                        )
                    )
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if rows[left] == rows[right]:
                continue
            actions.append(
                RepairAction(
                    kind="SWAP",
                    instructions=(AlgebraInstruction(OP_SWAP, left, right),),
                    pivot_row=left,
                    target_row=right,
                    pivot_column=-1,
                )
            )
    return tuple(
        sorted(
            actions,
            key=lambda action: (
                action.kind,
                action.pivot_row,
                action.target_row,
                action.pivot_column,
                action.action_sha256,
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class WeakLocalScorer:
    """A noisy geometry-independent proxy for a weak local neural policy."""

    seed: int
    noise_scale: float = 8.0
    relabel_action_kinds: bool = False

    @property
    def policy_sha256(self) -> str:
        return _digest(asdict(self))

    def _kind(self, kind: str) -> str:
        if not self.relabel_action_kinds:
            return kind
        labels = ("ELIMINATE", "NORMALIZE", "SWAP")
        shuffled = list(labels)
        random.Random(self.seed ^ 0x5A17).shuffle(shuffled)
        return dict(zip(labels, shuffled, strict=True))[kind]

    def score(
        self,
        *,
        before: StructuralPotential,
        after: StructuralPotential,
        action: RepairAction,
        state_sha256: str,
    ) -> float:
        """Return a lower-is-better local action score."""

        kind_bias = {
            "ELIMINATE": -0.25,
            "NORMALIZE": -0.15,
            "SWAP": 0.0,
        }[self._kind(action.kind)]
        delta = before.weighted_total - after.weighted_total
        distance = abs(action.pivot_row - action.target_row)
        noise_raw = sha256(
            (f"{self.seed}:{state_sha256}:{action.action_sha256}").encode("ascii")
        ).digest()
        unit = int.from_bytes(noise_raw[:8], "big") / (2**64 - 1)
        noise = (2.0 * unit - 1.0) * self.noise_scale
        return (
            0.03 * after.weighted_total
            - 0.08 * delta
            + 0.08 * len(action.instructions)
            + 0.03 * distance
            + kind_bias
            + noise
        )


@dataclass(frozen=True, slots=True)
class SearchBudget:
    max_nodes_expanded: int
    max_edges_considered: int
    max_depth: int
    max_frontier: int
    beam_width: int
    max_program_instructions: int

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if _plain_int(value, label=name) < 1:
                raise SearchFalsifierError(f"{name} must be positive")
        if self.beam_width > self.max_frontier:
            raise SearchFalsifierError("beam width cannot exceed the frontier hard cap")


@dataclass(frozen=True, slots=True)
class SearchReceipt:
    schema: str
    status: str
    strategy: str
    guidance: str
    candidate_packet_sha256: str
    policy_sha256: str
    budget_sha256: str
    completed: bool
    termination: str
    nodes_expanded: int
    nodes_generated: int
    nodes_deduplicated: int
    nodes_pruned: int
    edges_considered: int
    edges_legal: int
    edges_rejected: int
    maximum_depth_reached: int
    peak_frontier: int
    final_program_instructions: int
    max_nodes_expanded: int
    max_edges_considered: int
    max_depth: int
    max_frontier: int
    beam_width: int
    max_program_instructions: int
    search_trace_sha256: str
    program_sha256: str

    @property
    def hard_budgets_respected(self) -> bool:
        return (
            self.nodes_expanded <= self.max_nodes_expanded
            and self.edges_considered <= self.max_edges_considered
            and self.maximum_depth_reached <= self.max_depth
            and self.peak_frontier <= self.max_frontier
            and self.final_program_instructions <= self.max_program_instructions
        )

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(asdict(self)) + b"\n"


@dataclass(frozen=True, slots=True)
class CandidateSearchResult:
    program: tuple[AlgebraInstruction, ...] | None
    receipt: SearchReceipt


@dataclass(slots=True)
class _Counters:
    nodes_expanded: int = 0
    nodes_generated: int = 0
    nodes_deduplicated: int = 0
    nodes_pruned: int = 0
    edges_considered: int = 0
    edges_legal: int = 0
    edges_rejected: int = 0
    maximum_depth_reached: int = 0
    peak_frontier: int = 1


@dataclass(frozen=True, slots=True)
class _Node:
    state: AlgebraMachineState
    program: tuple[AlgebraInstruction, ...]
    depth: int
    cumulative_policy_score: float
    rank: float

    @property
    def state_sha256(self) -> str:
        # Registers are scratch values.  Every admitted macro overwrites every
        # register it consumes, so row equality is the exact future-behavior
        # quotient for this action grammar.
        return _digest([list(row) for row in self.state.rows])

    @property
    def program_sha256(self) -> str:
        return _program_sha256(self.program)


def _program_sha256(program: Sequence[AlgebraInstruction]) -> str:
    return _digest([instruction.canonical_data() for instruction in program])


def _guidance_value(
    state: AlgebraMachineState,
    potential: StructuralPotential,
    *,
    mode: str,
    seed: int,
) -> int:
    if mode == GUIDANCE_STRUCTURAL:
        return potential.weighted_total
    if mode != GUIDANCE_RANDOM_RELABEL:
        raise SearchFalsifierError("unknown search guidance mode")
    ceiling = max(
        1,
        16 * len(state.rows) * len(state.rows)
        + 4 * len(state.rows) * len(state.rows[0]),
    )
    material = _canonical_bytes(
        {
            "potential": potential.canonical_data(),
            "rows": [list(row) for row in state.rows],
            "seed": seed,
        }
    )
    return int.from_bytes(sha256(material).digest()[:8], "big") % ceiling


def _make_receipt(
    *,
    packet: SealedAlgebraPacket,
    policy: WeakLocalScorer,
    budget: SearchBudget,
    strategy: str,
    guidance: str,
    counters: _Counters,
    completed: bool,
    termination: str,
    program: tuple[AlgebraInstruction, ...] | None,
    trace_sha256: str,
) -> SearchReceipt:
    frozen_program = program or ()
    receipt = SearchReceipt(
        schema=SEARCH_RECEIPT_SCHEMA,
        status=STATUS,
        strategy=strategy,
        guidance=guidance,
        candidate_packet_sha256=_digest(packet.canonical_data()),
        policy_sha256=policy.policy_sha256,
        budget_sha256=_digest(asdict(budget)),
        completed=completed,
        termination=termination,
        nodes_expanded=counters.nodes_expanded,
        nodes_generated=counters.nodes_generated,
        nodes_deduplicated=counters.nodes_deduplicated,
        nodes_pruned=counters.nodes_pruned,
        edges_considered=counters.edges_considered,
        edges_legal=counters.edges_legal,
        edges_rejected=counters.edges_rejected,
        maximum_depth_reached=counters.maximum_depth_reached,
        peak_frontier=counters.peak_frontier,
        final_program_instructions=len(frozen_program),
        max_nodes_expanded=budget.max_nodes_expanded,
        max_edges_considered=budget.max_edges_considered,
        max_depth=budget.max_depth,
        max_frontier=budget.max_frontier,
        beam_width=budget.beam_width,
        max_program_instructions=budget.max_program_instructions,
        search_trace_sha256=trace_sha256,
        program_sha256=_program_sha256(frozen_program),
    )
    if not receipt.hard_budgets_respected:
        raise RuntimeError("search escaped a declared hard resource budget")
    return receipt


def _finish_program(
    node: _Node,
    budget: SearchBudget,
) -> tuple[AlgebraInstruction, ...] | None:
    program = (*node.program, AlgebraInstruction(OP_HALT))
    if len(program) > budget.max_program_instructions:
        return None
    return program


def _apply_action(
    packet: SealedAlgebraPacket,
    node: _Node,
    action: RepairAction,
    *,
    budget: SearchBudget,
) -> tuple[AlgebraMachineState, tuple[AlgebraInstruction, ...]] | None:
    program = (*node.program, *action.instructions)
    if len(program) + 1 > budget.max_program_instructions:
        return None
    state = execute_program(
        packet.rows,
        program,
        register_count=packet.register_count,
        maximum_instructions=budget.max_program_instructions,
    )
    return state, program


def greedy_candidate_search(
    packet: SealedAlgebraPacket,
    policy: WeakLocalScorer,
    budget: SearchBudget,
) -> CandidateSearchResult:
    """Run policy-only greedy decoding without any assessor invocation."""

    packet.validate()
    budget.validate()
    root_state = execute_program(
        packet.rows,
        (),
        register_count=packet.register_count,
    )
    node = _Node(
        state=root_state,
        program=(),
        depth=0,
        cumulative_policy_score=0.0,
        rank=0.0,
    )
    visited = {node.state_sha256}
    counters = _Counters()
    trace = sha256()
    trace.update(STRATEGY_GREEDY.encode("ascii"))
    termination = "max_depth"
    completed_program: tuple[AlgebraInstruction, ...] | None = None

    for depth in range(budget.max_depth + 1):
        counters.maximum_depth_reached = max(
            counters.maximum_depth_reached,
            depth,
        )
        potential = structural_potential(node.state.rows)
        if potential.solved:
            completed_program = _finish_program(node, budget)
            termination = (
                "candidate_goal"
                if completed_program is not None
                else "instruction_budget"
            )
            break
        if depth == budget.max_depth:
            break
        if counters.nodes_expanded >= budget.max_nodes_expanded:
            termination = "node_budget"
            break
        counters.nodes_expanded += 1
        candidates: list[
            tuple[float, str, AlgebraMachineState, tuple[AlgebraInstruction, ...]]
        ] = []
        for action in enumerate_legal_repair_actions(node.state):
            if counters.edges_considered >= budget.max_edges_considered:
                termination = "edge_budget"
                break
            counters.edges_considered += 1
            try:
                applied = _apply_action(
                    packet,
                    node,
                    action,
                    budget=budget,
                )
            except AlgebraMachineError:
                counters.edges_rejected += 1
                continue
            if applied is None:
                counters.nodes_pruned += 1
                continue
            state, program = applied
            counters.edges_legal += 1
            counters.nodes_generated += 1
            state_sha = _digest([list(row) for row in state.rows])
            trace.update(
                _canonical_bytes([node.state_sha256, action.action_sha256, state_sha])
            )
            if state_sha in visited:
                counters.nodes_deduplicated += 1
                continue
            after = structural_potential(state.rows)
            score = policy.score(
                before=potential,
                after=after,
                action=action,
                state_sha256=node.state_sha256,
            )
            candidates.append((score, action.action_sha256, state, program))
        if termination == "edge_budget":
            break
        counters.peak_frontier = max(
            counters.peak_frontier,
            min(len(candidates), budget.max_frontier),
        )
        if not candidates:
            termination = "dead_end"
            break
        if len(candidates) > budget.max_frontier:
            counters.nodes_pruned += len(candidates) - budget.max_frontier
            candidates = sorted(candidates)[: budget.max_frontier]
        score, _, state, program = min(candidates)
        node = _Node(
            state=state,
            program=program,
            depth=depth + 1,
            cumulative_policy_score=node.cumulative_policy_score + score,
            rank=score,
        )
        visited.add(node.state_sha256)

    receipt = _make_receipt(
        packet=packet,
        policy=policy,
        budget=budget,
        strategy=STRATEGY_GREEDY,
        guidance="local_policy_only",
        counters=counters,
        completed=completed_program is not None,
        termination=termination,
        program=completed_program,
        trace_sha256=trace.hexdigest(),
    )
    return CandidateSearchResult(
        program=completed_program,
        receipt=receipt,
    )


def bounded_beam_candidate_search(
    packet: SealedAlgebraPacket,
    policy: WeakLocalScorer,
    budget: SearchBudget,
    *,
    guidance: str = GUIDANCE_STRUCTURAL,
    guidance_seed: int = 0,
) -> CandidateSearchResult:
    """Search legal repairs with fixed node, edge, depth, and frontier caps."""

    packet.validate()
    budget.validate()
    root_state = execute_program(
        packet.rows,
        (),
        register_count=packet.register_count,
    )
    root_potential = structural_potential(root_state.rows)
    root = _Node(
        state=root_state,
        program=(),
        depth=0,
        cumulative_policy_score=0.0,
        rank=float(
            _guidance_value(
                root_state,
                root_potential,
                mode=guidance,
                seed=guidance_seed,
            )
        ),
    )
    frontier = [root]
    best_rank = {root.state_sha256: root.rank}
    counters = _Counters()
    trace = sha256()
    trace.update(STRATEGY_BEAM.encode("ascii"))
    trace.update(guidance.encode("ascii"))
    termination = "frontier_exhausted"
    completed_program: tuple[AlgebraInstruction, ...] | None = None

    for depth in range(budget.max_depth + 1):
        counters.maximum_depth_reached = max(
            counters.maximum_depth_reached,
            depth,
        )
        frontier.sort(
            key=lambda node: (
                node.rank,
                node.state_sha256,
                node.program_sha256,
            )
        )
        counters.peak_frontier = max(
            counters.peak_frontier,
            len(frontier),
        )
        for node in frontier:
            if structural_potential(node.state.rows).solved:
                completed_program = _finish_program(node, budget)
                termination = (
                    "candidate_goal"
                    if completed_program is not None
                    else "instruction_budget"
                )
                break
        if completed_program is not None or termination == "instruction_budget":
            break
        if depth == budget.max_depth:
            termination = "max_depth"
            break

        candidate_nodes: dict[str, _Node] = {}
        stopped = False
        for node in frontier[: budget.beam_width]:
            if counters.nodes_expanded >= budget.max_nodes_expanded:
                termination = "node_budget"
                stopped = True
                break
            counters.nodes_expanded += 1
            before = structural_potential(node.state.rows)
            for action in enumerate_legal_repair_actions(node.state):
                if counters.edges_considered >= budget.max_edges_considered:
                    termination = "edge_budget"
                    stopped = True
                    break
                counters.edges_considered += 1
                try:
                    applied = _apply_action(
                        packet,
                        node,
                        action,
                        budget=budget,
                    )
                except AlgebraMachineError:
                    counters.edges_rejected += 1
                    continue
                if applied is None:
                    counters.nodes_pruned += 1
                    continue
                state, program = applied
                counters.edges_legal += 1
                counters.nodes_generated += 1
                after = structural_potential(state.rows)
                local_score = policy.score(
                    before=before,
                    after=after,
                    action=action,
                    state_sha256=node.state_sha256,
                )
                cumulative = node.cumulative_policy_score + local_score
                guidance_value = _guidance_value(
                    state,
                    after,
                    mode=guidance,
                    seed=guidance_seed,
                )
                rank = 6.0 * guidance_value + 0.20 * cumulative + 0.05 * (depth + 1)
                child = _Node(
                    state=state,
                    program=program,
                    depth=depth + 1,
                    cumulative_policy_score=cumulative,
                    rank=rank,
                )
                state_sha = child.state_sha256
                trace.update(
                    _canonical_bytes(
                        [
                            node.state_sha256,
                            action.action_sha256,
                            state_sha,
                            round(rank, 12),
                        ]
                    )
                )
                prior_rank = best_rank.get(state_sha)
                if prior_rank is not None and prior_rank <= rank:
                    counters.nodes_deduplicated += 1
                    continue
                prior_child = candidate_nodes.get(state_sha)
                if prior_child is not None and prior_child.rank <= rank:
                    counters.nodes_deduplicated += 1
                    continue
                best_rank[state_sha] = rank
                candidate_nodes[state_sha] = child
            if stopped:
                break
        if stopped:
            break
        ordered = sorted(
            candidate_nodes.values(),
            key=lambda node: (
                node.rank,
                node.state_sha256,
                node.program_sha256,
            ),
        )
        keep = min(budget.beam_width, budget.max_frontier)
        if len(ordered) > keep:
            counters.nodes_pruned += len(ordered) - keep
        frontier = ordered[:keep]
        if not frontier:
            termination = "frontier_exhausted"
            break

    receipt = _make_receipt(
        packet=packet,
        policy=policy,
        budget=budget,
        strategy=STRATEGY_BEAM,
        guidance=guidance,
        counters=counters,
        completed=completed_program is not None,
        termination=termination,
        program=completed_program,
        trace_sha256=trace.hexdigest(),
    )
    return CandidateSearchResult(
        program=completed_program,
        receipt=receipt,
    )


@dataclass(frozen=True, slots=True)
class AssessmentReceipt:
    schema: str
    passed: bool
    reason: str
    candidate_packet_sha256: str
    search_receipt_sha256: str
    program_sha256: str
    output_sha256: str | None
    rank: int | None
    executed_instructions: int
    verifier_gates: tuple[tuple[str, bool], ...]


def assess_candidate_program(
    packet: SealedAlgebraPacket,
    result: CandidateSearchResult,
) -> AssessmentReceipt:
    """Separately replay and verify a completed program after candidate exit."""

    packet.validate()
    packet_sha = _digest(packet.canonical_data())
    receipt_sha = sha256(result.receipt.canonical_bytes()).hexdigest()
    if result.receipt.candidate_packet_sha256 != packet_sha:
        raise SearchFalsifierError("search receipt binds a different packet")
    if result.program is None:
        return AssessmentReceipt(
            schema=ASSESSMENT_SCHEMA,
            passed=False,
            reason="candidate_did_not_finish",
            candidate_packet_sha256=packet_sha,
            search_receipt_sha256=receipt_sha,
            program_sha256=_program_sha256(()),
            output_sha256=None,
            rank=None,
            executed_instructions=0,
            verifier_gates=(),
        )
    program_sha = _program_sha256(result.program)
    if result.receipt.program_sha256 != program_sha:
        raise SearchFalsifierError("search receipt binds a different program")
    try:
        state = execute_program(
            packet.rows,
            result.program,
            register_count=packet.register_count,
            maximum_instructions=result.receipt.max_program_instructions,
        )
        verification = verify_reduction_program(packet.rows, state)
    except AlgebraMachineError as error:
        return AssessmentReceipt(
            schema=ASSESSMENT_SCHEMA,
            passed=False,
            reason=f"independent_verifier_rejected:{type(error).__name__}",
            candidate_packet_sha256=packet_sha,
            search_receipt_sha256=receipt_sha,
            program_sha256=program_sha,
            output_sha256=None,
            rank=None,
            executed_instructions=len(result.program),
            verifier_gates=(),
        )
    return AssessmentReceipt(
        schema=ASSESSMENT_SCHEMA,
        passed=verification.passed,
        reason="independent_verifier_accepted",
        candidate_packet_sha256=packet_sha,
        search_receipt_sha256=receipt_sha,
        program_sha256=program_sha,
        output_sha256=verification.output_sha256,
        rank=verification.rank,
        executed_instructions=verification.executed_instructions,
        verifier_gates=verification.gates,
    )


@dataclass(frozen=True, slots=True)
class GeometryCase:
    packet: SealedAlgebraPacket

    @property
    def matrix_sha256(self) -> str:
        return _digest([list(row) for row in self.packet.rows])


def generate_geometry_cases(
    *,
    seed: int,
    count: int,
    minimum_rows: int,
    maximum_rows: int,
    minimum_columns: int,
    maximum_columns: int,
    register_count: int = 4,
) -> tuple[GeometryCase, ...]:
    """Generate deterministic matrix-only cases with no expert artifacts."""

    for label, value in (
        ("count", count),
        ("minimum rows", minimum_rows),
        ("maximum rows", maximum_rows),
        ("minimum columns", minimum_columns),
        ("maximum columns", maximum_columns),
    ):
        _plain_int(value, label=label)
    if count < 1:
        raise SearchFalsifierError("case count must be positive")
    if not 2 <= minimum_rows <= maximum_rows:
        raise SearchFalsifierError("row geometry bounds differ")
    if not 2 <= minimum_columns <= maximum_columns:
        raise SearchFalsifierError("column geometry bounds differ")
    if maximum_columns < maximum_rows:
        raise SearchFalsifierError("columns must admit every row geometry")
    rng = random.Random(seed)
    cases: list[GeometryCase] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    while len(cases) < count:
        row_count = rng.randint(minimum_rows, maximum_rows)
        column_count = rng.randint(
            max(row_count, minimum_columns),
            maximum_columns,
        )
        rows = tuple(
            tuple(
                0 if rng.random() < 0.48 else rng.randrange(1, FIELD_MODULUS)
                for _ in range(column_count)
            )
            for _ in range(row_count)
        )
        if rows in seen or not any(value for row in rows for value in row):
            continue
        seen.add(rows)
        cases.append(
            GeometryCase(
                packet=SealedAlgebraPacket.from_rows(
                    rows,
                    register_count=register_count,
                )
            )
        )
    return tuple(cases)


@dataclass(frozen=True, slots=True)
class CaseComparison:
    matrix_sha256: str
    row_count: int
    column_count: int
    greedy_passed: bool
    search_passed: bool
    random_relabel_passed: bool
    greedy_receipt_sha256: str
    search_receipt_sha256: str
    random_relabel_receipt_sha256: str
    greedy_nodes: int
    search_nodes: int
    random_relabel_nodes: int
    greedy_edges: int
    search_edges: int
    random_relabel_edges: int
    search_depth: int
    random_relabel_depth: int


@dataclass(frozen=True, slots=True)
class FalsifierReport:
    schema: str
    status: str
    reasoning_claim_authorized: bool
    candidate_runtime_boundary: str
    assessor_boundary: str
    candidate_packet_fields: tuple[str, ...]
    forbidden_candidate_fields: tuple[str, ...]
    seed: int
    policy_sha256: str
    random_relabel_policy_sha256: str
    development_cases: int
    evaluation_cases: int
    development_maximum_rows: int
    development_maximum_columns: int
    evaluation_minimum_rows: int
    evaluation_minimum_columns: int
    evaluation_maximum_rows: int
    evaluation_maximum_columns: int
    candidate_oracle_calls: int
    greedy_certified: int
    search_certified: int
    random_relabel_certified: int
    development_manifest_sha256: str
    evaluation_manifest_sha256: str
    strict_geometry_holdout: bool
    hard_resource_gate: bool
    search_absolute_gate: bool
    search_beats_greedy_gate: bool
    search_beats_random_relabel_gate: bool
    mechanics_promotion_gate: bool
    cases: tuple[CaseComparison, ...]

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(asdict(self)) + b"\n"


def _case_manifest(cases: Iterable[GeometryCase]) -> str:
    return sha256(
        ("\n".join(case.matrix_sha256 for case in cases) + "\n").encode("ascii")
    ).hexdigest()


def _receipt_sha256(receipt: SearchReceipt) -> str:
    return sha256(receipt.canonical_bytes()).hexdigest()


def run_falsifier_benchmark(
    *,
    seed: int,
    development_count: int,
    evaluation_count: int,
    development_maximum_rows: int,
    development_maximum_columns: int,
    evaluation_minimum_rows: int,
    evaluation_minimum_columns: int,
    evaluation_maximum_rows: int,
    evaluation_maximum_columns: int,
    greedy_budget: SearchBudget,
    search_budget: SearchBudget,
) -> FalsifierReport:
    """Compare greedy, meaningful search, and matched relabelled search."""

    strict_holdout = (
        development_maximum_rows < evaluation_minimum_rows
        and development_maximum_columns < evaluation_minimum_columns
    )
    if not strict_holdout:
        raise SearchFalsifierError(
            "evaluation geometry must be strictly larger than development"
        )
    development = generate_geometry_cases(
        seed=seed ^ 0xD3E,
        count=development_count,
        minimum_rows=2,
        maximum_rows=development_maximum_rows,
        minimum_columns=2,
        maximum_columns=development_maximum_columns,
    )
    evaluation = generate_geometry_cases(
        seed=seed ^ 0xE7A,
        count=evaluation_count,
        minimum_rows=evaluation_minimum_rows,
        maximum_rows=evaluation_maximum_rows,
        minimum_columns=evaluation_minimum_columns,
        maximum_columns=evaluation_maximum_columns,
    )
    if {case.matrix_sha256 for case in development} & {
        case.matrix_sha256 for case in evaluation
    }:
        raise RuntimeError("development and evaluation matrices overlap")

    policy = WeakLocalScorer(seed=seed)
    random_policy = WeakLocalScorer(
        seed=seed,
        relabel_action_kinds=True,
    )
    comparisons: list[CaseComparison] = []
    resource_gate = True
    for case in evaluation:
        greedy = greedy_candidate_search(
            case.packet,
            policy,
            greedy_budget,
        )
        search = bounded_beam_candidate_search(
            case.packet,
            policy,
            search_budget,
            guidance=GUIDANCE_STRUCTURAL,
            guidance_seed=seed,
        )
        control = bounded_beam_candidate_search(
            case.packet,
            random_policy,
            search_budget,
            guidance=GUIDANCE_RANDOM_RELABEL,
            guidance_seed=seed ^ 0xBAD5EED,
        )
        greedy_assessment = assess_candidate_program(case.packet, greedy)
        search_assessment = assess_candidate_program(case.packet, search)
        control_assessment = assess_candidate_program(case.packet, control)
        resource_gate = resource_gate and all(
            result.receipt.hard_budgets_respected
            for result in (greedy, search, control)
        )
        comparisons.append(
            CaseComparison(
                matrix_sha256=case.matrix_sha256,
                row_count=len(case.packet.rows),
                column_count=len(case.packet.rows[0]),
                greedy_passed=greedy_assessment.passed,
                search_passed=search_assessment.passed,
                random_relabel_passed=control_assessment.passed,
                greedy_receipt_sha256=_receipt_sha256(greedy.receipt),
                search_receipt_sha256=_receipt_sha256(search.receipt),
                random_relabel_receipt_sha256=_receipt_sha256(control.receipt),
                greedy_nodes=greedy.receipt.nodes_expanded,
                search_nodes=search.receipt.nodes_expanded,
                random_relabel_nodes=control.receipt.nodes_expanded,
                greedy_edges=greedy.receipt.edges_considered,
                search_edges=search.receipt.edges_considered,
                random_relabel_edges=control.receipt.edges_considered,
                search_depth=search.receipt.maximum_depth_reached,
                random_relabel_depth=control.receipt.maximum_depth_reached,
            )
        )

    greedy_certified = sum(case.greedy_passed for case in comparisons)
    search_certified = sum(case.search_passed for case in comparisons)
    control_certified = sum(case.random_relabel_passed for case in comparisons)
    search_absolute = search_certified == evaluation_count
    beats_greedy = search_certified > greedy_certified
    beats_control = search_certified > control_certified
    mechanics_gate = (
        strict_holdout
        and resource_gate
        and search_absolute
        and beats_greedy
        and beats_control
    )
    return FalsifierReport(
        schema=BENCHMARK_SCHEMA,
        status=STATUS,
        reasoning_claim_authorized=False,
        candidate_runtime_boundary=(
            "sealed_algebra_packet_plus_legal_primitive_transitions_"
            "plus_source_independent_local_potential"
        ),
        assessor_boundary=(
            "post_search_separate_primitive_replay_and_rref_verification"
        ),
        candidate_packet_fields=(
            "field_modulus",
            "register_count",
            "rows",
            "schema",
            "seal_sha256",
        ),
        forbidden_candidate_fields=tuple(sorted(FORBIDDEN_CANDIDATE_FIELDS)),
        seed=seed,
        policy_sha256=policy.policy_sha256,
        random_relabel_policy_sha256=random_policy.policy_sha256,
        development_cases=len(development),
        evaluation_cases=len(evaluation),
        development_maximum_rows=development_maximum_rows,
        development_maximum_columns=development_maximum_columns,
        evaluation_minimum_rows=evaluation_minimum_rows,
        evaluation_minimum_columns=evaluation_minimum_columns,
        evaluation_maximum_rows=evaluation_maximum_rows,
        evaluation_maximum_columns=evaluation_maximum_columns,
        candidate_oracle_calls=0,
        greedy_certified=greedy_certified,
        search_certified=search_certified,
        random_relabel_certified=control_certified,
        development_manifest_sha256=_case_manifest(development),
        evaluation_manifest_sha256=_case_manifest(evaluation),
        strict_geometry_holdout=strict_holdout,
        hard_resource_gate=resource_gate,
        search_absolute_gate=search_absolute,
        search_beats_greedy_gate=beats_greedy,
        search_beats_random_relabel_gate=beats_control,
        mechanics_promotion_gate=mechanics_gate,
        cases=tuple(comparisons),
    )


def _default_greedy_budget() -> SearchBudget:
    return SearchBudget(
        max_nodes_expanded=48,
        max_edges_considered=2_000,
        max_depth=36,
        max_frontier=128,
        beam_width=1,
        max_program_instructions=160,
    )


def _default_search_budget() -> SearchBudget:
    return SearchBudget(
        max_nodes_expanded=2_048,
        max_edges_considered=50_000,
        max_depth=36,
        max_frontier=96,
        beam_width=96,
        max_program_instructions=160,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--development-count", type=int, default=8)
    parser.add_argument("--evaluation-count", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_falsifier_benchmark(
        seed=args.seed,
        development_count=args.development_count,
        evaluation_count=args.evaluation_count,
        development_maximum_rows=3,
        development_maximum_columns=4,
        evaluation_minimum_rows=4,
        evaluation_minimum_columns=5,
        evaluation_maximum_rows=4,
        evaluation_maximum_columns=6,
        greedy_budget=_default_greedy_budget(),
        search_budget=_default_search_budget(),
    )
    if args.output is None:
        print(report.canonical_bytes().decode("ascii"), end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(report.canonical_bytes())


if __name__ == "__main__":
    main()
