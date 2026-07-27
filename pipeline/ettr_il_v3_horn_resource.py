"""Deterministic broad-corpus Horn and resource episodes for ETTR-IL v3.

This module is assessor-side, pure, and CPU-only.  It deliberately does not
render, tokenize, materialize files, load a model, or require strict semantic
checkerboards.  Existing v2 primary and replay engines remain the semantic
oracles.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
import hashlib
from itertools import product
from typing import Iterable, Sequence, TypeAlias

from cross_ontology_horn_board import (
    THEORIES as HORN_THEORIES,
    GroundAtom,
    all_ground_atoms,
    challenge_initials,
    reference_theory_state as horn_reference_theory_state,
)
from cross_ontology_resource_board import (
    OPERATOR_SYMBOL_COUNT,
    PLACE_SPECS,
    THEORIES as RESOURCE_THEORIES,
    Marking,
    reference_theory_state as resource_reference_theory_state,
)
from ettr_il_v2_candidate_search import (
    canonical_json_bytes,
    semantic_command_value,
    semantic_world_value,
)
from ettr_il_v2_semantics import (
    HornCommand,
    HornExecution,
    HornPolicy,
    HornWorld,
    Ontology,
    ResourceCommand,
    ResourceExecution,
    ResourcePolicy,
    ResourceWorld,
    SemanticError,
    SemanticQuery,
    StepOutcome,
    TerminalDisposition,
    enumerate_queries,
    evaluate_query,
    execute_semantics,
    replay_semantics,
)


PROTOCOL = "R12-ETTR-IL-v3-initializer"
EPISODE_SCHEMA = "r12-ettr-il-v3-semantic-episode-v1"
COUNTERFACTUAL_SCHEMA = "r12-ettr-il-v3-counterfactual-v1"
MASTER_SEED = hashlib.sha256(
    b"R12-ETTR-IL-v3|horn-resource|broad-corpus|2026-07-27"
).digest()
MAX_PACKET_SLOTS = 64
MAX_PACKET_EDGES = 256
MAX_TRACE_STEPS = 64

SemanticWorld: TypeAlias = HornWorld | ResourceWorld
SemanticCommand: TypeAlias = HornCommand | ResourceCommand
SemanticExecution: TypeAlias = HornExecution | ResourceExecution


class V3EpisodeError(ValueError):
    """A requested v3 episode is malformed or fails exact admission."""


class CurriculumStage(StrEnum):
    COMPILER_GROUNDING = "compiler_grounding"
    ATOMIC_TRANSITIONS = "atomic_transactions"
    DEPENDENT_COMPOSITION = "dependent_composition"
    QUERY_COUNTERFACTUAL_GROUNDING = "query_counterfactual_grounding"
    CLOSED_LOOP = "closed_loop_invariance"


class CounterfactualAxis(StrEnum):
    WORLD = "world"
    COMMAND = "command"
    QUERY = "query"


@dataclass(frozen=True, slots=True)
class CapacityReceipt:
    """Exact generic-packet and materializer-trace capacity accounting."""

    initial_active_slots: int
    terminal_active_slots: int
    initial_edges: int
    terminal_edges: int
    encoded_trace_steps: int

    def validate(self) -> None:
        if (
            not 0 < self.initial_active_slots <= MAX_PACKET_SLOTS
            or not 0 < self.terminal_active_slots <= MAX_PACKET_SLOTS
            or not 0 <= self.initial_edges <= MAX_PACKET_EDGES
            or not 0 <= self.terminal_edges <= MAX_PACKET_EDGES
            or not 0 < self.encoded_trace_steps <= MAX_TRACE_STEPS
        ):
            raise V3EpisodeError("episode leaves ETTR packet or trace capacity")

    def assessor_value(self) -> dict[str, int]:
        return {
            "encoded_trace_steps": self.encoded_trace_steps,
            "initial_active_slots": self.initial_active_slots,
            "initial_edges": self.initial_edges,
            "terminal_active_slots": self.terminal_active_slots,
            "terminal_edges": self.terminal_edges,
        }


@dataclass(frozen=True, slots=True)
class CoverageMetadata:
    """Selection-facing coverage facts, never candidate-visible metadata."""

    ontology: str
    stage: str
    theory_index: int
    policy: str
    depth: int
    disposition: str
    status: str
    operation_histogram: tuple[tuple[str, int], ...]
    outcome_histogram: tuple[tuple[str, int], ...]
    prefix_dependent_steps: int
    state_change_count: int
    initial_item_count: int
    terminal_item_count: int
    query_ops: tuple[str, ...]
    query_answers: tuple[bool, ...]
    trace_length_bin: str
    packet_density_bin: str

    def assessor_value(self) -> dict[str, object]:
        return {
            "depth": self.depth,
            "disposition": self.disposition,
            "initial_item_count": self.initial_item_count,
            "ontology": self.ontology,
            "operation_histogram": [list(item) for item in self.operation_histogram],
            "outcome_histogram": [list(item) for item in self.outcome_histogram],
            "packet_density_bin": self.packet_density_bin,
            "policy": self.policy,
            "prefix_dependent_steps": self.prefix_dependent_steps,
            "query_answers": list(self.query_answers),
            "query_ops": list(self.query_ops),
            "stage": self.stage,
            "state_change_count": self.state_change_count,
            "status": self.status,
            "terminal_item_count": self.terminal_item_count,
            "theory_index": self.theory_index,
            "trace_length_bin": self.trace_length_bin,
        }


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """One exact broad-corpus semantic episode and its sealed targets."""

    episode_id: str
    stage: CurriculumStage
    ontology: Ontology
    world: SemanticWorld
    command: SemanticCommand
    primary: SemanticExecution
    replay: SemanticExecution
    queries: tuple[SemanticQuery, SemanticQuery]
    answers: tuple[bool, bool]
    capacity: CapacityReceipt
    coverage: CoverageMetadata

    def assessor_value(self) -> dict[str, object]:
        return {
            "answers": list(self.answers),
            "capacity": self.capacity.assessor_value(),
            "command": semantic_command_value(self.command),
            "coverage": self.coverage.assessor_value(),
            "execution": _execution_value(self.primary),
            "ontology": self.ontology.value,
            "protocol": PROTOCOL,
            "queries": [query.assessor_value() for query in self.queries],
            "schema": EPISODE_SCHEMA,
            "stage": self.stage.value,
            "world": semantic_world_value(self.world),
        }


@dataclass(frozen=True, slots=True)
class CounterfactualRecord:
    """A minimal controlled factor change with an independently scored flip."""

    counterfactual_id: str
    source_episode_id: str
    axis: CounterfactualAxis
    query_index: int
    semantic_distance: int
    world: SemanticWorld
    command: SemanticCommand
    query: SemanticQuery
    primary: SemanticExecution
    replay: SemanticExecution
    answer_before: bool
    answer_after: bool
    capacity: CapacityReceipt

    def assessor_value(self) -> dict[str, object]:
        return {
            "answer_after": self.answer_after,
            "answer_before": self.answer_before,
            "axis": self.axis.value,
            "capacity": self.capacity.assessor_value(),
            "command": semantic_command_value(self.command),
            "execution": _execution_value(self.primary),
            "protocol": PROTOCOL,
            "query": self.query.assessor_value(),
            "query_index": self.query_index,
            "schema": COUNTERFACTUAL_SCHEMA,
            "semantic_distance": self.semantic_distance,
            "source_episode_id": self.source_episode_id,
            "world": semantic_world_value(self.world),
        }


@dataclass(frozen=True, slots=True)
class CounterfactualBundle:
    """One same-query WORLD/COMMAND/QUERY intervention bundle."""

    query_index: int
    world: CounterfactualRecord
    command: CounterfactualRecord
    query: CounterfactualRecord


def _hash_value(domain: bytes, value: object) -> bytes:
    return hashlib.sha256(
        MASTER_SEED + b"|" + domain + b"|" + canonical_json_bytes(value)
    ).digest()


def _evidence_id(ontology: Ontology, theory_index: int) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "ontology": ontology.value,
                "protocol": PROTOCOL,
                "theory_index": theory_index,
            }
        )
    ).hexdigest()


def _atom_value(atom: GroundAtom) -> list[object]:
    return [atom.predicate, list(atom.arguments)]


def _marking_value(marking: Marking) -> list[int]:
    return list(marking.multiplicities)


def _execution_value(execution: SemanticExecution) -> dict[str, object]:
    if type(execution) is HornExecution:
        return {
            "disposition": execution.disposition.value,
            "snapshots": [
                [_atom_value(atom) for atom in snapshot]
                for snapshot in execution.snapshots
            ],
            "steps": [
                {
                    "after": [_atom_value(atom) for atom in step.after],
                    "before": [_atom_value(atom) for atom in step.before],
                    "index": step.index,
                    "operation": _atom_value(step.operation),
                    "outcome": step.outcome.value,
                    "prefix_dependent": step.prefix_dependent,
                }
                for step in execution.steps
            ],
        }
    if type(execution) is ResourceExecution:
        return {
            "cursor": execution.cursor,
            "disposition": execution.disposition.value,
            "snapshots": [_marking_value(snapshot) for snapshot in execution.snapshots],
            "status": execution.status.value,
            "steps": [
                {
                    "after": _marking_value(step.after),
                    "before": _marking_value(step.before),
                    "cursor_after": step.cursor_after,
                    "cursor_before": step.cursor_before,
                    "index": step.index,
                    "operation": step.operation,
                    "outcome": step.outcome.value,
                    "prefix_dependent": step.prefix_dependent,
                }
                for step in execution.steps
            ],
        }
    raise V3EpisodeError("execution ontology differs")


def full_resource_markings() -> tuple[Marking, ...]:
    """Return all ``4^4`` capacity-valid guarded-resource markings."""

    return tuple(
        Marking(tuple(values))
        for values in product(*(range(place.capacity + 1) for place in PLACE_SPECS))
    )


def horn_worlds(
    theory_index: int,
    *,
    policy: HornPolicy = HornPolicy.PERSISTENT,
) -> tuple[HornWorld, ...]:
    if (
        type(theory_index) is not int
        or not 0 <= theory_index < len(HORN_THEORIES)
        or type(policy) is not HornPolicy
    ):
        raise V3EpisodeError("Horn world request differs")
    evidence_id = _evidence_id(Ontology.HORN, theory_index)
    return tuple(
        HornWorld(evidence_id, theory_index, initial, policy)
        for initial in challenge_initials()
    )


def resource_worlds(
    theory_index: int,
    *,
    policy: ResourcePolicy = ResourcePolicy.ATOMIC_DEADLOCK,
) -> tuple[ResourceWorld, ...]:
    if (
        type(theory_index) is not int
        or not 0 <= theory_index < len(RESOURCE_THEORIES)
        or type(policy) is not ResourcePolicy
    ):
        raise V3EpisodeError("resource world request differs")
    evidence_id = _evidence_id(Ontology.RESOURCE, theory_index)
    return tuple(
        ResourceWorld(evidence_id, theory_index, marking, policy)
        for marking in full_resource_markings()
    )


def _ontology(world: SemanticWorld) -> Ontology:
    if type(world) is HornWorld:
        return Ontology.HORN
    if type(world) is ResourceWorld:
        return Ontology.RESOURCE
    raise V3EpisodeError("world ontology differs")


def _operation_alphabet(world: SemanticWorld) -> tuple[object, ...]:
    if type(world) is HornWorld:
        return tuple(all_ground_atoms())
    if type(world) is ResourceWorld:
        return tuple(range(OPERATOR_SYMBOL_COUNT))
    raise V3EpisodeError("world ontology differs")


def _command_from_operations(
    world: SemanticWorld,
    operations: Sequence[object],
) -> SemanticCommand:
    values = tuple(operations)
    if type(world) is HornWorld:
        if not values or any(type(value) is not GroundAtom for value in values):
            raise V3EpisodeError("Horn command operations differ")
        return HornCommand(len(values), values)  # type: ignore[arg-type]
    if type(world) is ResourceWorld:
        if not values or any(type(value) is not int for value in values):
            raise V3EpisodeError("resource command operations differ")
        return ResourceCommand(len(values), values)  # type: ignore[arg-type]
    raise V3EpisodeError("world ontology differs")


def _execute_equal(
    world: SemanticWorld,
    command: SemanticCommand,
    *,
    require_dependent: bool,
) -> tuple[SemanticExecution, SemanticExecution]:
    try:
        primary = execute_semantics(
            world,
            command,
            require_dependent=require_dependent,
        )
        replay = replay_semantics(
            world,
            command,
            require_dependent=require_dependent,
        )
    except (SemanticError, TypeError, ValueError) as exc:
        raise V3EpisodeError("episode fails semantic execution") from exc
    if primary != replay:
        raise V3EpisodeError("primary and replay executions differ")
    if primary.disposition is not TerminalDisposition.ANSWER:
        raise V3EpisodeError("broad-corpus episode is not answerable")
    return primary, replay


def _trace_change_count(execution: SemanticExecution) -> int:
    if type(execution) is HornExecution:
        return sum(len(set(step.before) ^ set(step.after)) for step in execution.steps)
    if type(execution) is ResourceExecution:
        return sum(
            sum(
                left != right
                for left, right in zip(
                    step.before.multiplicities,
                    step.after.multiplicities,
                    strict=True,
                )
            )
            for step in execution.steps
        )
    raise V3EpisodeError("execution ontology differs")


def _capacity_receipt(execution: SemanticExecution) -> CapacityReceipt:
    changes = _trace_change_count(execution)
    depth = execution.command.depth
    encoded_trace_steps = 2 * depth + changes + 2
    if type(execution) is HornExecution:
        static = horn_reference_theory_state(execution.world.theory_index)
        initial_slots = len(static.cells) + 6
        terminal_slots = initial_slots + 8
        initial_edges = len(static.edges) + len(execution.world.initial)
        terminal_edges = len(static.edges) + len(execution.terminal)
    elif type(execution) is ResourceExecution:
        static = resource_reference_theory_state(execution.world.theory_index)
        initial_slots = len(static.cells) + 4
        terminal_slots = initial_slots + 8
        initial_edges = terminal_edges = len(static.edges)
    else:
        raise V3EpisodeError("execution ontology differs")
    receipt = CapacityReceipt(
        initial_active_slots=initial_slots,
        terminal_active_slots=terminal_slots,
        initial_edges=initial_edges,
        terminal_edges=terminal_edges,
        encoded_trace_steps=encoded_trace_steps,
    )
    receipt.validate()
    return receipt


def _stage_contract(
    stage: CurriculumStage,
    execution: SemanticExecution,
) -> None:
    depth = execution.command.depth
    if (
        stage
        in {
            CurriculumStage.COMPILER_GROUNDING,
            CurriculumStage.ATOMIC_TRANSITIONS,
        }
        and depth != 1
    ):
        raise V3EpisodeError(f"{stage.value} requires depth one")
    if stage in {
        CurriculumStage.DEPENDENT_COMPOSITION,
        CurriculumStage.CLOSED_LOOP,
    }:
        if not 2 <= depth <= 6:
            raise V3EpisodeError(f"{stage.value} requires depth 2..6")
        if len(execution.steps) != depth or any(
            step.outcome is not StepOutcome.APPLIED or not step.prefix_dependent
            for step in execution.steps
        ):
            raise V3EpisodeError(
                f"{stage.value} requires a fully dependent applied trajectory"
            )


def _validate_stage_depth(stage: CurriculumStage, depth: int) -> None:
    if type(stage) is not CurriculumStage:
        raise V3EpisodeError("curriculum stage differs")
    if type(depth) is not int or not 1 <= depth <= 6:
        raise V3EpisodeError("episode depth differs")
    if (
        stage
        in {
            CurriculumStage.COMPILER_GROUNDING,
            CurriculumStage.ATOMIC_TRANSITIONS,
        }
        and depth != 1
    ):
        raise V3EpisodeError(f"{stage.value} requires depth one")
    if (
        stage
        in {
            CurriculumStage.DEPENDENT_COMPOSITION,
            CurriculumStage.CLOSED_LOOP,
        }
        and not 2 <= depth <= 6
    ):
        raise V3EpisodeError(f"{stage.value} requires depth 2..6")


def _ranked_queries(execution: SemanticExecution) -> tuple[SemanticQuery, ...]:
    ontology = _ontology(execution.world)
    context = {
        "command": semantic_command_value(execution.command),
        "execution": _execution_value(execution),
        "world": semantic_world_value(execution.world),
    }
    return tuple(
        sorted(
            enumerate_queries(ontology),
            key=lambda query: (
                _hash_value(
                    b"query-rank",
                    {"context": context, "query": query.assessor_value()},
                ),
                canonical_json_bytes(query.assessor_value()),
            ),
        )
    )


def _select_queries(
    execution: SemanticExecution,
) -> tuple[tuple[SemanticQuery, SemanticQuery], tuple[bool, bool]]:
    ranked = _ranked_queries(execution)
    by_answer = {
        answer: tuple(
            query for query in ranked if evaluate_query(query, execution) is answer
        )
        for answer in (False, True)
    }
    if by_answer[False] and by_answer[True]:
        queries = (by_answer[False][0], by_answer[True][0])
    elif len(ranked) >= 2:
        queries = (ranked[0], ranked[1])
    else:
        raise V3EpisodeError("query grammar cannot supply two denotations")
    return queries, tuple(evaluate_query(query, execution) for query in queries)  # type: ignore[return-value]


def _validate_queries(
    execution: SemanticExecution,
    queries: Iterable[SemanticQuery],
) -> tuple[tuple[SemanticQuery, SemanticQuery], tuple[bool, bool]]:
    values = tuple(queries)
    ontology = _ontology(execution.world)
    if (
        len(values) != 2
        or values[0] == values[1]
        or any(
            type(query) is not SemanticQuery or query.ontology is not ontology
            for query in values
        )
    ):
        raise V3EpisodeError("episode queries differ")
    answers = tuple(evaluate_query(query, execution) for query in values)
    return (values[0], values[1]), answers  # type: ignore[return-value]


def _coverage(
    stage: CurriculumStage,
    execution: SemanticExecution,
    queries: tuple[SemanticQuery, SemanticQuery],
    answers: tuple[bool, bool],
    capacity: CapacityReceipt,
) -> CoverageMetadata:
    operations = Counter(
        (
            f"{step.operation.predicate}:{','.join(map(str, step.operation.arguments))}"
            if type(execution) is HornExecution
            else str(step.operation)
        )
        for step in execution.steps
    )
    outcomes = Counter(step.outcome.value for step in execution.steps)
    if type(execution) is HornExecution:
        status = "closure"
        initial_items = len(execution.world.initial)
        terminal_items = len(execution.terminal)
    elif type(execution) is ResourceExecution:
        status = execution.status.value
        initial_items = sum(execution.world.initial.multiplicities)
        terminal_items = sum(execution.terminal.multiplicities)
    else:
        raise V3EpisodeError("execution ontology differs")
    if capacity.encoded_trace_steps <= 16:
        trace_bin = "short"
    elif capacity.encoded_trace_steps <= 32:
        trace_bin = "medium"
    else:
        trace_bin = "long"
    density = capacity.terminal_active_slots / MAX_PACKET_SLOTS
    packet_bin = "sparse" if density < 0.5 else "medium" if density < 0.75 else "dense"
    return CoverageMetadata(
        ontology=_ontology(execution.world).value,
        stage=stage.value,
        theory_index=execution.world.theory_index,
        policy=execution.world.policy.value,
        depth=execution.command.depth,
        disposition=execution.disposition.value,
        status=status,
        operation_histogram=tuple(sorted(operations.items())),
        outcome_histogram=tuple(sorted(outcomes.items())),
        prefix_dependent_steps=sum(step.prefix_dependent for step in execution.steps),
        state_change_count=_trace_change_count(execution),
        initial_item_count=initial_items,
        terminal_item_count=terminal_items,
        query_ops=tuple(query.op.value for query in queries),
        query_answers=answers,
        trace_length_bin=trace_bin,
        packet_density_bin=packet_bin,
    )


def build_episode(
    *,
    stage: CurriculumStage,
    world: SemanticWorld,
    command: SemanticCommand,
    queries: Iterable[SemanticQuery] | None = None,
) -> EpisodeRecord:
    """Execute, replay, bound, label, and content-address one episode."""

    if type(stage) is not CurriculumStage:
        raise V3EpisodeError("curriculum stage differs")
    if (
        type(world) is HornWorld
        and type(command) is not HornCommand
        or type(world) is ResourceWorld
        and type(command) is not ResourceCommand
    ):
        raise V3EpisodeError("world and command ontologies differ")
    require_dependent = stage in {
        CurriculumStage.DEPENDENT_COMPOSITION,
        CurriculumStage.CLOSED_LOOP,
    }
    primary, replay = _execute_equal(
        world,
        command,
        require_dependent=require_dependent,
    )
    _stage_contract(stage, primary)
    query_pair, answers = (
        _select_queries(primary)
        if queries is None
        else _validate_queries(primary, queries)
    )
    capacity = _capacity_receipt(primary)
    coverage = _coverage(
        stage,
        primary,
        query_pair,
        answers,
        capacity,
    )
    provisional = EpisodeRecord(
        episode_id="",
        stage=stage,
        ontology=_ontology(world),
        world=world,
        command=command,
        primary=primary,
        replay=replay,
        queries=query_pair,
        answers=answers,
        capacity=capacity,
        coverage=coverage,
    )
    episode_id = hashlib.sha256(
        canonical_json_bytes(provisional.assessor_value())
    ).hexdigest()
    result = EpisodeRecord(
        episode_id=episode_id,
        stage=stage,
        ontology=provisional.ontology,
        world=world,
        command=command,
        primary=primary,
        replay=replay,
        queries=query_pair,
        answers=answers,
        capacity=capacity,
        coverage=coverage,
    )
    if stage is CurriculumStage.QUERY_COUNTERFACTUAL_GROUNDING and all(
        minimal_query_counterfactual(result, query_index=query_index) is None
        for query_index in range(2)
    ):
        raise V3EpisodeError(
            "query/counterfactual grounding lacks a query intervention"
        )
    if stage is CurriculumStage.CLOSED_LOOP and counterfactual_bundle(result) is None:
        raise V3EpisodeError("closed-loop episode lacks all interventions")
    return result


def admitted_commands(
    world: SemanticWorld,
    *,
    depth: int,
    beam_width: int = 64,
    require_dependent: bool,
) -> tuple[SemanticCommand, ...]:
    """Return a deterministic hash-ranked command beam for one world."""

    if type(depth) is not int or not 1 <= depth <= 6:
        raise V3EpisodeError("command depth differs")
    if type(beam_width) is not int or beam_width < 1:
        raise V3EpisodeError("beam width differs")
    alphabet = _operation_alphabet(world)
    survivors: tuple[tuple[object, ...], ...] = ((),)
    for prefix_depth in range(1, depth + 1):
        admitted: dict[bytes, tuple[object, ...]] = {}
        for prefix in survivors:
            for operation in alphabet:
                operations = (*prefix, operation)
                command = _command_from_operations(world, operations)
                try:
                    execution, _ = _execute_equal(
                        world,
                        command,
                        require_dependent=require_dependent,
                    )
                    if require_dependent and (
                        len(execution.steps) != prefix_depth
                        or any(
                            step.outcome is not StepOutcome.APPLIED
                            or not step.prefix_dependent
                            for step in execution.steps
                        )
                    ):
                        continue
                    _capacity_receipt(execution)
                except V3EpisodeError:
                    continue
                key = canonical_json_bytes(semantic_command_value(command))
                admitted.setdefault(key, operations)
        survivors = tuple(
            sorted(
                admitted.values(),
                key=lambda operations: (
                    _hash_value(
                        b"command-beam",
                        {
                            "depth": prefix_depth,
                            "operations": semantic_command_value(
                                _command_from_operations(world, operations)
                            ),
                            "world": semantic_world_value(world),
                        },
                    ),
                    canonical_json_bytes(
                        semantic_command_value(
                            _command_from_operations(world, operations)
                        )
                    ),
                ),
            )[:beam_width]
        )
        if not survivors:
            break
    return tuple(
        _command_from_operations(world, operations)
        for operations in survivors
        if len(operations) == depth
    )


def _ordered_worlds(
    worlds: Iterable[SemanticWorld],
    *,
    bucket_index: int,
    bucket_count: int,
) -> tuple[SemanticWorld, ...]:
    if (
        type(bucket_count) is not int
        or bucket_count < 1
        or type(bucket_index) is not int
        or not 0 <= bucket_index < bucket_count
    ):
        raise V3EpisodeError("deterministic bucket differs")
    ranked = sorted(
        worlds,
        key=lambda world: (
            _hash_value(b"world-rank", semantic_world_value(world)),
            canonical_json_bytes(semantic_world_value(world)),
        ),
    )
    return tuple(
        world
        for world in ranked
        if int.from_bytes(
            _hash_value(b"world-bucket", semantic_world_value(world))[:8],
            "big",
        )
        % bucket_count
        == bucket_index
    )


def _generate_stage(
    *,
    stage: CurriculumStage,
    worlds: Iterable[SemanticWorld],
    depth: int,
    limit: int | None,
    beam_width: int,
    bucket_index: int,
    bucket_count: int,
) -> tuple[EpisodeRecord, ...]:
    _validate_stage_depth(stage, depth)
    if limit is not None and (type(limit) is not int or limit < 1):
        raise V3EpisodeError("episode limit differs")
    ordered_worlds = _ordered_worlds(
        worlds,
        bucket_index=bucket_index,
        bucket_count=bucket_count,
    )
    records: dict[str, EpisodeRecord] = {}

    def admit(world: SemanticWorld, command: SemanticCommand) -> bool:
        try:
            episode = build_episode(
                stage=stage,
                world=world,
                command=command,
            )
        except V3EpisodeError:
            return False
        records.setdefault(episode.episode_id, episode)
        return limit is not None and len(records) >= limit

    require_dependent = stage in {
        CurriculumStage.DEPENDENT_COMPOSITION,
        CurriculumStage.CLOSED_LOOP,
    }
    if stage is CurriculumStage.COMPILER_GROUNDING:
        command_cache: dict[SemanticWorld, tuple[SemanticCommand, ...]] = {}
        for command_index in range(beam_width):
            progressed = False
            for world in ordered_worlds:
                commands = command_cache.get(world)
                if commands is None:
                    commands = admitted_commands(
                        world,
                        depth=depth,
                        beam_width=beam_width,
                        require_dependent=False,
                    )
                    command_cache[world] = commands
                if command_index >= len(commands):
                    continue
                progressed = True
                if admit(world, commands[command_index]):
                    return tuple(records.values())
            if not progressed:
                break
        return tuple(records.values())

    for world in ordered_worlds:
        commands = admitted_commands(
            world,
            depth=depth,
            beam_width=beam_width,
            require_dependent=require_dependent,
        )
        for command in commands:
            if admit(world, command):
                return tuple(records.values())
    return tuple(records.values())


def generate_horn_episodes(
    *,
    stage: CurriculumStage,
    theory_index: int,
    depth: int = 1,
    policy: HornPolicy = HornPolicy.PERSISTENT,
    limit: int | None = None,
    beam_width: int = 64,
    bucket_index: int = 0,
    bucket_count: int = 1,
) -> tuple[EpisodeRecord, ...]:
    """Generate one deterministic Horn stage/bucket without checkerboard gating."""

    return _generate_stage(
        stage=stage,
        worlds=horn_worlds(theory_index, policy=policy),
        depth=depth,
        limit=limit,
        beam_width=beam_width,
        bucket_index=bucket_index,
        bucket_count=bucket_count,
    )


def generate_resource_episodes(
    *,
    stage: CurriculumStage,
    theory_index: int,
    depth: int = 1,
    policy: ResourcePolicy = ResourcePolicy.ATOMIC_DEADLOCK,
    limit: int | None = None,
    beam_width: int = 64,
    bucket_index: int = 0,
    bucket_count: int = 1,
) -> tuple[EpisodeRecord, ...]:
    """Generate one deterministic resource stage/bucket over all 256 markings."""

    return _generate_stage(
        stage=stage,
        worlds=resource_worlds(theory_index, policy=policy),
        depth=depth,
        limit=limit,
        beam_width=beam_width,
        bucket_index=bucket_index,
        bucket_count=bucket_count,
    )


def _query_distance(left: SemanticQuery, right: SemanticQuery) -> int:
    return (
        int(left.op is not right.op)
        + abs(len(left.args) - len(right.args))
        + sum(lvalue != rvalue for lvalue, rvalue in zip(left.args, right.args))
    )


def _counterfactual_record(
    episode: EpisodeRecord,
    *,
    axis: CounterfactualAxis,
    query_index: int,
    semantic_distance: int,
    world: SemanticWorld,
    command: SemanticCommand,
    query: SemanticQuery,
) -> CounterfactualRecord | None:
    before = episode.answers[query_index]
    try:
        primary, replay = _execute_equal(
            world,
            command,
            require_dependent=False,
        )
        after = evaluate_query(query, primary)
        capacity = _capacity_receipt(primary)
    except (SemanticError, V3EpisodeError, ValueError):
        return None
    if after is before:
        return None
    provisional = CounterfactualRecord(
        counterfactual_id="",
        source_episode_id=episode.episode_id,
        axis=axis,
        query_index=query_index,
        semantic_distance=semantic_distance,
        world=world,
        command=command,
        query=query,
        primary=primary,
        replay=replay,
        answer_before=before,
        answer_after=after,
        capacity=capacity,
    )
    counterfactual_id = hashlib.sha256(
        canonical_json_bytes(provisional.assessor_value())
    ).hexdigest()
    return CounterfactualRecord(
        counterfactual_id=counterfactual_id,
        source_episode_id=episode.episode_id,
        axis=axis,
        query_index=query_index,
        semantic_distance=semantic_distance,
        world=world,
        command=command,
        query=query,
        primary=primary,
        replay=replay,
        answer_before=before,
        answer_after=after,
        capacity=capacity,
    )


def minimal_world_counterfactual(
    episode: EpisodeRecord,
    *,
    query_index: int = 0,
) -> CounterfactualRecord | None:
    """Return the smallest canonical WORLD edit that flips one query."""

    if type(query_index) is not int or not 0 <= query_index < 2:
        raise V3EpisodeError("query index differs")
    world = episode.world
    candidates: list[tuple[int, SemanticWorld]] = []
    if type(world) is HornWorld:
        initial = set(world.initial)
        for atom in sorted(initial):
            if len(initial) > 1:
                candidates.append(
                    (
                        1,
                        HornWorld(
                            world.evidence_id,
                            world.theory_index,
                            tuple(sorted(initial - {atom})),
                            world.policy,
                        ),
                    )
                )
        for atom in all_ground_atoms():
            if atom not in initial:
                candidates.append(
                    (
                        1,
                        HornWorld(
                            world.evidence_id,
                            world.theory_index,
                            tuple(sorted((*initial, atom))),
                            world.policy,
                        ),
                    )
                )
        for removed in sorted(initial):
            for added in all_ground_atoms():
                if added not in initial:
                    candidates.append(
                        (
                            2,
                            HornWorld(
                                world.evidence_id,
                                world.theory_index,
                                tuple(sorted((initial - {removed}) | {added})),
                                world.policy,
                            ),
                        )
                    )
    elif type(world) is ResourceWorld:
        values = world.initial.multiplicities
        for place, previous in enumerate(values):
            for successor in range(PLACE_SPECS[place].capacity + 1):
                if successor == previous:
                    continue
                changed = list(values)
                changed[place] = successor
                candidates.append(
                    (
                        abs(successor - previous),
                        ResourceWorld(
                            world.evidence_id,
                            world.theory_index,
                            Marking(tuple(changed)),
                            world.policy,
                        ),
                    )
                )
    else:
        raise V3EpisodeError("episode world ontology differs")
    query = episode.queries[query_index]
    for distance, candidate in sorted(
        candidates,
        key=lambda item: (
            item[0],
            _hash_value(b"world-counterfactual", semantic_world_value(item[1])),
            canonical_json_bytes(semantic_world_value(item[1])),
        ),
    ):
        result = _counterfactual_record(
            episode,
            axis=CounterfactualAxis.WORLD,
            query_index=query_index,
            semantic_distance=distance,
            world=candidate,
            command=episode.command,
            query=query,
        )
        if result is not None:
            return result
    return None


def minimal_command_counterfactual(
    episode: EpisodeRecord,
    *,
    query_index: int = 0,
) -> CounterfactualRecord | None:
    """Return the first canonical one-operation COMMAND edit that flips a query."""

    if type(query_index) is not int or not 0 <= query_index < 2:
        raise V3EpisodeError("query index differs")
    candidates: list[SemanticCommand] = []
    operations = episode.command.operations
    for position, previous in enumerate(operations):
        for successor in _operation_alphabet(episode.world):
            if successor == previous:
                continue
            changed = list(operations)
            changed[position] = successor
            candidates.append(_command_from_operations(episode.world, changed))
    query = episode.queries[query_index]
    for candidate in sorted(
        candidates,
        key=lambda command: (
            _hash_value(
                b"command-counterfactual",
                semantic_command_value(command),
            ),
            canonical_json_bytes(semantic_command_value(command)),
        ),
    ):
        result = _counterfactual_record(
            episode,
            axis=CounterfactualAxis.COMMAND,
            query_index=query_index,
            semantic_distance=1,
            world=episode.world,
            command=candidate,
            query=query,
        )
        if result is not None:
            return result
    return None


def minimal_query_counterfactual(
    episode: EpisodeRecord,
    *,
    query_index: int = 0,
) -> CounterfactualRecord | None:
    """Return the minimum canonical QUERY edit that flips the fixed execution."""

    if type(query_index) is not int or not 0 <= query_index < 2:
        raise V3EpisodeError("query index differs")
    source = episode.queries[query_index]
    candidates = tuple(
        query for query in enumerate_queries(episode.ontology) if query != source
    )
    for query in sorted(
        candidates,
        key=lambda candidate: (
            _query_distance(source, candidate),
            _hash_value(b"query-counterfactual", candidate.assessor_value()),
            canonical_json_bytes(candidate.assessor_value()),
        ),
    ):
        result = _counterfactual_record(
            episode,
            axis=CounterfactualAxis.QUERY,
            query_index=query_index,
            semantic_distance=_query_distance(source, query),
            world=episode.world,
            command=episode.command,
            query=query,
        )
        if result is not None:
            return result
    return None


def counterfactual_bundle(
    episode: EpisodeRecord,
) -> CounterfactualBundle | None:
    """Find one query with independently valid WORLD/COMMAND/QUERY flips."""

    for query_index in range(2):
        world = minimal_world_counterfactual(
            episode,
            query_index=query_index,
        )
        command = minimal_command_counterfactual(
            episode,
            query_index=query_index,
        )
        query = minimal_query_counterfactual(
            episode,
            query_index=query_index,
        )
        if world is not None and command is not None and query is not None:
            return CounterfactualBundle(
                query_index=query_index,
                world=world,
                command=command,
                query=query,
            )
    return None


def population_sha256(episodes: Iterable[EpisodeRecord]) -> str:
    """Hash an ordered or unordered episode population canonically."""

    values = tuple(
        sorted(
            (episode.episode_id for episode in episodes),
        )
    )
    if len(values) != len(set(values)):
        raise V3EpisodeError("population repeats an episode ID")
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "episode_ids": list(values),
                "protocol": PROTOCOL,
                "schema": EPISODE_SCHEMA,
            }
        )
    ).hexdigest()


def domain_cardinality_receipt() -> dict[str, object]:
    """Return static broad-domain cardinalities and a content hash."""

    horn_world_count = len(challenge_initials())
    resource_marking_count = len(full_resource_markings())
    receipt: dict[str, object] = {
        "horn": {
            "atomic_operations": len(all_ground_atoms()),
            "raw_atomic_pairs_per_policy": (
                len(HORN_THEORIES) * horn_world_count * len(all_ground_atoms())
            ),
            "theories": len(HORN_THEORIES),
            "worlds_per_theory_policy": horn_world_count,
        },
        "protocol": PROTOCOL,
        "resource": {
            "atomic_operations": OPERATOR_SYMBOL_COUNT,
            "raw_atomic_pairs_per_policy": (
                len(RESOURCE_THEORIES) * resource_marking_count * OPERATOR_SYMBOL_COUNT
            ),
            "theories": len(RESOURCE_THEORIES),
            "worlds_per_theory_policy": resource_marking_count,
        },
        "schema": "r12-ettr-il-v3-horn-resource-domain-receipt-v1",
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        canonical_json_bytes(receipt)
    ).hexdigest()
    return receipt


__all__ = [
    "COUNTERFACTUAL_SCHEMA",
    "EPISODE_SCHEMA",
    "MAX_PACKET_EDGES",
    "MAX_PACKET_SLOTS",
    "MAX_TRACE_STEPS",
    "PROTOCOL",
    "CapacityReceipt",
    "CounterfactualAxis",
    "CounterfactualBundle",
    "CounterfactualRecord",
    "CoverageMetadata",
    "CurriculumStage",
    "EpisodeRecord",
    "V3EpisodeError",
    "admitted_commands",
    "build_episode",
    "counterfactual_bundle",
    "domain_cardinality_receipt",
    "full_resource_markings",
    "generate_horn_episodes",
    "generate_resource_episodes",
    "horn_worlds",
    "minimal_command_counterfactual",
    "minimal_query_counterfactual",
    "minimal_world_counterfactual",
    "population_sha256",
    "resource_worlds",
]
