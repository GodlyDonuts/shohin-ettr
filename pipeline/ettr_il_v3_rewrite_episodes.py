"""Curriculum episodes over the bounded ETTR-IL-v3 local rewrite domain.

The primitive rewrite module owns the two independent execution engines. This
module adds deterministic episode generation, curriculum admission, query
selection, and minimal factor interventions. It is CPU-only and performs no
filesystem, network, model, checkpoint, optimizer, or accelerator access.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

from ettr_il_v3_horn_resource import CurriculumStage
from ettr_il_v3_protocol import MASTER_SEED, PROTOCOL
from ettr_il_v3_rewrite import (
    MAX_DEPTH,
    MIN_DEPTH,
    LocalOperation,
    RewriteCommand,
    RewriteExecution,
    RewriteWorld,
    StepOutcome,
    StructuralQuery,
    TerminalDisposition,
    canonical_json_bytes,
    evaluate_query_primary,
    evaluate_query_replay,
    execute_primary,
    execute_replay,
    iter_worlds,
    primitive_operations,
    structural_queries,
)


SCHEMA = "r12-ettr-il-v3-rewrite-episode-v1"
COUNTERFACTUAL_SCHEMA = "r12-ettr-il-v3-rewrite-counterfactual-v1"


class RewriteEpisodeError(ValueError):
    """A local-rewrite episode violates the frozen v3 curriculum."""


@dataclass(frozen=True, slots=True)
class RewriteCapacity:
    initial_active_slots: int
    terminal_active_slots: int
    encoded_trace_steps: int

    def to_value(self) -> dict[str, int]:
        if (
            self.initial_active_slots != 6
            or self.terminal_active_slots != 6
            or not 1 <= self.encoded_trace_steps <= 64
        ):
            raise RewriteEpisodeError("rewrite episode exceeds ETTR capacity")
        return {
            "encoded_trace_steps": self.encoded_trace_steps,
            "initial_active_slots": self.initial_active_slots,
            "terminal_active_slots": self.terminal_active_slots,
        }


@dataclass(frozen=True, slots=True)
class RewriteCoverage:
    stage: str
    theory_index: int
    depth: int
    disposition: str
    applied_count: int
    blocked_count: int
    rejected_count: int
    prefix_dependent_steps: int
    changed_slots: int
    query_ops: tuple[str, str]
    query_answers: tuple[bool, bool]

    def to_value(self) -> dict[str, object]:
        return {
            "applied_count": self.applied_count,
            "blocked_count": self.blocked_count,
            "changed_slots": self.changed_slots,
            "depth": self.depth,
            "disposition": self.disposition,
            "prefix_dependent_steps": self.prefix_dependent_steps,
            "query_answers": list(self.query_answers),
            "query_ops": list(self.query_ops),
            "rejected_count": self.rejected_count,
            "stage": self.stage,
            "theory_index": self.theory_index,
        }


@dataclass(frozen=True, slots=True)
class RewriteEpisode:
    episode_id: str
    stage: CurriculumStage
    world: RewriteWorld
    command: RewriteCommand
    primary: RewriteExecution
    replay: RewriteExecution
    queries: tuple[StructuralQuery, StructuralQuery]
    answers: tuple[bool, bool]
    capacity: RewriteCapacity
    coverage: RewriteCoverage

    def assessor_value(self) -> dict[str, object]:
        return {
            "answers": list(self.answers),
            "capacity": self.capacity.to_value(),
            "command": self.command.to_value(),
            "coverage": self.coverage.to_value(),
            "execution": self.primary.to_value(),
            "protocol": PROTOCOL,
            "queries": [query.to_value() for query in self.queries],
            "schema": SCHEMA,
            "stage": self.stage.value,
            "world": self.world.to_value(),
        }


@dataclass(frozen=True, slots=True)
class RewriteCounterfactual:
    counterfactual_id: str
    source_episode_id: str
    axis: str
    query_index: int
    semantic_distance: int
    world: RewriteWorld
    command: RewriteCommand
    query: StructuralQuery
    primary: RewriteExecution
    replay: RewriteExecution
    answer_before: bool
    answer_after: bool

    def to_value(self) -> dict[str, object]:
        return {
            "answer_after": self.answer_after,
            "answer_before": self.answer_before,
            "axis": self.axis,
            "command": self.command.to_value(),
            "counterfactual_id": self.counterfactual_id,
            "execution": self.primary.to_value(),
            "protocol": PROTOCOL,
            "query": self.query.to_value(),
            "query_index": self.query_index,
            "schema": COUNTERFACTUAL_SCHEMA,
            "semantic_distance": self.semantic_distance,
            "source_episode_id": self.source_episode_id,
            "world": self.world.to_value(),
        }


@dataclass(frozen=True, slots=True)
class RewriteCounterfactualBundle:
    query_index: int
    world: RewriteCounterfactual
    command: RewriteCounterfactual
    query: RewriteCounterfactual


def _rank(domain: bytes, value: object) -> bytes:
    return hashlib.sha256(
        MASTER_SEED + b"|" + domain + b"|" + canonical_json_bytes(value)
    ).digest()


def _execute_equal(
    world: RewriteWorld,
    command: RewriteCommand,
) -> tuple[RewriteExecution, RewriteExecution]:
    primary = execute_primary(world, command)
    replay = execute_replay(world, command)
    if primary != replay:
        raise RewriteEpisodeError("rewrite primary and replay differ")
    return primary, replay


def _prefix_dependent_count(execution: RewriteExecution) -> int:
    count = 0
    for index, step in enumerate(execution.steps):
        if index == 0 or step.outcome is not StepOutcome.APPLIED:
            continue
        primitive = RewriteCommand((step.operation,))
        initial_outcome = execute_primary(execution.world, primitive).steps[0].outcome
        if initial_outcome is not StepOutcome.APPLIED:
            count += 1
    return count


def _validate_stage(
    stage: CurriculumStage,
    execution: RewriteExecution,
) -> None:
    depth = execution.command.depth
    if stage in {
        CurriculumStage.COMPILER_GROUNDING,
        CurriculumStage.ATOMIC_TRANSITIONS,
    } and depth != 1:
        raise RewriteEpisodeError(f"{stage.value} requires depth one")
    if stage in {
        CurriculumStage.DEPENDENT_COMPOSITION,
        CurriculumStage.CLOSED_LOOP,
    }:
        if not 2 <= depth <= MAX_DEPTH:
            raise RewriteEpisodeError(f"{stage.value} requires depth 2..6")
        if (
            execution.disposition is not TerminalDisposition.ANSWER
            or len(execution.steps) != depth
            or any(step.outcome is not StepOutcome.APPLIED for step in execution.steps)
            or _prefix_dependent_count(execution) != depth - 1
        ):
            raise RewriteEpisodeError(
                f"{stage.value} requires a fully applied prefix-dependent path"
            )


def _query_rank(
    execution: RewriteExecution,
    query: StructuralQuery,
) -> tuple[bytes, bytes]:
    return (
        _rank(
            b"rewrite-query",
            {
                "execution": execution.to_value(),
                "query": query.to_value(),
            },
        ),
        query.canonical_bytes(),
    )


def _select_queries(
    execution: RewriteExecution,
) -> tuple[tuple[StructuralQuery, StructuralQuery], tuple[bool, bool]]:
    if execution.disposition is not TerminalDisposition.ANSWER:
        queries = structural_queries()
        return (queries[0], queries[1]), (False, False)
    ranked = tuple(
        sorted(
            structural_queries(),
            key=lambda query: _query_rank(execution, query),
        )
    )
    by_answer = {
        answer: tuple(
            query
            for query in ranked
            if evaluate_query_primary(execution, query) is answer
        )
        for answer in (False, True)
    }
    if by_answer[False] and by_answer[True]:
        selected = (by_answer[False][0], by_answer[True][0])
    else:
        selected = (ranked[0], ranked[1])
    answers = tuple(
        evaluate_query_primary(execution, query) for query in selected
    )
    if answers != tuple(
        evaluate_query_replay(execution, query) for query in selected
    ):
        raise RewriteEpisodeError("rewrite query evaluators differ")
    return selected, answers  # type: ignore[return-value]


def build_rewrite_episode(
    *,
    stage: CurriculumStage,
    world: RewriteWorld,
    command: RewriteCommand,
) -> RewriteEpisode:
    if type(stage) is not CurriculumStage:
        raise RewriteEpisodeError("curriculum stage differs")
    primary, replay = _execute_equal(world, command)
    _validate_stage(stage, primary)
    queries, answers = _select_queries(primary)
    changed_slots = sum(
        before != after
        for before, after in zip(world.registers, primary.terminal, strict=True)
    )
    capacity = RewriteCapacity(
        initial_active_slots=6,
        terminal_active_slots=6,
        encoded_trace_steps=2 * len(primary.steps) + changed_slots + 2,
    )
    capacity.to_value()
    coverage = RewriteCoverage(
        stage=stage.value,
        theory_index=world.theory_index,
        depth=command.depth,
        disposition=primary.disposition.value,
        applied_count=primary.applied_count,
        blocked_count=primary.blocked_count,
        rejected_count=sum(
            step.outcome is StepOutcome.REJECTED for step in primary.steps
        ),
        prefix_dependent_steps=_prefix_dependent_count(primary),
        changed_slots=changed_slots,
        query_ops=(queries[0].op.value, queries[1].op.value),
        query_answers=answers,
    )
    provisional = RewriteEpisode(
        episode_id="",
        stage=stage,
        world=world,
        command=command,
        primary=primary,
        replay=replay,
        queries=queries,
        answers=answers,
        capacity=capacity,
        coverage=coverage,
    )
    episode_id = hashlib.sha256(
        canonical_json_bytes(provisional.assessor_value())
    ).hexdigest()
    return RewriteEpisode(
        episode_id=episode_id,
        stage=stage,
        world=world,
        command=command,
        primary=primary,
        replay=replay,
        queries=queries,
        answers=answers,
        capacity=capacity,
        coverage=coverage,
    )


def _counterfactual(
    episode: RewriteEpisode,
    *,
    axis: str,
    query_index: int,
    semantic_distance: int,
    world: RewriteWorld,
    command: RewriteCommand,
    query: StructuralQuery,
) -> RewriteCounterfactual | None:
    if episode.primary.disposition is not TerminalDisposition.ANSWER:
        return None
    try:
        primary, replay = _execute_equal(world, command)
        if primary.disposition is not TerminalDisposition.ANSWER:
            return None
        answer_after = evaluate_query_primary(primary, query)
        if answer_after != evaluate_query_replay(replay, query):
            raise RewriteEpisodeError("counterfactual query replay differs")
    except (RewriteEpisodeError, ValueError):
        return None
    answer_before = episode.answers[query_index]
    if answer_before is answer_after:
        return None
    provisional = {
        "answer_after": answer_after,
        "answer_before": answer_before,
        "axis": axis,
        "command": command.to_value(),
        "protocol": PROTOCOL,
        "query": query.to_value(),
        "query_index": query_index,
        "schema": COUNTERFACTUAL_SCHEMA,
        "semantic_distance": semantic_distance,
        "source_episode_id": episode.episode_id,
        "world": world.to_value(),
    }
    counterfactual_id = hashlib.sha256(
        canonical_json_bytes(provisional)
    ).hexdigest()
    return RewriteCounterfactual(
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
        answer_before=answer_before,
        answer_after=answer_after,
    )


def minimal_world_counterfactual(
    episode: RewriteEpisode,
    *,
    query_index: int = 0,
) -> RewriteCounterfactual | None:
    if not 0 <= query_index < 2:
        raise RewriteEpisodeError("query index differs")
    query = episode.queries[query_index]
    candidates = []
    for slot, value in enumerate(episode.world.registers):
        for replacement in range(4):
            if replacement == value:
                continue
            registers = list(episode.world.registers)
            registers[slot] = replacement
            world = RewriteWorld(episode.world.theory_index, tuple(registers))
            candidates.append(world)
    for world in sorted(
        candidates,
        key=lambda candidate: (
            _rank(b"rewrite-world-counterfactual", candidate.to_value()),
            candidate.canonical_bytes(),
        ),
    ):
        result = _counterfactual(
            episode,
            axis="world",
            query_index=query_index,
            semantic_distance=1,
            world=world,
            command=episode.command,
            query=query,
        )
        if result is not None:
            return result
    return None


def minimal_command_counterfactual(
    episode: RewriteEpisode,
    *,
    query_index: int = 0,
) -> RewriteCounterfactual | None:
    if not 0 <= query_index < 2:
        raise RewriteEpisodeError("query index differs")
    query = episode.queries[query_index]
    candidates: dict[bytes, RewriteCommand] = {}
    for index in range(episode.command.depth):
        for operation in primitive_operations():
            if operation == episode.command.operations[index]:
                continue
            operations = list(episode.command.operations)
            operations[index] = operation
            candidate = RewriteCommand(tuple(operations))
            candidates.setdefault(candidate.canonical_bytes(), candidate)
    for command in sorted(
        candidates.values(),
        key=lambda candidate: (
            _rank(b"rewrite-command-counterfactual", candidate.to_value()),
            candidate.canonical_bytes(),
        ),
    ):
        result = _counterfactual(
            episode,
            axis="command",
            query_index=query_index,
            semantic_distance=1,
            world=episode.world,
            command=command,
            query=query,
        )
        if result is not None:
            return result
    return None


def _query_distance(left: StructuralQuery, right: StructuralQuery) -> int:
    if left.op is not right.op:
        return 1 + max(len(left.arguments), len(right.arguments))
    return sum(
        left_value != right_value
        for left_value, right_value in zip(
            left.arguments,
            right.arguments,
            strict=True,
        )
    )


def minimal_query_counterfactual(
    episode: RewriteEpisode,
    *,
    query_index: int = 0,
) -> RewriteCounterfactual | None:
    if not 0 <= query_index < 2:
        raise RewriteEpisodeError("query index differs")
    if episode.primary.disposition is not TerminalDisposition.ANSWER:
        return None
    source = episode.queries[query_index]
    candidates = tuple(query for query in structural_queries() if query != source)
    for query in sorted(
        candidates,
        key=lambda candidate: (
            _query_distance(source, candidate),
            _rank(b"rewrite-query-counterfactual", candidate.to_value()),
            candidate.canonical_bytes(),
        ),
    ):
        result = _counterfactual(
            episode,
            axis="query",
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
    episode: RewriteEpisode,
) -> RewriteCounterfactualBundle | None:
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
            return RewriteCounterfactualBundle(
                query_index=query_index,
                world=world,
                command=command,
                query=query,
            )
    return None


def _candidate_commands(
    world: RewriteWorld,
    *,
    stage: CurriculumStage,
    depth: int,
    beam_width: int,
) -> tuple[RewriteCommand, ...]:
    if not MIN_DEPTH <= depth <= MAX_DEPTH:
        raise RewriteEpisodeError("depth differs")
    if beam_width < 1:
        raise RewriteEpisodeError("beam width differs")
    if depth == 1:
        operations = primitive_operations(
            include_reject_controls=stage is CurriculumStage.ATOMIC_TRANSITIONS
        )
        return tuple(RewriteCommand((operation,)) for operation in operations)

    survivors: tuple[tuple[LocalOperation, ...], ...] = ((),)
    valid_operations = primitive_operations(include_reject_controls=False)
    for prefix_depth in range(1, depth + 1):
        next_survivors: dict[bytes, tuple[LocalOperation, ...]] = {}
        for prefix in survivors:
            for operation in valid_operations:
                operations = (*prefix, operation)
                command = RewriteCommand(operations)
                execution = execute_primary(world, command)
                if (
                    len(execution.steps) != prefix_depth
                    or any(
                        step.outcome is not StepOutcome.APPLIED
                        for step in execution.steps
                    )
                    or _prefix_dependent_count(execution) != prefix_depth - 1
                ):
                    continue
                next_survivors.setdefault(command.canonical_bytes(), operations)
        survivors = tuple(
            sorted(
                next_survivors.values(),
                key=lambda operations: (
                    _rank(
                        b"rewrite-command-beam",
                        RewriteCommand(operations).to_value(),
                    ),
                    RewriteCommand(operations).canonical_bytes(),
                ),
            )[:beam_width]
        )
        if not survivors:
            break
    return tuple(
        RewriteCommand(operations)
        for operations in survivors
        if len(operations) == depth
    )


def _ordered_worlds(
    theory_index: int,
    *,
    bucket_index: int,
    bucket_count: int,
) -> tuple[RewriteWorld, ...]:
    if bucket_count < 1 or not 0 <= bucket_index < bucket_count:
        raise RewriteEpisodeError("bucket differs")
    worlds = sorted(
        iter_worlds(theory_index),
        key=lambda world: (
            _rank(b"rewrite-world-rank", world.to_value()),
            world.canonical_bytes(),
        ),
    )
    return tuple(
        world
        for world in worlds
        if int.from_bytes(
            _rank(b"rewrite-world-bucket", world.to_value())[:8],
            "big",
        )
        % bucket_count
        == bucket_index
    )


def generate_rewrite_episodes(
    *,
    stage: CurriculumStage,
    theory_index: int,
    depth: int = 1,
    limit: int | None = None,
    beam_width: int = 64,
    bucket_index: int = 0,
    bucket_count: int = 1,
) -> tuple[RewriteEpisode, ...]:
    """Generate one deterministic theory/stage/bucket population."""

    if limit is not None and limit < 1:
        raise RewriteEpisodeError("limit differs")
    if stage in {
        CurriculumStage.COMPILER_GROUNDING,
        CurriculumStage.ATOMIC_TRANSITIONS,
    } and depth != 1:
        raise RewriteEpisodeError(f"{stage.value} requires depth one")
    if stage in {
        CurriculumStage.DEPENDENT_COMPOSITION,
        CurriculumStage.CLOSED_LOOP,
    } and not 2 <= depth <= MAX_DEPTH:
        raise RewriteEpisodeError(f"{stage.value} requires depth 2..6")

    records: dict[str, RewriteEpisode] = {}
    for world in _ordered_worlds(
        theory_index,
        bucket_index=bucket_index,
        bucket_count=bucket_count,
    ):
        commands = _candidate_commands(
            world,
            stage=stage,
            depth=depth,
            beam_width=beam_width,
        )
        if stage is CurriculumStage.COMPILER_GROUNDING:
            commands = commands[:1]
        for command in commands:
            try:
                episode = build_rewrite_episode(
                    stage=stage,
                    world=world,
                    command=command,
                )
                if (
                    stage is CurriculumStage.QUERY_COUNTERFACTUAL_GROUNDING
                    and all(
                        minimal_query_counterfactual(
                            episode,
                            query_index=query_index,
                        )
                        is None
                        for query_index in range(2)
                    )
                ):
                    continue
                if (
                    stage is CurriculumStage.CLOSED_LOOP
                    and counterfactual_bundle(episode) is None
                ):
                    continue
            except RewriteEpisodeError:
                continue
            records.setdefault(episode.episode_id, episode)
            if limit is not None and len(records) >= limit:
                return tuple(records.values())
    return tuple(records.values())


def population_sha256(episodes: Iterable[RewriteEpisode]) -> str:
    identifiers = tuple(sorted(episode.episode_id for episode in episodes))
    if len(identifiers) != len(set(identifiers)):
        raise RewriteEpisodeError("population repeats an episode ID")
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "episode_ids": list(identifiers),
                "protocol": PROTOCOL,
                "schema": SCHEMA,
            }
        )
    ).hexdigest()


__all__ = [
    "COUNTERFACTUAL_SCHEMA",
    "SCHEMA",
    "RewriteCapacity",
    "RewriteCounterfactual",
    "RewriteCounterfactualBundle",
    "RewriteCoverage",
    "RewriteEpisode",
    "RewriteEpisodeError",
    "build_rewrite_episode",
    "counterfactual_bundle",
    "generate_rewrite_episodes",
    "minimal_command_counterfactual",
    "minimal_query_counterfactual",
    "minimal_world_counterfactual",
    "population_sha256",
]
