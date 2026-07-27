"""Deterministic broad 2x2 causal rectangles for ETTR-IL-v3 candidates.

Each selected semantic episode supplies the factual ``W0,C0`` corner.  This
module chooses a same-theory WORLD neighbor and a same-depth COMMAND neighbor,
reruns both independent executors at all four corners, and requires a real
terminal-state consequence on both WORLD edges.  COMMAND source and packet
controls remain distinct by construction.

Answer labels are deliberately not required to flip.  That is the v3 broad
initializer contract: every factor intervention must change the machine
state, while answer-preserving queries receive invariance supervision.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import TypeAlias

from ettr_il_v2_candidate_search import (
    canonical_json_bytes,
    semantic_command_value,
    semantic_world_value,
)
from ettr_il_v2_semantics import (
    HornCommand,
    HornExecution,
    HornWorld,
    ResourceCommand,
    ResourceExecution,
    ResourceWorld,
    SemanticQuery,
    TerminalDisposition as LegacyDisposition,
    execute_semantics,
    replay_semantics,
)
from ettr_il_v3_horn_resource import (
    EpisodeRecord,
    admitted_commands,
    horn_worlds,
    resource_worlds,
)
from ettr_il_v3_protocol import MASTER_SEED, PROTOCOL
from ettr_il_v3_reconstruct import ReconstructedCandidate
from ettr_il_v3_rewrite import (
    RewriteCommand,
    RewriteExecution,
    RewriteWorld,
    StructuralQuery,
    TerminalDisposition as RewriteDisposition,
    execute_primary,
    execute_replay,
    primitive_operations,
)
from ettr_il_v3_rewrite_episodes import RewriteEpisode


LegacyWorld: TypeAlias = HornWorld | ResourceWorld
LegacyCommand: TypeAlias = HornCommand | ResourceCommand
LegacyExecution: TypeAlias = HornExecution | ResourceExecution
WorldPair: TypeAlias = tuple[LegacyWorld, LegacyWorld] | tuple[
    RewriteWorld,
    RewriteWorld,
]
CommandPair: TypeAlias = tuple[LegacyCommand, LegacyCommand] | tuple[
    RewriteCommand,
    RewriteCommand,
]
QueryPair: TypeAlias = tuple[SemanticQuery, SemanticQuery] | tuple[
    StructuralQuery,
    StructuralQuery,
]
ExecutionGrid: TypeAlias = tuple[
    tuple[LegacyExecution, LegacyExecution],
    tuple[LegacyExecution, LegacyExecution],
] | tuple[
    tuple[RewriteExecution, RewriteExecution],
    tuple[RewriteExecution, RewriteExecution],
]


class RectangleError(ValueError):
    """A candidate cannot form a packet-effective broad causal rectangle."""


@dataclass(frozen=True, slots=True)
class SemanticRectangleBundle:
    """One typed 2x2 board with primary and independent replay executions."""

    semantic_rectangle_id: str
    episode_id: str
    family: str
    worlds: WorldPair
    commands: CommandPair
    queries: QueryPair
    primary: ExecutionGrid
    replay: ExecutionGrid


def _rank(domain: bytes, episode_id: str, value: object) -> bytes:
    return hashlib.sha256(
        MASTER_SEED
        + b"|"
        + domain
        + b"|"
        + episode_id.encode("ascii")
        + b"|"
        + canonical_json_bytes(value)
    ).digest()


def _legacy_execute(
    world: LegacyWorld,
    command: LegacyCommand,
) -> tuple[LegacyExecution, LegacyExecution] | None:
    try:
        primary = execute_semantics(
            world,
            command,
            require_dependent=False,
        )
        replay = replay_semantics(
            world,
            command,
            require_dependent=False,
        )
    except (TypeError, ValueError):
        return None
    if (
        primary != replay
        or primary.disposition is not LegacyDisposition.ANSWER
    ):
        return None
    return primary, replay  # type: ignore[return-value]


def _legacy_terminal_key(execution: LegacyExecution) -> object:
    if type(execution) is HornExecution:
        return {
            "disposition": execution.disposition.value,
            "terminal": [
                [atom.predicate, list(atom.arguments)]
                for atom in execution.terminal
            ],
        }
    if type(execution) is ResourceExecution:
        return {
            "cursor": execution.cursor,
            "disposition": execution.disposition.value,
            "status": execution.status.value,
            "terminal": list(execution.terminal.multiplicities),
        }
    raise RectangleError("legacy execution ontology differs")


def _legacy_world_candidates(
    episode: EpisodeRecord,
) -> tuple[LegacyWorld, ...]:
    if type(episode.world) is HornWorld:
        values = horn_worlds(
            episode.world.theory_index,
            policy=episode.world.policy,
        )
    elif type(episode.world) is ResourceWorld:
        values = resource_worlds(
            episode.world.theory_index,
            policy=episode.world.policy,
        )
    else:
        raise RectangleError("legacy episode world type differs")
    return tuple(
        sorted(
            (world for world in values if world != episode.world),
            key=lambda world: (
                _rank(
                    b"rectangle-world",
                    episode.episode_id,
                    semantic_world_value(world),
                ),
                canonical_json_bytes(semantic_world_value(world)),
            ),
        )
    )


def _legacy_command_candidates(
    episode: EpisodeRecord,
) -> tuple[LegacyCommand, ...]:
    generated = admitted_commands(
        episode.world,
        depth=episode.command.depth,
        beam_width=64,
        require_dependent=False,
    )
    values = {
        canonical_json_bytes(semantic_command_value(command)): command
        for command in generated
        if command != episode.command
    }
    return tuple(
        values[key]
        for key in sorted(
            values,
            key=lambda value: (
                _rank(
                    b"rectangle-command",
                    episode.episode_id,
                    semantic_command_value(values[value]),
                ),
                value,
            ),
        )
    )


def _legacy_rectangle(
    candidate: ReconstructedCandidate,
    episode: EpisodeRecord,
) -> SemanticRectangleBundle:
    factual = _legacy_execute(episode.world, episode.command)
    if factual is None or factual[0] != episode.primary or factual[1] != episode.replay:
        raise RectangleError("stored factual execution differs on reconstruction")
    for world in _legacy_world_candidates(episode):
        world_factual = _legacy_execute(world, episode.command)
        if (
            world_factual is None
            or _legacy_terminal_key(world_factual[0])
            == _legacy_terminal_key(factual[0])
        ):
            continue
        for command in _legacy_command_candidates(episode):
            source_command = _legacy_execute(episode.world, command)
            neighbor_command = _legacy_execute(world, command)
            if (
                source_command is None
                or neighbor_command is None
                or _legacy_terminal_key(source_command[0])
                == _legacy_terminal_key(neighbor_command[0])
            ):
                continue
            worlds: tuple[LegacyWorld, LegacyWorld] = (episode.world, world)
            commands: tuple[LegacyCommand, LegacyCommand] = (
                episode.command,
                command,
            )
            primary = (
                (factual[0], source_command[0]),
                (world_factual[0], neighbor_command[0]),
            )
            replay = (
                (factual[1], source_command[1]),
                (world_factual[1], neighbor_command[1]),
            )
            rectangle_id = hashlib.sha256(
                canonical_json_bytes(
                    {
                        "commands": [
                            semantic_command_value(item)
                            for item in commands
                        ],
                        "episode_id": episode.episode_id,
                        "family": candidate.family,
                        "protocol": PROTOCOL,
                        "worlds": [
                            semantic_world_value(item)
                            for item in worlds
                        ],
                    }
                )
            ).hexdigest()
            return SemanticRectangleBundle(
                semantic_rectangle_id=rectangle_id,
                episode_id=episode.episode_id,
                family=candidate.family,
                worlds=worlds,
                commands=commands,
                queries=episode.queries,
                primary=primary,
                replay=replay,
            )
    raise RectangleError("legacy causal-neighbor search exhausted")


def _rewrite_execute(
    world: RewriteWorld,
    command: RewriteCommand,
) -> tuple[RewriteExecution, RewriteExecution] | None:
    try:
        primary = execute_primary(world, command)
        replay = execute_replay(world, command)
    except (TypeError, ValueError):
        return None
    if (
        primary != replay
        or primary.disposition is not RewriteDisposition.ANSWER
    ):
        return None
    return primary, replay


def _rewrite_world_candidates(
    episode: RewriteEpisode,
) -> tuple[RewriteWorld, ...]:
    values: list[RewriteWorld] = []
    for slot, previous in enumerate(episode.world.registers):
        for successor in range(4):
            if successor == previous:
                continue
            registers = list(episode.world.registers)
            registers[slot] = successor
            values.append(
                RewriteWorld(
                    episode.world.theory_index,
                    tuple(registers),  # type: ignore[arg-type]
                )
            )
    return tuple(
        sorted(
            values,
            key=lambda world: (
                _rank(
                    b"rectangle-world",
                    episode.episode_id,
                    world.to_value(),
                ),
                world.canonical_bytes(),
            ),
        )
    )


def _rewrite_command_candidates(
    episode: RewriteEpisode,
) -> tuple[RewriteCommand, ...]:
    valid_operations = primitive_operations(True)
    values: dict[bytes, RewriteCommand] = {}
    for position, previous in enumerate(episode.command.operations):
        for successor in valid_operations:
            if successor == previous:
                continue
            operations = list(episode.command.operations)
            operations[position] = successor
            command = RewriteCommand(tuple(operations))
            values.setdefault(command.canonical_bytes(), command)
    return tuple(
        values[key]
        for key in sorted(
            values,
            key=lambda value: (
                _rank(
                    b"rectangle-command",
                    episode.episode_id,
                    values[value].to_value(),
                ),
                value,
            ),
        )
    )


def _rewrite_rectangle(
    candidate: ReconstructedCandidate,
    episode: RewriteEpisode,
) -> SemanticRectangleBundle:
    factual = _rewrite_execute(episode.world, episode.command)
    if factual is None or factual[0] != episode.primary or factual[1] != episode.replay:
        raise RectangleError("stored rewrite execution differs")
    for world in _rewrite_world_candidates(episode):
        world_factual = _rewrite_execute(world, episode.command)
        if world_factual is None or world_factual[0].terminal == factual[0].terminal:
            continue
        for command in _rewrite_command_candidates(episode):
            source_command = _rewrite_execute(episode.world, command)
            neighbor_command = _rewrite_execute(world, command)
            if (
                source_command is None
                or neighbor_command is None
                or source_command[0].terminal == neighbor_command[0].terminal
            ):
                continue
            worlds = (episode.world, world)
            commands = (episode.command, command)
            primary = (
                (factual[0], source_command[0]),
                (world_factual[0], neighbor_command[0]),
            )
            replay = (
                (factual[1], source_command[1]),
                (world_factual[1], neighbor_command[1]),
            )
            rectangle_id = hashlib.sha256(
                canonical_json_bytes(
                    {
                        "commands": [item.to_value() for item in commands],
                        "episode_id": episode.episode_id,
                        "family": candidate.family,
                        "protocol": PROTOCOL,
                        "worlds": [item.to_value() for item in worlds],
                    }
                )
            ).hexdigest()
            return SemanticRectangleBundle(
                semantic_rectangle_id=rectangle_id,
                episode_id=episode.episode_id,
                family=candidate.family,
                worlds=worlds,
                commands=commands,
                queries=episode.queries,
                primary=primary,
                replay=replay,
            )
    raise RectangleError("rewrite causal-neighbor search exhausted")


def build_causal_rectangle(
    candidate: ReconstructedCandidate,
) -> SemanticRectangleBundle:
    """Construct one deterministic packet-effective broad causal rectangle."""

    if not isinstance(candidate, ReconstructedCandidate):
        raise TypeError("candidate must be a ReconstructedCandidate")
    if isinstance(candidate.episode, RewriteEpisode):
        return _rewrite_rectangle(candidate, candidate.episode)
    if isinstance(candidate.episode, EpisodeRecord):
        return _legacy_rectangle(candidate, candidate.episode)
    raise RectangleError("candidate episode type differs")


__all__ = [
    "CommandPair",
    "ExecutionGrid",
    "QueryPair",
    "RectangleError",
    "SemanticRectangleBundle",
    "WorldPair",
    "build_causal_rectangle",
]
