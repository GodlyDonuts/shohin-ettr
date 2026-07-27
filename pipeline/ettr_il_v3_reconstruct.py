"""Strict reconstruction of selected ETTR-IL-v3 semantic candidates.

Production shards intentionally contain canonical assessor values rather than
pickled Python objects.  This module is the sole inverse boundary: it rebuilds
the typed finite-semantic objects, reruns both executors, and requires the
resulting assessor record and content hash to match the stored candidate.

It is CPU-only and performs no filesystem, network, tokenizer, model,
checkpoint, optimizer, or accelerator access.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping

from cross_ontology_horn_board import GroundAtom
from cross_ontology_resource_board import Marking
from ettr_il_v2_candidate_search import canonical_json_bytes
from ettr_il_v2_semantics import (
    HornCommand,
    HornPolicy,
    HornWorld,
    Ontology,
    QueryOp as LegacyQueryOp,
    ResourceCommand,
    ResourcePolicy,
    ResourceWorld,
    SemanticQuery,
)
from ettr_il_v3_horn_resource import (
    CurriculumStage,
    EpisodeRecord,
    build_episode,
)
from ettr_il_v3_protocol import FAMILIES, PROTOCOL
from ettr_il_v3_rewrite import (
    Direction,
    LocalOperation,
    QueryOp as RewriteQueryOp,
    RewriteCommand,
    RewriteWorld,
    StructuralQuery,
)
from ettr_il_v3_rewrite_episodes import (
    RewriteEpisode,
    build_rewrite_episode,
)


ROW_SCHEMA = "r12-ettr-il-v3-semantic-candidate-v1"


class ReconstructionError(ValueError):
    """A stored semantic candidate cannot be exactly reconstructed."""


@dataclass(frozen=True, slots=True)
class ReconstructedCandidate:
    """One typed, independently replayed candidate and its storage identity."""

    episode_id: str
    split: str
    family: str
    stage: CurriculumStage
    depth: int
    ordinal: int
    episode: EpisodeRecord | RewriteEpisode
    row: Mapping[str, object]


def _exact_mapping(
    value: object,
    fields: set[str],
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ReconstructionError(f"{name} fields differ")
    return value


def _exact_list(value: object, name: str) -> list[object]:
    if type(value) is not list:
        raise ReconstructionError(f"{name} must be an exact list")
    return value


def _exact_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise ReconstructionError(f"{name} must be an exact integer")
    return value


def _exact_text(value: object, name: str) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise ReconstructionError(f"{name} must be nonempty ASCII")
    return value


def _ground_atom(value: object, name: str) -> GroundAtom:
    pair = _exact_list(value, name)
    if len(pair) != 2:
        raise ReconstructionError(f"{name} fields differ")
    arguments = _exact_list(pair[1], f"{name}.arguments")
    return GroundAtom(
        _exact_int(pair[0], f"{name}.predicate"),
        tuple(
            _exact_int(argument, f"{name}.arguments[{index}]")
            for index, argument in enumerate(arguments)
        ),
    )


def _semantic_query(value: object, name: str) -> SemanticQuery:
    item = _exact_mapping(value, {"args", "op"}, name)
    arguments = _exact_list(item["args"], f"{name}.args")
    try:
        operation = LegacyQueryOp(_exact_text(item["op"], f"{name}.op"))
    except ValueError as exc:
        raise ReconstructionError(f"{name}.op differs") from exc
    return SemanticQuery(
        operation,
        tuple(
            _exact_int(argument, f"{name}.args[{index}]")
            for index, argument in enumerate(arguments)
        ),
    )


def _legacy_episode(
    family: str,
    stage: CurriculumStage,
    value: Mapping[str, object],
) -> EpisodeRecord:
    world_value = _exact_mapping(
        value.get("world"),
        {"evidence_id", "initial", "ontology", "policy", "theory_index"},
        "episode.world",
    )
    command_value = _exact_mapping(
        value.get("command"),
        {"depth", "ontology", "operations"},
        "episode.command",
    )
    queries_value = _exact_list(value.get("queries"), "episode.queries")
    if len(queries_value) != 2:
        raise ReconstructionError("episode.queries must contain two queries")
    queries = (
        _semantic_query(queries_value[0], "episode.queries[0]"),
        _semantic_query(queries_value[1], "episode.queries[1]"),
    )
    theory_index = _exact_int(
        world_value["theory_index"],
        "episode.world.theory_index",
    )
    evidence_id = _exact_text(
        world_value["evidence_id"],
        "episode.world.evidence_id",
    )
    depth = _exact_int(command_value["depth"], "episode.command.depth")
    operations = _exact_list(
        command_value["operations"],
        "episode.command.operations",
    )
    if family == "horn":
        if (
            world_value["ontology"] != Ontology.HORN.value
            or command_value["ontology"] != Ontology.HORN.value
        ):
            raise ReconstructionError("Horn ontology tag differs")
        initial = _exact_list(world_value["initial"], "episode.world.initial")
        try:
            policy = HornPolicy(
                _exact_text(world_value["policy"], "episode.world.policy")
            )
        except ValueError as exc:
            raise ReconstructionError("Horn policy differs") from exc
        world = HornWorld(
            evidence_id,
            theory_index,
            tuple(
                _ground_atom(atom, f"episode.world.initial[{index}]")
                for index, atom in enumerate(initial)
            ),
            policy,
        )
        command = HornCommand(
            depth,
            tuple(
                _ground_atom(
                    operation,
                    f"episode.command.operations[{index}]",
                )
                for index, operation in enumerate(operations)
            ),
        )
    elif family == "resource":
        if (
            world_value["ontology"] != Ontology.RESOURCE.value
            or command_value["ontology"] != Ontology.RESOURCE.value
        ):
            raise ReconstructionError("resource ontology tag differs")
        initial = _exact_list(world_value["initial"], "episode.world.initial")
        try:
            policy = ResourcePolicy(
                _exact_text(world_value["policy"], "episode.world.policy")
            )
        except ValueError as exc:
            raise ReconstructionError("resource policy differs") from exc
        world = ResourceWorld(
            evidence_id,
            theory_index,
            Marking(
                tuple(
                    _exact_int(item, f"episode.world.initial[{index}]")
                    for index, item in enumerate(initial)
                )
            ),
            policy,
        )
        command = ResourceCommand(
            depth,
            tuple(
                _exact_int(
                    operation,
                    f"episode.command.operations[{index}]",
                )
                for index, operation in enumerate(operations)
            ),
        )
    else:
        raise ReconstructionError("legacy episode family differs")
    return build_episode(
        stage=stage,
        world=world,
        command=command,
        queries=queries,
    )


def _rewrite_operation(value: object, name: str) -> LocalOperation:
    item = _exact_mapping(
        value,
        {"direction", "law_slot", "schema", "site"},
        name,
    )
    try:
        direction = Direction(_exact_text(item["direction"], f"{name}.direction"))
    except ValueError as exc:
        raise ReconstructionError(f"{name}.direction differs") from exc
    return LocalOperation(
        law_slot=_exact_int(item["law_slot"], f"{name}.law_slot"),
        site=_exact_int(item["site"], f"{name}.site"),
        direction=direction,
    )


def _rewrite_query(value: object, name: str) -> StructuralQuery:
    item = _exact_mapping(value, {"arguments", "op", "schema"}, name)
    arguments = _exact_list(item["arguments"], f"{name}.arguments")
    try:
        operation = RewriteQueryOp(_exact_text(item["op"], f"{name}.op"))
    except ValueError as exc:
        raise ReconstructionError(f"{name}.op differs") from exc
    return StructuralQuery(
        operation,
        tuple(
            _exact_int(argument, f"{name}.arguments[{index}]")
            for index, argument in enumerate(arguments)
        ),
    )


def _rewrite_episode(
    stage: CurriculumStage,
    value: Mapping[str, object],
) -> RewriteEpisode:
    world_value = _exact_mapping(
        value.get("world"),
        {"registers", "schema", "theory_index", "world_index"},
        "episode.world",
    )
    command_value = _exact_mapping(
        value.get("command"),
        {"depth", "operations", "schema"},
        "episode.command",
    )
    registers = _exact_list(world_value["registers"], "episode.world.registers")
    operations = _exact_list(
        command_value["operations"],
        "episode.command.operations",
    )
    queries_value = _exact_list(value.get("queries"), "episode.queries")
    if len(queries_value) != 2:
        raise ReconstructionError("episode.queries must contain two queries")
    world = RewriteWorld(
        _exact_int(world_value["theory_index"], "episode.world.theory_index"),
        tuple(
            _exact_int(register, f"episode.world.registers[{index}]")
            for index, register in enumerate(registers)
        ),  # type: ignore[arg-type]
    )
    command = RewriteCommand(
        tuple(
            _rewrite_operation(
                operation,
                f"episode.command.operations[{index}]",
            )
            for index, operation in enumerate(operations)
        )
    )
    if (
        world.index
        != _exact_int(world_value["world_index"], "episode.world.world_index")
        or command.depth
        != _exact_int(command_value["depth"], "episode.command.depth")
    ):
        raise ReconstructionError("rewrite derived identity differs")
    queries = (
        _rewrite_query(queries_value[0], "episode.queries[0]"),
        _rewrite_query(queries_value[1], "episode.queries[1]"),
    )
    rebuilt = build_rewrite_episode(
        stage=stage,
        world=world,
        command=command,
    )
    if rebuilt.queries != queries:
        raise ReconstructionError("rewrite selected queries differ")
    return rebuilt


def reconstruct_candidate(value: object) -> ReconstructedCandidate:
    """Rebuild and independently re-execute one canonical candidate row."""

    row = _exact_mapping(
        value,
        {
            "cell",
            "episode",
            "episode_id",
            "ordinal",
            "owner",
            "protocol",
            "schema",
        },
        "candidate",
    )
    if row["schema"] != ROW_SCHEMA or row["protocol"] != PROTOCOL:
        raise ReconstructionError("candidate protocol or schema differs")
    cell = _exact_mapping(
        row["cell"],
        {
            "candidate_target",
            "depth",
            "family",
            "index",
            "owner_skip",
            "selected_quota",
            "split",
            "stage",
        },
        "candidate.cell",
    )
    family = _exact_text(cell["family"], "candidate.cell.family")
    if family not in FAMILIES:
        raise ReconstructionError("candidate family differs")
    try:
        stage = CurriculumStage(
            _exact_text(cell["stage"], "candidate.cell.stage")
        )
    except ValueError as exc:
        raise ReconstructionError("candidate stage differs") from exc
    episode_value = _exact_mapping(
        row["episode"],
        {
            "answers",
            "capacity",
            "command",
            "coverage",
            "execution",
            "protocol",
            "queries",
            "schema",
            "stage",
            "world",
            *(() if family == "local_rewrite" else ("ontology",)),
        },
        "candidate.episode",
    )
    rebuilt = (
        _rewrite_episode(stage, episode_value)
        if family == "local_rewrite"
        else _legacy_episode(family, stage, episode_value)
    )
    if rebuilt.assessor_value() != dict(episode_value):
        raise ReconstructionError("reconstructed assessor episode differs")
    episode_id = _exact_text(row["episode_id"], "candidate.episode_id")
    observed_id = hashlib.sha256(canonical_json_bytes(episode_value)).hexdigest()
    if episode_id != observed_id or rebuilt.episode_id != episode_id:
        raise ReconstructionError("candidate episode identity differs")
    if rebuilt.stage is not stage:
        raise ReconstructionError("candidate stage binding differs")
    depth = _exact_int(cell["depth"], "candidate.cell.depth")
    if rebuilt.command.depth != depth:
        raise ReconstructionError("candidate depth binding differs")
    split = _exact_text(cell["split"], "candidate.cell.split")
    if row["owner"] != split.removesuffix("_reserve"):
        raise ReconstructionError("candidate owner binding differs")
    return ReconstructedCandidate(
        episode_id=episode_id,
        split=split,
        family=family,
        stage=stage,
        depth=depth,
        ordinal=_exact_int(row["ordinal"], "candidate.ordinal"),
        episode=rebuilt,
        row=row,
    )


__all__ = [
    "ROW_SCHEMA",
    "ReconstructedCandidate",
    "ReconstructionError",
    "reconstruct_candidate",
]
