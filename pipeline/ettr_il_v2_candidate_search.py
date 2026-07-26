"""Deterministic semantic-candidate search for R12-ETTR-IL-v2."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from itertools import combinations
import json
from typing import Iterable, Sequence

from cross_ontology_horn_board import (
    GroundAtom,
    all_ground_atoms,
    challenge_initials as horn_initials,
)
from cross_ontology_resource_board import (
    Marking,
    input_markings as resource_initials,
)
from cross_ontology_rewrite_board import (
    GroundTerm,
    challenge_terms as rewrite_initials,
)
from ettr_il_v2_semantics import (
    CHECKERBOARD_PATTERNS,
    Command,
    Execution,
    HornCommand,
    HornExecution,
    HornPolicy,
    HornWorld,
    Ontology,
    ResourceCommand,
    ResourceExecution,
    ResourcePolicy,
    ResourceWorld,
    RewriteCommand,
    RewriteExecution,
    RewritePolicy,
    RewriteWorld,
    SelectedQueries,
    SemanticAdmissionError,
    SemanticError,
    SemanticRectangle,
    TerminalDisposition,
    World,
    enumerate_queries,
    evaluate_query,
    execute_semantics,
    replay_semantics,
    select_queries,
    MASTER_SEED,
)


PROTOCOL = "R12-ETTR-IL-v2"
EVIDENCE_SCHEMA = "r12-ettr-il-v2-evidence-id-v1"


class CandidateSearchError(ValueError):
    """The frozen candidate domain cannot satisfy semantic admission."""


@dataclass(frozen=True, slots=True)
class SemanticCandidate:
    ontology: Ontology
    theory_index: int
    depth: int
    worlds: tuple[World, World]
    commands: tuple[Command, Command]
    rectangle: SemanticRectangle
    queries: SelectedQueries


@dataclass(frozen=True, slots=True)
class BeamReceipt:
    """Deterministic command-beam accounting for one owned world pair."""

    fold: int
    split: str
    ontology: Ontology
    theory_index: int
    target_depth: int
    world_ids: tuple[str, str]
    operation_alphabet_size: int
    raw_template_count: int
    prefix_survivors: tuple[int, ...]
    retained_survivors: tuple[int, ...]
    final_command_count: int


@dataclass(frozen=True, slots=True)
class CandidateScanReceipt:
    """Accounting for a deterministic finite prefix of one semantic cell."""

    fold: int
    split: str
    ontology: Ontology
    theory_index: int
    depth: int
    owned_world_count: int
    owned_world_pair_count: int
    scanned_world_pairs: int
    emitted_command_pairs: int
    execution_pass_count: int
    checkerboard_pass_count: int
    unique_core_count: int
    stopped_after: int | None
    exhausted: bool


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _atom_value(atom: GroundAtom) -> list[object]:
    return [atom.predicate, list(atom.arguments)]


def _term_value(term: GroundTerm) -> list[object]:
    return [
        term.type_index,
        term.constructor_index,
        [_term_value(child) for child in term.children],
    ]


def _initial_value(ontology: Ontology, initial: object) -> object:
    if ontology is Ontology.HORN:
        if not isinstance(initial, tuple):
            raise CandidateSearchError("Horn initial state differs")
        return [_atom_value(atom) for atom in initial]
    if ontology is Ontology.REWRITE:
        if not isinstance(initial, GroundTerm):
            raise CandidateSearchError("rewrite initial term differs")
        return _term_value(initial)
    if not isinstance(initial, Marking):
        raise CandidateSearchError("resource initial marking differs")
    return list(initial.multiplicities)


def _evidence_id(
    ontology: Ontology,
    theory_index: int,
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "ontology": ontology.value,
                "protocol": PROTOCOL,
                "schema": EVIDENCE_SCHEMA,
                "theory_index": theory_index,
            }
        )
    ).hexdigest()


def semantic_world_value(world: World) -> dict[str, object]:
    if isinstance(world, HornWorld):
        ontology = Ontology.HORN
    elif isinstance(world, RewriteWorld):
        ontology = Ontology.REWRITE
    elif isinstance(world, ResourceWorld):
        ontology = Ontology.RESOURCE
    else:
        raise CandidateSearchError("semantic world type differs")
    return {
        "evidence_id": world.evidence_id,
        "initial": _initial_value(ontology, world.initial),
        "ontology": ontology.value,
        "policy": world.policy.value,
        "theory_index": world.theory_index,
    }


def semantic_command_value(command: Command) -> dict[str, object]:
    if isinstance(command, HornCommand):
        ontology = Ontology.HORN
        operations: object = [
            _atom_value(operation) for operation in command.operations
        ]
    elif isinstance(command, RewriteCommand):
        ontology = Ontology.REWRITE
        operations = list(command.operations)
    elif isinstance(command, ResourceCommand):
        ontology = Ontology.RESOURCE
        operations = list(command.operations)
    else:
        raise CandidateSearchError("semantic command type differs")
    return {
        "depth": command.depth,
        "ontology": ontology.value,
        "operations": operations,
    }


def semantic_candidate_value(
    candidate: SemanticCandidate,
) -> dict[str, object]:
    if not isinstance(candidate, SemanticCandidate):
        raise CandidateSearchError("semantic candidate type differs")
    return {
        "commands": [
            semantic_command_value(command)
            for command in candidate.commands
        ],
        "depth": candidate.depth,
        "ontology": candidate.ontology.value,
        "queries": [
            {
                "args": list(query.args),
                "op": query.op.value,
            }
            for query in (
                candidate.queries.slot_0,
                candidate.queries.slot_1,
            )
        ],
        "theory_index": candidate.theory_index,
        "worlds": [
            semantic_world_value(world) for world in candidate.worlds
        ],
    }


def semantic_core_id(candidate: SemanticCandidate) -> str:
    return hashlib.sha256(
        canonical_json_bytes(semantic_candidate_value(candidate))
    ).hexdigest()


def semantic_world_id(world: World) -> str:
    return hashlib.sha256(
        canonical_json_bytes(semantic_world_value(world))
    ).hexdigest()


def semantic_command_id(command: Command) -> str:
    return hashlib.sha256(
        canonical_json_bytes(semantic_command_value(command))
    ).hexdigest()


def terminal_observation_value(execution: Execution) -> dict[str, object]:
    if isinstance(execution, HornExecution):
        return {
            "facts": [_atom_value(atom) for atom in execution.terminal],
            "ontology": Ontology.HORN.value,
        }
    if isinstance(execution, RewriteExecution):
        return {
            "normal_forms": [
                _term_value(term)
                for term in execution.terminal_normal_forms
            ],
            "ontology": Ontology.REWRITE.value,
        }
    if isinstance(execution, ResourceExecution):
        return {
            "cursor": execution.cursor,
            "marking": list(execution.terminal.multiplicities),
            "ontology": Ontology.RESOURCE.value,
            "status": execution.status.value,
        }
    raise CandidateSearchError("terminal execution type differs")


def _theory_count(ontology: Ontology) -> int:
    if ontology is Ontology.HORN:
        from cross_ontology_horn_board import THEORIES  # noqa: PLC0415
    elif ontology is Ontology.REWRITE:
        from cross_ontology_rewrite_board import THEORIES  # noqa: PLC0415
    else:
        from cross_ontology_resource_board import THEORIES  # noqa: PLC0415
    return len(THEORIES)


def _worlds(
    ontology: Ontology,
    theory_index: int,
) -> tuple[World, ...]:
    if ontology is Ontology.HORN:
        return tuple(
            HornWorld(
                _evidence_id(ontology, theory_index),
                theory_index,
                initial,
                HornPolicy.PERSISTENT,
            )
            for initial in horn_initials()
        )
    if ontology is Ontology.REWRITE:
        return tuple(
            RewriteWorld(
                _evidence_id(ontology, theory_index),
                theory_index,
                initial,
                RewritePolicy.CONTEXTUAL,
            )
            for initial in rewrite_initials()
            if initial.type_index == 0
        )
    return tuple(
        ResourceWorld(
            _evidence_id(ontology, theory_index),
            theory_index,
            initial,
            ResourcePolicy.ATOMIC_DEADLOCK,
        )
        for initial in resource_initials()
    )


def _operation_alphabet(ontology: Ontology) -> tuple[object, ...]:
    if ontology is Ontology.HORN:
        return tuple(all_ground_atoms())
    if ontology is Ontology.REWRITE:
        return (0, 1)
    return (0, 1, 2)


def _command_from_operations(
    ontology: Ontology,
    operations: Sequence[object],
) -> Command:
    values = tuple(operations)
    if ontology is Ontology.HORN:
        if not all(isinstance(value, GroundAtom) for value in values):
            raise CandidateSearchError("Horn operation sequence differs")
        return HornCommand(len(values), values)  # type: ignore[arg-type]
    if not all(type(value) is int for value in values):
        raise CandidateSearchError("integer operation sequence differs")
    if ontology is Ontology.REWRITE:
        return RewriteCommand(len(values), values)  # type: ignore[arg-type]
    return ResourceCommand(len(values), values)  # type: ignore[arg-type]


def _primitive_commands(ontology: Ontology) -> tuple[Command, ...]:
    return tuple(
        _command_from_operations(ontology, (operation,))
        for operation in _operation_alphabet(ontology)
    )


def _admitted_execution(
    world: World,
    command: Command,
    *,
    require_dependent: bool = False,
) -> Execution | None:
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
    except (SemanticError, TypeError, ValueError):
        return None
    if primary != replay:
        raise CandidateSearchError("primary and replay executions differ")
    if primary.disposition is not TerminalDisposition.ANSWER:
        return None
    return primary


@lru_cache(maxsize=3)
def terminal_witness_universe(
    ontology: Ontology,
) -> tuple[Execution, ...]:
    if type(ontology) is not Ontology:
        raise CandidateSearchError("ontology differs")
    observations: dict[bytes, Execution] = {}
    commands = _primitive_commands(ontology)
    for theory_index in range(_theory_count(ontology)):
        for world in _worlds(ontology, theory_index):
            for command in commands:
                execution = _admitted_execution(world, command)
                if execution is None:
                    continue
                payload = canonical_json_bytes(
                    terminal_observation_value(execution)
                )
                observations.setdefault(payload, execution)
    if not observations:
        raise CandidateSearchError(
            f"{ontology.value} terminal witness universe is empty"
        )
    return tuple(observations[key] for key in sorted(observations))


def _query_flip_codes(
    executions: tuple[Execution | None, ...],
    ontology: Ontology,
) -> tuple[int, ...]:
    if any(value is None for value in executions):
        return ()
    cells = tuple(
        value for value in executions if value is not None
    )
    codes: list[int] = []
    for query in enumerate_queries(ontology):
        labels = tuple(evaluate_query(query, cell) for cell in cells)
        if labels[0] == labels[1]:
            codes.append(0)
        elif labels[:2] == (False, True):
            codes.append(1)
        else:
            codes.append(2)
    return tuple(codes)


def _candidate_from_cells(
    ontology: Ontology,
    theory_index: int,
    worlds: tuple[World, World],
    commands: tuple[Command, Command],
    cells: tuple[Execution, Execution, Execution, Execution],
    universe: tuple[Execution, ...],
    *,
    depth: int,
) -> SemanticCandidate | None:
    rectangle = SemanticRectangle(cells)
    try:
        selected = select_queries(
            rectangle,
            bounded_terminal_universe=universe,
        )
    except SemanticAdmissionError:
        return None
    if (
        selected.slot_0_labels not in CHECKERBOARD_PATTERNS
        or selected.slot_1_labels not in CHECKERBOARD_PATTERNS
    ):
        raise AssertionError("selected query escaped checkerboard admission")
    return SemanticCandidate(
        ontology=ontology,
        theory_index=theory_index,
        depth=depth,
        worlds=worlds,
        commands=commands,
        rectangle=rectangle,
        queries=selected,
    )


def find_first_depth1_checkerboard(
    ontology: Ontology,
    *,
    bounded_terminal_universe: Iterable[Execution] | None = None,
) -> SemanticCandidate:
    """Find the first deterministic strict checkerboard in the depth-1 domain."""

    if type(ontology) is not Ontology:
        raise CandidateSearchError("ontology differs")
    universe = (
        terminal_witness_universe(ontology)
        if bounded_terminal_universe is None
        else tuple(bounded_terminal_universe)
    )
    if not universe:
        raise CandidateSearchError("query denotation universe is empty")
    commands = _primitive_commands(ontology)
    for theory_index in range(_theory_count(ontology)):
        worlds = _worlds(ontology, theory_index)
        grid: dict[tuple[int, int], Execution] = {}
        for world_index, world in enumerate(worlds):
            for command_index, command in enumerate(commands):
                execution = _admitted_execution(world, command)
                if execution is not None:
                    grid[world_index, command_index] = execution
        for command_left, command_right in combinations(
            range(len(commands)),
            2,
        ):
            codes_by_world: dict[int, tuple[int, ...]] = {}
            for world_index in range(len(worlds)):
                left = grid.get((world_index, command_left))
                right = grid.get((world_index, command_right))
                if left is None or right is None:
                    continue
                codes_by_world[world_index] = _query_flip_codes(
                    (left, right),
                    ontology,
                )
            for world_left, world_right in combinations(
                sorted(codes_by_world),
                2,
            ):
                first = codes_by_world[world_left]
                second = codes_by_world[world_right]
                opposite = sum(
                    left != 0 and right != 0 and left != right
                    for left, right in zip(first, second, strict=True)
                )
                if opposite < 2:
                    continue
                cells = (
                    grid[world_left, command_left],
                    grid[world_left, command_right],
                    grid[world_right, command_left],
                    grid[world_right, command_right],
                )
                candidate = _candidate_from_cells(
                    ontology,
                    theory_index,
                    (worlds[world_left], worlds[world_right]),
                    (commands[command_left], commands[command_right]),
                    cells,
                    universe,
                    depth=1,
                )
                if candidate is not None:
                    return candidate
    raise CandidateSearchError(
        f"{ontology.value} has no strict depth-1 checkerboard"
    )


def world_owner(
    world: World,
    *,
    fold: int,
    ontology: Ontology,
) -> int:
    """Return the immutable train/development/confirmation owner in ``0..2``."""

    if type(fold) is not int or fold not in (0, 1, 2):
        raise CandidateSearchError("fold differs")
    if type(ontology) is not Ontology:
        raise CandidateSearchError("ontology differs")
    payload = (
        MASTER_SEED
        + b"|world-owner|"
        + str(fold).encode("ascii")
        + b"|"
        + ontology.value.encode("ascii")
        + b"|"
        + canonical_json_bytes(semantic_world_value(world))
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 3


def owned_worlds(
    ontology: Ontology,
    *,
    theory_index: int,
    fold: int,
    split: str,
) -> tuple[World, ...]:
    if split not in ("train", "development", "confirmation"):
        raise CandidateSearchError("split differs")
    if type(theory_index) is not int or not 0 <= theory_index < _theory_count(
        ontology
    ):
        raise CandidateSearchError("theory index differs")
    owner = ("train", "development", "confirmation").index(split)
    return tuple(
        world
        for world in _worlds(ontology, theory_index)
        if world_owner(world, fold=fold, ontology=ontology) == owner
    )


def _beam_rank(
    *,
    fold: int,
    split: str,
    ontology: Ontology,
    theory_index: int,
    depth: int,
    worlds: tuple[World, World],
    operations: tuple[object, ...],
) -> tuple[bytes, bytes]:
    sequence = canonical_json_bytes(
        [
            semantic_command_value(
                _command_from_operations(ontology, operations)
            )
        ]
    )
    context = canonical_json_bytes(
        {
            "depth": depth,
            "fold": fold,
            "ontology": ontology.value,
            "sequence_sha256": hashlib.sha256(sequence).hexdigest(),
            "split": split,
            "theory_index": theory_index,
            "worlds": [
                semantic_world_value(world) for world in worlds
            ],
        }
    )
    return hashlib.sha256(MASTER_SEED + b"|beam|" + context).digest(), sequence


def beam_commands(
    *,
    fold: int,
    split: str,
    ontology: Ontology,
    theory_index: int,
    depth: int,
    worlds: tuple[World, World],
    beam_width: int = 64,
) -> tuple[tuple[Command, ...], BeamReceipt]:
    """Construct the frozen hash-ranked dependent command beam.

    Prefixes survive only when both independent semantic implementations agree
    and every operation is dependent in both worlds.  Ranking happens after
    each extension, which makes the returned beam independent of iterator or
    process scheduling.
    """

    if type(depth) is not int or not 1 <= depth <= 6:
        raise CandidateSearchError("depth differs")
    if type(beam_width) is not int or beam_width < 2:
        raise CandidateSearchError("beam width differs")
    if (
        type(worlds) is not tuple
        or len(worlds) != 2
        or semantic_world_id(worlds[0]) == semantic_world_id(worlds[1])
    ):
        raise CandidateSearchError("world pair differs")
    alphabet = _operation_alphabet(ontology)
    survivors: tuple[tuple[object, ...], ...] = ((),)
    prefix_counts: list[int] = []
    retained_counts: list[int] = []
    for prefix_depth in range(1, depth + 1):
        admitted: dict[bytes, tuple[object, ...]] = {}
        for prefix in survivors:
            for operation in alphabet:
                sequence = (*prefix, operation)
                command = _command_from_operations(ontology, sequence)
                if all(
                    _admitted_execution(
                        world,
                        command,
                        require_dependent=True,
                    )
                    is not None
                    for world in worlds
                ):
                    canonical = canonical_json_bytes(
                        semantic_command_value(command)
                    )
                    admitted.setdefault(canonical, sequence)
        prefix_counts.append(len(admitted))
        ranked = sorted(
            admitted.values(),
            key=lambda sequence: _beam_rank(
                fold=fold,
                split=split,
                ontology=ontology,
                theory_index=theory_index,
                depth=prefix_depth,
                worlds=worlds,
                operations=sequence,
            ),
        )
        survivors = tuple(ranked[:beam_width])
        retained_counts.append(len(survivors))
        if not survivors:
            break
    commands = tuple(
        _command_from_operations(ontology, sequence)
        for sequence in survivors
        if len(sequence) == depth
    )
    world_ids = tuple(semantic_world_id(world) for world in worlds)
    return commands, BeamReceipt(
        fold=fold,
        split=split,
        ontology=ontology,
        theory_index=theory_index,
        target_depth=depth,
        world_ids=(world_ids[0], world_ids[1]),
        operation_alphabet_size=len(alphabet),
        raw_template_count=len(alphabet) ** depth,
        prefix_survivors=tuple(prefix_counts),
        retained_survivors=tuple(retained_counts),
        final_command_count=len(commands),
    )


def scan_admissible_candidates(
    *,
    fold: int,
    split: str,
    ontology: Ontology,
    theory_index: int,
    depth: int,
    stop_after: int | None = None,
    beam_width: int = 64,
    bounded_terminal_universe: Iterable[Execution] | None = None,
) -> tuple[tuple[SemanticCandidate, ...], CandidateScanReceipt]:
    """Scan a deterministic world/command prefix and return unique cores.

    ``stop_after`` is a constructive certificate boundary: every earlier
    world pair and command pair is still visited in canonical order and every
    rejection remains counted.  ``None`` exhausts the finite owned domain.
    """

    if stop_after is not None and (
        type(stop_after) is not int or stop_after < 1
    ):
        raise CandidateSearchError("stop_after differs")
    universe = (
        terminal_witness_universe(ontology)
        if bounded_terminal_universe is None
        else tuple(bounded_terminal_universe)
    )
    if not universe:
        raise CandidateSearchError("query denotation universe is empty")
    worlds = owned_worlds(
        ontology,
        theory_index=theory_index,
        fold=fold,
        split=split,
    )
    ordered_worlds = tuple(
        sorted(
            worlds,
            key=lambda world: canonical_json_bytes(
                semantic_world_value(world)
            ),
        )
    )
    world_pairs = tuple(combinations(ordered_worlds, 2))
    found: dict[str, SemanticCandidate] = {}
    emitted = 0
    execution_pass = 0
    checkerboard_pass = 0
    scanned_world_pairs = 0
    exhausted = True
    for pair in world_pairs:
        scanned_world_pairs += 1
        commands, _ = beam_commands(
            fold=fold,
            split=split,
            ontology=ontology,
            theory_index=theory_index,
            depth=depth,
            worlds=pair,
            beam_width=beam_width,
        )
        ordered_commands = tuple(
            sorted(
                commands,
                key=lambda command: canonical_json_bytes(
                    semantic_command_value(command)
                ),
            )
        )
        execution_grid = {
            (world_index, command_index): _admitted_execution(
                world,
                command,
                require_dependent=True,
            )
            for world_index, world in enumerate(pair)
            for command_index, command in enumerate(ordered_commands)
        }
        for command_left, command_right in combinations(
            range(len(ordered_commands)),
            2,
        ):
            emitted += 1
            cell_values = tuple(
                execution_grid[(world_index, command_index)]
                for world_index in range(2)
                for command_index in (command_left, command_right)
            )
            if any(value is None for value in cell_values):
                continue
            execution_pass += 1
            cells = tuple(value for value in cell_values if value is not None)
            candidate = _candidate_from_cells(
                ontology,
                theory_index,
                pair,
                (
                    ordered_commands[command_left],
                    ordered_commands[command_right],
                ),
                cells,  # type: ignore[arg-type]
                universe,
                depth=depth,
            )
            if candidate is None:
                continue
            checkerboard_pass += 1
            found.setdefault(semantic_core_id(candidate), candidate)
            if stop_after is not None and len(found) >= stop_after:
                exhausted = False
                break
        if not exhausted:
            break
    return tuple(found.values()), CandidateScanReceipt(
        fold=fold,
        split=split,
        ontology=ontology,
        theory_index=theory_index,
        depth=depth,
        owned_world_count=len(ordered_worlds),
        owned_world_pair_count=len(world_pairs),
        scanned_world_pairs=scanned_world_pairs,
        emitted_command_pairs=emitted,
        execution_pass_count=execution_pass,
        checkerboard_pass_count=checkerboard_pass,
        unique_core_count=len(found),
        stopped_after=stop_after,
        exhausted=exhausted,
    )


__all__ = [
    "CandidateSearchError",
    "CandidateScanReceipt",
    "EVIDENCE_SCHEMA",
    "BeamReceipt",
    "PROTOCOL",
    "SemanticCandidate",
    "beam_commands",
    "canonical_json_bytes",
    "find_first_depth1_checkerboard",
    "owned_worlds",
    "scan_admissible_candidates",
    "semantic_candidate_value",
    "semantic_command_value",
    "semantic_command_id",
    "semantic_core_id",
    "semantic_world_value",
    "semantic_world_id",
    "terminal_observation_value",
    "terminal_witness_universe",
    "world_owner",
]
