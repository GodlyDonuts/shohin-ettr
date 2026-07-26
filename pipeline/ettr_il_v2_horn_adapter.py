"""Strict Horn-to-generic adapter for ETTR isolated learnability v2.

The adapter is assessor-side and CPU-only.  It accepts already-rendered,
cell-local candidate bytes, but it never renders, tokenizes, selects data,
opens a checkpoint, or starts training.  Horn semantics are recomputed through
both independent executors before the frozen generic packet projection is
constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TypeAlias

from cross_ontology_horn_board import (
    OBJECT_TYPES,
    PREDICATES,
    THEORIES,
    GroundAtom,
    all_ground_atoms,
    execute_closure,
    reference_theory_state,
)
from cross_ontology_schema import ObjectCell, ReactorState, RelationEdge
from ettr_il_v2_materialize import (
    Disposition,
    GenericCell,
    GenericCommand,
    GenericCorner,
    GenericEdge,
    GenericMutation,
    GenericOperationTrace,
    GenericPacket,
    GenericQuery,
    GenericSemanticRectangle,
    GenericWorld,
    Opcode,
    ValueRef,
)
from ettr_il_v2_semantics import (
    CHECKERBOARD_PATTERNS,
    HornCommand,
    HornExecution,
    HornPolicy,
    HornWorld,
    Ontology,
    SemanticQuery,
    TerminalDisposition,
    evaluate_query,
    execute_horn,
    replay_horn,
)


_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_STATIC_LIMIT = 32
_OBJECT_BASE = 32
_COMMAND_BASE = 48
_COMMAND_CAPACITY = 6
_CURSOR_SLOT = 54
_OUTCOME_SLOT = 55

WorldPair: TypeAlias = tuple[HornWorld, HornWorld]
CommandPair: TypeAlias = tuple[HornCommand, HornCommand]
ExecutionMatrix: TypeAlias = tuple[
    tuple[HornExecution, HornExecution],
    tuple[HornExecution, HornExecution],
]
SourceMatrix: TypeAlias = tuple[tuple[bytes, bytes], tuple[bytes, bytes]]
QueryPair: TypeAlias = tuple[SemanticQuery, SemanticQuery]


class HornAdapterError(ValueError):
    """Horn semantics or their generic projection fail v2 admission."""


@dataclass(frozen=True, slots=True)
class _StaticProjection:
    cells: tuple[GenericCell, ...]
    edges: tuple[GenericEdge, ...]
    root: int | None


def _require_exact_pair(value: object, name: str) -> tuple[object, object]:
    if type(value) is not tuple or len(value) != 2:
        raise HornAdapterError(f"{name} must be an exact pair")
    return value


def _require_execution_matrix(
    value: object,
    name: str,
) -> ExecutionMatrix:
    outer = _require_exact_pair(value, name)
    rows = tuple(
        _require_exact_pair(row, f"{name} row {index}")
        for index, row in enumerate(outer)
    )
    if any(type(item) is not HornExecution for row in rows for item in row):
        raise HornAdapterError(f"{name} contains a non-Horn execution")
    return rows  # type: ignore[return-value]


def _require_ascii_bytes(value: object, name: str) -> bytes:
    if type(value) is not bytes or not value:
        raise HornAdapterError(f"{name} must be nonempty exact bytes")
    try:
        value.decode("ascii")
    except UnicodeDecodeError as exc:
        raise HornAdapterError(f"{name} must be strict ASCII") from exc
    return value


def _require_source_matrix(value: object, name: str) -> SourceMatrix:
    outer = _require_exact_pair(value, name)
    rows: list[tuple[bytes, bytes]] = []
    for row_index, row in enumerate(outer):
        pair = _require_exact_pair(row, f"{name} row {row_index}")
        rows.append(
            (
                _require_ascii_bytes(
                    pair[0],
                    f"{name} {row_index}/0",
                ),
                _require_ascii_bytes(
                    pair[1],
                    f"{name} {row_index}/1",
                ),
            )
        )
    flattened = tuple(item for row in rows for item in row)
    if len(set(flattened)) != 4:
        raise HornAdapterError(f"{name} cell-local bytes are not all distinct")
    return tuple(rows)  # type: ignore[return-value]


def _require_identifier(value: object, name: str, *, digest: bool) -> str:
    if (
        type(value) is not str
        or not value
        or not value.isascii()
        or (digest and _HEX64.fullmatch(value) is None)
    ):
        raise HornAdapterError(f"{name} differs")
    return value


def _project_static_theory(theory_index: int) -> _StaticProjection:
    state = reference_theory_state(theory_index)
    if type(state) is not ReactorState:
        raise HornAdapterError("Horn reference theory state type differs")
    if (
        state.capacity != _STATIC_LIMIT
        or not 1 <= state.type_count <= 4
        or tuple(sorted(state.cells, key=lambda cell: cell.slot)) != state.cells
        or tuple(
            sorted(
                state.edges,
                key=lambda edge: (edge.relation_index, edge.arguments),
            )
        )
        != state.edges
        or tuple(
            sorted(state.relation_specs, key=lambda relation: relation.index)
        )
        != state.relation_specs
    ):
        raise HornAdapterError("Horn reference theory geometry is noncanonical")
    if any(type(cell) is not ObjectCell for cell in state.cells):
        raise HornAdapterError("Horn reference theory cell type differs")
    if any(
        cell.slot >= _STATIC_LIMIT or cell.type_index >= 4
        for cell in state.cells
    ):
        raise HornAdapterError("Horn static theory leaves slots/types 0..31/0..3")
    raw_values = {cell.value for cell in state.cells}
    if not raw_values or len(raw_values) > 32:
        raise HornAdapterError("Horn static theory value rank capacity differs")

    relation_specs = {
        relation.index: relation
        for relation in state.relation_specs
    }
    if (
        len(relation_specs) != len(state.relation_specs)
        or any(
            relation.index > 7
            or len(relation.argument_types) not in {1, 2}
            for relation in state.relation_specs
        )
    ):
        raise HornAdapterError("Horn static relation projection differs")
    generic_edges: list[GenericEdge] = []
    for edge in state.edges:
        if type(edge) is not RelationEdge:
            raise HornAdapterError("Horn reference theory edge type differs")
        relation = relation_specs.get(edge.relation_index)
        if (
            relation is None
            or len(edge.arguments) != len(relation.argument_types)
        ):
            raise HornAdapterError("Horn static edge arity differs")
        if len(edge.arguments) == 1:
            source = target = edge.arguments[0]
        else:
            source, target = edge.arguments
        generic_edges.append(
            GenericEdge(edge.relation_index, source, target)
        )

    cells = tuple(
        GenericCell(
            cell.slot,
            cell.type_index,
            ValueRef.static(cell.value),
        )
        for cell in state.cells
    )
    edges = tuple(sorted(generic_edges))
    projection = _StaticProjection(cells, edges, state.root)

    expected_cells = tuple(
        (cell.slot, cell.type_index, cell.value)
        for cell in state.cells
    )
    observed_cells = tuple(
        (cell.slot, cell.type_index, cell.value.index)
        for cell in projection.cells
    )
    expected_edges = tuple(
        (
            edge.relation_index,
            edge.arguments[0],
            edge.arguments[-1],
        )
        for edge in state.edges
    )
    observed_edges = tuple(
        (edge.relation, edge.source, edge.target)
        for edge in projection.edges
    )
    if (
        observed_cells != expected_cells
        or observed_edges != expected_edges
        or projection.root != state.root
    ):
        raise HornAdapterError("Horn static theory projection is not lossless")
    return projection


def _validate_atom(atom: object) -> GroundAtom:
    if type(atom) is not GroundAtom:
        raise HornAdapterError("Horn fact type differs")
    if not 0 <= atom.predicate < len(PREDICATES):
        raise HornAdapterError("Horn fact predicate differs")
    required_types = PREDICATES[atom.predicate].argument_types
    if type(atom.arguments) is not tuple or len(atom.arguments) != len(
        required_types
    ):
        raise HornAdapterError("Horn fact arity differs")
    if len(atom.arguments) not in {1, 2}:
        raise HornAdapterError("Horn fact cannot enter the binary ledger")
    for object_index, required_type in zip(
        atom.arguments,
        required_types,
        strict=True,
    ):
        if (
            type(object_index) is not int
            or not 0 <= object_index < len(OBJECT_TYPES)
            or OBJECT_TYPES[object_index] != required_type
        ):
            raise HornAdapterError("Horn fact typing differs")
    return atom


def _fact_edge(atom: GroundAtom) -> GenericEdge:
    atom = _validate_atom(atom)
    source = _OBJECT_BASE + atom.arguments[0]
    target = (
        source
        if len(atom.arguments) == 1
        else _OBJECT_BASE + atom.arguments[1]
    )
    return GenericEdge(8 + atom.predicate, source, target)


def _runtime_cells() -> tuple[GenericCell, ...]:
    return tuple(
        GenericCell(
            _OBJECT_BASE + object_index,
            4,
            ValueRef.local_id(object_index),
        )
        for object_index in range(len(OBJECT_TYPES))
    )


def _fact_edges(facts: tuple[GroundAtom, ...]) -> tuple[GenericEdge, ...]:
    if type(facts) is not tuple:
        raise HornAdapterError("Horn fact state is not an exact tuple")
    for atom in facts:
        _validate_atom(atom)
    if tuple(sorted(set(facts))) != facts:
        raise HornAdapterError("Horn fact state is not sorted unique")
    edges = tuple(sorted(_fact_edge(atom) for atom in facts))
    if len(edges) != len(set(edges)):
        raise HornAdapterError("Horn fact projection collides")
    return edges


def _initial_packet(
    static: _StaticProjection,
    world: HornWorld,
) -> GenericPacket:
    return GenericPacket(
        cells=static.cells + _runtime_cells(),
        edges=tuple(sorted((*static.edges, *_fact_edges(world.initial)))),
        root=static.root,
        committed=False,
        halted=False,
    )


def _command_atom_indices(command: HornCommand) -> tuple[int, ...]:
    atom_indices = {
        atom: index for index, atom in enumerate(all_ground_atoms())
    }
    result: list[int] = []
    for operation in command.operations:
        try:
            result.append(atom_indices[operation])
        except KeyError as exc:
            raise HornAdapterError(
                "Horn command atom leaves the canonical catalog"
            ) from exc
    if len(result) != command.depth or not 1 <= len(result) <= 6:
        raise HornAdapterError("Horn command depth differs")
    return tuple(result)


def _fact_mutations(
    before: tuple[GroundAtom, ...],
    after: tuple[GroundAtom, ...],
) -> tuple[GenericMutation, ...]:
    before_edges = set(_fact_edges(before))
    after_edges = set(_fact_edges(after))
    removed = sorted(before_edges - after_edges)
    added = sorted(after_edges - before_edges)
    return tuple(
        GenericMutation(
            Opcode.UNLINK,
            source=edge.source,
            target=edge.target,
            relation=edge.relation,
        )
        for edge in removed
    ) + tuple(
        GenericMutation(
            Opcode.LINK,
            source=edge.source,
            target=edge.target,
            relation=edge.relation,
        )
        for edge in added
    )


def _operation_traces(execution: HornExecution) -> tuple[GenericOperationTrace, ...]:
    if (
        len(execution.steps) != execution.command.depth
        or len(execution.snapshots) != execution.command.depth + 1
    ):
        raise HornAdapterError("Horn execution depth differs")
    traces: list[GenericOperationTrace] = []
    for index, step in enumerate(execution.steps, start=1):
        if (
            step.index != index
            or step.operation != execution.command.operations[index - 1]
            or step.before != execution.snapshots[index - 1]
            or step.after != execution.snapshots[index]
        ):
            raise HornAdapterError("Horn operation snapshot binding differs")
        traces.append(
            GenericOperationTrace(
                mutations=_fact_mutations(step.before, step.after),
                cursor=index,
            )
        )
    return tuple(traces)


def _disposition(
    value: TerminalDisposition,
) -> tuple[Disposition, ValueRef, bool, bool]:
    if value is TerminalDisposition.ANSWER:
        return Disposition.ANSWER, ValueRef.execute(), True, False
    if value is TerminalDisposition.ABSTAIN:
        return Disposition.ABSTAIN, ValueRef.abstain(), False, True
    if value is TerminalDisposition.REJECT:
        return Disposition.REJECT, ValueRef.reject(), True, True
    raise HornAdapterError("Horn terminal disposition differs")


def _terminal_packet(
    static: _StaticProjection,
    execution: HornExecution,
) -> GenericPacket:
    disposition, outcome, committed, halted = _disposition(
        execution.disposition
    )
    del disposition
    atom_indices = _command_atom_indices(execution.command)
    command_cells = tuple(
        GenericCell(
            _COMMAND_BASE + index,
            5,
            (
                ValueRef.command_atom(atom_indices[index])
                if index < len(atom_indices)
                else ValueRef.empty()
            ),
        )
        for index in range(_COMMAND_CAPACITY)
    )
    control_cells = (
        GenericCell(
            _CURSOR_SLOT,
            6,
            ValueRef.small_uint(execution.command.depth),
        ),
        GenericCell(_OUTCOME_SLOT, 6, outcome),
    )
    return GenericPacket(
        cells=static.cells + _runtime_cells() + command_cells + control_cells,
        edges=tuple(
            sorted((*static.edges, *_fact_edges(execution.terminal)))
        ),
        root=static.root,
        committed=committed,
        halted=halted,
    )


def _answer_pair(
    queries: QueryPair,
    primary: HornExecution,
    replay: HornExecution,
) -> tuple[bool | None, bool | None]:
    if primary.disposition is not replay.disposition:
        raise HornAdapterError("Horn primary/replay disposition differs")
    if primary.disposition is not TerminalDisposition.ANSWER:
        return (None, None)
    primary_answers = tuple(evaluate_query(query, primary) for query in queries)
    replay_answers = tuple(evaluate_query(query, replay) for query in queries)
    if primary_answers != replay_answers:
        raise HornAdapterError("Horn primary/replay answer labels differ")
    return primary_answers  # type: ignore[return-value]


def _validate_execution(
    world: HornWorld,
    command: HornCommand,
    primary: HornExecution,
    replay: HornExecution,
    *,
    world_index: int,
    command_index: int,
) -> None:
    name = f"Horn corner {world_index}{command_index}"
    if (
        primary.world != world
        or primary.command != command
        or replay.world != world
        or replay.command != command
    ):
        raise HornAdapterError(f"{name} world/command binding differs")
    expected_primary = execute_horn(world, command)
    expected_replay = replay_horn(world, command)
    full_state = tuple(world.initial)
    asserted = frozenset(world.initial)
    reference_snapshots: list[tuple[GroundAtom, ...]] = [world.initial]
    for operation in command.operations:
        asserted = asserted | {operation}
        full_state = execute_closure(
            THEORIES[world.theory_index],
            tuple(sorted((*full_state, operation))),
        )
        reference_snapshots.append(
            full_state
            if world.policy is HornPolicy.PERSISTENT
            else tuple(atom for atom in full_state if atom not in asserted)
        )
    if expected_primary != expected_replay:
        raise HornAdapterError(f"{name} independent executors disagree")
    if expected_primary.snapshots != tuple(reference_snapshots):
        raise HornAdapterError(f"{name} canonical execute_closure differs")
    if primary != expected_primary:
        raise HornAdapterError(f"{name} primary execution differs")
    if replay != expected_replay:
        raise HornAdapterError(f"{name} replay execution differs")


def _validate_generic_corner(
    corner: GenericCorner,
    *,
    static: _StaticProjection,
    world: HornWorld,
    command: HornCommand,
    primary: HornExecution,
    replay: HornExecution,
    queries: QueryPair,
) -> None:
    expected_traces = _operation_traces(primary)
    if corner.operation_traces != expected_traces:
        raise HornAdapterError("Horn generic operation mutations differ")

    current = set(_fact_edges(world.initial))
    for step, trace in zip(primary.steps, corner.operation_traces, strict=True):
        for mutation in trace.mutations:
            edge = GenericEdge(
                mutation.relation,
                mutation.source,
                mutation.target,
            )
            if mutation.opcode is Opcode.UNLINK:
                if edge not in current:
                    raise HornAdapterError("Horn trace unlinks a missing fact")
                current.remove(edge)
            elif mutation.opcode is Opcode.LINK:
                if edge in current:
                    raise HornAdapterError("Horn trace relinks an existing fact")
                current.add(edge)
            else:
                raise HornAdapterError("Horn fact mutation opcode differs")
        if current != set(_fact_edges(step.after)):
            raise HornAdapterError("Horn generic operation replay differs")

    expected_packet = _terminal_packet(static, primary)
    if corner.terminal_packet != expected_packet:
        raise HornAdapterError("Horn generic terminal packet differs")
    disposition, outcome, _, _ = _disposition(primary.disposition)
    if corner.disposition is not disposition or corner.outcome != outcome:
        raise HornAdapterError("Horn generic disposition/outcome differs")
    expected_answers = _answer_pair(queries, primary, replay)
    if corner.answers != expected_answers:
        raise HornAdapterError("Horn generic answer labels differ")


def adapt_horn_semantic_rectangle(
    *,
    semantic_rectangle_id: str,
    presentation_id: str,
    worlds: WorldPair,
    commands: CommandPair,
    primary_executions: ExecutionMatrix,
    replay_executions: ExecutionMatrix,
    queries: QueryPair,
    world_sources: SourceMatrix,
    command_sources: SourceMatrix,
    query_prefixes: SourceMatrix,
) -> GenericSemanticRectangle:
    """Project one exact Horn 2x2 semantic board into the generic v2 schema."""

    rectangle_id = _require_identifier(
        semantic_rectangle_id,
        "semantic_rectangle_id",
        digest=True,
    )
    presentation = _require_identifier(
        presentation_id,
        "presentation_id",
        digest=False,
    )
    world_values = _require_exact_pair(worlds, "worlds")
    command_values = _require_exact_pair(commands, "commands")
    if any(type(world) is not HornWorld for world in world_values):
        raise HornAdapterError("worlds contain a non-Horn world")
    if any(type(command) is not HornCommand for command in command_values):
        raise HornAdapterError("commands contain a non-Horn command")
    typed_worlds: WorldPair = world_values  # type: ignore[assignment]
    typed_commands: CommandPair = command_values  # type: ignore[assignment]
    if (
        typed_worlds[0] == typed_worlds[1]
        or typed_commands[0] == typed_commands[1]
        or typed_worlds[0].theory_index != typed_worlds[1].theory_index
        or typed_worlds[0].evidence_id != typed_worlds[1].evidence_id
        or typed_worlds[0].policy is not typed_worlds[1].policy
        or typed_commands[0].depth != typed_commands[1].depth
    ):
        raise HornAdapterError("Horn rectangle factor contract differs")

    query_values = _require_exact_pair(queries, "queries")
    if (
        any(type(query) is not SemanticQuery for query in query_values)
        or any(query.ontology is not Ontology.HORN for query in query_values)
        or query_values[0] == query_values[1]
    ):
        raise HornAdapterError("Horn query semantics differ")
    typed_queries: QueryPair = query_values  # type: ignore[assignment]
    world_bytes = _require_source_matrix(world_sources, "WORLD sources")
    command_bytes = _require_source_matrix(command_sources, "COMMAND sources")
    query_bytes = _require_source_matrix(query_prefixes, "QUERY prefixes")
    primary_matrix = _require_execution_matrix(
        primary_executions,
        "primary executions",
    )
    replay_matrix = _require_execution_matrix(
        replay_executions,
        "replay executions",
    )

    static = _project_static_theory(typed_worlds[0].theory_index)
    generic_worlds = tuple(
        GenericWorld(
            sources=world_bytes[world_index],
            initial_packet=_initial_packet(static, world),
        )
        for world_index, world in enumerate(typed_worlds)
    )
    generic_commands = tuple(
        GenericCommand(
            sources=command_bytes[command_index],
            command_atoms=_command_atom_indices(command),
        )
        for command_index, command in enumerate(typed_commands)
    )
    generic_queries = tuple(
        GenericQuery(query_bytes[index]) for index in range(2)
    )

    corner_rows: list[list[GenericCorner]] = [[], []]
    label_columns: list[list[bool]] = [[], []]
    for world_index, world in enumerate(typed_worlds):
        for command_index, command in enumerate(typed_commands):
            primary = primary_matrix[world_index][command_index]
            replay = replay_matrix[world_index][command_index]
            _validate_execution(
                world,
                command,
                primary,
                replay,
                world_index=world_index,
                command_index=command_index,
            )
            disposition, outcome, _, _ = _disposition(primary.disposition)
            answers = _answer_pair(typed_queries, primary, replay)
            corner = GenericCorner(
                operation_traces=_operation_traces(primary),
                terminal_packet=_terminal_packet(static, primary),
                disposition=disposition,
                outcome=outcome,
                answers=answers,
            )
            _validate_generic_corner(
                corner,
                static=static,
                world=world,
                command=command,
                primary=primary,
                replay=replay,
                queries=typed_queries,
            )
            corner_rows[world_index].append(corner)
            for query_index, answer in enumerate(answers):
                if type(answer) is not bool:
                    raise HornAdapterError(
                        "Horn causal rectangle is not answerable"
                    )
                label_columns[query_index].append(answer)

    for query_index, labels in enumerate(label_columns):
        if tuple(labels) not in CHECKERBOARD_PATTERNS:
            raise HornAdapterError(
                f"Horn query {query_index} labels are not a strict checkerboard"
            )

    rectangle = GenericSemanticRectangle(
        semantic_rectangle_id=rectangle_id,
        presentation_id=presentation,
        worlds=generic_worlds,  # type: ignore[arg-type]
        commands=generic_commands,  # type: ignore[arg-type]
        queries=generic_queries,  # type: ignore[arg-type]
        corners=(
            (corner_rows[0][0], corner_rows[0][1]),
            (corner_rows[1][0], corner_rows[1][1]),
        ),
    )
    for world_index, world in enumerate(typed_worlds):
        if rectangle.worlds[world_index].initial_packet != _initial_packet(
            static,
            world,
        ):
            raise HornAdapterError("Horn generic initial packet differs")
    return rectangle


__all__ = [
    "CommandPair",
    "ExecutionMatrix",
    "HornAdapterError",
    "QueryPair",
    "SourceMatrix",
    "WorldPair",
    "adapt_horn_semantic_rectangle",
]
