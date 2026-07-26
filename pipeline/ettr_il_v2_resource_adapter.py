"""Strict CPU-only resource adapter for ETTR IL v2.

This module performs the resource-specific projection between the sealed
semantic oracle and the ontology-neutral materializer input.  It does not
render surfaces, select cases, tokenize data, access models, or run jobs.
All candidate-visible bytes and identifiers are supplied by the caller.
"""

from __future__ import annotations

from typing import TypeAlias
import re

from cross_ontology_resource_board import (
    Marking,
    ProcessStatus,
    reference_theory_state,
)
from cross_ontology_schema import ReactorState
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
    ValueKind,
    ValueRef,
)
from ettr_il_v2_semantics import (
    Ontology,
    ResourceCommand,
    ResourceExecution,
    ResourceWorld,
    SemanticAdmissionError,
    SemanticQuery,
    SemanticRectangle,
    StepOutcome,
    TerminalDisposition,
    checkerboard_labels,
    execute_resource,
    replay_resource,
)


_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_RESOURCE_SLOTS = (32, 33, 34, 35)
_COMMAND_SLOTS = (48, 49, 50, 51, 52, 53)

CellBytes: TypeAlias = tuple[
    tuple[bytes, bytes],
    tuple[bytes, bytes],
]
ResourceExecutions: TypeAlias = tuple[
    tuple[ResourceExecution, ResourceExecution],
    tuple[ResourceExecution, ResourceExecution],
]


class ResourceAdapterError(ValueError):
    """A resource oracle rectangle cannot be projected losslessly."""


def _exact_pair(value: object, name: str) -> tuple[object, object]:
    if type(value) is not tuple or len(value) != 2:
        raise ResourceAdapterError(f"{name} must be an exact pair")
    return value


def _surface_bytes(value: object, name: str) -> bytes:
    if type(value) is not bytes or not value:
        raise ResourceAdapterError(f"{name} must be nonempty exact bytes")
    try:
        value.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ResourceAdapterError(f"{name} must be strict ASCII") from exc
    return value


def _cell_bytes(value: object, name: str) -> CellBytes:
    rows = _exact_pair(value, name)
    result: list[tuple[bytes, bytes]] = []
    for row_index, row in enumerate(rows):
        cells = _exact_pair(row, f"{name}[{row_index}]")
        result.append(
            (
                _surface_bytes(cells[0], f"{name}[{row_index}][0]"),
                _surface_bytes(cells[1], f"{name}[{row_index}][1]"),
            )
        )
    return (result[0], result[1])


def _validate_surface_contrast(
    values: CellBytes,
    name: str,
) -> None:
    for row in range(2):
        if values[row][0] == values[row][1]:
            raise ResourceAdapterError(
                f"{name} row {row} lacks cell-local contrast"
            )
    for column in range(2):
        if values[0][column] == values[1][column]:
            raise ResourceAdapterError(
                f"{name} column {column} lacks cell-local contrast"
            )


def _identifier(value: object, name: str, *, digest: bool) -> str:
    if type(value) is not str or not value:
        raise ResourceAdapterError(f"{name} differs")
    if not value.isascii() or any(
        ord(character) < 0x21 or ord(character) > 0x7E
        for character in value
    ):
        raise ResourceAdapterError(f"{name} is not printable strict ASCII")
    if digest and _HEX_64.fullmatch(value) is None:
        raise ResourceAdapterError(f"{name} is not lowercase SHA-256")
    return value


def _static_projection(theory_index: int) -> tuple[
    tuple[GenericCell, ...],
    tuple[GenericEdge, ...],
    int | None,
]:
    state = reference_theory_state(theory_index)
    if type(state) is not ReactorState:
        raise ResourceAdapterError("reference theory state type differs")
    if state.capacity != 32 or state.type_count != 4:
        raise ResourceAdapterError("reference theory geometry differs")

    cells: list[GenericCell] = []
    active_slots: set[int] = set()
    for cell in sorted(state.cells, key=lambda item: item.slot):
        if (
            not 0 <= cell.slot < 32
            or not 0 <= cell.type_index < 4
            or cell.slot in active_slots
        ):
            raise ResourceAdapterError("reference theory cell differs")
        active_slots.add(cell.slot)
        cells.append(
            GenericCell(
                slot=cell.slot,
                type_index=cell.type_index,
                value=ValueRef.static(cell.value),
            )
        )
    if not cells:
        raise ResourceAdapterError("reference theory has no static cells")

    edges: list[GenericEdge] = []
    for edge in sorted(
        state.edges,
        key=lambda item: (item.relation_index, item.arguments),
    ):
        if (
            not 0 <= edge.relation_index < 8
            or any(slot not in active_slots for slot in edge.arguments)
        ):
            raise ResourceAdapterError("reference theory edge differs")
        if len(edge.arguments) == 1:
            source = target = edge.arguments[0]
        elif len(edge.arguments) == 2:
            source, target = edge.arguments
        else:
            raise ResourceAdapterError(
                "reference theory relation is not unary or binary"
            )
        edges.append(GenericEdge(edge.relation_index, source, target))
    if len(edges) != len(set(edges)):
        raise ResourceAdapterError("reference theory projection repeats an edge")
    if state.root is not None and state.root not in active_slots:
        raise ResourceAdapterError("reference theory root differs")
    return tuple(cells), tuple(edges), state.root


def _runtime_cells(marking: Marking) -> tuple[GenericCell, ...]:
    if type(marking) is not Marking or len(marking.multiplicities) != 4:
        raise ResourceAdapterError("resource marking differs")
    return tuple(
        GenericCell(
            slot=slot,
            type_index=4,
            value=ValueRef.small_uint(multiplicity),
        )
        for slot, multiplicity in zip(
            _RESOURCE_SLOTS,
            marking.multiplicities,
            strict=True,
        )
    )


def _process_status_value(status: ProcessStatus) -> ValueRef:
    if status is ProcessStatus.HALT:
        return ValueRef(ValueKind.PROCESS_HALT)
    if status is ProcessStatus.DEADLOCK:
        return ValueRef(ValueKind.PROCESS_DEADLOCK)
    raise ResourceAdapterError("resource process status differs")


def _initial_packet(world: ResourceWorld) -> GenericPacket:
    static_cells, static_edges, root = _static_projection(world.theory_index)
    return GenericPacket(
        cells=static_cells + _runtime_cells(world.initial),
        edges=static_edges,
        root=root,
        committed=False,
        halted=False,
    )


def _terminal_packet(execution: ResourceExecution) -> GenericPacket:
    static_cells, static_edges, root = _static_projection(
        execution.world.theory_index
    )
    command_cells = tuple(
        GenericCell(
            slot=slot,
            type_index=5,
            value=(
                ValueRef.command_atom(execution.command.operations[index])
                if index < execution.command.depth
                else ValueRef.empty()
            ),
        )
        for index, slot in enumerate(_COMMAND_SLOTS)
    )
    status_value = _process_status_value(execution.status)
    return GenericPacket(
        cells=(
            static_cells
            + _runtime_cells(execution.terminal)
            + command_cells
            + (
                GenericCell(
                    slot=54,
                    type_index=6,
                    value=ValueRef.small_uint(execution.cursor),
                ),
                GenericCell(slot=55, type_index=6, value=status_value),
            )
        ),
        edges=static_edges,
        root=root,
        committed=True,
        halted=False,
    )


def _changed_place_mutations(
    before: Marking,
    after: Marking,
) -> tuple[GenericMutation, ...]:
    if type(before) is not Marking or type(after) is not Marking:
        raise ResourceAdapterError("resource operation marking differs")
    return tuple(
        GenericMutation(
            opcode=Opcode.WRITE,
            source=32 + place,
            value=ValueRef.small_uint(successor),
        )
        for place, (previous, successor) in enumerate(
            zip(
                before.multiplicities,
                after.multiplicities,
                strict=True,
            )
        )
        if previous != successor
    )


def _operation_traces(
    execution: ResourceExecution,
) -> tuple[GenericOperationTrace, ...]:
    semantic_steps = execution.steps
    if len(semantic_steps) > execution.command.depth:
        raise ResourceAdapterError("resource execution has excess steps")
    if semantic_steps and tuple(
        step.index for step in semantic_steps
    ) != tuple(range(1, len(semantic_steps) + 1)):
        raise ResourceAdapterError("resource step indices differ")
    if len(semantic_steps) < execution.command.depth:
        if (
            execution.status is not ProcessStatus.DEADLOCK
            or not semantic_steps
            or semantic_steps[-1].outcome is not StepOutcome.DEADLOCK
        ):
            raise ResourceAdapterError(
                "only atomic deadlock may leave a disclosed command suffix"
            )

    traces: list[GenericOperationTrace] = []
    for position in range(execution.command.depth):
        if position < len(semantic_steps):
            step = semantic_steps[position]
            if step.operation != execution.command.operations[position]:
                raise ResourceAdapterError(
                    "resource step and command operation differ"
                )
            mutations = _changed_place_mutations(step.before, step.after)
            cursor = step.cursor_after
        else:
            # The command remains candidate-visible after atomic deadlock.
            # Section 8.3 requires every disclosed atom and the frozen oracle
            # cursor, but no further ontology mutation.
            mutations = ()
            cursor = execution.cursor
        if not 0 <= cursor <= 15:
            raise ResourceAdapterError("resource cursor leaves SMALL_UINT")
        traces.append(
            GenericOperationTrace(
                mutations=mutations,
                cursor=cursor,
            )
        )
    return tuple(traces)


def _validated_executions(
    worlds: tuple[ResourceWorld, ResourceWorld],
    commands: tuple[ResourceCommand, ResourceCommand],
    supplied: ResourceExecutions,
) -> tuple[
    tuple[ResourceExecution, ResourceExecution],
    tuple[ResourceExecution, ResourceExecution],
]:
    validated: list[tuple[ResourceExecution, ResourceExecution]] = []
    for world_index, world in enumerate(worlds):
        row: list[ResourceExecution] = []
        for command_index, command in enumerate(commands):
            execution = supplied[world_index][command_index]
            if type(execution) is not ResourceExecution:
                raise ResourceAdapterError("resource execution type differs")
            try:
                primary = execute_resource(world, command)
                replay = replay_resource(world, command)
            except (ValueError, SemanticAdmissionError) as exc:
                raise ResourceAdapterError(
                    f"resource corner {world_index}{command_index} "
                    "fails primary/replay admission"
                ) from exc
            if primary != replay:
                raise ResourceAdapterError(
                    f"resource corner {world_index}{command_index} "
                    "primary/replay differs"
                )
            if execution != primary:
                raise ResourceAdapterError(
                    f"resource corner {world_index}{command_index} "
                    "supplied execution differs"
                )
            if (
                execution.disposition is not TerminalDisposition.ANSWER
                or execution.status
                not in {ProcessStatus.HALT, ProcessStatus.DEADLOCK}
            ):
                raise ResourceAdapterError(
                    f"resource corner {world_index}{command_index} "
                    "terminal contract differs"
                )
            row.append(execution)
        validated.append((row[0], row[1]))
    return (validated[0], validated[1])


def adapt_resource_rectangle(
    *,
    semantic_rectangle_id: str,
    presentation_id: str,
    worlds: tuple[ResourceWorld, ResourceWorld],
    commands: tuple[ResourceCommand, ResourceCommand],
    executions: ResourceExecutions,
    queries: tuple[SemanticQuery, SemanticQuery],
    cell_world_sources: CellBytes,
    cell_command_sources: CellBytes,
    query_prefixes: tuple[tuple[bytes, bytes], tuple[bytes, bytes]],
) -> GenericSemanticRectangle:
    """Project one already-selected resource rectangle without rendering it.

    Cell-local source matrices use WORLD-major, COMMAND-minor indexing:
    ``cell_*_sources[world][command]``.  Query prefixes use
    ``query_prefixes[query][paraphrase]``.
    """

    rectangle_id = _identifier(
        semantic_rectangle_id,
        "semantic_rectangle_id",
        digest=True,
    )
    presentation = _identifier(
        presentation_id,
        "presentation_id",
        digest=False,
    )
    world_values = _exact_pair(worlds, "worlds")
    command_values = _exact_pair(commands, "commands")
    query_values = _exact_pair(queries, "queries")
    execution_rows = _exact_pair(executions, "executions")
    supplied_executions: list[
        tuple[ResourceExecution, ResourceExecution]
    ] = []
    for row_index, row in enumerate(execution_rows):
        values = _exact_pair(row, f"executions[{row_index}]")
        supplied_executions.append((values[0], values[1]))  # type: ignore[list-item]

    typed_worlds: list[ResourceWorld] = []
    for world in world_values:
        if type(world) is not ResourceWorld:
            raise ResourceAdapterError("resource world type differs")
        typed_worlds.append(world)
    typed_commands: list[ResourceCommand] = []
    for command in command_values:
        if type(command) is not ResourceCommand:
            raise ResourceAdapterError("resource command type differs")
        typed_commands.append(command)
    typed_queries: list[SemanticQuery] = []
    for query in query_values:
        if type(query) is not SemanticQuery or query.ontology is not Ontology.RESOURCE:
            raise ResourceAdapterError("resource query type differs")
        typed_queries.append(query)
    if typed_queries[0] == typed_queries[1]:
        raise ResourceAdapterError("resource query semantics are identical")

    world_source_matrix = _cell_bytes(
        cell_world_sources,
        "cell_world_sources",
    )
    command_source_matrix = _cell_bytes(
        cell_command_sources,
        "cell_command_sources",
    )
    query_source_matrix = _cell_bytes(query_prefixes, "query_prefixes")
    _validate_surface_contrast(world_source_matrix, "WORLD sources")
    _validate_surface_contrast(command_source_matrix, "COMMAND sources")
    for query_index, prefixes in enumerate(query_source_matrix):
        if prefixes[0] == prefixes[1]:
            raise ResourceAdapterError(
                f"query {query_index} paraphrases are identical"
            )

    typed_world_pair = (typed_worlds[0], typed_worlds[1])
    typed_command_pair = (typed_commands[0], typed_commands[1])
    validated = _validated_executions(
        typed_world_pair,
        typed_command_pair,
        (supplied_executions[0], supplied_executions[1]),
    )
    flat = (
        validated[0][0],
        validated[0][1],
        validated[1][0],
        validated[1][1],
    )
    try:
        semantic_rectangle = SemanticRectangle(flat)
        label_rows = tuple(
            checkerboard_labels(query, semantic_rectangle)
            for query in typed_queries
        )
    except (ValueError, SemanticAdmissionError) as exc:
        raise ResourceAdapterError(
            "resource query labels do not form strict checkerboards"
        ) from exc

    generic_worlds = tuple(
        GenericWorld(
            sources=world_source_matrix[world_index],
            initial_packet=_initial_packet(world),
        )
        for world_index, world in enumerate(typed_world_pair)
    )
    generic_commands = tuple(
        GenericCommand(
            sources=(
                command_source_matrix[0][command_index],
                command_source_matrix[1][command_index],
            ),
            command_atoms=command.operations,
        )
        for command_index, command in enumerate(typed_command_pair)
    )
    generic_queries = tuple(
        GenericQuery(prefixes=query_source_matrix[index])
        for index in range(2)
    )
    generic_corners: list[list[GenericCorner]] = [[], []]
    for world_index in range(2):
        for command_index in range(2):
            execution = validated[world_index][command_index]
            outcome = _process_status_value(execution.status)
            generic_corners[world_index].append(
                GenericCorner(
                    operation_traces=_operation_traces(execution),
                    terminal_packet=_terminal_packet(execution),
                    disposition=Disposition.ANSWER,
                    outcome=outcome,
                    answers=(
                        label_rows[0][2 * world_index + command_index],
                        label_rows[1][2 * world_index + command_index],
                    ),
                )
            )

    return GenericSemanticRectangle(
        semantic_rectangle_id=rectangle_id,
        presentation_id=presentation,
        worlds=(generic_worlds[0], generic_worlds[1]),
        commands=(generic_commands[0], generic_commands[1]),
        queries=(generic_queries[0], generic_queries[1]),
        corners=(
            (generic_corners[0][0], generic_corners[0][1]),
            (generic_corners[1][0], generic_corners[1][1]),
        ),
    )


__all__ = [
    "CellBytes",
    "ResourceAdapterError",
    "ResourceExecutions",
    "adapt_resource_rectangle",
]
