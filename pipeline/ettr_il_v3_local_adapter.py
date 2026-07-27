"""Project bounded local-rewrite v3 rectangles into generic ETTR packets.

The v3 local-rewrite family is intentionally distinct from the legacy term
rewriter, so it cannot use the v2 rewrite adapter.  This module gives its two
opaque laws a fixed typed static packet, maps six registers into the sixteen
runtime slots, maps each fixed-width operation into one command atom, and
emits exact WRITE traces independently checked by the generic materializer.
"""

from __future__ import annotations

from typing import TypeAlias

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
from ettr_il_v3_rectangles import SemanticRectangleBundle
from ettr_il_v3_rewrite import (
    Direction,
    LOCAL_LAWS,
    THEORIES,
    RewriteCommand,
    RewriteExecution,
    RewriteWorld,
    StructuralQuery,
    TerminalDisposition,
    evaluate_query_primary,
    evaluate_query_replay,
    execute_primary,
    execute_replay,
)


ByteGrid: TypeAlias = tuple[tuple[bytes, bytes], tuple[bytes, bytes]]
_RUNTIME_SLOTS = tuple(range(32, 38))
_COMMAND_SLOTS = tuple(range(48, 54))
_CURSOR_SLOT = 54
_OUTCOME_SLOT = 55


class LocalAdapterError(ValueError):
    """A local-rewrite rectangle cannot be projected without semantic loss."""


def _pair(value: object, name: str) -> tuple[object, object]:
    if type(value) is not tuple or len(value) != 2:
        raise LocalAdapterError(f"{name} is not an exact pair")
    return value


def _byte_grid(value: object, name: str) -> ByteGrid:
    outer = _pair(value, name)
    result: list[tuple[bytes, bytes]] = []
    for row_index, row in enumerate(outer):
        items = _pair(row, f"{name}[{row_index}]")
        if any(type(item) is not bytes or not item for item in items):
            raise LocalAdapterError(f"{name}[{row_index}] bytes differ")
        if items[0] == items[1]:
            raise LocalAdapterError(f"{name}[{row_index}] variants are identical")
        try:
            items[0].decode("ascii")
            items[1].decode("ascii")
        except UnicodeDecodeError as exc:
            raise LocalAdapterError(f"{name} is not ASCII") from exc
        result.append((items[0], items[1]))  # type: ignore[list-item]
    return result[0], result[1]


def _static_projection(
    theory_index: int,
) -> tuple[tuple[GenericCell, ...], tuple[GenericEdge, ...], int]:
    try:
        theory = THEORIES[theory_index]
    except (IndexError, TypeError) as exc:
        raise LocalAdapterError("rewrite theory index differs") from exc
    cells: list[GenericCell] = []
    edges: list[GenericEdge] = []
    for local_slot, law_index in enumerate(theory.law_indices):
        law = LOCAL_LAWS[law_index]
        base = 5 * local_slot
        values = (
            100 + law.index,
            200 + law.forward_source[0],
            300 + law.forward_source[1],
            400 + law.forward_target[0],
            500 + law.forward_target[1],
        )
        cells.extend(
            GenericCell(base + offset, 0, ValueRef.static(value))
            for offset, value in enumerate(values)
        )
        edges.extend(
            GenericEdge(relation, base, base + relation + 1)
            for relation in range(4)
        )
    return tuple(cells), tuple(edges), 0


def _runtime_cells(registers: tuple[int, ...]) -> tuple[GenericCell, ...]:
    if len(registers) != len(_RUNTIME_SLOTS):
        raise LocalAdapterError("rewrite register width differs")
    return tuple(
        GenericCell(slot, 4, ValueRef.local_id(value))
        for slot, value in zip(_RUNTIME_SLOTS, registers, strict=True)
    )


def _command_atom(command: RewriteCommand, position: int) -> int:
    operation = command.operations[position]
    direction = 0 if operation.direction is Direction.FORWARD else 1
    return (operation.law_slot * 8 + operation.site) * 2 + direction


def _initial_packet(world: RewriteWorld) -> GenericPacket:
    static_cells, static_edges, root = _static_projection(world.theory_index)
    return GenericPacket(
        cells=static_cells + _runtime_cells(world.registers),
        edges=static_edges,
        root=root,
    )


def _terminal_packet(execution: RewriteExecution) -> GenericPacket:
    static_cells, static_edges, root = _static_projection(
        execution.world.theory_index
    )
    command_cells = tuple(
        GenericCell(
            slot,
            5,
            (
                ValueRef.command_atom(_command_atom(execution.command, index))
                if index < execution.command.depth
                else ValueRef.empty()
            ),
        )
        for index, slot in enumerate(_COMMAND_SLOTS)
    )
    return GenericPacket(
        cells=(
            static_cells
            + _runtime_cells(execution.terminal)
            + command_cells
            + (
                GenericCell(
                    _CURSOR_SLOT,
                    6,
                    ValueRef.small_uint(execution.command.depth),
                ),
                GenericCell(_OUTCOME_SLOT, 6, ValueRef.execute()),
            )
        ),
        edges=static_edges,
        root=root,
        committed=True,
        halted=False,
    )


def _operation_traces(
    execution: RewriteExecution,
) -> tuple[GenericOperationTrace, ...]:
    if (
        execution.disposition is not TerminalDisposition.ANSWER
        or len(execution.steps) != execution.command.depth
        or len(execution.snapshots) != execution.command.depth + 1
    ):
        raise LocalAdapterError("rewrite execution depth or disposition differs")
    traces: list[GenericOperationTrace] = []
    for index, step in enumerate(execution.steps, start=1):
        if (
            step.index != index
            or step.operation != execution.command.operations[index - 1]
            or step.before != execution.snapshots[index - 1]
            or step.after != execution.snapshots[index]
        ):
            raise LocalAdapterError("rewrite step binding differs")
        traces.append(
            GenericOperationTrace(
                mutations=tuple(
                    GenericMutation(
                        Opcode.WRITE,
                        source=_RUNTIME_SLOTS[slot],
                        value=ValueRef.local_id(after),
                    )
                    for slot, (before, after) in enumerate(
                        zip(step.before, step.after, strict=True)
                    )
                    if before != after
                ),
                cursor=index,
            )
        )
    return tuple(traces)


def _answers(
    execution: RewriteExecution,
    replay: RewriteExecution,
    queries: tuple[StructuralQuery, StructuralQuery],
) -> tuple[bool, bool]:
    primary_answers = tuple(
        evaluate_query_primary(execution, query)
        for query in queries
    )
    replay_answers = tuple(
        evaluate_query_replay(replay, query)
        for query in queries
    )
    if primary_answers != replay_answers:
        raise LocalAdapterError("rewrite primary/replay query answers differ")
    return primary_answers  # type: ignore[return-value]


def adapt_local_rewrite_rectangle(
    rectangle: SemanticRectangleBundle,
    *,
    presentation_id: str,
    world_sources: ByteGrid,
    command_sources: ByteGrid,
    query_prefixes: ByteGrid,
    require_query_checkerboard: bool = True,
) -> GenericSemanticRectangle:
    """Project one exact local-rewrite rectangle into generic ETTR inputs."""

    if (
        not isinstance(rectangle, SemanticRectangleBundle)
        or rectangle.family != "local_rewrite"
        or type(presentation_id) is not str
        or not presentation_id
        or not presentation_id.isascii()
        or not isinstance(require_query_checkerboard, bool)
    ):
        raise LocalAdapterError("local rectangle request differs")
    worlds = _pair(rectangle.worlds, "worlds")
    commands = _pair(rectangle.commands, "commands")
    queries = _pair(rectangle.queries, "queries")
    if (
        any(type(world) is not RewriteWorld for world in worlds)
        or any(type(command) is not RewriteCommand for command in commands)
        or any(type(query) is not StructuralQuery for query in queries)
    ):
        raise LocalAdapterError("local rectangle semantic types differ")
    typed_worlds: tuple[RewriteWorld, RewriteWorld] = worlds  # type: ignore[assignment]
    typed_commands: tuple[RewriteCommand, RewriteCommand] = commands  # type: ignore[assignment]
    typed_queries: tuple[StructuralQuery, StructuralQuery] = queries  # type: ignore[assignment]
    if (
        typed_worlds[0] == typed_worlds[1]
        or typed_commands[0] == typed_commands[1]
        or typed_queries[0] == typed_queries[1]
        or typed_worlds[0].theory_index != typed_worlds[1].theory_index
        or typed_commands[0].depth != typed_commands[1].depth
    ):
        raise LocalAdapterError("local rectangle factor geometry differs")

    primary = rectangle.primary
    replay = rectangle.replay
    corners: list[list[GenericCorner]] = [[], []]
    labels: list[list[tuple[bool, bool]]] = [[], []]
    for world_index in range(2):
        for command_index in range(2):
            supplied_primary = primary[world_index][command_index]
            supplied_replay = replay[world_index][command_index]
            if (
                type(supplied_primary) is not RewriteExecution
                or type(supplied_replay) is not RewriteExecution
            ):
                raise LocalAdapterError("local execution type differs")
            expected_primary = execute_primary(
                typed_worlds[world_index],
                typed_commands[command_index],
            )
            expected_replay = execute_replay(
                typed_worlds[world_index],
                typed_commands[command_index],
            )
            if (
                supplied_primary != expected_primary
                or supplied_replay != expected_replay
                or supplied_primary != supplied_replay
                or supplied_primary.disposition is not TerminalDisposition.ANSWER
            ):
                raise LocalAdapterError("local primary/replay execution differs")
            answer_pair = _answers(
                supplied_primary,
                supplied_replay,
                typed_queries,
            )
            labels[world_index].append(answer_pair)
            corners[world_index].append(
                GenericCorner(
                    operation_traces=_operation_traces(supplied_primary),
                    terminal_packet=_terminal_packet(supplied_primary),
                    disposition=Disposition.ANSWER,
                    outcome=ValueRef.execute(),
                    answers=answer_pair,
                )
            )
    if require_query_checkerboard:
        for query_index in range(2):
            values = (
                labels[0][0][query_index],
                labels[0][1][query_index],
                labels[1][0][query_index],
                labels[1][1][query_index],
            )
            if any(
                values[left] == values[right]
                for left, right in ((0, 1), (2, 3), (0, 2), (1, 3))
            ):
                raise LocalAdapterError("local query lacks checkerboard contrast")

    world_bytes = _byte_grid(world_sources, "world_sources")
    command_bytes = _byte_grid(command_sources, "command_sources")
    query_bytes = _byte_grid(query_prefixes, "query_prefixes")
    if any(
        world_bytes[0][index] == world_bytes[1][index]
        for index in range(2)
    ):
        raise LocalAdapterError("local WORLD factors have identical bytes")
    if any(
        command_bytes[0][index] == command_bytes[1][index]
        for index in range(2)
    ):
        raise LocalAdapterError("local COMMAND factors have identical bytes")
    generic_worlds = tuple(
        GenericWorld(world_bytes[index], _initial_packet(world))
        for index, world in enumerate(typed_worlds)
    )
    generic_commands = tuple(
        GenericCommand(
            command_bytes[index],
            tuple(
                _command_atom(command, position)
                for position in range(command.depth)
            ),
        )
        for index, command in enumerate(typed_commands)
    )
    return GenericSemanticRectangle(
        semantic_rectangle_id=rectangle.semantic_rectangle_id,
        presentation_id=presentation_id,
        worlds=(generic_worlds[0], generic_worlds[1]),
        commands=(generic_commands[0], generic_commands[1]),
        queries=(
            GenericQuery(query_bytes[0]),
            GenericQuery(query_bytes[1]),
        ),
        corners=(
            (corners[0][0], corners[0][1]),
            (corners[1][0], corners[1][1]),
        ),
    )


__all__ = [
    "ByteGrid",
    "LocalAdapterError",
    "adapt_local_rewrite_rectangle",
]
