"""Strict rewrite-to-generic adapter for ETTR isolated learnability v2.

This CPU-only module is the ontology boundary between the sealed rewrite
assessor and :mod:`ettr_il_v2_materialize`.  It does not render, parse, select,
tokenize, train, open checkpoints, or access datasets.  Callers supply the
already-admitted cell-local source bytes and both primary and independent
semantic executions.
"""

from __future__ import annotations

from typing import TypeAlias

from cross_ontology_rewrite_board import (
    CONSTRUCTORS,
    GroundTerm,
    reference_theory_state,
)
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
    Ontology,
    RewriteCommand,
    RewriteExecution,
    RewriteWorld,
    SemanticQuery,
    StepOutcome,
    TerminalDisposition,
    evaluate_query,
    execute_rewrite,
    replay_rewrite,
)


class RewriteAdapterError(ValueError):
    """A rewrite rectangle cannot be projected without semantic loss."""


ExecutionGrid: TypeAlias = tuple[
    tuple[RewriteExecution, RewriteExecution],
    tuple[RewriteExecution, RewriteExecution],
]
ByteGrid: TypeAlias = tuple[tuple[bytes, bytes], tuple[bytes, bytes]]

_RUNTIME_SLOTS = tuple(range(32, 36))
_COMMAND_SLOTS = tuple(range(48, 54))
_CURSOR_SLOT = 54
_OUTCOME_SLOT = 55
_FULL_SLOT_SUPPORT = (True,) * 64
_FULL_RELATION_SUPPORT = (True,) * (16 * 64 * 64)


def _exact_pair(value: object, name: str) -> tuple[object, object]:
    if type(value) is not tuple or len(value) != 2:
        raise RewriteAdapterError(f"{name} is not an exact pair")
    return value


def _ascii_identifier(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or not value.isascii()
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in value)
    ):
        raise RewriteAdapterError(f"{name} differs")
    return value


def _surface_pair(value: object, name: str) -> tuple[bytes, bytes]:
    pair = _exact_pair(value, name)
    result: list[bytes] = []
    for index, item in enumerate(pair):
        if type(item) is not bytes or not item:
            raise RewriteAdapterError(f"{name}[{index}] differs")
        try:
            item.decode("ascii")
        except UnicodeDecodeError as exc:
            raise RewriteAdapterError(f"{name}[{index}] is not ASCII") from exc
        result.append(item)
    if result[0] == result[1]:
        raise RewriteAdapterError(f"{name} variants are identical")
    return result[0], result[1]


def _surface_grid(value: object, name: str) -> ByteGrid:
    outer = _exact_pair(value, name)
    return (
        _surface_pair(outer[0], f"{name}[0]"),
        _surface_pair(outer[1], f"{name}[1]"),
    )


def _validate_term(term: object) -> GroundTerm:
    if type(term) is not GroundTerm:
        raise RewriteAdapterError("rewrite term type differs")
    if (
        type(term.constructor_index) is not int
        or not 0 <= term.constructor_index < len(CONSTRUCTORS)
        or type(term.type_index) is not int
    ):
        raise RewriteAdapterError("rewrite term constructor differs")
    constructor = CONSTRUCTORS[term.constructor_index]
    if term.type_index != constructor.result_type or type(term.children) is not tuple:
        raise RewriteAdapterError("rewrite term typing differs")
    if len(term.children) != len(constructor.argument_types):
        raise RewriteAdapterError("rewrite term arity differs")
    for child, required_type in zip(
        term.children,
        constructor.argument_types,
        strict=True,
    ):
        validated = _validate_term(child)
        if validated.type_index != required_type:
            raise RewriteAdapterError("rewrite child typing differs")
    return term


def _term_values(term: GroundTerm) -> tuple[ValueRef, ValueRef, ValueRef, ValueRef]:
    """Project one canonical term tree to four fixed preorder registers."""

    preorder: list[int] = []

    def visit(node: GroundTerm) -> None:
        validated = _validate_term(node)
        preorder.append(validated.constructor_index)
        for child in validated.children:
            visit(child)

    visit(term)
    if len(preorder) > len(_RUNTIME_SLOTS):
        raise RewriteAdapterError("rewrite term exceeds four runtime registers")
    values = [ValueRef.local_id(index) for index in preorder]
    values.extend(ValueRef.empty() for _ in range(4 - len(values)))
    return values[0], values[1], values[2], values[3]


def _static_projection(theory_index: int) -> tuple[
    tuple[GenericCell, ...],
    tuple[GenericEdge, ...],
    int | None,
]:
    state = reference_theory_state(theory_index)
    if state.capacity != 32 or state.type_count != 4:
        raise RewriteAdapterError("rewrite static theory geometry differs")
    relation_specs = {spec.index: spec for spec in state.relation_specs}
    if (
        len(relation_specs) != len(state.relation_specs)
        or any(
            not 0 <= spec.index <= 7
            or len(spec.argument_types) not in {1, 2}
            or any(not 0 <= item < 4 for item in spec.argument_types)
            for spec in state.relation_specs
        )
    ):
        raise RewriteAdapterError("rewrite static relation schema differs")
    cells = tuple(
        GenericCell(cell.slot, cell.type_index, ValueRef.static(cell.value))
        for cell in state.cells
    )
    if (
        not cells
        or tuple(cell.slot for cell in cells)
        != tuple(sorted(cell.slot for cell in cells))
        or len({cell.slot for cell in cells}) != len(cells)
        or any(not 0 <= cell.slot < 32 for cell in cells)
    ):
        raise RewriteAdapterError("rewrite static cells differ")
    active = {cell.slot: cell.type_index for cell in cells}
    edges: list[GenericEdge] = []
    for edge in state.edges:
        spec = relation_specs.get(edge.relation_index)
        if spec is None or len(edge.arguments) != len(spec.argument_types):
            raise RewriteAdapterError("rewrite static edge schema differs")
        for slot, required_type in zip(
            edge.arguments,
            spec.argument_types,
            strict=True,
        ):
            if active.get(slot) != required_type:
                raise RewriteAdapterError("rewrite static edge typing differs")
        if len(edge.arguments) == 1:
            source = target = edge.arguments[0]
        else:
            source, target = edge.arguments
        edges.append(GenericEdge(edge.relation_index, source, target))
    projected_edges = tuple(sorted(edges))
    if len(projected_edges) != len(set(projected_edges)):
        raise RewriteAdapterError("rewrite static edges are not unique")
    if state.root is not None and state.root not in active:
        raise RewriteAdapterError("rewrite static root differs")
    return cells, projected_edges, state.root


def _packet(
    static_cells: tuple[GenericCell, ...],
    static_edges: tuple[GenericEdge, ...],
    static_root: int | None,
    runtime: tuple[ValueRef, ValueRef, ValueRef, ValueRef],
    *,
    command: RewriteCommand | None = None,
    disposition: TerminalDisposition | None = None,
) -> GenericPacket:
    cells = static_cells + tuple(
        GenericCell(slot, 4, value)
        for slot, value in zip(_RUNTIME_SLOTS, runtime, strict=True)
    )
    committed = False
    halted = False
    if command is not None:
        if disposition is None:
            raise RewriteAdapterError("terminal disposition is absent")
        cells += tuple(
            GenericCell(
                slot,
                5,
                (
                    ValueRef.command_atom(command.operations[index])
                    if index < command.depth
                    else ValueRef.empty()
                ),
            )
            for index, slot in enumerate(_COMMAND_SLOTS)
        )
        cells += (
            GenericCell(_CURSOR_SLOT, 6, ValueRef.small_uint(command.depth)),
            GenericCell(
                _OUTCOME_SLOT,
                6,
                {
                    TerminalDisposition.ANSWER: ValueRef.execute(),
                    TerminalDisposition.ABSTAIN: ValueRef.abstain(),
                    TerminalDisposition.REJECT: ValueRef.reject(),
                }[disposition],
            ),
        )
        committed, halted = {
            TerminalDisposition.ANSWER: (True, False),
            TerminalDisposition.ABSTAIN: (False, True),
            TerminalDisposition.REJECT: (True, True),
        }[disposition]
    return GenericPacket(
        cells=cells,
        edges=static_edges,
        root=static_root,
        committed=committed,
        halted=halted,
        slot_support=_FULL_SLOT_SUPPORT,
        relation_support=_FULL_RELATION_SUPPORT,
    )


def _expected_wrap(before: GroundTerm, operation: int) -> GroundTerm:
    return GroundTerm(
        type_index=0,
        constructor_index=5,
        children=(
            before,
            GroundTerm(type_index=0, constructor_index=operation, children=()),
        ),
    )


def _trace_and_terminal_runtime(
    execution: RewriteExecution,
) -> tuple[
    tuple[GenericOperationTrace, ...],
    tuple[ValueRef, ValueRef, ValueRef, ValueRef],
]:
    command = execution.command
    current_term = execution.world.initial
    current_values = _term_values(current_term)
    traces: list[GenericOperationTrace] = []
    if (
        type(execution.snapshots) is not tuple
        or not execution.snapshots
        or execution.snapshots[0].index != 0
        or execution.snapshots[0].normal_forms != (current_term,)
        or type(execution.steps) is not tuple
        or len(execution.snapshots) != len(execution.steps) + 1
    ):
        raise RewriteAdapterError("rewrite execution snapshots differ")
    terminal_seen = False
    for position, operation in enumerate(command.operations, start=1):
        mutations: tuple[GenericMutation, ...] = ()
        if position <= len(execution.steps):
            step = execution.steps[position - 1]
            snapshot = execution.snapshots[position]
            if (
                step.index != position
                or step.operation != operation
                or step.before != current_term
                or step.wrapped != _expected_wrap(current_term, operation)
                or snapshot.index != position
                or snapshot.normal_forms != step.normal_forms
            ):
                raise RewriteAdapterError("rewrite operation record differs")
            if step.outcome is StepOutcome.APPLIED:
                if terminal_seen or len(step.normal_forms) != 1:
                    raise RewriteAdapterError("applied rewrite snapshot differs")
                successor = step.normal_forms[0]
                next_values = _term_values(successor)
                mutations = tuple(
                    GenericMutation(
                        opcode=Opcode.WRITE,
                        source=slot,
                        value=after,
                    )
                    for slot, before, after in zip(
                        _RUNTIME_SLOTS,
                        current_values,
                        next_values,
                        strict=True,
                    )
                    if before != after
                )
                if not mutations:
                    raise RewriteAdapterError("applied rewrite changes no register")
                current_term = successor
                current_values = next_values
            elif step.outcome is StepOutcome.AMBIGUOUS:
                if terminal_seen or len(step.normal_forms) <= 1:
                    raise RewriteAdapterError("ambiguous rewrite snapshot differs")
                terminal_seen = True
            elif step.outcome is StepOutcome.REJECTED:
                if terminal_seen or step.normal_forms:
                    raise RewriteAdapterError("rejected rewrite snapshot differs")
                terminal_seen = True
            else:
                raise RewriteAdapterError("rewrite step outcome differs")
        elif not terminal_seen:
            raise RewriteAdapterError("rewrite execution omits a disclosed operation")
        traces.append(GenericOperationTrace(mutations=mutations, cursor=position))
    if len(execution.steps) > command.depth:
        raise RewriteAdapterError("rewrite execution exceeds command depth")
    expected_disposition = (
        TerminalDisposition.ABSTAIN
        if execution.steps
        and execution.steps[-1].outcome is StepOutcome.AMBIGUOUS
        else (
            TerminalDisposition.REJECT
            if execution.steps
            and execution.steps[-1].outcome is StepOutcome.REJECTED
            else TerminalDisposition.ANSWER
        )
    )
    if execution.disposition is not expected_disposition:
        raise RewriteAdapterError("rewrite terminal disposition differs")
    if execution.disposition is TerminalDisposition.ANSWER and (
        len(execution.steps) != command.depth or terminal_seen
    ):
        raise RewriteAdapterError("answering rewrite did not execute every operation")
    return tuple(traces), current_values


def _disposition(value: TerminalDisposition) -> Disposition:
    return {
        TerminalDisposition.ANSWER: Disposition.ANSWER,
        TerminalDisposition.ABSTAIN: Disposition.ABSTAIN,
        TerminalDisposition.REJECT: Disposition.REJECT,
    }[value]


def _corner(
    execution: RewriteExecution,
    queries: tuple[SemanticQuery, SemanticQuery],
    static: tuple[tuple[GenericCell, ...], tuple[GenericEdge, ...], int | None],
) -> GenericCorner:
    traces, terminal_runtime = _trace_and_terminal_runtime(execution)
    answers: tuple[bool | None, bool | None]
    if execution.disposition is TerminalDisposition.ANSWER:
        answers = (
            evaluate_query(queries[0], execution),
            evaluate_query(queries[1], execution),
        )
    else:
        answers = (None, None)
    terminal = _packet(
        *static,
        terminal_runtime,
        command=execution.command,
        disposition=execution.disposition,
    )
    return GenericCorner(
        operation_traces=traces,
        terminal_packet=terminal,
        disposition=_disposition(execution.disposition),
        outcome={
            TerminalDisposition.ANSWER: ValueRef.execute(),
            TerminalDisposition.ABSTAIN: ValueRef.abstain(),
            TerminalDisposition.REJECT: ValueRef.reject(),
        }[execution.disposition],
        answers=answers,
    )


def _execution_grid(value: object, name: str) -> ExecutionGrid:
    outer = _exact_pair(value, name)
    rows = (
        _exact_pair(outer[0], f"{name}[0]"),
        _exact_pair(outer[1], f"{name}[1]"),
    )
    if any(type(item) is not RewriteExecution for row in rows for item in row):
        raise RewriteAdapterError(f"{name} contains a non-rewrite execution")
    return (
        (rows[0][0], rows[0][1]),  # type: ignore[return-value]
        (rows[1][0], rows[1][1]),  # type: ignore[return-value]
    )


def _answer_code(corner: GenericCorner, query_index: int) -> int:
    if corner.disposition is Disposition.ANSWER:
        answer = corner.answers[query_index]
        if type(answer) is not bool:
            raise RewriteAdapterError("answering corner lacks a Boolean label")
        return int(answer)
    if corner.answers[query_index] is not None:
        raise RewriteAdapterError("non-answering corner exposes a Boolean label")
    return 2 if corner.disposition is Disposition.ABSTAIN else 3


def _validate_all_edges(
    corners: tuple[
        tuple[GenericCorner, GenericCorner],
        tuple[GenericCorner, GenericCorner],
    ],
) -> None:
    for query_index in range(2):
        labels = tuple(
            _answer_code(corners[world][command], query_index)
            for world in range(2)
            for command in range(2)
        )
        if any(
            labels[left] == labels[right]
            for left, right in ((0, 1), (2, 3), (0, 2), (1, 3))
        ):
            raise RewriteAdapterError(
                f"query {query_index} labels lack all-edge contrast: {labels!r}"
            )


def adapt_rewrite_rectangle(
    *,
    semantic_rectangle_id: str,
    presentation_id: str,
    worlds: tuple[RewriteWorld, RewriteWorld],
    commands: tuple[RewriteCommand, RewriteCommand],
    primary_executions: ExecutionGrid,
    replay_executions: ExecutionGrid,
    queries: tuple[SemanticQuery, SemanticQuery],
    world_sources: ByteGrid,
    command_sources: ByteGrid,
    query_prefixes: ByteGrid,
) -> GenericSemanticRectangle:
    """Adapt one already-selected rewrite rectangle to generic ETTR inputs.

    ``world_sources[w][c]`` is the cell-local rendering of semantic world
    ``w`` under ``world-<c>``.  ``command_sources[c][w]`` is the rendering of
    semantic command ``c`` under ``command-<w>``.  This function validates but
    never creates or parses those source bytes.
    """

    semantic_id = _ascii_identifier(
        semantic_rectangle_id,
        "semantic_rectangle_id",
    )
    presentation = _ascii_identifier(presentation_id, "presentation_id")
    world_pair = _exact_pair(worlds, "worlds")
    command_pair = _exact_pair(commands, "commands")
    query_pair = _exact_pair(queries, "queries")
    if any(type(world) is not RewriteWorld for world in world_pair):
        raise RewriteAdapterError("worlds contain a non-rewrite world")
    if any(type(command) is not RewriteCommand for command in command_pair):
        raise RewriteAdapterError("commands contain a non-rewrite command")
    if any(
        type(query) is not SemanticQuery or query.ontology is not Ontology.REWRITE
        for query in query_pair
    ):
        raise RewriteAdapterError("queries contain a non-rewrite query")
    typed_worlds: tuple[RewriteWorld, RewriteWorld] = (  # type: ignore[assignment]
        world_pair[0],
        world_pair[1],
    )
    typed_commands: tuple[RewriteCommand, RewriteCommand] = (  # type: ignore[assignment]
        command_pair[0],
        command_pair[1],
    )
    typed_queries: tuple[SemanticQuery, SemanticQuery] = (  # type: ignore[assignment]
        query_pair[0],
        query_pair[1],
    )
    if (
        typed_worlds[0] == typed_worlds[1]
        or typed_commands[0] == typed_commands[1]
        or typed_queries[0] == typed_queries[1]
    ):
        raise RewriteAdapterError("rectangle semantic factors are not distinct")
    if (
        typed_worlds[0].theory_index != typed_worlds[1].theory_index
        or typed_worlds[0].evidence_id != typed_worlds[1].evidence_id
        or typed_worlds[0].policy is not typed_worlds[1].policy
    ):
        raise RewriteAdapterError("worlds do not share one theory/evidence/policy")
    if typed_commands[0].depth != typed_commands[1].depth:
        raise RewriteAdapterError("commands do not share one depth stratum")

    primary = _execution_grid(primary_executions, "primary_executions")
    replay = _execution_grid(replay_executions, "replay_executions")
    for world_index in range(2):
        for command_index in range(2):
            supplied_primary = primary[world_index][command_index]
            supplied_replay = replay[world_index][command_index]
            world = typed_worlds[world_index]
            command = typed_commands[command_index]
            if (
                supplied_primary.world != world
                or supplied_primary.command != command
                or supplied_replay.world != world
                or supplied_replay.command != command
            ):
                raise RewriteAdapterError(
                    f"corner {world_index}{command_index} binding differs"
                )
            recomputed_primary = execute_rewrite(world, command)
            recomputed_replay = replay_rewrite(world, command)
            if (
                supplied_primary != recomputed_primary
                or supplied_replay != recomputed_replay
                or supplied_primary != supplied_replay
            ):
                raise RewriteAdapterError(
                    f"corner {world_index}{command_index} primary/replay differs"
                )

    world_bytes = _surface_grid(world_sources, "world_sources")
    command_bytes = _surface_grid(command_sources, "command_sources")
    prefixes = _surface_grid(query_prefixes, "query_prefixes")
    if any(
        world_bytes[0][nuisance] == world_bytes[1][nuisance]
        for nuisance in range(2)
    ):
        raise RewriteAdapterError("WORLD factors have identical cell-local bytes")
    if any(
        command_bytes[0][nuisance] == command_bytes[1][nuisance]
        for nuisance in range(2)
    ):
        raise RewriteAdapterError("COMMAND factors have identical cell-local bytes")

    static = _static_projection(typed_worlds[0].theory_index)
    generic_worlds = tuple(
        GenericWorld(
            sources=world_bytes[index],
            initial_packet=_packet(
                *static,
                _term_values(world.initial),
            ),
        )
        for index, world in enumerate(typed_worlds)
    )
    if generic_worlds[0].initial_packet == generic_worlds[1].initial_packet:
        raise RewriteAdapterError("WORLD initial packets are identical")
    generic_commands = tuple(
        GenericCommand(
            sources=command_bytes[index],
            command_atoms=command.operations,
        )
        for index, command in enumerate(typed_commands)
    )
    corners = (
        (
            _corner(primary[0][0], typed_queries, static),
            _corner(primary[0][1], typed_queries, static),
        ),
        (
            _corner(primary[1][0], typed_queries, static),
            _corner(primary[1][1], typed_queries, static),
        ),
    )
    if any(
        corners[left_w][left_c].terminal_packet
        == corners[right_w][right_c].terminal_packet
        for (left_w, left_c), (right_w, right_c) in (
            ((0, 0), (0, 1)),
            ((1, 0), (1, 1)),
            ((0, 0), (1, 0)),
            ((0, 1), (1, 1)),
        )
    ):
        raise RewriteAdapterError("terminal packets lack all-edge contrast")
    _validate_all_edges(corners)
    return GenericSemanticRectangle(
        semantic_rectangle_id=semantic_id,
        presentation_id=presentation,
        worlds=(generic_worlds[0], generic_worlds[1]),
        commands=(generic_commands[0], generic_commands[1]),
        queries=(
            GenericQuery(prefixes[0]),
            GenericQuery(prefixes[1]),
        ),
        corners=corners,
    )


__all__ = [
    "ByteGrid",
    "ExecutionGrid",
    "RewriteAdapterError",
    "adapt_rewrite_rectangle",
]
