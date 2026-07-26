from __future__ import annotations

from dataclasses import replace

import pytest

from cross_ontology_rewrite_board import GroundTerm, reference_theory_state
from ettr_il_v2_materialize import (
    Disposition,
    MaterializationRequest,
    Opcode,
    ValueKind,
    materialize_ettr_il_v2,
)
import ettr_il_v2_rewrite_adapter as adapter
from ettr_il_v2_rewrite_adapter import (
    RewriteAdapterError,
    adapt_rewrite_rectangle,
)
from ettr_il_v2_semantics import (
    QueryOp,
    RewriteCommand,
    RewritePolicy,
    RewriteWorld,
    SemanticQuery,
    TerminalDisposition,
    execute_rewrite,
    replay_rewrite,
)


EVIDENCE_ID = "0" * 64
WORLD_SOURCES = (
    (b"WORLD-W0-world-0", b"WORLD-W0-world-1"),
    (b"WORLD-W1-world-0", b"WORLD-W1-world-1"),
)
COMMAND_SOURCES = (
    (b"COMMAND-C0-command-0", b"COMMAND-C0-command-1"),
    (b"COMMAND-C1-command-0", b"COMMAND-C1-command-1"),
)
QUERY_PREFIXES = (
    (b"Does q0 hold? ", b"q0 verdict: "),
    (b"Does q1 hold? ", b"q1 verdict: "),
)


class _Encoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _ByteTokenizer:
    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> _Encoding:
        assert add_special_tokens is False
        return _Encoding([ord(character) + 1 for character in text])


def _inputs() -> dict[str, object]:
    worlds = (
        RewriteWorld(
            EVIDENCE_ID,
            1,
            GroundTerm(0, 0, ()),
            RewritePolicy.CONTEXTUAL,
        ),
        RewriteWorld(
            EVIDENCE_ID,
            1,
            GroundTerm(0, 1, ()),
            RewritePolicy.CONTEXTUAL,
        ),
    )
    commands = (
        RewriteCommand(1, (0,)),
        RewriteCommand(1, (1,)),
    )
    primary = tuple(
        tuple(execute_rewrite(world, command) for command in commands)
        for world in worlds
    )
    replay = tuple(
        tuple(replay_rewrite(world, command) for command in commands)
        for world in worlds
    )
    return {
        "semantic_rectangle_id": "rewrite-rectangle-0",
        "presentation_id": "rewrite-presentation-0",
        "worlds": worlds,
        "commands": commands,
        "primary_executions": primary,
        "replay_executions": replay,
        "queries": (
            SemanticQuery(QueryOp.REWRITE_ROOT_IS, (4,)),
            SemanticQuery(QueryOp.REWRITE_ROOT_IS, (5,)),
        ),
        "world_sources": WORLD_SOURCES,
        "command_sources": COMMAND_SOURCES,
        "query_prefixes": QUERY_PREFIXES,
    }


def test_projects_static_terms_traces_terminals_and_answers_exactly() -> None:
    rectangle = adapt_rewrite_rectangle(**_inputs())
    reference = reference_theory_state(1)

    assert rectangle.semantic_rectangle_id == "rewrite-rectangle-0"
    assert rectangle.presentation_id == "rewrite-presentation-0"
    assert rectangle.worlds[0].sources == WORLD_SOURCES[0]
    assert rectangle.commands[1].sources == COMMAND_SOURCES[1]
    assert rectangle.queries[0].prefixes == QUERY_PREFIXES[0]
    assert rectangle.commands[0].command_atoms == (0,)
    assert rectangle.commands[1].command_atoms == (1,)

    static_cells = rectangle.worlds[0].initial_packet.cells[: len(reference.cells)]
    assert tuple(
        (cell.slot, cell.type_index, cell.value.kind, cell.value.index)
        for cell in static_cells
    ) == tuple(
        (cell.slot, cell.type_index, ValueKind.STATIC_RAW, cell.value)
        for cell in reference.cells
    )
    assert tuple(
        (edge.relation, edge.source, edge.target)
        for edge in rectangle.worlds[0].initial_packet.edges
    ) == tuple(
        (
            edge.relation_index,
            edge.arguments[0],
            edge.arguments[-1],
        )
        for edge in reference.edges
    )
    assert rectangle.worlds[0].initial_packet.root == reference.root
    assert all(rectangle.worlds[0].initial_packet.slot_support or ())
    assert all(rectangle.worlds[0].initial_packet.relation_support or ())

    world_0_runtime = rectangle.worlds[0].initial_packet.cells[-4:]
    world_1_runtime = rectangle.worlds[1].initial_packet.cells[-4:]
    assert [cell.value.index for cell in world_0_runtime] == [0, None, None, None]
    assert [cell.value.index for cell in world_1_runtime] == [1, None, None, None]

    corner_00 = rectangle.corners[0][0]
    corner_01 = rectangle.corners[0][1]
    assert [mutation.source for mutation in corner_00.operation_traces[0].mutations] == [
        32,
        33,
    ]
    assert [
        mutation.value.index for mutation in corner_00.operation_traces[0].mutations
    ] == [4, 0]
    assert [
        mutation.source for mutation in corner_01.operation_traces[0].mutations
    ] == [32, 33, 34]
    assert all(
        mutation.opcode is Opcode.WRITE
        for row in rectangle.corners
        for corner in row
        for trace in corner.operation_traces
        for mutation in trace.mutations
    )
    assert corner_00.disposition is Disposition.ANSWER
    assert corner_00.answers == (True, False)
    assert rectangle.corners[0][1].answers == (False, True)
    assert rectangle.corners[1][0].answers == (False, True)
    assert rectangle.corners[1][1].answers == (True, False)

    terminal_controls = {
        cell.slot: cell for cell in corner_00.terminal_packet.cells if cell.slot >= 48
    }
    assert terminal_controls[48].value.kind is ValueKind.COMMAND_ATOM
    assert terminal_controls[48].value.index == 0
    assert all(
        terminal_controls[slot].value.kind is ValueKind.EMPTY
        for slot in range(49, 54)
    )
    assert terminal_controls[54].value.kind is ValueKind.SMALL_UINT
    assert terminal_controls[54].value.index == 1
    assert terminal_controls[55].value.kind is ValueKind.EXECUTE
    assert corner_00.terminal_packet.committed
    assert not corner_00.terminal_packet.halted


def test_adapter_output_passes_full_cpu_materializer_replay() -> None:
    rectangle = adapt_rewrite_rectangle(**_inputs())
    batch = materialize_ettr_il_v2(
        MaterializationRequest(
            manifest_sha256="a" * 64,
            dataset_sha256="b" * 64,
            vocab_size=512,
            rectangles=(rectangle,),
        ),
        _ByteTokenizer(),
    )

    assert batch.episodes.world.tokens.shape == (16, 192)
    assert batch.transaction_targets.step_mask.sum(dim=1).tolist() == [
        6,
        7,
        7,
        6,
    ] * 4
    assert batch.causal_rectangles.rows.shape == (4, 2, 2)


def test_rejects_supplied_or_recomputed_primary_replay_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs()
    replay = inputs["replay_executions"]
    assert isinstance(replay, tuple)
    bad = replace(replay[0][0], snapshots=replay[0][0].snapshots[:-1])
    inputs["replay_executions"] = ((bad, replay[0][1]), replay[1])
    with pytest.raises(RewriteAdapterError, match="primary/replay differs"):
        adapt_rewrite_rectangle(**inputs)

    inputs = _inputs()
    original = adapter.replay_rewrite

    def divergent(world: RewriteWorld, command: RewriteCommand):
        result = original(world, command)
        return replace(result, snapshots=result.snapshots[:-1])

    monkeypatch.setattr(adapter, "replay_rewrite", divergent)
    with pytest.raises(RewriteAdapterError, match="primary/replay differs"):
        adapt_rewrite_rectangle(**inputs)


def test_rejects_wrong_corner_binding_depth_and_theory_identity() -> None:
    inputs = _inputs()
    primary = inputs["primary_executions"]
    assert isinstance(primary, tuple)
    inputs["primary_executions"] = (
        (primary[0][1], primary[0][0]),
        primary[1],
    )
    with pytest.raises(RewriteAdapterError, match="binding differs"):
        adapt_rewrite_rectangle(**inputs)

    inputs = _inputs()
    inputs["commands"] = (
        inputs["commands"][0],
        RewriteCommand(2, (1, 0)),
    )
    with pytest.raises(RewriteAdapterError, match="depth stratum"):
        adapt_rewrite_rectangle(**inputs)

    inputs = _inputs()
    worlds = inputs["worlds"]
    assert isinstance(worlds, tuple)
    inputs["worlds"] = (
        worlds[0],
        replace(worlds[1], theory_index=2),
    )
    with pytest.raises(RewriteAdapterError, match="theory/evidence/policy"):
        adapt_rewrite_rectangle(**inputs)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        (
            "world_sources",
            ((b"same", b"same"), WORLD_SOURCES[1]),
            "variants are identical",
        ),
        (
            "command_sources",
            ((b"COMMAND-C0-command-0", b"\xff"), COMMAND_SOURCES[1]),
            "not ASCII",
        ),
        (
            "query_prefixes",
            ((b"", b"q0"), QUERY_PREFIXES[1]),
            "differs",
        ),
    ],
)
def test_rejects_non_cell_local_or_non_ascii_surface_bytes(
    field: str,
    replacement: object,
    message: str,
) -> None:
    inputs = _inputs()
    inputs[field] = replacement
    with pytest.raises(RewriteAdapterError, match=message):
        adapt_rewrite_rectangle(**inputs)


def test_rejects_non_checkerboard_query_without_selecting_a_replacement() -> None:
    inputs = _inputs()
    inputs["queries"] = (
        SemanticQuery(QueryOp.REWRITE_ROOT_IS, (4,)),
        SemanticQuery(QueryOp.REWRITE_CONTAINS, (0,)),
    )
    with pytest.raises(RewriteAdapterError, match="lack all-edge contrast"):
        adapt_rewrite_rectangle(**inputs)


def test_rejects_lossy_or_malformed_static_theory_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = reference_theory_state(1)
    monkeypatch.setattr(
        adapter,
        "reference_theory_state",
        lambda _index: replace(state, type_count=5),
    )
    with pytest.raises(RewriteAdapterError, match="geometry differs"):
        adapt_rewrite_rectangle(**_inputs())


def test_canonical_term_projection_rejects_more_than_four_nodes() -> None:
    leaf = GroundTerm(0, 0, ())
    five_nodes = GroundTerm(
        0,
        5,
        (
            GroundTerm(0, 5, (leaf, leaf)),
            leaf,
        ),
    )
    with pytest.raises(RewriteAdapterError, match="exceeds four"):
        adapter._term_values(five_nodes)


def test_ambiguous_execution_freezes_runtime_but_discloses_full_command() -> None:
    initial = GroundTerm(
        0,
        4,
        (GroundTerm(0, 4, (GroundTerm(0, 0, ()),)),),
    )
    world = RewriteWorld(
        EVIDENCE_ID,
        0,
        initial,
        RewritePolicy.CONTEXTUAL,
    )
    command = RewriteCommand(3, (0, 1, 0))
    execution = execute_rewrite(
        world,
        command,
        require_dependent=False,
    )

    traces, terminal_runtime = adapter._trace_and_terminal_runtime(execution)

    assert execution.disposition is TerminalDisposition.ABSTAIN
    assert len(execution.steps) == 1
    assert len(traces) == command.depth
    assert [trace.cursor for trace in traces] == [1, 2, 3]
    assert all(not trace.mutations for trace in traces)
    assert terminal_runtime == adapter._term_values(initial)
