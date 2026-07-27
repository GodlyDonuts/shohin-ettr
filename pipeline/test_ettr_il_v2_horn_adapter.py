from __future__ import annotations

from dataclasses import replace

import pytest

import ettr_il_v2_horn_adapter as adapter
from cross_ontology_horn_board import (
    THEORIES,
    GroundAtom,
    all_ground_atoms,
    reference_theory_state,
)
from cross_ontology_schema import ReactorState
from ettr_il_v2_horn_adapter import (
    HornAdapterError,
    adapt_horn_semantic_rectangle,
)
from ettr_il_v2_materialize import (
    Disposition,
    MaterializationRequest,
    Opcode,
    ValueKind,
    materialize_ettr_il_v2,
)
from ettr_il_v2_semantics import (
    HornCommand,
    HornPolicy,
    HornWorld,
    QueryOp,
    SemanticQuery,
    execute_horn,
    replay_horn,
)


EVIDENCE_ID = "0" * 64
RECTANGLE_ID = "a" * 64


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


def _board() -> dict[str, object]:
    worlds = (
        HornWorld(
            EVIDENCE_ID,
            0,
            (
                GroundAtom(3, (1, 3)),
                GroundAtom(4, (3, 2)),
            ),
            HornPolicy.PERSISTENT,
        ),
        HornWorld(
            EVIDENCE_ID,
            0,
            (
                GroundAtom(3, (0, 3)),
                GroundAtom(4, (3, 2)),
            ),
            HornPolicy.PERSISTENT,
        ),
    )
    commands = (
        HornCommand(1, (GroundAtom(0, (0,)),)),
        HornCommand(1, (GroundAtom(0, (1,)),)),
    )
    primary = tuple(
        tuple(execute_horn(world, command) for command in commands)
        for world in worlds
    )
    replay = tuple(
        tuple(replay_horn(world, command) for command in commands)
        for world in worlds
    )
    return {
        "semantic_rectangle_id": RECTANGLE_ID,
        "presentation_id": "horn-base-renderer-0",
        "worlds": worlds,
        "commands": commands,
        "primary_executions": primary,
        "replay_executions": replay,
        "queries": (
            SemanticQuery(QueryOp.HORN_HAS, (0, 2)),
            SemanticQuery(QueryOp.HORN_COUNT_GE, (5,)),
        ),
        "world_sources": (
            (b"WORLD-0-cell-C0\n", b"WORLD-0-cell-C1\n"),
            (b"WORLD-1-cell-C0\n", b"WORLD-1-cell-C1\n"),
        ),
        "command_sources": (
            (b"COMMAND-0-cell-W0\n", b"COMMAND-0-cell-W1\n"),
            (b"COMMAND-1-cell-W0\n", b"COMMAND-1-cell-W1\n"),
        ),
        "query_prefixes": (
            (b"Q0 paraphrase A: ", b"Q0 paraphrase B: "),
            (b"Q1 paraphrase A: ", b"Q1 paraphrase B: "),
        ),
    }


def test_projects_exact_static_dynamic_trace_terminal_and_labels() -> None:
    rectangle = adapt_horn_semantic_rectangle(**_board())  # type: ignore[arg-type]
    state = reference_theory_state(0)
    initial = rectangle.worlds[0].initial_packet

    assert tuple(
        (cell.slot, cell.type_index, cell.value.index)
        for cell in initial.cells[: len(state.cells)]
    ) == tuple(
        (cell.slot, cell.type_index, cell.value)
        for cell in state.cells
    )
    assert tuple(
        (cell.slot, cell.type_index, cell.value.kind, cell.value.index)
        for cell in initial.cells[len(state.cells) :]
    ) == tuple(
        (32 + index, 4, ValueKind.LOCAL_ID, index)
        for index in range(6)
    )
    static_edges = {
        (
            edge.relation_index,
            edge.arguments[0],
            edge.arguments[-1],
        )
        for edge in state.edges
    }
    assert static_edges <= {
        (edge.relation, edge.source, edge.target)
        for edge in initial.edges
    }
    assert {
        (edge.relation, edge.source, edge.target)
        for edge in initial.edges
        if edge.relation >= 8
    } == {
        (11, 33, 35),
        (12, 35, 34),
    }
    assert initial.root == state.root
    assert initial.committed is False
    assert initial.halted is False

    atom_index = all_ground_atoms().index(GroundAtom(0, (0,)))
    assert rectangle.commands[0].command_atoms == (atom_index,)
    corner = rectangle.corners[0][0]
    assert tuple(mutation.opcode for mutation in corner.operation_traces[0].mutations) == (
        Opcode.LINK,
        Opcode.LINK,
    )
    assert tuple(
        (
            mutation.relation,
            mutation.source,
            mutation.target,
        )
        for mutation in corner.operation_traces[0].mutations
    ) == (
        (8, 32, 32),
        (9, 32, 32),
    )
    assert corner.operation_traces[0].cursor == 1
    assert corner.disposition is Disposition.ANSWER
    assert corner.outcome.kind is ValueKind.EXECUTE
    assert corner.answers == (False, False)
    assert rectangle.corners[0][1].answers == (True, True)
    assert rectangle.corners[1][0].answers == (True, True)
    assert rectangle.corners[1][1].answers == (False, False)

    terminal_cells = {
        cell.slot: cell for cell in corner.terminal_packet.cells
    }
    assert tuple(
        terminal_cells[48 + index].value.kind for index in range(6)
    ) == (
        ValueKind.COMMAND_ATOM,
        ValueKind.EMPTY,
        ValueKind.EMPTY,
        ValueKind.EMPTY,
        ValueKind.EMPTY,
        ValueKind.EMPTY,
    )
    assert terminal_cells[54].value.index == 1
    assert terminal_cells[55].value.kind is ValueKind.EXECUTE
    assert corner.terminal_packet.committed is True
    assert corner.terminal_packet.halted is False


def test_adapter_output_materializes_and_replays_on_cpu() -> None:
    rectangle = adapt_horn_semantic_rectangle(**_board())  # type: ignore[arg-type]
    batch = materialize_ettr_il_v2(
        MaterializationRequest(
            manifest_sha256="b" * 64,
            dataset_sha256="c" * 64,
            vocab_size=512,
            rectangles=(rectangle,),
        ),
        _ByteTokenizer(),
    )

    assert batch.episodes.world.tokens.shape[0] == 16
    assert batch.transaction_targets.step_mask.sum(dim=1).tolist() == [
        6,
        9,
        9,
        6,
    ] * 4
    assert batch.packet_targets.committed.eq(False).all()
    assert batch.terminal_packet_targets.committed.eq(True).all()
    assert batch.terminal_packet_targets.halted.eq(False).all()
    assert batch.causal_rectangles.rows.shape == (4, 2, 2)
    assert batch.episodes.world.tokens.device.type == "cpu"


@pytest.mark.parametrize("which", ["primary", "replay"])
def test_rejects_tampered_primary_or_replay_execution(which: str) -> None:
    board = _board()
    key = f"{which}_executions"
    executions = board[key]
    assert isinstance(executions, tuple)
    first = executions[0][0]
    tampered = replace(
        first,
        snapshots=first.snapshots[:-1] + (first.snapshots[0],),
    )
    board[key] = (
        (tampered, executions[0][1]),
        executions[1],
    )

    with pytest.raises(
        HornAdapterError,
        match=f"{which} execution differs",
    ):
        adapt_horn_semantic_rectangle(**board)  # type: ignore[arg-type]


def test_rejects_noncanonical_or_out_of_geometry_reference_theory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = reference_theory_state(0)
    hostile = ReactorState(
        capacity=state.capacity,
        type_count=5,
        relation_specs=state.relation_specs,
        cells=state.cells,
        edges=state.edges,
        root=state.root,
        committed_steps=state.committed_steps,
        halted=state.halted,
    )
    monkeypatch.setattr(adapter, "reference_theory_state", lambda _: hostile)

    with pytest.raises(HornAdapterError, match="geometry is noncanonical"):
        adapt_horn_semantic_rectangle(**_board())  # type: ignore[arg-type]


@pytest.mark.parametrize("theory_index", range(len(THEORIES)))
def test_all_horn_reference_theories_project_losslessly(
    theory_index: int,
) -> None:
    projection = adapter._project_static_theory(theory_index)
    state = reference_theory_state(theory_index)

    assert len(projection.cells) == len(state.cells)
    assert len(projection.edges) == len(state.edges)
    assert projection.root == state.root


def test_rejects_divergence_from_canonical_execute_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = adapter.execute_closure

    def hostile(*args: object) -> tuple[GroundAtom, ...]:
        result = original(*args)  # type: ignore[arg-type]
        return result[:-1]

    monkeypatch.setattr(adapter, "execute_closure", hostile)
    with pytest.raises(HornAdapterError, match="execute_closure differs"):
        adapt_horn_semantic_rectangle(**_board())  # type: ignore[arg-type]


def test_internal_replay_rejects_hostile_operation_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = adapter._operation_traces

    def hostile(execution: object) -> tuple[object, ...]:
        traces = original(execution)  # type: ignore[arg-type]
        first = traces[0]
        mutation = replace(
            first.mutations[0],
            relation=first.mutations[0].relation + 1,
        )
        return (replace(first, mutations=(mutation, *first.mutations[1:])),)

    monkeypatch.setattr(adapter, "_operation_traces", hostile)
    with pytest.raises(
        HornAdapterError,
        match="relinks an existing fact|operation replay differs",
    ):
        adapt_horn_semantic_rectangle(**_board())  # type: ignore[arg-type]


def test_independent_label_validation_rejects_hostile_corner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = adapter._answer_pair
    calls = 0

    def hostile(*args: object) -> tuple[bool | None, bool | None]:
        nonlocal calls
        calls += 1
        answers = original(*args)  # type: ignore[arg-type]
        if calls == 1:
            return (not bool(answers[0]), answers[1])
        return answers

    monkeypatch.setattr(adapter, "_answer_pair", hostile)
    with pytest.raises(HornAdapterError, match="answer labels differ"):
        adapt_horn_semantic_rectangle(**_board())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("semantic_rectangle_id", "not-a-digest", "semantic_rectangle_id"),
        (
            "world_sources",
            ((b"same", b"same"), (b"other", b"fourth")),
            "not all distinct",
        ),
        (
            "command_sources",
            ((b"ok", b"\xff"), (b"three", b"four")),
            "strict ASCII",
        ),
        (
            "query_prefixes",
            ((b"same", b"same"), (b"three", b"four")),
            "not all distinct",
        ),
    ],
)
def test_rejects_malformed_ids_or_external_surface_bytes(
    field: str,
    value: object,
    match: str,
) -> None:
    board = _board()
    board[field] = value
    with pytest.raises(HornAdapterError, match=match):
        adapt_horn_semantic_rectangle(**board)  # type: ignore[arg-type]


def test_rejects_non_checkerboard_answer_semantics() -> None:
    board = _board()
    board["queries"] = (
        SemanticQuery(QueryOp.HORN_HAS, (3, 0, 3)),
        board["queries"][1],  # type: ignore[index]
    )

    with pytest.raises(HornAdapterError, match="not a strict checkerboard"):
        adapt_horn_semantic_rectangle(**board)  # type: ignore[arg-type]

    broad = adapt_horn_semantic_rectangle(
        **board,  # type: ignore[arg-type]
        require_query_checkerboard=False,
    )
    assert len(broad.corners) == 2
