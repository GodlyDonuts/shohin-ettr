from __future__ import annotations

from dataclasses import replace

import pytest

import ettr_il_v2_resource_adapter as adapter
from cross_ontology_resource_board import Marking
from ettr_il_v2_materialize import (
    Disposition,
    MaterializationRequest,
    Opcode,
    ValueKind,
    materialize_ettr_il_v2,
)
from ettr_il_v2_resource_adapter import (
    ResourceAdapterError,
    adapt_resource_rectangle,
)
from ettr_il_v2_semantics import (
    QueryOp,
    ResourceCommand,
    ResourcePolicy,
    ResourceWorld,
    SemanticQuery,
    execute_resource,
)


EVIDENCE_ID = "0" * 64
RECTANGLE_ID = "1" * 64


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
        ResourceWorld(
            EVIDENCE_ID,
            0,
            Marking((1, 0, 1, 0)),
            ResourcePolicy.SKIP_BLOCKED,
        ),
        ResourceWorld(
            EVIDENCE_ID,
            3,
            Marking((1, 0, 0, 2)),
            ResourcePolicy.SKIP_BLOCKED,
        ),
    )
    commands = (
        ResourceCommand(2, (0, 0)),
        ResourceCommand(2, (0, 1)),
    )
    executions = tuple(
        tuple(execute_resource(world, command) for command in commands)
        for world in worlds
    )
    return {
        "semantic_rectangle_id": RECTANGLE_ID,
        "presentation_id": "resource-base-r0",
        "worlds": worlds,
        "commands": commands,
        "executions": executions,
        "queries": (
            SemanticQuery(QueryOp.RESOURCE_PLACE_GE, (2, 1)),
            SemanticQuery(QueryOp.RESOURCE_PLACE_GE, (3, 1)),
        ),
        "cell_world_sources": (
            (b"W0-for-C0", b"W0-for-C1"),
            (b"W1-for-C0", b"W1-for-C1"),
        ),
        "cell_command_sources": (
            (b"C0-for-W0", b"C1-for-W0"),
            (b"C0-for-W1", b"C1-for-W1"),
        ),
        "query_prefixes": (
            (b"Q0 form zero: ", b"Q0 form one: "),
            (b"Q1 form zero: ", b"Q1 form one: "),
        ),
    }


def _deadlock_suffix_inputs() -> dict[str, object]:
    values = _inputs()
    worlds = (
        ResourceWorld(
            EVIDENCE_ID,
            0,
            Marking((2, 0, 0, 0)),
            ResourcePolicy.ATOMIC_DEADLOCK,
        ),
        ResourceWorld(
            EVIDENCE_ID,
            0,
            Marking((2, 2, 1, 0)),
            ResourcePolicy.ATOMIC_DEADLOCK,
        ),
    )
    commands = (
        ResourceCommand(3, (0, 0, 0)),
        ResourceCommand(3, (0, 1, 0)),
    )
    values["worlds"] = worlds
    values["commands"] = commands
    values["executions"] = tuple(
        tuple(execute_resource(world, command) for command in commands)
        for world in worlds
    )
    values["queries"] = (
        SemanticQuery(QueryOp.RESOURCE_PLACE_GE, (0, 1)),
        SemanticQuery(QueryOp.RESOURCE_CURSOR_GE, (2,)),
    )
    return values


def _rectangle():
    return adapt_resource_rectangle(**_inputs())  # type: ignore[arg-type]


def test_projects_static_runtime_trace_terminal_and_answers_exactly() -> None:
    rectangle = _rectangle()

    initial = rectangle.worlds[0].initial_packet
    assert len(initial.cells) == 24
    static = tuple(cell for cell in initial.cells if cell.slot < 32)
    runtime = tuple(cell for cell in initial.cells if 32 <= cell.slot <= 35)
    assert len(static) == 20
    assert tuple(cell.slot for cell in static) == tuple(range(20))
    assert all(cell.type_index < 4 for cell in static)
    assert all(cell.value.kind is ValueKind.STATIC_RAW for cell in static)
    assert tuple(cell.value.index for cell in runtime) == (1, 0, 1, 0)
    assert initial.root == 0
    assert not initial.committed
    assert not initial.halted
    assert len(initial.edges) == 27

    # W0/C0 applies its first operation and then skips a blocked second one.
    # The blocked operation remains represented by its command/cursor writes.
    corner = rectangle.corners[0][0]
    assert len(corner.operation_traces) == 2
    assert tuple(trace.cursor for trace in corner.operation_traces) == (1, 2)
    assert corner.operation_traces[1].mutations == ()
    assert corner.disposition is Disposition.ANSWER
    assert corner.outcome.kind is ValueKind.PROCESS_HALT
    assert corner.answers == (True, False)

    first_mutations = corner.operation_traces[0].mutations
    assert tuple(mutation.opcode for mutation in first_mutations) == (
        Opcode.WRITE,
        Opcode.WRITE,
    )
    assert tuple(mutation.source for mutation in first_mutations) == (32, 33)
    assert tuple(mutation.value.index for mutation in first_mutations) == (0, 1)

    terminal = corner.terminal_packet
    by_slot = {cell.slot: cell for cell in terminal.cells}
    assert tuple(by_slot[slot].value.index for slot in range(32, 36)) == (
        0,
        1,
        1,
        0,
    )
    assert tuple(by_slot[slot].value.index for slot in range(48, 54)) == (
        0,
        0,
        None,
        None,
        None,
        None,
    )
    assert by_slot[54].value.index == 2
    assert by_slot[55].value.kind is ValueKind.PROCESS_HALT
    assert terminal.committed
    assert not terminal.halted

    assert tuple(
        rectangle.corners[w][c].answers[0]
        for w in range(2)
        for c in range(2)
    ) == (True, False, False, True)
    assert tuple(
        rectangle.corners[w][c].answers[1]
        for w in range(2)
        for c in range(2)
    ) == (False, True, True, False)


def test_atomic_deadlock_keeps_disclosed_suffix_and_frozen_cursor() -> None:
    rectangle = adapt_resource_rectangle(
        **_deadlock_suffix_inputs()  # type: ignore[arg-type]
    )
    deadlock = rectangle.corners[0][1]

    assert len(deadlock.operation_traces) == 3
    assert tuple(trace.cursor for trace in deadlock.operation_traces) == (1, 1, 1)
    assert deadlock.operation_traces[1].mutations == ()
    assert deadlock.operation_traces[2].mutations == ()
    assert deadlock.outcome.kind is ValueKind.PROCESS_DEADLOCK
    assert deadlock.terminal_packet.committed
    assert not deadlock.terminal_packet.halted


def test_output_materializes_end_to_end_on_cpu() -> None:
    batch = materialize_ettr_il_v2(
        MaterializationRequest(
            manifest_sha256="a" * 64,
            dataset_sha256="b" * 64,
            vocab_size=512,
            rectangles=(_rectangle(),),
        ),
        _ByteTokenizer(),
    )

    assert batch.episodes.world.tokens.shape == (16, 192)
    assert batch.episodes.command.tokens.shape == (16, 96)
    assert batch.episodes.query.tokens.shape == (16, 48)
    assert batch.causal_rectangles.rows.shape == (4, 2, 2)
    assert batch.packet_targets.slot_mask.all()
    assert batch.terminal_packet_targets.relation_mask.all()
    assert batch.transaction_targets.step_mask.sum(dim=1).max().item() <= 64
    assert batch.episodes.world.tokens.device.type == "cpu"


def test_primary_or_supplied_execution_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _inputs()
    executions = values["executions"]
    assert isinstance(executions, tuple)
    corner = executions[0][0]
    values["executions"] = (
        (replace(corner, cursor=corner.cursor + 1), executions[0][1]),
        executions[1],
    )
    with pytest.raises(ResourceAdapterError, match="supplied execution differs"):
        adapt_resource_rectangle(**values)  # type: ignore[arg-type]

    values = _inputs()
    original = adapter.replay_resource

    def hostile_replay(world, command):
        replay = original(world, command)
        return replace(replay, cursor=replay.cursor + 1)

    monkeypatch.setattr(adapter, "replay_resource", hostile_replay)
    with pytest.raises(ResourceAdapterError, match="primary/replay differs"):
        adapt_resource_rectangle(**values)  # type: ignore[arg-type]


def test_invalid_static_projection_and_surface_inputs_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _inputs()
    monkeypatch.setattr(adapter, "reference_theory_state", lambda _: object())
    with pytest.raises(ResourceAdapterError, match="state type differs"):
        adapt_resource_rectangle(**values)  # type: ignore[arg-type]

    monkeypatch.undo()
    values = _inputs()
    values["cell_world_sources"] = (
        (b"same", b"same"),
        (b"W1-C0", b"W1-C1"),
    )
    with pytest.raises(ResourceAdapterError, match="lacks cell-local contrast"):
        adapt_resource_rectangle(**values)  # type: ignore[arg-type]

    values = _inputs()
    values["query_prefixes"] = (
        (b"same", b"same"),
        (b"q1-a", b"q1-b"),
    )
    with pytest.raises(ResourceAdapterError, match="paraphrases are identical"):
        adapt_resource_rectangle(**values)  # type: ignore[arg-type]


def test_first_operation_deadlock_preserves_zero_oracle_cursor() -> None:
    values = _inputs()
    worlds = (
        ResourceWorld(
            EVIDENCE_ID,
            0,
            Marking((1, 1, 1, 0)),
            ResourcePolicy.ATOMIC_DEADLOCK,
        ),
        ResourceWorld(
            EVIDENCE_ID,
            0,
            Marking((2, 0, 0, 0)),
            ResourcePolicy.ATOMIC_DEADLOCK,
        ),
    )
    commands = (
        ResourceCommand(2, (0, 0)),
        ResourceCommand(2, (1, 0)),
    )
    values["worlds"] = worlds
    values["commands"] = commands
    values["executions"] = tuple(
        tuple(execute_resource(world, command) for command in commands)
        for world in worlds
    )
    values["queries"] = (
        SemanticQuery(QueryOp.RESOURCE_CURSOR_GE, (2,)),
        SemanticQuery(QueryOp.RESOURCE_HALT, ()),
    )

    rectangle = adapt_resource_rectangle(**values)  # type: ignore[arg-type]
    first_deadlock = rectangle.corners[1][1]
    assert first_deadlock.outcome.kind is ValueKind.PROCESS_DEADLOCK
    assert tuple(trace.cursor for trace in first_deadlock.operation_traces) == (
        0,
        0,
    )
    assert all(not trace.mutations for trace in first_deadlock.operation_traces)


def test_non_checkerboard_queries_and_bad_ids_fail_closed() -> None:
    values = _inputs()
    values["queries"] = (
        SemanticQuery(QueryOp.RESOURCE_PLACE_GE, (0, 1)),
        SemanticQuery(QueryOp.RESOURCE_PLACE_GE, (1, 1)),
    )
    with pytest.raises(ResourceAdapterError, match="strict checkerboards"):
        adapt_resource_rectangle(**values)  # type: ignore[arg-type]

    values = _inputs()
    values["semantic_rectangle_id"] = "not-a-digest"
    with pytest.raises(ResourceAdapterError, match="lowercase SHA-256"):
        adapt_resource_rectangle(**values)  # type: ignore[arg-type]
