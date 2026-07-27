from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from ettr_il_v2_materialize import (
    COMMAND_WIDTH,
    QUERY_WIDTH,
    WORLD_WIDTH,
    Disposition,
    GenericCell,
    GenericCommand,
    GenericCorner,
    GenericEdge,
    GenericInvariantPair,
    GenericMutation,
    GenericOperationTrace,
    GenericPacket,
    GenericQuery,
    GenericSemanticRectangle,
    GenericWorld,
    MaterializationError,
    MaterializationRequest,
    Opcode,
    ValueKind,
    ValueRef,
    materialize_ettr_il_v2,
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


class _SplitAnswerTokenizer(_ByteTokenizer):
    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> _Encoding:
        ids = super().encode(text, add_special_tokens).ids
        expanded: list[int] = []
        for character, token_id in zip(text, ids, strict=True):
            if character in {"0", "1", "2", "3"}:
                expanded.extend((token_id, 400))
            else:
                expanded.append(token_id)
        return _Encoding(expanded)


TOKENIZER = _ByteTokenizer()


def _initial_packet(world: int) -> GenericPacket:
    return GenericPacket(
        cells=(
            GenericCell(0, 0, ValueRef.static(100)),
            GenericCell(1, 1, ValueRef.static(-5)),
            GenericCell(32, 4, ValueRef.local_id(world)),
            GenericCell(33, 4, ValueRef.local_id(3)),
        ),
        edges=(GenericEdge(8, 32, 33),),
        root=0,
    )


def _terminal_packet(
    world: int,
    command: int,
    result: int,
    *,
    edge: bool = True,
) -> GenericPacket:
    cells = list(_initial_packet(world).cells)
    cells[2] = GenericCell(32, 4, ValueRef.local_id(result))
    cells.extend(
        GenericCell(
            slot,
            5,
            ValueRef.command_atom(10 + command)
            if slot == 48
            else ValueRef.empty(),
        )
        for slot in range(48, 54)
    )
    cells.extend(
        (
            GenericCell(54, 6, ValueRef.small_uint(1)),
            GenericCell(55, 6, ValueRef.execute()),
        )
    )
    return GenericPacket(
        cells=tuple(cells),
        edges=(GenericEdge(8, 32, 33),) if edge else (),
        root=0,
        committed=True,
        halted=False,
    )


def _rectangle(
    name: str,
    *,
    source_suffix: bytes = b"",
) -> GenericSemanticRectangle:
    worlds = (
        GenericWorld(
            (
                b"WORLD-A-P0" + source_suffix,
                b"WORLD-A-P1" + source_suffix,
            ),
            _initial_packet(0),
        ),
        GenericWorld(
            (
                b"WORLD-B-P0" + source_suffix,
                b"WORLD-B-P1" + source_suffix,
            ),
            _initial_packet(1),
        ),
    )
    commands = (
        GenericCommand(
            (
                b"COMMAND-A-P0" + source_suffix,
                b"COMMAND-A-P1" + source_suffix,
            ),
            (10,),
        ),
        GenericCommand(
            (
                b"COMMAND-B-P0" + source_suffix,
                b"COMMAND-B-P1" + source_suffix,
            ),
            (11,),
        ),
    )
    corners: list[list[GenericCorner]] = [[], []]
    for world in range(2):
        for command in range(2):
            result = 8 + 2 * world + command
            corners[world].append(
                GenericCorner(
                    operation_traces=(
                        GenericOperationTrace(
                            mutations=(
                                GenericMutation(
                                    Opcode.WRITE,
                                    source=32,
                                    value=ValueRef.local_id(result),
                                ),
                            ),
                            cursor=1,
                        ),
                    ),
                    terminal_packet=_terminal_packet(
                        world,
                        command,
                        result,
                    ),
                    disposition=Disposition.ANSWER,
                    outcome=ValueRef.execute(),
                    answers=(
                        bool(world ^ command),
                        not bool(world ^ command),
                    ),
                )
            )
    return GenericSemanticRectangle(
        semantic_rectangle_id=name,
        presentation_id=f"presentation-{name}",
        worlds=worlds,
        commands=commands,
        queries=(
            GenericQuery((b"Is alpha true? ", b"Alpha verdict: ")),
            GenericQuery((b"Is beta true? ", b"Beta verdict: ")),
        ),
        corners=(
            (corners[0][0], corners[0][1]),
            (corners[1][0], corners[1][1]),
        ),
    )


def _request(
    *rectangles: GenericSemanticRectangle,
    pairs: tuple[GenericInvariantPair, ...] = (),
) -> MaterializationRequest:
    return MaterializationRequest(
        manifest_sha256="a" * 64,
        dataset_sha256="b" * 64,
        vocab_size=512,
        rectangles=tuple(rectangles),
        invariant_pairs=pairs,
    )


def test_materializes_exact_geometry_ranking_replay_and_rectangles() -> None:
    batch = materialize_ettr_il_v2(_request(_rectangle("r0")), TOKENIZER)

    assert batch.episodes.world.tokens.shape == (16, WORLD_WIDTH)
    assert batch.episodes.command.tokens.shape == (16, COMMAND_WIDTH)
    assert batch.episodes.query.tokens.shape == (16, QUERY_WIDTH)
    assert batch.transaction_targets.opcode.shape == (16, 64)
    assert batch.causal_rectangles.rows.tolist() == [
        [[0, 1], [2, 3]],
        [[4, 5], [6, 7]],
        [[8, 9], [10, 11]],
        [[12, 13], [14, 15]],
    ]
    assert batch.packet_targets.slot_mask.all()
    assert batch.packet_targets.relation_mask.all()
    assert batch.terminal_packet_targets.slot_mask.all()
    assert batch.terminal_packet_targets.relation_mask.all()
    # Numeric sorting maps -5 -> rank 0/code 1 and 100 -> rank 1/code 2.
    assert batch.packet_targets.value_code[0, 0].item() == 2
    assert batch.packet_targets.value_code[0, 1].item() == 1
    # Materializer-owned command, cursor, and outcome cells are active.
    assert batch.packet_targets.active[0, 48:56].all()
    assert batch.packet_targets.value_code[0, 48:56].eq(0).all()
    valid = batch.transaction_targets.step_mask[0]
    assert valid.sum().item() == 5
    assert batch.transaction_targets.opcode[0, valid].tolist() == [1, 1, 1, 1, 6]
    assert batch.transaction_targets.source[0, valid].tolist() == [48, 32, 54, 55, 0]
    assert batch.transaction_targets.committed[0, valid].tolist() == [
        False,
        False,
        False,
        False,
        True,
    ]
    assert batch.transaction_targets.committed[0, ~valid].all()
    read = batch.episodes.query_read_index
    targets = batch.episodes.query.targets.gather(1, read[:, None]).squeeze(1)
    assert targets[0].item() != targets[1].item()
    assert targets[0].item() != targets[2].item()
    assert batch.equivariance is None
    assert all(tensor.device.type == "cpu" for tensor in (
        batch.episodes.world.tokens,
        batch.packet_targets.active,
        batch.transaction_targets.opcode,
    ))


def test_lossless_invariant_pair_builds_identity_alignment_for_all_rows() -> None:
    left = _rectangle("left")
    right = replace(
        _rectangle("right", source_suffix=b"-VARIANT"),
        presentation_id="presentation-right-variant",
    )
    batch = materialize_ettr_il_v2(
        _request(
            left,
            right,
            pairs=(GenericInvariantPair(0, 1),),
        ),
        TOKENIZER,
    )

    assert batch.equivariance is not None
    assert batch.equivariance.left_index.tolist() == list(range(16))
    assert batch.equivariance.right_index.tolist() == list(range(16, 32))
    assert batch.equivariance.slot_permutation.eq(torch.arange(64)).all()
    assert batch.equivariance.type_permutation.eq(torch.arange(8)).all()
    assert batch.equivariance.relation_permutation.eq(torch.arange(16)).all()
    assert batch.equivariance.value_permutation.eq(torch.arange(256)).all()
    assert batch.equivariance.step_mask.sum(-1).eq(5).all()


def test_trace_over_64_steps_fails_before_any_torch_construction() -> None:
    rectangle = _rectangle("too-long")
    mutation = GenericMutation(
        Opcode.WRITE,
        source=32,
        value=ValueRef.local_id(7),
    )
    operations = tuple(
        GenericOperationTrace(
            mutations=(mutation,) * 9,
            cursor=index + 1,
        )
        for index in range(6)
    )
    command = replace(rectangle.commands[0], command_atoms=(0, 1, 2, 3, 4, 5))
    corner = replace(rectangle.corners[0][0], operation_traces=operations)
    rectangle = replace(
        rectangle,
        commands=(command, rectangle.commands[1]),
        corners=(
            (corner, rectangle.corners[0][1]),
            rectangle.corners[1],
        ),
    )

    with pytest.raises(MaterializationError, match="exceeds 64 steps"):
        materialize_ettr_il_v2(_request(rectangle), TOKENIZER)


def test_fixed_packet_capacity_and_reserved_slots_fail_closed() -> None:
    rectangle = _rectangle("capacity")
    hostile = replace(
        rectangle.worlds[0].initial_packet,
        cells=rectangle.worlds[0].initial_packet.cells
        + (GenericCell(56, 4, ValueRef.local_id(0)),),
    )
    rectangle = replace(
        rectangle,
        worlds=(
            replace(rectangle.worlds[0], initial_packet=hostile),
            rectangle.worlds[1],
        ),
    )

    with pytest.raises(MaterializationError, match="reserved slot"):
        materialize_ettr_il_v2(_request(rectangle), TOKENIZER)


def test_first_operation_deadlock_cursor_zero_is_exactly_constrained() -> None:
    rectangle = _rectangle("deadlock-zero")
    deadlocked = replace(
        rectangle.corners[0][0],
        operation_traces=(
            GenericOperationTrace(mutations=(), cursor=0),
        ),
        terminal_packet=replace(
            rectangle.worlds[0].initial_packet,
            cells=rectangle.worlds[0].initial_packet.cells
            + tuple(
                GenericCell(
                    slot,
                    5,
                    ValueRef.command_atom(10)
                    if slot == 48
                    else ValueRef.empty(),
                )
                for slot in range(48, 54)
            )
            + (
                GenericCell(54, 6, ValueRef.small_uint(0)),
                GenericCell(
                    55,
                    6,
                    ValueRef(ValueKind.PROCESS_DEADLOCK),
                ),
            ),
            committed=True,
        ),
        outcome=ValueRef(ValueKind.PROCESS_DEADLOCK),
    )
    deadlock_rectangle = replace(
        rectangle,
        corners=(
            (deadlocked, rectangle.corners[0][1]),
            rectangle.corners[1],
        ),
    )
    batch = materialize_ettr_il_v2(_request(deadlock_rectangle), TOKENIZER)
    valid = batch.transaction_targets.step_mask[0]
    assert batch.transaction_targets.source[0, valid].tolist() == [
        48,
        54,
        55,
        0,
    ]
    assert batch.transaction_targets.value_code[0, valid].tolist()[1] == 65

    hostile = replace(
        deadlocked,
        operation_traces=(
            GenericOperationTrace(
                mutations=(
                    GenericMutation(
                        Opcode.WRITE,
                        source=32,
                        value=ValueRef.local_id(7),
                    ),
                ),
                cursor=0,
            ),
        ),
    )
    hostile_rectangle = replace(
        deadlock_rectangle,
        corners=(
            (hostile, rectangle.corners[0][1]),
            rectangle.corners[1],
        ),
    )
    with pytest.raises(MaterializationError, match="after the deadlock"):
        materialize_ettr_il_v2(_request(hostile_rectangle), TOKENIZER)


def test_non_single_token_answer_boundary_fails_closed() -> None:
    with pytest.raises(MaterializationError, match="one next token"):
        materialize_ettr_il_v2(
            _request(_rectangle("split-answer")),
            _SplitAnswerTokenizer(),
        )


def test_invalid_query_edge_contrast_fails_closed() -> None:
    rectangle = _rectangle("answer-edge")
    left_answers = rectangle.corners[0][0].answers
    hostile = replace(rectangle.corners[0][1], answers=left_answers)
    rectangle = replace(
        rectangle,
        corners=(
            (rectangle.corners[0][0], hostile),
            rectangle.corners[1],
        ),
    )

    with pytest.raises(MaterializationError, match="query-label contrast"):
        materialize_ettr_il_v2(_request(rectangle), TOKENIZER)


def test_v3_broad_mode_accepts_answer_invariant_packet_effects() -> None:
    rectangle = _rectangle("broad-answer-edge")
    left_answers = rectangle.corners[0][0].answers
    broad_corner = replace(rectangle.corners[0][1], answers=left_answers)
    rectangle = replace(
        rectangle,
        corners=(
            (rectangle.corners[0][0], broad_corner),
            rectangle.corners[1],
        ),
    )
    request = replace(
        _request(rectangle),
        require_query_checkerboard=False,
    )
    batch = materialize_ettr_il_v2(request, TOKENIZER)
    assert batch.causal_rectangles.rows.shape == (4, 2, 2)
    assert batch.terminal_packet_targets.value_code.shape[0] == 16


def test_query_checkerboard_mode_must_be_boolean() -> None:
    with pytest.raises(MaterializationError, match="request differs"):
        materialize_ettr_il_v2(
            replace(
                _request(_rectangle("bad-checkerboard-mode")),
                require_query_checkerboard=1,
            ),
            TOKENIZER,
        )


def test_invalid_terminal_edge_contrast_fails_closed() -> None:
    rectangle = _rectangle("packet-edge")
    command = replace(
        rectangle.commands[1],
        command_atoms=rectangle.commands[0].command_atoms,
    )
    copied = rectangle.corners[0][0]
    copied_world_one = replace(
        rectangle.corners[1][0],
        answers=rectangle.corners[1][1].answers,
    )
    rectangle = replace(
        rectangle,
        commands=(rectangle.commands[0], command),
        corners=(
            (rectangle.corners[0][0], copied),
            (rectangle.corners[1][0], copied_world_one),
        ),
    )

    with pytest.raises(MaterializationError, match="terminal packet contrast"):
        materialize_ettr_il_v2(_request(rectangle), TOKENIZER)


def test_independent_replay_rejects_oracle_terminal_mismatch() -> None:
    rectangle = _rectangle("replay")
    terminal = rectangle.corners[0][0].terminal_packet
    cells = tuple(
        replace(cell, value=ValueRef.local_id(31))
        if cell.slot == 32
        else cell
        for cell in terminal.cells
    )
    hostile = replace(
        rectangle.corners[0][0],
        terminal_packet=replace(terminal, cells=cells),
    )
    rectangle = replace(
        rectangle,
        corners=(
            (hostile, rectangle.corners[0][1]),
            rectangle.corners[1],
        ),
    )

    with pytest.raises(MaterializationError, match="independent replay differs"):
        materialize_ettr_il_v2(_request(rectangle), TOKENIZER)


def test_lossy_or_nonidentity_alignment_fails_closed() -> None:
    left = _rectangle("align-left")
    right = _rectangle("align-right", source_suffix=b"-ALT")
    lossy = (0, 0) + tuple(range(2, 64))
    request = _request(
        left,
        right,
        pairs=(GenericInvariantPair(0, 1, slot_permutation=lossy),),
    )

    with pytest.raises(MaterializationError, match="alignment is lossy"):
        materialize_ettr_il_v2(request, TOKENIZER)


def test_partial_packet_support_fails_closed() -> None:
    rectangle = _rectangle("partial-support")
    support = (False,) + (True,) * 63
    packet = replace(
        rectangle.worlds[0].initial_packet,
        slot_support=support,
    )
    rectangle = replace(
        rectangle,
        worlds=(
            replace(rectangle.worlds[0], initial_packet=packet),
            rectangle.worlds[1],
        ),
    )

    with pytest.raises(MaterializationError, match="partial slot support"):
        materialize_ettr_il_v2(_request(rectangle), TOKENIZER)


def test_partial_relation_support_fails_closed() -> None:
    rectangle = _rectangle("partial-relation-support")
    relation_support = (True,) * (16 * 64 * 64 - 1) + (False,)
    packet = replace(
        rectangle.worlds[0].initial_packet,
        relation_support=relation_support,
    )
    rectangle = replace(
        rectangle,
        worlds=(
            replace(rectangle.worlds[0], initial_packet=packet),
            rectangle.worlds[1],
        ),
    )

    with pytest.raises(MaterializationError, match="partial relation support"):
        materialize_ettr_il_v2(_request(rectangle), TOKENIZER)


def test_relation_edge_capacity_fails_closed_without_clipping() -> None:
    rectangle = _rectangle("edge-capacity")
    initial = rectangle.worlds[0].initial_packet
    cells = initial.cells + (
        GenericCell(34, 4, ValueRef.local_id(4)),
    )
    active_slots = (0, 1, 32, 33, 34)
    edges = tuple(
        GenericEdge(relation, source, target)
        for relation in range(16)
        for source in active_slots
        for target in active_slots
    )[:257]
    packet = replace(initial, cells=cells, edges=edges)
    rectangle = replace(
        rectangle,
        worlds=(
            replace(rectangle.worlds[0], initial_packet=packet),
            rectangle.worlds[1],
        ),
    )

    with pytest.raises(MaterializationError, match="256-edge capacity"):
        materialize_ettr_il_v2(_request(rectangle), TOKENIZER)


def test_exact_world_width_is_fail_closed_not_truncated() -> None:
    rectangle = _rectangle("wide")
    rectangle = replace(
        rectangle,
        worlds=(
            replace(
                rectangle.worlds[0],
                sources=(
                    b"x" * (WORLD_WIDTH + 1),
                    rectangle.worlds[0].sources[1],
                ),
            ),
            rectangle.worlds[1],
        ),
    )

    with pytest.raises(MaterializationError, match="192-token width"):
        materialize_ettr_il_v2(_request(rectangle), TOKENIZER)


def test_noncanonical_generic_mutation_operands_fail_closed() -> None:
    rectangle = _rectangle("operands")
    operation = rectangle.corners[0][0].operation_traces[0]
    mutation = replace(operation.mutations[0], target=1)
    corner = replace(
        rectangle.corners[0][0],
        operation_traces=(
            replace(operation, mutations=(mutation,)),
        ),
    )
    rectangle = replace(
        rectangle,
        corners=(
            (corner, rectangle.corners[0][1]),
            rectangle.corners[1],
        ),
    )

    with pytest.raises(MaterializationError, match="noncanonical unused operands"):
        materialize_ettr_il_v2(_request(rectangle), TOKENIZER)
