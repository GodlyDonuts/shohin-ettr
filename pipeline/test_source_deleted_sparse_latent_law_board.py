from __future__ import annotations

from dataclasses import asdict

import pytest

from source_deleted_sparse_latent_law_board import (
    FAMILIES,
    SealedSparseLawMachine,
    SparseLatentLawBoardError,
    build_frozen_board,
    compile_source,
    decode_query,
    execute_query,
    generate_episode,
)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("renderer", range(6))
def test_sparse_law_compiles_hidden_query_exactly(
    family: str,
    renderer: int,
) -> None:
    row = generate_episode(
        seed=811,
        split="development",
        family=family,
        renderer=renderer,
        cell="joint",
        cardinality=16,
        action_count=4,
    )
    source = row.candidate.source
    machine = compile_source(source)
    del source
    assert execute_query(machine, row.candidate.query) == row.supervisor.answer
    assert (
        SealedSparseLawMachine.from_deployed_wire(
            machine.deployed_wire()
        )
        == machine
    )
    state, actions = decode_query(machine, row.candidate.query)
    for action in actions:
        assert state not in machine.visible_inputs[action]
        state = machine.transition[action][state]


def test_every_sparse_record_is_an_identifying_witness() -> None:
    row = generate_episode(
        seed=812,
        split="development",
        family="gray_conjugate_affine",
        renderer=5,
        cell="renderer",
        cardinality=8,
        action_count=3,
    )
    lines = row.candidate.source.splitlines()
    assert len(lines) - 1 <= row.supervisor.complete_records // 2
    compile_source(row.candidate.source)
    for record_index in range(1, len(lines)):
        reduced = "\n".join(
            line
            for index, line in enumerate(lines)
            if index != record_index
        )
        with pytest.raises(
            SparseLatentLawBoardError,
            match="not uniquely identifiable",
        ):
            compile_source(reduced)


def test_renderer_orbit_seals_identical_sparse_packet() -> None:
    rows = [
        generate_episode(
            seed=813,
            split="development",
            family="bitwise_rotate_xor",
            renderer=renderer,
            cell="renderer",
            cardinality=16,
            action_count=3,
        )
        for renderer in range(6)
    ]
    machines = [
        compile_source(row.candidate.source)
        for row in rows
    ]
    assert len({machine.packet_sha256 for machine in machines}) == 1
    assert len({asdict(row.supervisor)["answer"] for row in rows}) == 1


def test_frozen_sparse_board_has_unique_laws_and_no_family_leaks() -> None:
    board = build_frozen_board(
        seed=20260725,
        train_per_renderer=1,
        development_per_cell=1,
    )
    assert len(board) == 30
    assert len(
        {row.supervisor.law_sha256 for row in board}
    ) == len(board)
    for row in board:
        payload = row.candidate.source + row.candidate.query
        assert all(family not in payload for family in FAMILIES)
        assert (
            row.supervisor.visible_records * 2
            <= row.supervisor.complete_records
        )
    train_maps = {
        digest
        for row in board
        if row.supervisor.split == "train"
        for digest in row.supervisor.action_law_sha256
    }
    development_maps = {
        digest
        for row in board
        if row.supervisor.split == "development"
        for digest in row.supervisor.action_law_sha256
    }
    assert train_maps.isdisjoint(development_maps)
