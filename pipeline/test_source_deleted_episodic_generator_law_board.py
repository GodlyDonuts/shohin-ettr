from __future__ import annotations

from dataclasses import asdict

import pytest

from source_deleted_episodic_generator_law_board import (
    DEVELOPMENT_CELLS,
    FAMILIES,
    HELD_OUT_FAMILY,
    TRAIN_FAMILIES,
    EpisodicGeneratorLawBoardError,
    SealedEpisodicGeneratorMachine,
    build_episode_closure,
    build_frozen_board,
    compile_source,
    compose_support_word,
    decode_query,
    execute_query,
    generate_episode,
)


@pytest.mark.parametrize("family", FAMILIES)
def test_episode_local_closure_executes_hidden_targets(
    family: str,
) -> None:
    cell = (
        "joint"
        if family == HELD_OUT_FAMILY
        else "composition"
    )
    row = generate_episode(
        seed=811,
        split="development",
        family=family,
        renderer=5,
        cell=cell,
        cardinality=16,
    )
    source = row.candidate.source
    machine = compile_source(source)
    source_bytes = source.encode("ascii")
    del source
    assert (
        execute_query(machine, row.candidate.query)
        == row.supervisor.answer
    )
    assert source_bytes not in machine.deployed_wire()
    assert (
        SealedEpisodicGeneratorMachine.from_deployed_wire(
            machine.deployed_wire()
        )
        == machine
    )
    closure = {
        entry.transition: entry.word
        for entry in build_episode_closure(
            row.supervisor.support_transition
        )
    }
    for target, word in zip(
        row.supervisor.target_transition,
        row.supervisor.target_composition_words,
        strict=True,
    ):
        assert compose_support_word(
            row.supervisor.support_transition,
            word,
        ) == target
        assert closure[target] == word
    state, actions = decode_query(
        machine,
        row.candidate.query,
    )
    for action in actions:
        assert state not in machine.visible_inputs[action]
        state = machine.transition[action][state]


def test_every_sparse_target_record_is_necessary() -> None:
    row = generate_episode(
        seed=812,
        split="development",
        family=HELD_OUT_FAMILY,
        renderer=4,
        cell="joint",
        cardinality=16,
    )
    machine = compile_source(row.candidate.source)
    lines = row.candidate.source.splitlines()
    target_record_indexes = [
        index
        for index, line in enumerate(lines)
        if any(
            target_key in line
            for target_key in machine.target_keys
        )
    ]
    assert len(target_record_indexes) == (
        row.supervisor.target_visible_records
    )
    assert len(target_record_indexes) < (
        row.supervisor.target_complete_records
    )
    for record_index in target_record_indexes:
        reduced = "\n".join(
            line
            for index, line in enumerate(lines)
            if index != record_index
        )
        with pytest.raises(
            EpisodicGeneratorLawBoardError
        ):
            compile_source(reduced)


@pytest.mark.parametrize("family", FAMILIES)
def test_six_renderer_orbit_seals_identical_packet(
    family: str,
) -> None:
    cell = (
        "joint"
        if family == HELD_OUT_FAMILY
        else "composition"
    )
    rows = [
        generate_episode(
            seed=813,
            split="development",
            family=family,
            renderer=renderer,
            cell=cell,
            cardinality=8,
        )
        for renderer in range(6)
    ]
    machines = [
        compile_source(row.candidate.source)
        for row in rows
    ]
    assert len(
        {machine.packet_sha256 for machine in machines}
    ) == 1
    assert len(
        {
            row.supervisor.law_sha256
            for row in rows
        }
    ) == 1
    assert all(
        execute_query(machine, row.candidate.query)
        == row.supervisor.answer
        for machine, row in zip(
            machines,
            rows,
            strict=True,
        )
    )


def test_frozen_board_has_factorized_cells_and_disjoint_laws() -> None:
    board = build_frozen_board(seed=20260725)
    assert len(board) == 26
    assert build_frozen_board(seed=20260725) == board
    train = [
        row
        for row in board
        if row.supervisor.split == "train"
    ]
    development = [
        row
        for row in board
        if row.supervisor.split == "development"
    ]
    assert {
        row.supervisor.family for row in train
    } == set(TRAIN_FAMILIES)
    assert all(
        row.supervisor.family != HELD_OUT_FAMILY
        for row in train
    )
    assert {
        row.supervisor.cell for row in development
    } == set(DEVELOPMENT_CELLS)
    held_out_rows = [
        row
        for row in development
        if row.supervisor.family == HELD_OUT_FAMILY
    ]
    assert {
        row.supervisor.cell for row in held_out_rows
    } == {"law", "joint"}
    target_laws = [
        digest
        for row in board
        for digest in row.supervisor.target_law_sha256
    ]
    target_maps = [
        digest
        for row in board
        for digest in row.supervisor.target_map_sha256
    ]
    assert len(set(target_laws)) == 2 * len(board)
    assert len(set(target_maps)) == 2 * len(board)
    train_laws = {
        digest
        for row in train
        for digest in row.supervisor.target_law_sha256
    }
    development_laws = {
        digest
        for row in development
        for digest in row.supervisor.target_law_sha256
    }
    assert train_laws.isdisjoint(development_laws)
    for row in board:
        payload = (
            row.candidate.source
            + row.candidate.query
        )
        assert all(
            family not in payload
            for family in FAMILIES
        )
        assert asdict(row.supervisor)["max_closure_depth"] == 6
