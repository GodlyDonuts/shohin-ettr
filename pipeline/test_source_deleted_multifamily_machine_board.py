from __future__ import annotations

from dataclasses import replace

import pytest

from source_deleted_multifamily_machine_board import (
    FAMILIES,
    HELD_OUT_RENDERER,
    MultiFamilyBoardError,
    TRAIN_RENDERERS,
    build_frozen_board,
    compile_source,
    execute_late_query,
    family_holdout_folds,
    generate_episode,
)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("renderer", range(4))
def test_exact_compilation_and_late_execution(family: str, renderer: int) -> None:
    episode = generate_episode(
        seed=101,
        split="development",
        family=family,
        renderer=renderer,
        cell="renderer" if renderer == HELD_OUT_RENDERER else "law",
    )
    machine = compile_source(episode.candidate.source)
    assert execute_late_query(machine, episode.candidate.query) == (
        episode.supervisor.answer
    )


def test_source_mutation_after_sealing_is_inert() -> None:
    episode = generate_episode(
        seed=202,
        split="development",
        family="permutation",
        renderer=HELD_OUT_RENDERER,
        cell="joint",
    )
    machine = compile_source(episode.candidate.source)
    expected = execute_late_query(machine, episode.candidate.query)
    mutated = replace(episode.candidate, source="destroyed after sealing")
    assert mutated.source != episode.candidate.source
    assert execute_late_query(machine, mutated.query) == expected


def test_renderer_orbit_preserves_law_query_and_answer() -> None:
    orbit = [
        generate_episode(
            seed=303,
            split="development",
        family="affine_modular",
            renderer=renderer,
            cell="renderer",
        )
        for renderer in range(4)
    ]
    assert len({row.supervisor.law_sha256 for row in orbit}) == 1
    assert len({row.supervisor.answer for row in orbit}) == 1
    assert len({row.candidate.source for row in orbit}) == 4
    for row in orbit:
        machine = compile_source(row.candidate.source)
        assert execute_late_query(machine, row.candidate.query) == (
            row.supervisor.answer
        )


def test_frozen_board_split_contract() -> None:
    board = build_frozen_board(
        seed=404,
        train_per_renderer=2,
        development_per_cell=2,
    )
    train = [row for row in board if row.supervisor.split == "train"]
    development = [
        row for row in board if row.supervisor.split == "development"
    ]
    assert {row.supervisor.renderer for row in train} == set(TRAIN_RENDERERS)
    assert all(row.supervisor.composition_length <= 4 for row in train)
    assert all(
        row.supervisor.composition_length >= 5
        for row in development
        if row.supervisor.cell in {"composition", "joint"}
    )
    assert all(
        row.supervisor.renderer == HELD_OUT_RENDERER
        for row in development
        if row.supervisor.cell in {"renderer", "joint"}
    )
    assert {
        row.supervisor.law_sha256 for row in train
    }.isdisjoint({row.supervisor.law_sha256 for row in development})
    assert len({row.supervisor.source_sha256 for row in board}) == len(board)
    assert len({row.supervisor.law_sha256 for row in board}) == len(board)
    assert len({row.supervisor.episode_seed for row in board}) == len(board)


def test_family_holdout_folds_are_complete() -> None:
    folds = family_holdout_folds()
    assert {fold["held_out_family"] for fold in folds} == set(FAMILIES)
    for fold in folds:
        held_out = fold["held_out_family"]
        fit = fold["fit_families"]
        assert held_out not in fit
        assert set(fit) | {held_out} == set(FAMILIES)


def test_fail_closed_contracts() -> None:
    with pytest.raises(MultiFamilyBoardError, match="confirmation"):
        generate_episode(
            seed=1,
            split="confirmation",
            family="affine_modular",
            renderer=0,
            cell="fit",
        )
    with pytest.raises(MultiFamilyBoardError, match="action count"):
        compile_source("from=s0; action=a0; to=s0")
