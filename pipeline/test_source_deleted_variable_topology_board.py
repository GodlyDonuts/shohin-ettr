from __future__ import annotations

from collections import Counter
import re

import pytest

from source_deleted_variable_topology_board import (
    DEVELOPMENT_CELLS,
    FAMILIES,
    RENDERERS,
    build_frozen_board,
    compile_source,
    execute_query,
    generate_episode,
)


_KEY = re.compile(r"h[0-9a-f]{20}")


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("renderer", range(len(RENDERERS)))
def test_every_renderer_executes_after_source_deletion(
    family: str,
    renderer: int,
) -> None:
    row = generate_episode(
        seed=700,
        split="development",
        family=family,
        renderer=renderer,
        cell="joint",
        cardinality=8,
        action_count=4,
    )
    machine = compile_source(row.candidate.source)
    query = row.candidate.query
    answer = row.supervisor.answer
    del row
    assert execute_query(machine, query) == answer


def test_renderer_orbit_seals_identically() -> None:
    rows = [
        generate_episode(
            seed=701,
            split="development",
            family="permutation",
            renderer=renderer,
            cell="renderer",
            cardinality=16,
            action_count=5,
        )
        for renderer in range(len(RENDERERS))
    ]
    assert len({compile_source(row.candidate.source).packet_sha256 for row in rows}) == 1
    assert len({row.supervisor.answer for row in rows}) == 1


def test_frozen_board_has_matched_cells_and_incidence_collisions() -> None:
    board = build_frozen_board(
        seed=20260725,
        train_per_renderer=4,
        development_per_cell=4,
    )
    assert len(board) == 132
    assert len({row.supervisor.law_sha256 for row in board}) == len(board)
    for family in FAMILIES:
        family_rows = [
            row for row in board if row.supervisor.family == family
        ]
        assert len(family_rows) == 44
        assert {
            row.supervisor.cell
            for row in family_rows
            if row.supervisor.split == "development"
        } == set(DEVELOPMENT_CELLS)
    collision_rows = [
        row for row in board if row.supervisor.incidence_collision
    ]
    assert len(collision_rows) == 24
    for row in collision_rows:
        counts = Counter(_KEY.findall(row.candidate.source))
        assert len(set(counts.values())) == 1
