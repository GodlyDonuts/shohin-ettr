from __future__ import annotations

import pytest

from ettr_il_v3_horn_resource import (
    CurriculumStage,
    generate_horn_episodes,
    generate_resource_episodes,
)
from ettr_il_v3_production import ProductionCell, _candidate_row
from ettr_il_v3_reconstruct import reconstruct_candidate
from ettr_il_v3_rectangles import build_causal_rectangle
from ettr_il_v3_rewrite import RewriteExecution
from ettr_il_v3_rewrite_episodes import generate_rewrite_episodes


def _candidate(family: str, stage: CurriculumStage, depth: int):
    if family == "horn":
        episode = generate_horn_episodes(
            stage=stage,
            theory_index=0,
            depth=depth,
            limit=1,
        )[0]
    elif family == "resource":
        episode = generate_resource_episodes(
            stage=stage,
            theory_index=0,
            depth=depth,
            limit=1,
        )[0]
    else:
        episode = generate_rewrite_episodes(
            stage=stage,
            theory_index=0,
            depth=depth,
            limit=1,
        )[0]
    cell = ProductionCell(
        index=0,
        split="train",
        family=family,
        stage=stage.value,
        depth=depth,
        selected_quota=1,
        candidate_target=1,
        owner_skip=0,
    )
    return reconstruct_candidate(_candidate_row(cell, episode, ordinal=0))


@pytest.mark.parametrize("family", ("horn", "resource", "local_rewrite"))
@pytest.mark.parametrize(
    ("stage", "depth"),
    (
        (CurriculumStage.ATOMIC_TRANSITIONS, 1),
        (CurriculumStage.DEPENDENT_COMPOSITION, 2),
    ),
)
def test_broad_rectangle_is_deterministic_and_replayed(
    family: str,
    stage: CurriculumStage,
    depth: int,
) -> None:
    candidate = _candidate(family, stage, depth)
    first = build_causal_rectangle(candidate)
    second = build_causal_rectangle(candidate)
    assert first == second
    assert first.episode_id == candidate.episode_id
    assert first.worlds[0] != first.worlds[1]
    assert first.commands[0] != first.commands[1]
    assert first.primary == first.replay
    assert len(first.semantic_rectangle_id) == 64
    if family == "local_rewrite":
        executions = first.primary
        assert all(
            type(item) is RewriteExecution
            for row in executions
            for item in row
        )
        assert executions[0][0].terminal != executions[1][0].terminal
        assert executions[0][1].terminal != executions[1][1].terminal
