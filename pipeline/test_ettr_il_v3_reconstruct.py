from __future__ import annotations

import pytest

from ettr_il_v3_horn_resource import (
    CurriculumStage,
    generate_horn_episodes,
    generate_resource_episodes,
)
from ettr_il_v3_production import ProductionCell, _candidate_row
from ettr_il_v3_reconstruct import (
    ReconstructionError,
    reconstruct_candidate,
)
from ettr_il_v3_rewrite_episodes import generate_rewrite_episodes


def _cell(family: str, depth: int) -> ProductionCell:
    return ProductionCell(
        index=0,
        split="train",
        family=family,
        stage="atomic_transactions",
        depth=depth,
        selected_quota=1,
        candidate_target=1,
        owner_skip=0,
    )


@pytest.mark.parametrize("family", ("horn", "resource", "local_rewrite"))
def test_reconstructs_and_replays_generated_candidate(family: str) -> None:
    stage = CurriculumStage.ATOMIC_TRANSITIONS
    if family == "horn":
        episode = generate_horn_episodes(
            stage=stage,
            theory_index=0,
            limit=1,
        )[0]
    elif family == "resource":
        episode = generate_resource_episodes(
            stage=stage,
            theory_index=0,
            limit=1,
        )[0]
    else:
        episode = generate_rewrite_episodes(
            stage=stage,
            theory_index=0,
            limit=1,
        )[0]
    row = _candidate_row(_cell(family, 1), episode, ordinal=0)
    rebuilt = reconstruct_candidate(row)
    assert rebuilt.episode == episode
    assert rebuilt.episode_id == episode.episode_id
    assert rebuilt.family == family


def test_reconstruction_rejects_tampered_semantics_and_identity() -> None:
    episode = generate_resource_episodes(
        stage=CurriculumStage.ATOMIC_TRANSITIONS,
        theory_index=0,
        limit=1,
    )[0]
    row = _candidate_row(_cell("resource", 1), episode, ordinal=0)
    broken = dict(row)
    broken["episode_id"] = "0" * 64
    with pytest.raises(ReconstructionError, match="identity differs"):
        reconstruct_candidate(broken)

    broken_cell = dict(row["cell"])
    broken_cell["depth"] = 2
    broken = dict(row)
    broken["cell"] = broken_cell
    with pytest.raises(ReconstructionError, match="depth binding differs"):
        reconstruct_candidate(broken)


def test_reconstruction_rejects_schema_drift() -> None:
    episode = generate_horn_episodes(
        stage=CurriculumStage.ATOMIC_TRANSITIONS,
        theory_index=0,
        limit=1,
    )[0]
    row = _candidate_row(_cell("horn", 1), episode, ordinal=0)
    with pytest.raises(ReconstructionError, match="fields differ"):
        reconstruct_candidate({**row, "unexpected": True})

    with pytest.raises(ReconstructionError, match="protocol or schema"):
        reconstruct_candidate({**row, "schema": "wrong"})
