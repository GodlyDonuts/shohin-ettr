"""Tests for the deterministic ETTR-IL-v3 CPU pilot."""

from __future__ import annotations

from ettr_il_v3_horn_resource import CurriculumStage
from ettr_il_v3_pilot import PilotCell, build_report, generate_cell, pilot_cells


COMMIT = "a" * 40
FREEZE = "b" * 64


def test_matrix_covers_every_family_stage_and_required_depth() -> None:
    cells = pilot_cells()
    assert len(cells) == 54
    for family in ("horn", "local_rewrite", "resource"):
        family_cells = tuple(cell for cell in cells if cell.family == family)
        assert {cell.stage for cell in family_cells} == set(CurriculumStage)
        assert {
            cell.depth
            for cell in family_cells
            if cell.stage is CurriculumStage.DEPENDENT_COMPOSITION
        } == set(range(2, 7))
        assert {
            cell.depth
            for cell in family_cells
            if cell.stage is CurriculumStage.QUERY_COUNTERFACTUAL_GROUNDING
        } == set(range(1, 7))
        assert {
            cell.depth
            for cell in family_cells
            if cell.stage is CurriculumStage.CLOSED_LOOP
        } == set(range(2, 7))


def test_three_family_smoke_cells_generate_exact_limits() -> None:
    cells = (
        PilotCell("horn", CurriculumStage.ATOMIC_TRANSITIONS, 1),
        PilotCell(
            "local_rewrite",
            CurriculumStage.QUERY_COUNTERFACTUAL_GROUNDING,
            1,
        ),
        PilotCell("resource", CurriculumStage.DEPENDENT_COMPOSITION, 2),
    )
    for cell in cells:
        episodes = generate_cell(
            cell,
            limit=2,
            beam_width=16,
            bucket_index=0,
            bucket_count=1,
        )
        assert len(episodes) == 2
        assert len({episode.episode_id for episode in episodes}) == 2


def test_report_is_self_bound_and_measures_resources() -> None:
    report = build_report(
        PilotCell("local_rewrite", CurriculumStage.ATOMIC_TRANSITIONS, 1),
        limit=2,
        beam_width=16,
        bucket_index=0,
        bucket_count=1,
        source_commit=COMMIT,
        protocol_freeze_sha256=FREEZE,
    )
    assert report["status"] == "pass"
    assert report["cores"] == 2
    assert report["primary_replay_mismatches"] == 0
    assert report["compressed_bytes"] > 0
    assert report["uncompressed_bytes"] >= report["compressed_bytes"]
    assert report["cpu_seconds"] >= 0
    assert report["wall_seconds"] >= 0
    assert len(report["report_sha256"]) == 64
