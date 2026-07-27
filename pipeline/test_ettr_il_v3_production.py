"""Tests for ETTR-IL-v3 production-cell generation."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from ettr_il_v3_production import (
    ALL_PRODUCTION_SPLITS,
    ProductionCell,
    base_owner_split,
    generate_production_cell,
    production_cells,
    stage_depths,
    write_production_cell,
)
from ettr_il_v3_protocol import (
    CURRICULUM_STAGES,
    FAMILIES,
    candidate_floor,
    split_stage_family_allocation,
)


COMMIT = "a" * 40
FREEZE = "b" * 64


def test_production_matrix_has_exact_frozen_candidate_marginals() -> None:
    cells = production_cells()
    assert len(cells) == 324
    assert tuple(cell.index for cell in cells) == tuple(range(324))
    for split in ALL_PRODUCTION_SPLITS:
        for family in FAMILIES:
            for stage in CURRICULUM_STAGES:
                matching = tuple(
                    cell
                    for cell in cells
                    if cell.split == split
                    and cell.family == family
                    and cell.stage == stage
                )
                quota = split_stage_family_allocation(split)[stage][family]
                assert {cell.depth for cell in matching} == set(
                    stage_depths(stage)
                )
                assert sum(cell.candidate_target for cell in matching) == (
                    candidate_floor(quota)
                )


def test_reserve_segments_skip_the_primary_candidate_segment() -> None:
    cells = production_cells()
    for reserve in (
        "train_reserve",
        "development_reserve",
        "confirmation_reserve",
    ):
        primary = base_owner_split(reserve)
        for reserve_cell in (cell for cell in cells if cell.split == reserve):
            primary_cell = next(
                cell
                for cell in cells
                if cell.split == primary
                and cell.family == reserve_cell.family
                and cell.stage == reserve_cell.stage
                and cell.depth == reserve_cell.depth
            )
            assert reserve_cell.owner_skip == primary_cell.candidate_target


def test_small_rewrite_cell_is_deterministic_and_writable(
    tmp_path: Path,
) -> None:
    cell = ProductionCell(
        index=0,
        split="train",
        family="local_rewrite",
        stage="atomic_transactions",
        depth=1,
        selected_quota=1,
        candidate_target=2,
        owner_skip=0,
    )
    first = generate_production_cell(cell, beam_width=16)
    second = generate_production_cell(cell, beam_width=16)
    assert tuple(item.episode_id for item in first) == tuple(
        item.episode_id for item in second
    )
    shard = tmp_path / "cell.jsonl.gz"
    report_path = tmp_path / "cell.report.json"
    report = write_production_cell(
        cell,
        first,
        source_commit=COMMIT,
        protocol_freeze_sha256=FREEZE,
        shard_path=shard,
        report_path=report_path,
    )
    assert report["row_count"] == 2
    rows = gzip.decompress(shard.read_bytes()).splitlines()
    assert len(rows) == 2
    assert all(json.loads(row)["owner"] == "train" for row in rows)
