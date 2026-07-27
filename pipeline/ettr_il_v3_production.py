#!/usr/bin/env python3
"""Generate one no-replace ETTR-IL-v3 semantic candidate production cell."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from ettr_il_v2_candidate_search import semantic_world_value
from ettr_il_v3_horn_resource import CurriculumStage, EpisodeRecord
from ettr_il_v3_pilot import (
    PilotCell,
    generate_cell,
)
from ettr_il_v3_protocol import (
    CURRICULUM_STAGES,
    FAMILIES,
    PROTOCOL,
    candidate_floor,
    canonical_json_bytes,
    cyclic_balanced_allocation,
    orbit_owner,
    split_stage_family_allocation,
)
from ettr_il_v3_rewrite_episodes import RewriteEpisode
from freeze_ettr_il_v3_protocol import load_and_verify_freeze


SCHEMA = "r12-ettr-il-v3-production-cell-v1"
ROW_SCHEMA = "r12-ettr-il-v3-semantic-candidate-v1"
PRIMARY_SPLITS = ("train", "development", "confirmation")
RESERVE_SPLITS = (
    "train_reserve",
    "development_reserve",
    "confirmation_reserve",
)
ALL_PRODUCTION_SPLITS = PRIMARY_SPLITS + RESERVE_SPLITS


class ProductionError(ValueError):
    """A production cell violates the frozen candidate protocol."""


@dataclass(frozen=True, slots=True)
class ProductionCell:
    index: int
    split: str
    family: str
    stage: str
    depth: int
    selected_quota: int
    candidate_target: int
    owner_skip: int

    def to_value(self) -> dict[str, object]:
        return {
            "candidate_target": self.candidate_target,
            "depth": self.depth,
            "family": self.family,
            "index": self.index,
            "owner_skip": self.owner_skip,
            "selected_quota": self.selected_quota,
            "split": self.split,
            "stage": self.stage,
        }


def stage_depths(stage: str) -> tuple[int, ...]:
    if stage in {"compiler_grounding", "atomic_transactions"}:
        return (1,)
    if stage == "query_counterfactual_grounding":
        return tuple(range(1, 7))
    if stage in {"dependent_composition", "closed_loop_invariance"}:
        return tuple(range(2, 7))
    raise ProductionError("production stage differs")


def base_owner_split(split: str) -> str:
    if split in PRIMARY_SPLITS:
        return split
    if split in RESERVE_SPLITS:
        return split.removesuffix("_reserve")
    raise ProductionError("production split differs")


def production_cells() -> tuple[ProductionCell, ...]:
    provisional: list[dict[str, object]] = []
    for split in ALL_PRODUCTION_SPLITS:
        matrix = split_stage_family_allocation(split)
        for family in FAMILIES:
            for stage in CURRICULUM_STAGES:
                selected_quota = matrix[stage][family]
                candidate_total = candidate_floor(selected_quota)
                depth_allocation = cyclic_balanced_allocation(
                    candidate_total,
                    tuple(str(depth) for depth in stage_depths(stage)),
                    context={
                        "axis": "production-depth",
                        "family": family,
                        "split": split,
                        "stage": stage,
                    },
                )
                for depth in stage_depths(stage):
                    provisional.append(
                        {
                            "candidate_target": depth_allocation[str(depth)],
                            "depth": depth,
                            "family": family,
                            "selected_quota": selected_quota,
                            "split": split,
                            "stage": stage,
                        }
                    )

    cells: list[ProductionCell] = []
    for index, item in enumerate(provisional):
        split = str(item["split"])
        skip = 0
        if split in RESERVE_SPLITS:
            primary = base_owner_split(split)
            skip = next(
                int(candidate["candidate_target"])
                for candidate in provisional
                if candidate["split"] == primary
                and candidate["family"] == item["family"]
                and candidate["stage"] == item["stage"]
                and candidate["depth"] == item["depth"]
            )
        cells.append(
            ProductionCell(
                index=index,
                split=split,
                family=str(item["family"]),
                stage=str(item["stage"]),
                depth=int(item["depth"]),
                selected_quota=int(item["selected_quota"]),
                candidate_target=int(item["candidate_target"]),
                owner_skip=skip,
            )
        )
    return tuple(cells)


def _episode_world_value(
    episode: EpisodeRecord | RewriteEpisode,
) -> Mapping[str, object]:
    if isinstance(episode, RewriteEpisode):
        return episode.world.to_value()
    value = semantic_world_value(episode.world)
    if not isinstance(value, Mapping):
        raise ProductionError("semantic world value differs")
    return value


def _episode_owner(
    episode: EpisodeRecord | RewriteEpisode,
    family: str,
) -> str:
    return orbit_owner(
        {
            "family": family,
            "world": dict(_episode_world_value(episode)),
        }
    )


def generate_production_cell(
    cell: ProductionCell,
    *,
    beam_width: int,
) -> tuple[EpisodeRecord | RewriteEpisode, ...]:
    required = cell.owner_skip + cell.candidate_target
    raw_limit = max(256, 4 * required)
    owner = base_owner_split(cell.split)
    selected: tuple[EpisodeRecord | RewriteEpisode, ...] = ()
    for _attempt in range(5):
        episodes = generate_cell(
            PilotCell(
                cell.family,
                CurriculumStage(cell.stage),
                cell.depth,
            ),
            limit=raw_limit,
            beam_width=beam_width,
            bucket_index=0,
            bucket_count=1,
        )
        owned = tuple(
            episode
            for episode in episodes
            if _episode_owner(episode, cell.family) == owner
        )
        if len(owned) >= required:
            selected = owned[
                cell.owner_skip : cell.owner_skip + cell.candidate_target
            ]
            break
        raw_limit *= 2
    if len(selected) != cell.candidate_target:
        raise ProductionError(
            f"production cell exhausted at {len(selected)} of "
            f"{cell.candidate_target} owned candidates"
        )
    identifiers = tuple(episode.episode_id for episode in selected)
    if len(set(identifiers)) != len(identifiers):
        raise ProductionError("production cell repeats an episode ID")
    return selected


def _candidate_row(
    cell: ProductionCell,
    episode: EpisodeRecord | RewriteEpisode,
    *,
    ordinal: int,
) -> dict[str, object]:
    return {
        "cell": cell.to_value(),
        "episode": episode.assessor_value(),
        "episode_id": episode.episode_id,
        "ordinal": ordinal,
        "owner": base_owner_split(cell.split),
        "protocol": PROTOCOL,
        "schema": ROW_SCHEMA,
    }


def _write_no_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def write_production_cell(
    cell: ProductionCell,
    episodes: Sequence[EpisodeRecord | RewriteEpisode],
    *,
    source_commit: str,
    protocol_freeze_sha256: str,
    shard_path: Path,
    report_path: Path,
) -> dict[str, object]:
    rows = tuple(
        canonical_json_bytes(_candidate_row(cell, episode, ordinal=ordinal))
        for ordinal, episode in enumerate(episodes)
    )
    compressed = gzip.compress(b"".join(rows), compresslevel=6, mtime=0)
    _write_no_replace(shard_path, compressed)
    shard_sha256 = hashlib.sha256(compressed).hexdigest()
    report: dict[str, object] = {
        "cell": cell.to_value(),
        "compressed_bytes": len(compressed),
        "protocol": PROTOCOL,
        "protocol_freeze_sha256": protocol_freeze_sha256,
        "row_count": len(rows),
        "schema": SCHEMA,
        "shard_name": shard_path.name,
        "shard_sha256": shard_sha256,
        "source_commit": source_commit,
        "status": "pass",
        "uncompressed_bytes": sum(len(row) for row in rows),
    }
    report["report_sha256"] = hashlib.sha256(
        canonical_json_bytes(report)
    ).hexdigest()
    try:
        _write_no_replace(report_path, canonical_json_bytes(report))
    except BaseException:
        shard_path.chmod(0o400)
        raise
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-index", type=int, required=True)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--protocol-freeze", type=Path, required=True)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cells = production_cells()
    if not 0 <= args.matrix_index < len(cells):
        raise ProductionError("production matrix index differs")
    freeze = load_and_verify_freeze(
        args.source_root,
        args.protocol_freeze,
        source_commit=args.source_commit,
    )
    cell = cells[args.matrix_index]
    episodes = generate_production_cell(
        cell,
        beam_width=args.beam_width,
    )
    report = write_production_cell(
        cell,
        episodes,
        source_commit=args.source_commit,
        protocol_freeze_sha256=str(freeze["freeze_sha256"]),
        shard_path=args.shard,
        report_path=args.report,
    )
    print(
        json.dumps(
            {
                "matrix_index": cell.index,
                "report_sha256": report["report_sha256"],
                "row_count": report["row_count"],
                "shard_sha256": report["shard_sha256"],
                "status": report["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
