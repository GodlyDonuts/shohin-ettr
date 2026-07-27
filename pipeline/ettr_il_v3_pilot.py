#!/usr/bin/env python3
"""Run one deterministic CPU pilot cell for the ETTR-IL-v3 initializer."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import time
from typing import Iterable, Sequence

from cross_ontology_horn_board import THEORIES as HORN_THEORIES
from cross_ontology_resource_board import THEORIES as RESOURCE_THEORIES
from ettr_il_v3_horn_resource import (
    CurriculumStage,
    EpisodeRecord,
    generate_horn_episodes,
    generate_resource_episodes,
)
from ettr_il_v3_protocol import FAMILIES, PROTOCOL, canonical_json_bytes
from ettr_il_v3_rewrite import THEORY_COUNT as REWRITE_THEORY_COUNT
from ettr_il_v3_rewrite_episodes import (
    RewriteEpisode,
    generate_rewrite_episodes,
)
from freeze_ettr_il_v3_protocol import load_and_verify_freeze


SCHEMA = "r12-ettr-il-v3-pilot-cell-v1"


class PilotError(ValueError):
    """A pilot cell is malformed or cannot meet its bounded target."""


@dataclass(frozen=True, slots=True)
class PilotCell:
    family: str
    stage: CurriculumStage
    depth: int

    def to_value(self) -> dict[str, object]:
        return {
            "depth": self.depth,
            "family": self.family,
            "stage": self.stage.value,
        }


def pilot_cells() -> tuple[PilotCell, ...]:
    cells: list[PilotCell] = []
    for family in FAMILIES:
        cells.append(
            PilotCell(family, CurriculumStage.COMPILER_GROUNDING, 1)
        )
        cells.append(
            PilotCell(family, CurriculumStage.ATOMIC_TRANSITIONS, 1)
        )
        cells.extend(
            PilotCell(
                family,
                CurriculumStage.DEPENDENT_COMPOSITION,
                depth,
            )
            for depth in range(2, 7)
        )
        cells.extend(
            PilotCell(
                family,
                CurriculumStage.QUERY_COUNTERFACTUAL_GROUNDING,
                depth,
            )
            for depth in range(1, 7)
        )
        cells.extend(
            PilotCell(family, CurriculumStage.CLOSED_LOOP, depth)
            for depth in range(2, 7)
        )
    return tuple(cells)


def _episode_value(episode: EpisodeRecord | RewriteEpisode) -> dict[str, object]:
    return episode.assessor_value()


def _theory_count(family: str) -> int:
    if family == "horn":
        return len(HORN_THEORIES)
    if family == "local_rewrite":
        return REWRITE_THEORY_COUNT
    if family == "resource":
        return len(RESOURCE_THEORIES)
    raise PilotError("pilot family differs")


def _generate_theory(
    cell: PilotCell,
    *,
    theory_index: int,
    limit: int,
    beam_width: int,
    bucket_index: int,
    bucket_count: int,
) -> tuple[EpisodeRecord | RewriteEpisode, ...]:
    common = {
        "stage": cell.stage,
        "theory_index": theory_index,
        "depth": cell.depth,
        "limit": limit,
        "beam_width": beam_width,
        "bucket_index": bucket_index,
        "bucket_count": bucket_count,
    }
    if cell.family == "horn":
        return generate_horn_episodes(**common)
    if cell.family == "local_rewrite":
        return generate_rewrite_episodes(**common)
    if cell.family == "resource":
        return generate_resource_episodes(**common)
    raise PilotError("pilot family differs")


def generate_cell(
    cell: PilotCell,
    *,
    limit: int,
    beam_width: int,
    bucket_index: int,
    bucket_count: int,
) -> tuple[EpisodeRecord | RewriteEpisode, ...]:
    if type(limit) is not int or limit < 1:
        raise PilotError("pilot limit differs")
    if type(beam_width) is not int or beam_width < 1:
        raise PilotError("pilot beam width differs")
    if (
        type(bucket_count) is not int
        or bucket_count < 1
        or type(bucket_index) is not int
        or not 0 <= bucket_index < bucket_count
    ):
        raise PilotError("pilot bucket differs")
    records: dict[str, EpisodeRecord | RewriteEpisode] = {}
    theory_indices = sorted(
        range(_theory_count(cell.family)),
        key=lambda theory_index: (
            hashlib.sha256(
                canonical_json_bytes(
                    {
                        "cell": cell.to_value(),
                        "protocol": PROTOCOL,
                        "theory_index": theory_index,
                    }
                )
            ).digest(),
            theory_index,
        ),
    )
    while len(records) < limit:
        previous_count = len(records)
        for theory_index in theory_indices:
            remaining = limit - len(records)
            generated = _generate_theory(
                cell,
                theory_index=theory_index,
                limit=remaining,
                beam_width=beam_width,
                bucket_index=bucket_index,
                bucket_count=bucket_count,
            )
            for episode in generated:
                records.setdefault(episode.episode_id, episode)
            if len(records) >= limit:
                break
        if len(records) == previous_count:
            break
    return tuple(records.values())


def _outcome_counts(
    episodes: Iterable[EpisodeRecord | RewriteEpisode],
) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                episode.primary.disposition.value for episode in episodes
            ).items()
        )
    )


def build_report(
    cell: PilotCell,
    *,
    limit: int,
    beam_width: int,
    bucket_index: int,
    bucket_count: int,
    source_commit: str,
    protocol_freeze_sha256: str,
) -> dict[str, object]:
    if (
        len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise PilotError("source commit differs")
    if (
        len(protocol_freeze_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in protocol_freeze_sha256
        )
    ):
        raise PilotError("protocol freeze SHA-256 differs")
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    episodes = generate_cell(
        cell,
        limit=limit,
        beam_width=beam_width,
        bucket_index=bucket_index,
        bucket_count=bucket_count,
    )
    cpu_seconds = time.process_time() - started_cpu
    wall_seconds = time.perf_counter() - started_wall
    if len(episodes) != limit:
        raise PilotError(
            f"pilot cell exhausted at {len(episodes)} of {limit} episodes"
        )
    identifiers = tuple(episode.episode_id for episode in episodes)
    if len(set(identifiers)) != len(identifiers):
        raise PilotError("pilot generated duplicate episode IDs")
    rows = tuple(canonical_json_bytes(_episode_value(episode)) for episode in episodes)
    uncompressed_bytes = sum(len(row) for row in rows)
    compressed_bytes = len(gzip.compress(b"".join(rows), mtime=0))
    primary_replay_mismatches = sum(
        episode.primary != episode.replay for episode in episodes
    )
    if primary_replay_mismatches:
        raise PilotError("pilot contains primary/replay mismatches")
    answer_counts = Counter(
        str(answer).lower()
        for episode in episodes
        for answer in episode.answers
    )
    report: dict[str, object] = {
        "beam_width": beam_width,
        "bucket_count": bucket_count,
        "bucket_index": bucket_index,
        "cell": cell.to_value(),
        "compressed_bytes": compressed_bytes,
        "compressed_bytes_per_core": compressed_bytes / len(episodes),
        "cpu_seconds": cpu_seconds,
        "cores": len(episodes),
        "cores_per_cpu_second": (
            len(episodes) / cpu_seconds if cpu_seconds > 0 else None
        ),
        "episode_population_sha256": hashlib.sha256(
            canonical_json_bytes(sorted(identifiers))
        ).hexdigest(),
        "outcome_counts": _outcome_counts(episodes),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "primary_replay_mismatches": primary_replay_mismatches,
        "protocol": PROTOCOL,
        "protocol_freeze_sha256": protocol_freeze_sha256,
        "query_answer_counts": dict(sorted(answer_counts.items())),
        "schema": SCHEMA,
        "source_commit": source_commit,
        "status": "pass",
        "uncompressed_bytes": uncompressed_bytes,
        "wall_seconds": wall_seconds,
    }
    report["report_sha256"] = hashlib.sha256(
        canonical_json_bytes(report)
    ).hexdigest()
    return report


def build_verified_report(
    cell: PilotCell,
    *,
    limit: int,
    beam_width: int,
    bucket_index: int,
    bucket_count: int,
    source_root: Path,
    source_commit: str,
    protocol_freeze: Path,
) -> dict[str, object]:
    freeze = load_and_verify_freeze(
        source_root,
        protocol_freeze,
        source_commit=source_commit,
    )
    return build_report(
        cell,
        limit=limit,
        beam_width=beam_width,
        bucket_index=bucket_index,
        bucket_count=bucket_count,
        source_commit=source_commit,
        protocol_freeze_sha256=str(freeze["freeze_sha256"]),
    )


def write_no_replace(path: Path, payload: bytes) -> None:
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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-index", type=int, required=True)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--bucket-index", type=int, default=0)
    parser.add_argument("--bucket-count", type=int, default=1)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--protocol-freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cells = pilot_cells()
    if not 0 <= args.matrix_index < len(cells):
        raise PilotError("matrix index differs")
    report = build_verified_report(
        cells[args.matrix_index],
        limit=args.limit,
        beam_width=args.beam_width,
        bucket_index=args.bucket_index,
        bucket_count=args.bucket_count,
        source_root=args.source_root,
        source_commit=args.source_commit,
        protocol_freeze=args.protocol_freeze,
    )
    write_no_replace(args.out, canonical_json_bytes(report))
    print(
        json.dumps(
            {
                "cores": report["cores"],
                "out": str(args.out),
                "report_sha256": report["report_sha256"],
                "status": report["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
