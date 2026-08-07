#!/usr/bin/env python3
"""Build only the conditional NPL1 development board and overlap audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

from diverge_npl1_data import (
    DEVELOPMENT_COUNT,
    DEVELOPMENT_SEED,
    SCHEMA,
    natural_assessor_record,
    natural_public_record,
)
from diverge_pl1_data import (
    Episode,
    build_split,
    episode_from_assessor_record,
    iter_program_identities,
)


REPORT_SCHEMA = "shohin-diverge-npl1-development-report-v1"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_prior(paths: list[Path]) -> tuple[Episode, ...]:
    episodes = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            episodes.extend(
                episode_from_assessor_record(json.loads(line)) for line in handle
            )
    return tuple(episodes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-assessor", type=Path, action="append", required=True)
    parser.add_argument("--development-count", type=int, default=DEVELOPMENT_COUNT)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing NPL1 development output: {args.output}")
    if args.development_count <= 0:
        raise SystemExit("NPL1 development count must be positive")

    episodes = build_split(
        split="npl1_development",
        seed=DEVELOPMENT_SEED,
        count=args.development_count,
    )
    prior = _load_prior(args.prior_assessor)
    aliases = {alias for episode in episodes for alias in episode.aliases}
    prior_aliases = {alias for episode in prior for alias in episode.aliases}
    programs = set(iter_program_identities(episodes))
    prior_programs = set(iter_program_identities(prior))
    overlap = {
        "aliases_with_pl1": len(aliases & prior_aliases),
        "programs_with_pl1": len(programs & prior_programs),
        "episode_ids_with_pl1": len(
            {episode.episode_id for episode in episodes}
            & {episode.episode_id for episode in prior}
        ),
    }
    if any(overlap.values()):
        raise SystemExit(f"NPL1 development overlaps PL1: {overlap}")

    public_rows = tuple(natural_public_record(episode) for episode in episodes)
    assessor_rows = tuple(natural_assessor_record(episode) for episode in episodes)
    args.output.mkdir(parents=True)
    public_path = args.output / "development_public.jsonl"
    assessor_path = args.output / "development_assessor.jsonl"
    _atomic_jsonl(public_path, public_rows)
    _atomic_jsonl(assessor_path, assessor_rows)
    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "board_schema": SCHEMA,
        "development_seed": DEVELOPMENT_SEED,
        "episodes": len(episodes),
        "public_path": str(public_path),
        "public_sha256": sha256_path(public_path),
        "assessor_path": str(assessor_path),
        "assessor_sha256": sha256_path(assessor_path),
        "prior_assessor_paths": [str(path) for path in args.prior_assessor],
        "prior_assessor_sha256": {
            str(path): sha256_path(path) for path in args.prior_assessor
        },
        "overlap": overlap,
        "candidate_contains_hidden_mapping_trace_or_terminal": False,
        "assessor_contains_hidden_mapping_trace_and_terminal": True,
        "confirmation_generated": False,
        "model_score_used_for_selection": False,
    }
    report_path = args.output / "report.json"
    _atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report_sha256": sha256_path(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
