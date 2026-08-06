#!/usr/bin/env python3
"""Materialize source-separated DIVERGE-PL1 episode boards."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

from diverge_pl1_data import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEED,
    OP_NAMES,
    PRIME,
    SCHEMA,
    TRAIN_SEED,
    Episode,
    build_split,
    iter_program_identities,
)


REPORT_SCHEMA = "shohin-diverge-pl1-data-report-v1"


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


def _write_split(output: Path, name: str, episodes: tuple[Episode, ...]) -> dict[str, object]:
    public_path = output / f"{name}_public.jsonl"
    assessor_path = output / f"{name}_assessor.jsonl"
    _atomic_jsonl(public_path, (episode.public_record() for episode in episodes))
    _atomic_jsonl(assessor_path, (episode.assessor_record() for episode in episodes))
    depths = Counter(
        len(program.symbols)
        for episode in episodes
        for program in (*episode.acquisition, *episode.transfer)
    )
    return {
        "episodes": len(episodes),
        "acquisition_programs": sum(len(episode.acquisition) for episode in episodes),
        "transfer_programs": sum(len(episode.transfer) for episode in episodes),
        "depth_counts": {str(key): value for key, value in sorted(depths.items())},
        "public_path": str(public_path),
        "public_sha256": sha256_path(public_path),
        "assessor_path": str(assessor_path),
        "assessor_sha256": sha256_path(assessor_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=2048)
    parser.add_argument("--development-count", type=int, default=256)
    parser.add_argument("--confirmation-count", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing PL1 output: {args.output}")
    if min(args.train_count, args.development_count, args.confirmation_count) <= 0:
        raise SystemExit("all split counts must be positive")

    train = build_split(split="train", seed=TRAIN_SEED, count=args.train_count)
    development = build_split(
        split="development",
        seed=DEVELOPMENT_SEED,
        count=args.development_count,
    )
    confirmations = {
        f"confirmation_seed_{seed}": build_split(
            split=f"confirmation_seed_{seed}",
            seed=seed,
            count=args.confirmation_count,
        )
        for seed in CONFIRMATION_SEEDS
    }
    groups = {"train": train, "development": development, **confirmations}

    alias_sets = {
        name: {alias for episode in episodes for alias in episode.aliases}
        for name, episodes in groups.items()
    }
    identity_sets = {
        name: set(iter_program_identities(episodes)) for name, episodes in groups.items()
    }
    overlap: dict[str, int] = {}
    names = tuple(groups)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap[f"aliases:{left}:{right}"] = len(alias_sets[left] & alias_sets[right])
            overlap[f"programs:{left}:{right}"] = len(
                identity_sets[left] & identity_sets[right]
            )
    if any(overlap.values()):
        raise SystemExit(f"PL1 split overlap: {overlap}")

    args.output.mkdir(parents=True)
    split_reports = {
        name: _write_split(args.output, name, episodes)
        for name, episodes in groups.items()
    }
    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "board_schema": SCHEMA,
        "prime": PRIME,
        "operations": list(OP_NAMES),
        "train_seed": TRAIN_SEED,
        "development_seed": DEVELOPMENT_SEED,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "split_reports": split_reports,
        "overlap": overlap,
        "candidate_files_contain_hidden_mapping_or_trace": False,
        "assessor_files_contain_hidden_mapping_and_trace": True,
        "model_score_used_for_selection": False,
    }
    report_path = args.output / "report.json"
    _atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report": str(report_path),
                "report_sha256": sha256_path(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
