#!/usr/bin/env python3
"""Build source-disjoint conditional confirmation data for DIVERGE-NPL2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from build_diverge_npl1_development import sha256_path
from diverge_npl1_data import (
    natural_assessor_record,
    natural_program_identities,
    natural_public_record,
    operation_aliases,
)
from diverge_pl1_data import (
    Episode,
    build_split,
    episode_from_assessor_record,
    iter_program_identities,
)


SCHEMA = "shohin-diverge-npl2-confirmation-report-v1"
CONFIRMATION_SEEDS = (
    2026080911,
    2026080912,
    2026080913,
    2026080914,
    2026080915,
)
EPISODES_PER_SEED = 256


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_prior(
    paths: Iterable[Path],
) -> tuple[tuple[Episode, ...], set[str], set[str], set[str]]:
    episodes: list[Episode] = []
    aliases: set[str] = set()
    programs: set[str] = set()
    ids: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if "oracle" in row:
                    episode = episode_from_assessor_record(row["oracle"])
                    public = row["public"]
                    aliases.update(str(value) for value in public["aliases"])
                    programs.update(natural_program_identities(episode))
                else:
                    episode = episode_from_assessor_record(row)
                    aliases.update(episode.aliases)
                    programs.update(iter_program_identities((episode,)))
                episodes.append(episode)
                ids.add(episode.episode_id)
    return tuple(episodes), aliases, programs, ids


def _aliases(episodes: Iterable[Episode]) -> set[str]:
    return {alias for episode in episodes for alias in operation_aliases(episode)}


def _ids(episodes: Iterable[Episode]) -> set[str]:
    return {episode.episode_id for episode in episodes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-assessor", type=Path, action="append", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing existing NPL2 confirmation output")
    prior, prior_aliases, prior_programs, prior_ids = _load_prior(
        args.prior_assessor
    )

    splits = {
        seed: build_split(
            split=f"npl2_confirmation_{seed}",
            seed=seed,
            count=EPISODES_PER_SEED,
        )
        for seed in CONFIRMATION_SEEDS
    }
    seen_aliases = set(prior_aliases)
    seen_ids = set(prior_ids)
    seen_programs = set(prior_programs)
    split_reports: dict[str, object] = {}
    args.output.mkdir(parents=True)
    for seed, episodes in splits.items():
        aliases = _aliases(episodes)
        ids = _ids(episodes)
        programs = {
            identity
            for episode in episodes
            for identity in natural_program_identities(episode)
        }
        overlap = {
            "aliases_with_prior_or_other_seed": len(aliases & seen_aliases),
            "episodes_with_prior_or_other_seed": len(ids & seen_ids),
            "programs_with_prior_or_other_seed": len(programs & seen_programs),
        }
        if any(overlap.values()):
            raise SystemExit(f"NPL2 confirmation overlap for {seed}: {overlap}")
        public_path = args.output / f"confirmation_seed_{seed}_public.jsonl"
        assessor_path = args.output / f"confirmation_seed_{seed}_assessor.jsonl"
        _atomic_jsonl(
            public_path, (natural_public_record(episode) for episode in episodes)
        )
        _atomic_jsonl(
            assessor_path,
            (natural_assessor_record(episode) for episode in episodes),
        )
        split_reports[str(seed)] = {
            "episodes": len(episodes),
            "public": public_path.name,
            "public_sha256": sha256_path(public_path),
            "assessor": assessor_path.name,
            "assessor_sha256": sha256_path(assessor_path),
            "overlap": overlap,
        }
        seen_aliases.update(aliases)
        seen_ids.update(ids)
        seen_programs.update(programs)

    report: dict[str, object] = {
        "schema": SCHEMA,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "episodes_per_seed": EPISODES_PER_SEED,
        "total_episodes": EPISODES_PER_SEED * len(CONFIRMATION_SEEDS),
        "split_reports": split_reports,
        "prior_assessor_sha256": {
            str(path): sha256_path(path) for path in args.prior_assessor
        },
        "candidate_contains_hidden_mapping_trace_or_terminal": False,
        "assessor_contains_hidden_mapping_trace_and_terminal": True,
        "generated_before_npl2_model_result": True,
        "model_score_used_for_selection": False,
    }
    report_path = args.output / "report.json"
    _atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report_sha256": sha256_path(report_path),
                "total_episodes": report["total_episodes"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
