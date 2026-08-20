#!/usr/bin/env python3
"""Measure revision activation and output-horizon geometry on scored ledgers."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any

from dense_public_official_scoring import LEDGER_SCHEMA, SCORE_SCHEMA, load_jsonl

REPORT_SCHEMA = "shohin-dense-public-revision-horizon-analysis-v1"
GENERATION_STAGES = (
    "direct_base",
    "draft",
    "unchanged_continuation",
    "trained_revision",
)
SCORED_STAGES = ("direct_base", "unchanged_continuation", "trained_revision")


class RevisionHorizonError(RuntimeError):
    """Generation or score custody differs from the horizon-analysis contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_generation(
    root: Path, benchmark: str
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    ledgers = {}
    hashes = {}
    expected_ids = None
    for stage in GENERATION_STAGES:
        path = root / benchmark / f"{stage}.jsonl"
        rows = load_jsonl(path)
        identities = [row.get("id") for row in rows]
        if (
            not rows
            or len(set(identities)) != len(rows)
            or any(
                row.get("schema") != LEDGER_SCHEMA
                or row.get("stage") != stage
                or row.get("benchmark") != benchmark
                or not isinstance(row.get("completion"), str)
                or not isinstance(row.get("prompt_sha256"), str)
                or not isinstance(row.get("max_token_exhausted"), bool)
                for row in rows
            )
        ):
            raise RevisionHorizonError(f"{benchmark}:{stage} generation differs")
        if expected_ids is None:
            expected_ids = identities
        elif identities != expected_ids:
            raise RevisionHorizonError(f"{benchmark}:{stage} identity order differs")
        ledgers[stage] = rows
        hashes[stage] = sha256_file(path)
    assert expected_ids is not None
    for unchanged, revision in zip(
        ledgers["unchanged_continuation"], ledgers["trained_revision"], strict=True
    ):
        if unchanged["prompt_sha256"] != revision["prompt_sha256"]:
            raise RevisionHorizonError(f"{benchmark} matched prompt differs")
    return ledgers, hashes


def _load_scores(
    root: Path, benchmark: str, expected_ids: list[str]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    scores = {}
    hashes = {}
    for stage in SCORED_STAGES:
        path = root / benchmark / f"{stage}.official-scores.jsonl"
        rows = load_jsonl(path)
        if [row.get("id") for row in rows] != expected_ids or any(
            row.get("schema") != SCORE_SCHEMA
            or row.get("stage") != stage
            or row.get("benchmark") != benchmark
            or isinstance(row.get("score"), bool)
            or not isinstance(row.get("score"), (int, float))
            or not math.isfinite(float(row["score"]))
            or not 0.0 <= float(row["score"]) <= 1.0
            or not isinstance(row.get("stratum"), str)
            for row in rows
        ):
            raise RevisionHorizonError(f"{benchmark}:{stage} score differs")
        scores[stage] = rows
        hashes[stage] = sha256_file(path)
    return scores, hashes


def _median(values: list[float | int]) -> float:
    if not values:
        raise RevisionHorizonError("cannot summarize an empty horizon")
    return float(statistics.median(values))


def _strata(
    scores: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    positions: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(scores["trained_revision"]):
        positions[row["stratum"]].append(index)
    output = {}
    for stratum, indices in sorted(positions.items()):
        unchanged = sum(
            float(scores["unchanged_continuation"][index]["score"]) for index in indices
        )
        revision = sum(
            float(scores["trained_revision"][index]["score"]) for index in indices
        )
        wins = sum(
            scores["trained_revision"][index]["score"]
            > scores["unchanged_continuation"][index]["score"]
            for index in indices
        )
        losses = sum(
            scores["trained_revision"][index]["score"]
            < scores["unchanged_continuation"][index]["score"]
            for index in indices
        )
        output[stratum] = {
            "rows": len(indices),
            "unchanged_score": 100 * unchanged / len(indices),
            "trained_revision_score": 100 * revision / len(indices),
            "paired_delta_points": 100 * (revision - unchanged) / len(indices),
            "wins": wins,
            "losses": losses,
        }
    return output


def analyze_benchmark(
    generation_root: Path, score_root: Path, benchmark: str
) -> dict[str, Any]:
    ledgers, generation_hashes = _load_generation(generation_root, benchmark)
    identities = [row["id"] for row in ledgers["trained_revision"]]
    scores, score_hashes = _load_scores(score_root, benchmark, identities)
    rows = len(identities)
    completions = {
        stage: [row["completion"] for row in ledgers[stage]]
        for stage in GENERATION_STAGES
    }
    unchanged_lengths = [len(text) for text in completions["unchanged_continuation"]]
    revision_lengths = [len(text) for text in completions["trained_revision"]]
    ratios = [
        revision / max(1, unchanged)
        for unchanged, revision in zip(unchanged_lengths, revision_lengths, strict=True)
    ]
    unchanged_scores = [float(row["score"]) for row in scores["unchanged_continuation"]]
    revision_scores = [float(row["score"]) for row in scores["trained_revision"]]
    direct_scores = [float(row["score"]) for row in scores["direct_base"]]
    wins = sum(r > u for u, r in zip(unchanged_scores, revision_scores, strict=True))
    losses = sum(r < u for u, r in zip(unchanged_scores, revision_scores, strict=True))
    unchanged_total = sum(unchanged_scores)
    retained = sum(
        revision
        for unchanged, revision in zip(unchanged_scores, revision_scores, strict=True)
        if unchanged > 0.0
    )
    return {
        "rows": rows,
        "prompt_hash_equal_rows": rows,
        "byte_changed_rows": sum(
            unchanged != revision
            for unchanged, revision in zip(
                completions["unchanged_continuation"],
                completions["trained_revision"],
                strict=True,
            )
        ),
        "byte_change_rate": sum(
            unchanged != revision
            for unchanged, revision in zip(
                completions["unchanged_continuation"],
                completions["trained_revision"],
                strict=True,
            )
        )
        / rows,
        "median_characters": {
            stage: _median([len(text) for text in completions[stage]])
            for stage in GENERATION_STAGES
        },
        "revision_to_unchanged_median_character_ratio": _median(ratios),
        "under_20_characters": {
            stage: sum(len(text.strip()) < 20 for text in completions[stage])
            for stage in GENERATION_STAGES
        },
        "max_token_exhausted": {
            stage: sum(row["max_token_exhausted"] for row in ledgers[stage])
            for stage in GENERATION_STAGES
        },
        "direct_base_score": 100 * sum(direct_scores) / rows,
        "unchanged_score": 100 * unchanged_total / rows,
        "trained_revision_score": 100 * sum(revision_scores) / rows,
        "paired_delta_points": 100 * (sum(revision_scores) - unchanged_total) / rows,
        "trained_revision_vs_direct_base_points": 100
        * (sum(revision_scores) - sum(direct_scores))
        / rows,
        "wins": wins,
        "losses": losses,
        "ties": rows - wins - losses,
        "unchanged_positive_score_retention": (
            retained / unchanged_total if unchanged_total else 1.0
        ),
        "strata": _strata(scores),
        "generation_sha256": generation_hashes,
        "official_score_sha256": score_hashes,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise RevisionHorizonError("refusing to replace horizon analysis")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--benchmark", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(set(args.benchmark)) != len(args.benchmark):
        raise RevisionHorizonError("benchmark is duplicated")
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "generation_root": str(args.generation_root.resolve()),
        "score_root": str(args.score_root.resolve()),
        "benchmarks": {
            benchmark: analyze_benchmark(
                args.generation_root, args.score_root, benchmark
            )
            for benchmark in args.benchmark
        },
    }
    atomic_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
