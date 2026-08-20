#!/usr/bin/env python3
"""Aggregate identity-bound official benchmark scores across matched Shohin arms."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "shohin-dense-public-campaign-manifest-v1"
SCORE_SCHEMA = "shohin-dense-public-official-score-v1"
REPORT_SCHEMA = "shohin-dense-public-official-aggregate-v1"
STAGES = ("direct_base", "unchanged_continuation", "trained_revision")
EPSILON = 1e-12


class OfficialAggregateError(RuntimeError):
    """Score custody, identity coverage, or value range differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def paired_sign_pvalue(wins: int, losses: int) -> float:
    trials = wins + losses
    if trials == 0:
        return 1.0
    tail = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(trials, i) for i in range(tail + 1)) / 2**trials)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise OfficialAggregateError("cannot summarize empty official scores")
    direct = sum(row["direct_base"] for row in rows)
    unchanged = sum(row["unchanged_continuation"] for row in rows)
    revision = sum(row["trained_revision"] for row in rows)
    wins = sum(
        row["trained_revision"] > row["unchanged_continuation"] + EPSILON
        for row in rows
    )
    losses = sum(
        row["unchanged_continuation"] > row["trained_revision"] + EPSILON
        for row in rows
    )
    count = len(rows)
    binary = all(
        score in {0.0, 1.0}
        for row in rows
        for score in (
            row["direct_base"],
            row["unchanged_continuation"],
            row["trained_revision"],
        )
    )
    result = {
        "rows": count,
        "direct_base_score": 100 * direct / count,
        "unchanged_score": 100 * unchanged / count,
        "trained_revision_score": 100 * revision / count,
        "trained_revision_vs_direct_base_points": 100 * (revision - direct) / count,
        "paired_delta_points": 100 * (revision - unchanged) / count,
        "wins": wins,
        "losses": losses,
        "ties": count - wins - losses,
        "paired_sign_test_two_sided_p": paired_sign_pvalue(wins, losses),
        "binary_metric": binary,
    }
    if binary:
        result["unchanged_correct_retention"] = (
            sum(
                row["unchanged_continuation"] == 1.0
                and row["trained_revision"] == 1.0
                for row in rows
            )
            / unchanged
            if unchanged
            else 1.0
        )
    else:
        result["unchanged_score_mass_retention"] = (
            sum(
                min(row["unchanged_continuation"], row["trained_revision"])
                for row in rows
            )
            / unchanged
            if unchanged
            else 1.0
        )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise OfficialAggregateError("campaign manifest schema differs")
    expected_by_benchmark: dict[str, list[str]] = {}
    for entry in manifest.get("benchmarks", []):
        benchmark = str(entry["name"])
        questions = load_jsonl(Path(entry["questions"]))
        if len(questions) != entry["rows"]:
            raise OfficialAggregateError(f"{benchmark} question coverage differs")
        if benchmark in expected_by_benchmark:
            raise OfficialAggregateError("campaign benchmark is duplicated")
        expected_by_benchmark[benchmark] = [str(row["id"]) for row in questions]
    expected = [
        (identity, benchmark)
        for benchmark, identities in expected_by_benchmark.items()
        for identity in identities
    ]
    if not expected or len({identity for identity, _ in expected}) != len(expected):
        raise OfficialAggregateError("campaign identities are empty or duplicated")

    ledgers: dict[str, list[dict[str, Any]]] = {stage: [] for stage in STAGES}
    score_hashes: dict[str, dict[str, str]] = {}
    for benchmark, expected_ids in expected_by_benchmark.items():
        score_hashes[benchmark] = {}
        for stage in STAGES:
            path = args.score_root / benchmark / f"{stage}.official-scores.jsonl"
            rows = load_jsonl(path)
            if len(rows) != len(expected_ids):
                raise OfficialAggregateError(
                    f"{benchmark}:{stage} official score coverage differs"
                )
            for row, identity in zip(rows, expected_ids, strict=True):
                score = row.get("score")
                if (
                    row.get("schema") != SCORE_SCHEMA
                    or row.get("stage") != stage
                    or row.get("id") != identity
                    or row.get("benchmark") != benchmark
                    or isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(float(score))
                    or not 0.0 <= float(score) <= 1.0
                    or not isinstance(row.get("metric"), str)
                    or not row["metric"]
                ):
                    raise OfficialAggregateError(
                        f"{benchmark}:{stage} official score row differs"
                    )
                row["score"] = float(score)
            ledgers[stage].extend(rows)
            score_hashes[benchmark][stage] = sha256_file(path)

    joined_by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    strata_by_benchmark: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, (identity, benchmark) in enumerate(expected):
        metrics = {ledgers[stage][index]["metric"] for stage in STAGES}
        strata = {str(ledgers[stage][index].get("stratum", "all")) for stage in STAGES}
        if len(metrics) != 1 or len(strata) != 1:
            raise OfficialAggregateError(f"{benchmark}:{identity} scorer binding differs")
        stratum = strata.pop()
        joined = {
            "id": identity,
            "metric": metrics.pop(),
            "stratum": stratum,
            **{stage: ledgers[stage][index]["score"] for stage in STAGES},
        }
        joined_by_benchmark[benchmark].append(joined)
        strata_by_benchmark[benchmark][stratum].append(joined)

    benchmarks = {}
    for entry in manifest["benchmarks"]:
        name = str(entry["name"])
        rows = joined_by_benchmark[name]
        benchmarks[name] = {
            "metrics": summarize(rows),
            "strata": {
                stratum: summarize(stratum_rows)
                for stratum, stratum_rows in sorted(strata_by_benchmark[name].items())
            },
        }
    macro = {
        key: sum(value["metrics"][key] for value in benchmarks.values())
        / len(benchmarks)
        for key in ("direct_base_score", "unchanged_score", "trained_revision_score")
    }
    macro["trained_revision_vs_direct_base_points"] = (
        macro["trained_revision_score"] - macro["direct_base_score"]
    )
    macro["paired_delta_points"] = (
        macro["trained_revision_score"] - macro["unchanged_score"]
    )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "official_score_sha256": score_hashes,
        "benchmarks": benchmarks,
        "standardized_overall": {
            "method": "unweighted_macro_average_of_official_primary_percentages",
            "benchmarks": len(benchmarks),
            **macro,
        },
    }
    if args.output.exists():
        raise OfficialAggregateError("refusing to replace official aggregate")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    report = run(parse_args())
    print(json.dumps(report["standardized_overall"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
