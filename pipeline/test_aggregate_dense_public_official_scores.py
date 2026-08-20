from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from aggregate_dense_public_official_scores import (
    MANIFEST_SCHEMA,
    SCORE_SCHEMA,
    OfficialAggregateError,
    run,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def fixture(tmp_path: Path) -> argparse.Namespace:
    questions = tmp_path / "questions.jsonl"
    identities = ["a" * 64, "b" * 64]
    write_jsonl(questions, [{"id": identity} for identity in identities])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": MANIFEST_SCHEMA,
                "benchmarks": [
                    {"name": "ruler", "questions": str(questions), "rows": 2}
                ],
            }
        )
    )
    root = tmp_path / "scores"
    values = {
        "direct_base": [0.0, 1.0],
        "unchanged_continuation": [0.0, 1.0],
        "trained_revision": [1.0, 1.0],
    }
    for stage, scores in values.items():
        write_jsonl(
            root / "ruler" / f"{stage}.official-scores.jsonl",
            [
                {
                    "schema": SCORE_SCHEMA,
                    "stage": stage,
                    "id": identity,
                    "benchmark": "ruler",
                    "metric": "official_string_match",
                    "stratum": f"{4 * (index + 1)}k",
                    "score": scores[index],
                }
                for index, identity in enumerate(identities)
            ],
        )
    return argparse.Namespace(
        manifest=manifest,
        score_root=root,
        output=tmp_path / "report.json",
    )


def test_aggregate_preserves_arms_and_strata(tmp_path: Path) -> None:
    report = run(fixture(tmp_path))
    metric = report["benchmarks"]["ruler"]["metrics"]
    assert metric["direct_base_score"] == 50.0
    assert metric["unchanged_score"] == 50.0
    assert metric["trained_revision_score"] == 100.0
    assert metric["wins"] == 1
    assert metric["losses"] == 0
    assert set(report["benchmarks"]["ruler"]["strata"]) == {"4k", "8k"}


def test_aggregate_rejects_nonfinite_or_out_of_range_scores(tmp_path: Path) -> None:
    args = fixture(tmp_path)
    direct = args.score_root / "ruler" / "direct_base.official-scores.jsonl"
    rows = [json.loads(line) for line in direct.read_text().splitlines()]
    rows[0]["score"] = 2.0
    direct.unlink()
    write_jsonl(direct, rows)
    with pytest.raises(OfficialAggregateError, match="official score row differs"):
        run(args)
