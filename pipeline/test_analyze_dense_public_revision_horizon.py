import json
from pathlib import Path

import pytest

from analyze_dense_public_revision_horizon import (
    REPORT_SCHEMA,
    RevisionHorizonError,
    analyze_benchmark,
)
from dense_public_official_scoring import LEDGER_SCHEMA, SCORE_SCHEMA


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    generation = tmp_path / "generation"
    scores = tmp_path / "scores"
    identities = ("a", "b")
    completions = {
        "direct_base": ("base-a", "base-b"),
        "draft": ("draft-a", "draft-b"),
        "unchanged_continuation": ("long unchanged answer", "same"),
        "trained_revision": ("x", "same"),
    }
    for stage, texts in completions.items():
        _jsonl(
            generation / "bench" / f"{stage}.jsonl",
            [
                {
                    "schema": LEDGER_SCHEMA,
                    "stage": stage,
                    "id": identity,
                    "benchmark": "bench",
                    "prompt_sha256": (
                        "p" if stage not in {"direct_base", "draft"} else stage
                    ),
                    "completion": text,
                    "max_token_exhausted": index == 0 and stage == "draft",
                }
                for index, (identity, text) in enumerate(zip(identities, texts))
            ],
        )
    stage_scores = {
        "direct_base": (1.0, 0.0),
        "unchanged_continuation": (1.0, 0.0),
        "trained_revision": (0.0, 1.0),
    }
    for stage, values in stage_scores.items():
        _jsonl(
            scores / "bench" / f"{stage}.official-scores.jsonl",
            [
                {
                    "schema": SCORE_SCHEMA,
                    "stage": stage,
                    "id": identity,
                    "benchmark": "bench",
                    "metric": "accuracy",
                    "stratum": "one" if index == 0 else "two",
                    "score": value,
                }
                for index, (identity, value) in enumerate(zip(identities, values))
            ],
        )
    return generation, scores


def test_analyze_benchmark_measures_activation_horizon_and_pairs(
    tmp_path: Path,
) -> None:
    generation, scores = _fixture(tmp_path)
    report = analyze_benchmark(generation, scores, "bench")
    assert report["rows"] == 2
    assert report["prompt_hash_equal_rows"] == 2
    assert report["byte_changed_rows"] == 1
    assert report["byte_change_rate"] == 0.5
    assert report["median_characters"]["trained_revision"] == 2.5
    assert report["under_20_characters"]["trained_revision"] == 2
    assert report["max_token_exhausted"]["draft"] == 1
    assert report["wins"] == report["losses"] == 1
    assert report["paired_delta_points"] == 0.0
    assert set(report["strata"]) == {"one", "two"}
    assert all(len(value) == 64 for value in report["generation_sha256"].values())
    assert all(len(value) == 64 for value in report["official_score_sha256"].values())


def test_analyze_benchmark_rejects_unmatched_prompts(tmp_path: Path) -> None:
    generation, scores = _fixture(tmp_path)
    path = generation / "bench" / "trained_revision.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["prompt_sha256"] = "different"
    _jsonl(path, rows)
    with pytest.raises(RevisionHorizonError, match="matched prompt differs"):
        analyze_benchmark(generation, scores, "bench")


def test_report_schema_is_frozen() -> None:
    assert REPORT_SCHEMA == "shohin-dense-public-revision-horizon-analysis-v1"
