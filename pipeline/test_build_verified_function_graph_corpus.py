from __future__ import annotations

import json
from pathlib import Path

from build_verified_function_graph_corpus import (
    _grams,
    _program_passes,
    generate_rows,
)


def test_generated_rows_are_deterministic_and_executable() -> None:
    first, first_counts = generate_rows(
        count=12,
        seed=31,
        shard_index=0,
        shard_count=1,
        blocked_grams=set(),
        ngram_width=13,
        timeout_seconds=2,
    )
    second, second_counts = generate_rows(
        count=12,
        seed=31,
        shard_index=0,
        shard_count=1,
        blocked_grams=set(),
        ngram_width=13,
        timeout_seconds=2,
    )
    assert first == second
    assert first_counts == second_counts
    assert {row["family"] for row in first} == {
        "list_pipeline",
        "string_pipeline",
        "number_theory",
        "record_pipeline",
    }
    for row in first:
        assert _program_passes(row["response"], row["tests"], 2)
        assert row["verification"] == "generated_reference_passes_randomized_tests"


def test_eval_ngram_overlap_is_rejected() -> None:
    baseline, _ = generate_rows(
        count=1,
        seed=7,
        shard_index=0,
        shard_count=1,
        blocked_grams=set(),
        ngram_width=13,
        timeout_seconds=2,
    )
    blocked = _grams(baseline[0]["question"], 13)
    rows, counters = generate_rows(
        count=1,
        seed=7,
        shard_index=0,
        shard_count=1,
        blocked_grams=blocked,
        ngram_width=13,
        timeout_seconds=2,
    )
    assert counters["eval_ngram_overlap"] == 1
    assert rows[0]["global_identity"] != baseline[0]["global_identity"]


def test_rows_round_trip_as_jsonl(tmp_path: Path) -> None:
    rows, _ = generate_rows(
        count=4,
        seed=11,
        shard_index=0,
        shard_count=1,
        blocked_grams=set(),
        ngram_width=13,
        timeout_seconds=2,
    )
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert len(path.read_text().splitlines()) == 4
