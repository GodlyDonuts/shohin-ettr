from __future__ import annotations

from collections import Counter

from build_verified_function_graph_corpus_v2 import GENERATORS, SCHEMA, generate_rows


def test_v2_generates_every_failure_aligned_family() -> None:
    rows, counters = generate_rows(
        count=len(GENERATORS) * 3,
        seed=20260804,
        shard_index=0,
        shard_count=1,
        blocked_grams=set(),
        ngram_width=13,
        timeout_seconds=2,
    )
    families = Counter(row["family"] for row in rows)
    assert set(families) == {
        "index_rewrite",
        "frequency_filter",
        "nested_support",
        "sentence_scan",
        "pair_scan",
        "rounded_affine",
        "set_relation",
    }
    assert set(families.values()) == {3}
    assert counters["execution_failure"] == 0
    assert all(row["schema"] == SCHEMA for row in rows)
    assert len({row["question"] for row in rows}) == len(rows)
