from __future__ import annotations

from build_sdr1_source_only_data import EVAL_SCHEMA, TRAIN_SCHEMA


def test_sdr1_schemas_are_distinct_from_vcr1() -> None:
    assert "source-only" in TRAIN_SCHEMA
    assert "source-only" in EVAL_SCHEMA
    assert TRAIN_SCHEMA != EVAL_SCHEMA
