from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.build_balanced_product_reasoning_mix import (
    BalancedMixError,
    build_balanced_mix,
    parse_weights,
)


def _write_source(path: Path) -> None:
    counts = {"math": 7, "code": 4, "procedural": 4, "teacher": 5}
    with path.open("w", encoding="utf-8") as handle:
        for group, count in counts.items():
            for index in range(count):
                handle.write(
                    json.dumps(
                        {
                            "question": f"{group} question {index}",
                            "response": f"{group} response {index}",
                            "training_group": group,
                        }
                    )
                    + "\n"
                )


def test_balanced_mix_is_exact_unique_and_reproducible(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_source(source)
    weights = parse_weights("math=.35,code=.2,procedural=.2,teacher=.25")
    first = build_balanced_mix(
        source,
        tmp_path / "first.jsonl",
        tmp_path / "first.report.json",
        weights,
        10,
        31,
    )
    second = build_balanced_mix(
        source,
        tmp_path / "second.jsonl",
        tmp_path / "second.report.json",
        weights,
        10,
        31,
    )
    assert first["selected_group_counts"] == {
        "code": 2,
        "math": 4,
        "procedural": 2,
        "teacher": 2,
    }
    assert first["output_sha256"] == second["output_sha256"]
    assert first["duplicate_questions"] == 0
    assert first["replayed_rows"] == 0


def test_balanced_mix_rejects_duplicate_selected_questions(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_source(source)
    with source.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "question": "math question 0",
                    "response": "duplicate",
                    "training_group": "code",
                }
            )
            + "\n"
        )
    with pytest.raises(BalancedMixError, match="duplicate questions"):
        build_balanced_mix(
            source,
            tmp_path / "bad.jsonl",
            tmp_path / "bad.report.json",
            parse_weights("math=.5,code=.5"),
            10,
            31,
        )
