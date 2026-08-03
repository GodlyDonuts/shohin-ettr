#!/usr/bin/env python3

import json
from pathlib import Path

import pytest

from build_product_outcome_router_board import build_board, normalized_question


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_board_is_disjoint_unique_and_answer_scored(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    excluded = tmp_path / "excluded.jsonl"
    write_rows(excluded, [{"question": "Already used?"}])
    write_rows(source, [
        {
            "question": "Already used?",
            "expected_answer_normalized": "1",
            "training_group": "math",
        },
        {
            "question": "Fresh equation",
            "expected_answer_normalized": "2",
            "training_group": "math",
            "verification": "expected_answer_match_v1",
        },
        {
            "question": "Fresh equation!",
            "expected_answer_normalized": "3",
            "training_group": "math",
        },
    ])
    rows, report = build_board(
        source_path=source,
        excluded_paths=[excluded],
        training_group="math",
        task="math500",
        count=1,
        seed=7,
    )
    assert rows[0]["answer"] == r"\boxed{2}"
    assert report["excluded_training_question"] == 1
    assert report["duplicate_normalized_question"] == 1


def test_short_answer_board_uses_evaluator_schema(tmp_path: Path) -> None:
    source = tmp_path / "science.jsonl"
    write_rows(source, [{
        "question": "Choose: (A) x (B) y",
        "expected_answer_normalized": r"\text{b}",
        "training_group": "science",
    }])
    rows, _ = build_board(
        source_path=source,
        excluded_paths=[],
        training_group="science",
        task="bbh_logic",
        count=1,
        seed=9,
    )
    assert rows[0]["input"].startswith("Choose")
    assert rows[0]["target"] == r"\text{b}"


def test_insufficient_unique_rows_fails(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    write_rows(source, [])
    with pytest.raises(ValueError, match="eligible unique rows"):
        build_board(
            source_path=source,
            excluded_paths=[],
            training_group="math",
            task="math500",
            count=1,
            seed=1,
        )


def test_normalized_question_is_punctuation_insensitive() -> None:
    assert normalized_question("A + B?") == normalized_question("a b")
