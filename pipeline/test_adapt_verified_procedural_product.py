from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.adapt_verified_procedural_product import (
    ProceduralAdmissionError,
    adapt_verified_procedural,
)


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_admission_filters_overlap_and_duplicate(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    evaluation = tmp_path / "eval.jsonl"
    output = tmp_path / "out.jsonl"
    report_path = tmp_path / "report.json"
    valid = {
        "question": "Derive the result.",
        "response": "<think>1 + 1 = 2</think>\nThe answer is 2.",
        "answer": "2",
        "source": "reasoning_gym_trace",
        "family": "chain_sum",
    }
    _write(
        source,
        [
            valid,
            {**valid, "question": "  derive   THE result. "},
            {**valid, "question": "Held out prompt"},
            {**valid, "question": "No trace", "response": "The answer is 2."},
        ],
    )
    _write(evaluation, [{"input": "Held out prompt", "target": "2"}])

    report = adapt_verified_procedural(
        source,
        output,
        report_path,
        eval_paths=[evaluation],
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["training_group"] == "procedural"
    assert rows[0]["verification"] == "reasoning_gym_answer_verified"
    assert report["counters"] == {
        "duplicate_question_dropped": 1,
        "eval_overlap_dropped": 1,
        "kept_rows": 1,
        "missing_verified_trace": 1,
        "raw_rows": 4,
    }
    assert json.loads(report_path.read_text()) == report


def test_admission_rejects_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "out.jsonl"
    report = tmp_path / "report.json"
    source.write_text("{}\n")
    output.write_text("occupied")

    with pytest.raises(ProceduralAdmissionError, match="must not already exist"):
        adapt_verified_procedural(source, output, report, eval_paths=[])
