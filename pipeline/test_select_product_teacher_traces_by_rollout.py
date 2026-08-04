import hashlib
import json
from pathlib import Path

import pytest

from select_product_teacher_traces_by_rollout import (
    RolloutTeacherSelectionError,
    select_teacher_traces,
)


def _identity(question: str) -> str:
    return hashlib.sha256(" ".join(question.split()).casefold().encode()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _teacher(question: str, answer: str = "2") -> dict:
    return {
        "question": question,
        "response": "A complete derivation. Final answer: 2",
        "training_group": "math",
        "expected_answer_normalized": answer,
        "verification": "expected_answer_match_v1",
        "source_identity_sha256": _identity(question),
    }


def _positive(question: str, answer: str = "2") -> dict:
    return {
        "question": question,
        "response": "A possibly truncated student response",
        "training_group": "math",
        "expected_answer_normalized": answer,
        "verification": "student_exact_answer_match_v1",
        "source_identity_sha256": _identity(question),
    }


def test_selects_complete_teacher_trace_in_teacher_order(tmp_path: Path) -> None:
    teacher = tmp_path / "teacher.jsonl"
    positives = tmp_path / "positives.jsonl"
    _write_jsonl(
        teacher,
        [_teacher("Question B"), _teacher("Question A"), _teacher("Question C")],
    )
    _write_jsonl(positives, [_positive("Question A"), _positive("Question B")])

    report = select_teacher_traces(
        teacher,
        positives,
        tmp_path / "output.jsonl",
        tmp_path / "report.json",
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "output.jsonl").read_text().splitlines()
    ]
    assert [row["question"] for row in rows] == ["Question B", "Question A"]
    assert all(row["response"].startswith("A complete") for row in rows)
    assert report["rows"] == 2
    assert report["counters"]["teacher_rows_not_selected"] == 1


def test_rejects_missing_teacher_identity(tmp_path: Path) -> None:
    teacher = tmp_path / "teacher.jsonl"
    positives = tmp_path / "positives.jsonl"
    _write_jsonl(teacher, [_teacher("Question A")])
    _write_jsonl(positives, [_positive("Missing")])
    with pytest.raises(RolloutTeacherSelectionError, match="absent"):
        select_teacher_traces(
            teacher,
            positives,
            tmp_path / "output.jsonl",
            tmp_path / "report.json",
        )


def test_rejects_answer_mismatch(tmp_path: Path) -> None:
    teacher = tmp_path / "teacher.jsonl"
    positives = tmp_path / "positives.jsonl"
    _write_jsonl(teacher, [_teacher("Question", "2")])
    _write_jsonl(positives, [_positive("Question", "3")])
    with pytest.raises(RolloutTeacherSelectionError, match="answer differs"):
        select_teacher_traces(
            teacher,
            positives,
            tmp_path / "output.jsonl",
            tmp_path / "report.json",
        )


def test_rejects_unverified_rollout_row(tmp_path: Path) -> None:
    teacher = tmp_path / "teacher.jsonl"
    positives = tmp_path / "positives.jsonl"
    row = _positive("Question")
    row["verification"] = "unknown"
    _write_jsonl(teacher, [_teacher("Question")])
    _write_jsonl(positives, [row])
    with pytest.raises(RolloutTeacherSelectionError, match="unverified"):
        select_teacher_traces(
            teacher,
            positives,
            tmp_path / "output.jsonl",
            tmp_path / "report.json",
        )
