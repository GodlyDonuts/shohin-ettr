import hashlib
import json
from pathlib import Path

import pytest

from materialize_product_teacher_trace_bank import (
    TeacherTraceBankError,
    materialize_teacher_trace_bank,
)


def _identity(question: str) -> str:
    return hashlib.sha256(" ".join(question.split()).casefold().encode()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _bank(question: str, answer: str = "2") -> dict:
    return {
        "schema": "shohin-product-rollout-bank-v1",
        "identity_sha256": _identity(question),
        "question": question,
        "training_group": "math",
        "expected_answer_normalized": answer,
    }


def _teacher(question: str, answer: str = "2") -> dict:
    return {
        "question": question,
        "response": "A complete verified derivation. Final answer: 2",
        "training_group": "math",
        "expected_answer_normalized": answer,
        "verification": "expected_answer_match_v1",
    }


def test_materializes_teacher_response_in_bank_order(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    source = tmp_path / "source.jsonl"
    _write_jsonl(bank, [_bank("Question B"), _bank("Question A")])
    _write_jsonl(source, [_teacher("Question A"), _teacher("Question B")])

    report = materialize_teacher_trace_bank(
        bank,
        source,
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
    assert report["counters"]["materialized_rows"] == 2


def test_rejects_missing_teacher_identity(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    source = tmp_path / "source.jsonl"
    _write_jsonl(bank, [_bank("Missing")])
    _write_jsonl(source, [_teacher("Other")])
    with pytest.raises(TeacherTraceBankError, match="absent"):
        materialize_teacher_trace_bank(
            bank,
            source,
            tmp_path / "output.jsonl",
            tmp_path / "report.json",
        )


def test_rejects_answer_mismatch(tmp_path: Path) -> None:
    bank = tmp_path / "bank.jsonl"
    source = tmp_path / "source.jsonl"
    _write_jsonl(bank, [_bank("Question", "2")])
    _write_jsonl(source, [_teacher("Question", "3")])
    with pytest.raises(TeacherTraceBankError, match="answer differs"):
        materialize_teacher_trace_bank(
            bank,
            source,
            tmp_path / "output.jsonl",
            tmp_path / "report.json",
        )
