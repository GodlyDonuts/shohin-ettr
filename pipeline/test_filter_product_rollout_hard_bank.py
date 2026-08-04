import json
from pathlib import Path

import pytest

from pipeline.filter_product_rollout_hard_bank import (
    ProductRolloutHardBankError,
    build_hard_bank,
)


def _identity(question: str) -> str:
    import hashlib

    return hashlib.sha256(question.casefold().encode()).hexdigest()


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _bank_row(question: str, group: str) -> dict:
    return {
        "schema": "shohin-product-rollout-bank-v1",
        "identity_sha256": _identity(question),
        "question": question,
        "answer": r"\boxed{1}",
        "expected_answer_normalized": "1",
        "task": "math500" if group == "math" else "bbh_logic",
        "training_group": group,
    }


def test_hard_bank_deduplicates_banks_and_removes_all_solved_rows(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    positives = tmp_path / "positives.jsonl"
    solved = _bank_row("Solved prompt", "math")
    hard = _bank_row("Hard prompt", "math")
    science = _bank_row("Science prompt", "science")
    _write(first, [solved, hard, science])
    _write(second, [hard])
    _write(
        positives,
        [
            {
                "question": solved["question"],
                "source_identity_sha256": solved["identity_sha256"],
            }
        ],
    )
    output = tmp_path / "hard.jsonl"
    report_path = tmp_path / "report.json"
    report = build_hard_bank(
        [first, second],
        [positives],
        output,
        report_path,
        group="math",
        seed=7,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows == [hard]
    assert report["rows"] == 1
    assert report["counters"]["duplicate_bank_rows"] == 1
    assert report["counters"]["selected_unsolved_prompts"] == 1


def test_hard_bank_rejects_conflicting_duplicate_identity(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    positives = tmp_path / "positives.jsonl"
    row = _bank_row("Question", "math")
    conflicting = dict(row, answer=r"\boxed{2}")
    _write(first, [row])
    _write(second, [conflicting])
    _write(
        positives,
        [
            {
                "question": "Other",
                "source_identity_sha256": _identity("Other"),
            }
        ],
    )
    with pytest.raises(ProductRolloutHardBankError, match="conflicting duplicate"):
        build_hard_bank(
            [first, second],
            [positives],
            tmp_path / "hard.jsonl",
            tmp_path / "report.json",
            group="math",
            seed=7,
        )
