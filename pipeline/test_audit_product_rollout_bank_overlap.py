import json
from pathlib import Path

from audit_product_rollout_bank_overlap import audit, normalized_words


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_normalization_is_case_and_punctuation_insensitive() -> None:
    assert normalized_words("A + B?") == normalized_words("a b")


def test_clean_disjoint_bank_passes(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.jsonl"
    reference = tmp_path / "reference.jsonl"
    write_rows(
        candidate,
        [{"question": "fresh one two three", "answer": "4", "task": "math500"}],
    )
    write_rows(reference, [{"question": "different four five six", "answer": "7"}])
    report = audit(candidate, [reference], ngram=3)
    assert report["admitted"] is True
    assert report["exact_reference_hits"] == 0


def test_overlap_duplicate_and_missing_answer_fail(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.jsonl"
    reference = tmp_path / "reference.jsonl"
    write_rows(
        candidate,
        [
            {"question": "Alpha beta gamma delta", "answer": "1"},
            {"question": "alpha beta gamma delta", "answer": "2"},
            {"question": "new alpha beta gamma sequence"},
        ],
    )
    write_rows(reference, [{"question": "Alpha beta gamma delta", "answer": "1"}])
    report = audit(candidate, [reference], ngram=3)
    assert report["admitted"] is False
    assert report["exact_reference_hits"] == 2
    assert report["rows_with_reference_ngram_hit"] == 3
    assert report["duplicate_normalized_rows"] == 1
    assert report["missing_answers"] == 1
