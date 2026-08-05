import hashlib

from aggregate_visible_code_repairs import (
    VisibleCodeRepairAggregateError,
    aggregate,
)


def _repair(original: str, text: str) -> dict:
    return {
        "task": "mbpp",
        "text": text,
        "original_identity_sha256": original,
        "repair_schema": "shohin-visible-code-repair-bank-v1",
    }


def _identity(text: str) -> str:
    return hashlib.sha256(f"mbpp\0{text}".encode()).hexdigest()


def _build(rows: int) -> dict:
    return {
        "schema": "shohin-visible-code-repair-bank-v1",
        "status": "complete",
        "repair_rows": rows,
        "source_total": rows + 10,
        "source_selected_correct": 10,
    }


def _eval(*results: dict) -> dict:
    correct = sum(bool(row["correct"]) for row in results)
    return {
        "status": "complete",
        "task": "mbpp",
        "total": len(results),
        "correct": correct,
        "results": list(results),
    }


def _result(text: str, correct: bool) -> dict:
    return {
        "identity_sha256": _identity(text),
        "correct": correct,
        "completion": "def f(): pass",
        "execution": {"passed": correct},
    }


def test_aggregates_complete_disjoint_repairs(tmp_path) -> None:
    rows = [_repair("a", "repair a"), _repair("b", "repair b")]
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    paths[0].write_text("a")
    paths[1].write_text("b")
    report = aggregate(
        rows,
        _build(2),
        [_eval(_result("repair a", True)), _eval(_result("repair b", False))],
        eval_paths=paths,
    )
    assert report["repair_correct"] == 1
    assert report["final_correct"] == 11
    assert report["gain"] == 1


def test_rejects_incomplete_coverage(tmp_path) -> None:
    path = tmp_path / "a.json"
    path.write_text("a")
    try:
        aggregate(
            [_repair("a", "repair a"), _repair("b", "repair b")],
            _build(2),
            [_eval(_result("repair a", True))],
            eval_paths=[path],
        )
    except VisibleCodeRepairAggregateError as exc:
        assert "coverage" in str(exc)
    else:
        raise AssertionError("expected incomplete coverage rejection")


def test_rejects_duplicate_evaluation_identity(tmp_path) -> None:
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    paths[0].write_text("a")
    paths[1].write_text("b")
    try:
        aggregate(
            [_repair("a", "repair a")],
            _build(1),
            [_eval(_result("repair a", True)), _eval(_result("repair a", True))],
            eval_paths=paths,
        )
    except VisibleCodeRepairAggregateError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("expected duplicate evaluation rejection")
