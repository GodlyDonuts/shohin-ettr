import json
from pathlib import Path

import pytest

from compare_q36_mtr_owner_previews import (
    Q36MTROwnerPreviewComparisonError,
    compare,
)


def _report(path: Path, outcomes: list[tuple[int, str, bool]]) -> Path:
    rows = [
        {
            "identity_sha256": f"{identity:064x}",
            "task": task,
            "correct": correct,
            "explicit_final_answer": True,
            "max_token_exhausted": False,
        }
        for identity, task, correct in outcomes
    ]
    path.write_text(
        json.dumps(
            {
                "schema": "shohin-q36-mtr-draft-preview-v1",
                "status": "complete",
                "split": "development",
                "rows": len(rows),
                "outcomes": rows,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_compare_paired_complementarity(tmp_path: Path) -> None:
    first = _report(
        tmp_path / "first.json",
        [(1, "math500", True), (2, "bbh_logic", False), (3, "mbpp", True)],
    )
    second = _report(
        tmp_path / "second.json",
        [(1, "math500", False), (2, "bbh_logic", True), (3, "mbpp", True)],
    )
    result = compare([first], [second], first_label="a", second_label="b")
    assert result["shared_identities"] == 3
    assert result["first_correct"] == result["second_correct"] == 2
    assert result["oracle_correct"] == 3
    assert result["paired_cells"] == {
        "both_wrong": 0,
        "first_only_correct": 1,
        "second_only_correct": 1,
        "both_correct": 1,
    }
    assert result["oracle_gain_over_best_points"] == pytest.approx(100 / 3)


def test_compare_rejects_task_mismatch(tmp_path: Path) -> None:
    first = _report(tmp_path / "first.json", [(1, "math500", True)])
    second = _report(tmp_path / "second.json", [(1, "bbh_logic", True)])
    with pytest.raises(Q36MTROwnerPreviewComparisonError, match="task"):
        compare([first], [second], first_label="a", second_label="b")


def test_compare_rejects_duplicate_label(tmp_path: Path) -> None:
    report = _report(tmp_path / "first.json", [(1, "math500", True)])
    with pytest.raises(Q36MTROwnerPreviewComparisonError, match="labels"):
        compare([report], [report], first_label="same", second_label="same")
