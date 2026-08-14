import json
from pathlib import Path

import pytest

from aggregate_q36_mtr_draft_previews import (
    Q36MTRDraftPreviewAggregateError,
    aggregate,
)


def _identity(value: int) -> str:
    return f"{value:064x}"


def _report(path: Path, start: int, rows: int = 3) -> Path:
    outcomes = []
    tasks = ("math500", "bbh_logic", "mbpp")
    for offset in range(rows):
        outcomes.append(
            {
                "identity_sha256": _identity(start + offset),
                "task": tasks[offset % len(tasks)],
                "correct": offset % 2 == 0,
                "explicit_final_answer": offset != 1,
                "max_token_exhausted": offset == 1,
            }
        )
    payload = {
        "schema": "shohin-q36-mtr-draft-preview-v1",
        "status": "complete",
        "interpretation": "exploratory_model_owned_draft_only_not_matched_gate",
        "split": "development",
        "rows": len(outcomes),
        "candidates_sha256": "a" * 64,
        "outcomes": outcomes,
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def test_aggregate_exact_disjoint_reports(tmp_path: Path) -> None:
    result = aggregate(
        [_report(tmp_path / "a.json", 1), _report(tmp_path / "b.json", 4)],
        label="owner-x",
    )
    assert result["rows"] == 6
    assert result["unique_identities"] == 6
    assert result["correct"] == 4
    assert result["max_token_exhausted"] == 2
    assert result["completion_status"]["exhausted_correct"] == 0
    assert result["completion_status"]["nonexhausted_correct"] == 4


def test_aggregate_rejects_duplicate_identity(tmp_path: Path) -> None:
    reports = [_report(tmp_path / "a.json", 1), _report(tmp_path / "b.json", 3)]
    with pytest.raises(Q36MTRDraftPreviewAggregateError, match="identity"):
        aggregate(reports, label="owner-x")


def test_aggregate_rejects_duplicate_report_path(tmp_path: Path) -> None:
    report = _report(tmp_path / "a.json", 1)
    with pytest.raises(Q36MTRDraftPreviewAggregateError, match="duplicate"):
        aggregate([report, report], label="owner-x")
