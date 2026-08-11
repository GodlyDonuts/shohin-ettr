"""CPU-only contract tests for the PCF1 learned commit policy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hf_pcf1_train_commit import (
    PAIR_SCHEMA,
    PCF1CommitError,
    balanced_strata,
    expected_outcome,
    load_pairs,
    summarize,
)


def _row(index: int, split: str, outcome: str = "both_wrong") -> dict:
    correctness = {
        "both_correct": (True, True),
        "revision_only": (True, False),
        "unchanged_only": (False, True),
        "both_wrong": (False, False),
    }[outcome]
    return {
        "schema": PAIR_SCHEMA,
        "identity_sha256": f"{index:064x}",
        "split": split,
        "task": ("math500", "bbh_logic", "mbpp")[index % 3],
        "question": f"question {index}",
        "outcome_class": outcome,
        "candidates": [
            {
                "lineage": "revision",
                "completion": f"revision {index}",
                "correct": correctness[0],
                "generated_tokens": 8,
                "max_token_exhausted": False,
            },
            {
                "lineage": "unchanged",
                "completion": f"unchanged {index}",
                "correct": correctness[1],
                "generated_tokens": 8,
                "max_token_exhausted": False,
            },
        ],
    }


def _write_pairs(path: Path) -> list[dict]:
    rows = [
        _row(index, "calibration_train", outcome)
        for index, outcome in enumerate(
            ("both_correct", "revision_only", "both_wrong", "unchanged_only"),
            start=1,
        )
    ]
    rows.append(_row(10, "calibration_development"))
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return rows


def test_load_pairs_and_balanced_strata_use_calibration_only(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pairs.jsonl"
    _write_pairs(path)
    rows = load_pairs(path)
    strata = balanced_strata(rows, 7)

    assert {row["split"] for row in rows} == {
        "calibration_train",
        "calibration_development",
    }
    assert {outcome for _, outcome in strata} == {
        "both_correct",
        "revision_only",
        "both_wrong",
        "unchanged_only",
    }


def test_load_pairs_rejects_sealed_path(tmp_path: Path) -> None:
    path = tmp_path / "holdout_pairs.jsonl"
    _write_pairs(path)
    with pytest.raises(PCF1CommitError, match="sealed path"):
        load_pairs(path)


def test_load_pairs_rejects_identity_reuse_across_splits(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    rows = _write_pairs(path)
    rows[-1]["identity_sha256"] = rows[0]["identity_sha256"]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(PCF1CommitError, match="duplicated"):
        load_pairs(path)


def test_summarize_reports_both_retention_denominators() -> None:
    rows = [
        _row(1, "confirmation", "both_correct"),
        _row(2, "confirmation", "revision_only"),
        _row(3, "confirmation", "unchanged_only"),
        _row(4, "confirmation", "both_wrong"),
    ]
    selections = {
        rows[0]["identity_sha256"]: (0, True, 1.0),
        rows[1]["identity_sha256"]: (0, True, 1.0),
        rows[2]["identity_sha256"]: (1, True, -1.0),
        rows[3]["identity_sha256"]: (0, True, 0.0),
    }
    overall = summarize(rows, selections, "confirmation")["overall"]

    assert overall["selected_correct"] == 3
    assert overall["revision_correct_retained"] == 2
    assert overall["revision_correct_retention"] == 1.0
    assert overall["unchanged_correct_retained"] == 2
    assert overall["unchanged_correct_retention"] == 1.0
    assert overall["order_consistency"] == 1.0


@pytest.mark.parametrize(
    ("revision", "unchanged", "outcome"),
    [
        (True, True, "both_correct"),
        (True, False, "revision_only"),
        (False, True, "unchanged_only"),
        (False, False, "both_wrong"),
    ],
)
def test_expected_outcome(revision: bool, unchanged: bool, outcome: str) -> None:
    assert expected_outcome(revision, unchanged) == outcome
