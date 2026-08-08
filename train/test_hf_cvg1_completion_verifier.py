from __future__ import annotations

import json
from pathlib import Path

import pytest

from hf_cvg1_completion_verifier import (
    CVG1VerifierError,
    EXPERT_LOGIT_MARGIN,
    PAIR_SCHEMA,
    build_balanced_strata,
    choose_lineage,
    load_pairs,
    metrics_from_scores,
    verifier_text,
)


def _row(index: int, *, split: str, outcome: str, task: str = "math") -> dict:
    correctness = {
        "base_only": (True, False),
        "both_correct": (True, True),
        "both_wrong": (False, False),
        "expert_only": (False, True),
    }[outcome]
    return {
        "schema": PAIR_SCHEMA,
        "identity_sha256": f"{index:064x}",
        "split": split,
        "task": task,
        "question": f"Question {index}",
        "outcome_class": outcome,
        "candidates": [
            {"lineage": "base", "completion": "base answer", "correct": correctness[0]},
            {
                "lineage": "expert",
                "completion": "expert answer",
                "correct": correctness[1],
            },
        ],
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_loader_orders_lineages_and_balances_all_outcomes(tmp_path: Path) -> None:
    rows = [
        _row(index, split="train", outcome=outcome)
        for index, outcome in enumerate(
            ("base_only", "both_correct", "both_wrong", "expert_only"), start=1
        )
    ]
    rows[0]["candidates"].reverse()
    path = tmp_path / "pairs.jsonl"
    _write(path, rows)
    loaded = load_pairs(path)
    assert [candidate["lineage"] for candidate in loaded[0]["candidates"]] == [
        "base",
        "expert",
    ]
    strata = build_balanced_strata(loaded, seed=7)
    assert {outcome for _, outcome in strata} == {
        "base_only",
        "both_correct",
        "both_wrong",
        "expert_only",
    }


def test_loader_rejects_false_outcome_or_duplicate_identity(tmp_path: Path) -> None:
    row = _row(1, split="train", outcome="base_only")
    row["outcome_class"] = "expert_only"
    path = tmp_path / "false.jsonl"
    _write(path, [row])
    with pytest.raises(CVG1VerifierError, match="outcome class"):
        load_pairs(path)

    duplicate = _row(2, split="train", outcome="both_wrong")
    path = tmp_path / "duplicate.jsonl"
    _write(path, [duplicate, duplicate])
    with pytest.raises(CVG1VerifierError, match="duplicated"):
        load_pairs(path)


def test_selector_defaults_to_base_without_sufficient_expert_evidence() -> None:
    assert choose_lineage(0.0, EXPERT_LOGIT_MARGIN - 1e-6) == 0
    assert choose_lineage(0.2, 0.2 + EXPERT_LOGIT_MARGIN) == 1
    assert choose_lineage(-2.0, -1.0) == 0


def test_metrics_gate_uses_whole_lineages_and_source_split() -> None:
    rows = [
        _row(1, split="holdout", outcome="base_only"),
        _row(2, split="holdout", outcome="expert_only"),
        _row(3, split="holdout", outcome="both_correct"),
        _row(4, split="holdout", outcome="both_wrong"),
        _row(5, split="train", outcome="expert_only"),
    ]
    scores = {
        rows[0]["identity_sha256"]: (4.0, -4.0),
        rows[1]["identity_sha256"]: (-4.0, 4.0),
        rows[2]["identity_sha256"]: (4.0, 4.0),
        rows[3]["identity_sha256"]: (-4.0, -4.0),
    }
    report = metrics_from_scores(rows, scores, split="holdout", seed=9)
    assert report["metrics"]["overall"]["selected_correct"] == 3
    assert report["metrics"]["overall"]["disagreement_selection_accuracy"] == 1.0
    assert report["metrics"]["overall"]["total"] == 4
    assert report["gate_pass"] is True


def test_verifier_prompt_excludes_task_and_gold_fields() -> None:
    text = verifier_text("What is 2+2?", "The answer is 4.")
    assert "What is 2+2?" in text
    assert "The answer is 4." in text
    assert "task" not in text.casefold()
    assert "gold" not in text.casefold()
