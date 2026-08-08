from __future__ import annotations

import json
from pathlib import Path

from hf_pcj1_pairwise_judge import (
    OUTCOMES,
    PAIR_SCHEMA,
    SPLITS,
    assigned_split,
    conservative_selection,
    load_partitioned_pairs,
    metrics_from_predictions,
    ordered_label,
    pairwise_text,
    partition_receipt,
    semantic_verdict,
)


def _row(index: int, outcome: str, split: str = "train") -> dict:
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
        "task": "math",
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


def test_new_partition_is_deterministic_and_complete(tmp_path: Path) -> None:
    rows: list[dict] = []
    index = 1
    for split in SPLITS:
        for outcome in OUTCOMES:
            while assigned_split(f"{index:064x}") != split:
                index += 1
            rows.append(_row(index, outcome))
            index += 1
    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    loaded = load_partitioned_pairs(path, 2026080811)
    assert {row["split"] for row in loaded} == set(SPLITS)
    for split in SPLITS:
        assert {row["outcome_class"] for row in loaded if row["split"] == split} == set(
            OUTCOMES
        )
    receipt = partition_receipt(loaded, 2026080811)
    assert sum(value["total"] for value in receipt["counts"].values()) == len(rows)


def test_order_labels_and_semantic_mapping() -> None:
    base_only = _row(1, "base_only")
    assert ordered_label(base_only, (0, 1)) == 0
    assert ordered_label(base_only, (1, 0)) == 2
    assert ordered_label(_row(2, "both_wrong"), (0, 1)) == 1
    assert semantic_verdict(0, (0, 1)) == "base"
    assert semantic_verdict(2, (1, 0)) == "base"


def test_conservative_selection_requires_order_agreement() -> None:
    assert conservative_selection(0, 2)[:2] == (0, True)
    assert conservative_selection(2, 0)[:2] == (1, True)
    assert conservative_selection(0, 0)[:2] == (1, False)
    assert conservative_selection(1, 1)[:2] == (1, True)


def test_metrics_gate_rewards_correct_conservative_switching() -> None:
    rows: list[dict] = []
    outcomes = ["base_only"] * 10 + ["expert_only"] * 20
    outcomes += ["both_correct"] * 30 + ["both_wrong"] * 40
    predictions: dict[str, tuple[int, int]] = {}
    for index, outcome in enumerate(outcomes, start=1):
        row = _row(index, outcome, split="holdout")
        rows.append(row)
        if outcome == "base_only":
            predictions[row["identity_sha256"]] = (0, 2)
        elif outcome == "expert_only":
            predictions[row["identity_sha256"]] = (2, 0)
        else:
            predictions[row["identity_sha256"]] = (1, 1)
    report = metrics_from_predictions(rows, predictions, split="holdout")
    assert report["gate_pass"] is True
    assert report["metrics"]["overall"]["selected_accuracy"] == 0.6
    assert report["metrics"]["overall"]["base_commit_rate"] == 0.1


def test_prompt_contains_only_joint_inference_fields() -> None:
    text = pairwise_text("What is 2+2?", "A says 4", "B says 5")
    assert "What is 2+2?" in text
    assert "A says 4" in text
    assert "B says 5" in text
    assert "benchmark" not in text.casefold()
    assert "gold" not in text.casefold()
    assert "correct" not in text.casefold()
