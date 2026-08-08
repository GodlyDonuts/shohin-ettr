from __future__ import annotations

from hf_idr1_evaluate_reviser import FROZEN_FLOORS, summarize


def _rows_results(split: str, generated_correct: int):
    floors = FROZEN_FLOORS[split]
    tasks = (
        ("math500", floors["math500"]),
        ("bbh_logic", floors["bbh_logic"]),
        ("mbpp", floors["mbpp"]),
    )
    rows = []
    results = []
    index = 0
    for task, correct in tasks:
        for task_index in range(correct):
            identity = f"{index:064x}"
            rows.append(
                {
                    "identity_sha256": identity,
                    "task": task,
                    "candidates": [
                        {"lineage": "base", "correct": False},
                        {"lineage": "expert", "correct": False},
                    ],
                }
            )
            results.append({"identity_sha256": identity, "correct": True})
            index += 1
    for _ in range(max(0, generated_correct - len(rows))):
        identity = f"{index:064x}"
        rows.append(
            {
                "identity_sha256": identity,
                "task": "math500",
                "candidates": [
                    {"lineage": "base", "correct": False},
                    {"lineage": "expert", "correct": False},
                ],
            }
        )
        results.append({"identity_sha256": identity, "correct": True})
        index += 1
    return rows, results


def test_idr1_frozen_holdout_floor_passes_at_boundary() -> None:
    rows, results = _rows_results("holdout", FROZEN_FLOORS["holdout"]["overall"])
    report = summarize(rows, results, "holdout")
    assert report["gate_pass"]
