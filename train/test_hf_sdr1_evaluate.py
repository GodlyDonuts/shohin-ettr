from __future__ import annotations

from hf_sdr1_evaluate import summarize


def _row(index: int, task: str, both_wrong: bool) -> dict:
    return {
        "identity_sha256": f"{index:064x}",
        "task": task,
        "candidates": [
            {"lineage": "base", "correct": not both_wrong},
            {"lineage": "expert", "correct": not both_wrong},
        ],
    }


def test_holdout_summary_applies_standalone_retention_gate() -> None:
    rows = []
    results = []
    index = 0
    for task, total, correct in (
        ("math500", 621, 300),
        ("bbh_logic", 625, 350),
        ("mbpp", 33, 25),
    ):
        for offset in range(total):
            index += 1
            row = _row(index, task, offset < 150)
            rows.append(row)
            results.append(
                {
                    "identity_sha256": row["identity_sha256"],
                    "correct": offset < correct,
                }
            )
    report = summarize(rows, results, "holdout")
    assert report["gate_pass"] is True
    assert report["metrics"]["overall"]["generated_correct"] == 675
