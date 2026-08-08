from __future__ import annotations

from hf_vcr1_evaluate_reviser import summarize


def _row(index: int, task: str, base: bool, expert: bool) -> dict:
    return {
        "identity_sha256": f"{index:064x}",
        "task": task,
        "candidates": [
            {"lineage": "base", "correct": base},
            {"lineage": "expert", "correct": expert},
        ],
    }


def test_summary_counts_repairs_and_conjunctive_gate() -> None:
    rows = []
    results = []
    index = 0
    for task in ("math500", "bbh_logic", "mbpp"):
        for offset in range(100):
            index += 1
            both_wrong = offset < 10
            row = _row(index, task, not both_wrong, not both_wrong)
            rows.append(row)
            results.append(
                {
                    "identity_sha256": row["identity_sha256"],
                    "correct": True,
                }
            )
    report = summarize(rows, results)
    assert report["gate_pass"] is True
    assert report["metrics"]["overall"]["both_wrong_repaired"] == 30
    assert report["metrics"]["overall"]["generated_correct"] == 300
