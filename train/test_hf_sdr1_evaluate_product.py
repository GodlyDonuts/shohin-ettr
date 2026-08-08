from __future__ import annotations

from hf_sdr1_evaluate_product import summarize


def _row(index: int, task: str, expert: bool) -> dict:
    return {
        "identity_sha256": f"{index:064x}",
        "task": task,
        "candidates": [
            {"lineage": "base", "correct": expert},
            {"lineage": "expert", "correct": expert},
        ],
    }


def test_standalone_product_summary_passes_frozen_gate() -> None:
    rows = []
    results = []
    index = 0
    counts = {
        "gsm8k": 100,
        "math500": 100,
        "humaneval": 20,
        "mbpp": 20,
        "gpqa": 198,
        "bbh_logic": 100,
        "aime": 30,
    }
    for task, total in counts.items():
        for offset in range(total):
            index += 1
            row = _row(index, task, offset < total // 2)
            rows.append(row)
            results.append(
                {
                    "identity_sha256": row["identity_sha256"],
                    "correct": offset < int(total * 0.8),
                }
            )
    report = summarize(rows, results)
    assert report["standalone_comparison"]["gate_pass"] is True
    assert report["arms"]["generated"]["solved"] >= 350
