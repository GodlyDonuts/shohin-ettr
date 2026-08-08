from __future__ import annotations

import pytest

from hf_vcr1_evaluate_product import VCR1ProductError, summarize


def _row(index: int, task: str, base: bool, expert: bool) -> dict:
    return {
        "identity_sha256": f"{index:064x}",
        "task": task,
        "candidates": [
            {"lineage": "expert", "correct": expert},
            {"lineage": "base", "correct": base},
        ],
    }


def _board(generated_fraction: float) -> tuple[list[dict], list[dict]]:
    rows = []
    results = []
    index = 0
    counts = {
        "gsm8k": 100,
        "math500": 100,
        "humaneval": 20,
        "mbpp": 20,
        "gpqa": 100,
        "bbh_logic": 100,
        "aime": 30,
    }
    for task, total in counts.items():
        for offset in range(total):
            index += 1
            row = _row(index, task, offset < total // 2, offset < total // 2)
            rows.append(row)
            results.append(
                {
                    "identity_sha256": row["identity_sha256"],
                    "correct": offset < int(total * generated_fraction),
                }
            )
    return rows, results


def test_summary_is_lineage_order_invariant_and_passes_full_gate() -> None:
    rows, results = _board(0.8)
    report = summarize(rows, results)
    assert report["comparison"]["gate_pass"] is True
    assert report["arms"]["generated"]["domains"]["code"]["correct"] == 32
    assert report["comparison"]["solved_delta"] == 132


def test_summary_rejects_incomplete_result_coverage() -> None:
    rows, results = _board(0.8)
    with pytest.raises(VCR1ProductError, match="coverage"):
        summarize(rows, results[:-1])
