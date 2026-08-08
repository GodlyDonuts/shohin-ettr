from __future__ import annotations

import json
from pathlib import Path

import pytest

from hf_cvg1_apply_completion_verifier import (
    CVG1ApplicationError,
    PAIR_SCHEMA,
    TASKS,
    _summarize,
    load_evaluation_pairs,
)


def _rows() -> list[dict]:
    rows: list[dict] = []
    for index, task in enumerate(TASKS):
        rows.append(
            {
                "schema": PAIR_SCHEMA,
                "identity_sha256": f"{index:064x}",
                "task": task,
                "question": f"question {index}",
                "candidates": [
                    {"lineage": "base", "completion": "base", "correct": False},
                    {"lineage": "expert", "completion": "expert", "correct": True},
                ],
            }
        )
    return rows


def test_loader_orders_lineages_and_requires_all_tasks(tmp_path: Path) -> None:
    rows = _rows()
    rows[0]["candidates"].reverse()
    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    loaded = load_evaluation_pairs(path)
    assert [item["lineage"] for item in loaded[0]["candidates"]] == [
        "base",
        "expert",
    ]

    path.write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]))
    with pytest.raises(CVG1ApplicationError, match="task coverage"):
        load_evaluation_pairs(path)


def test_summary_uses_one_whole_lineage_per_example() -> None:
    rows = _rows()
    selected = {row["identity_sha256"]: 1 for row in rows}
    report = _summarize(rows, selected)
    assert report["arms"]["selected"]["solved"] == 6
    assert report["arms"]["base"]["solved"] == 0
    assert report["comparison"]["strongest_single_lineage"] == "expert"
    assert report["comparison"]["solved_delta_selected_vs_strongest"] == 0
    assert report["comparison"]["development_gate_pass"] is False
