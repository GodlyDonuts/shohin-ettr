from __future__ import annotations

import json
from pathlib import Path

import pytest

from build_cvg1_evaluation_pairs import (
    CVG1EvaluationPairError,
    EVAL_SCHEMA,
    TASKS,
    build_pairs,
)


def _report(path: Path, task: str, lineage: str, *, mismatch: bool = False) -> None:
    rows = [
        {
            "identity_sha256": f"{TASKS.index(task) * 2 + index:064x}",
            "question": f"question {task} {index}" + (" changed" if mismatch else ""),
            "gold": str(index),
            "prediction": str(index),
            "completion": f"{lineage} completion {index}",
            "correct": bool(index),
        }
        for index in range(2)
    ]
    payload = {
        "schema": EVAL_SCHEMA,
        "status": "complete",
        "task": task,
        "data_sha256": f"data-{task}",
        "selection_sha256": f"selection-{task}",
        "generation_mode": "greedy",
        "generation_seed": 31,
        "max_new_tokens": 768,
        "generation_stop_token_ids": [1],
        "subset_seed": 20260802,
        "effective_enable_thinking": False,
        "total": 2,
        "results": rows,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixtures(root: Path, *, mismatch_task: str | None = None):
    base: list[Path] = []
    expert: list[Path] = []
    for task in TASKS:
        base_path = root / f"base-{task}.json"
        expert_path = root / f"expert-{task}.json"
        _report(base_path, task, "base")
        _report(expert_path, task, "expert", mismatch=task == mismatch_task)
        base.append(base_path)
        expert.append(expert_path)
    return base, expert


def test_build_pairs_preserves_two_whole_lineages(tmp_path: Path) -> None:
    base, expert = _fixtures(tmp_path)
    rows, report = build_pairs(base, expert)
    assert len(rows) == 14
    assert report["rows"] == 14
    assert report["inference_fields"] == ["question", "completion"]
    assert [candidate["lineage"] for candidate in rows[0]["candidates"]] == [
        "base",
        "expert",
    ]


def test_build_pairs_rejects_question_mismatch(tmp_path: Path) -> None:
    base, expert = _fixtures(tmp_path, mismatch_task="math500")
    with pytest.raises(CVG1EvaluationPairError, match="questions differ"):
        build_pairs(base, expert)
