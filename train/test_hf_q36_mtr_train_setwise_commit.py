from __future__ import annotations

import json
from pathlib import Path

import pytest

import hf_q36_mtr_train_setwise_commit as module


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    index = 0
    for task in module.TASKS:
        for pattern in module.PATTERNS:
            correct = [value == "1" for value in pattern]
            rows.append(
                {
                    "schema": module.ROW_SCHEMA,
                    "identity_sha256": f"{index:064x}",
                    "split": "calibration_train",
                    "task": task,
                    "question": f"question {index}",
                    "correctness_pattern": pattern,
                    "candidates": [
                        {
                            "lineage": lineage,
                            "completion": f"{lineage} completion {index}",
                            "correct": value,
                            "generated_tokens": 10,
                            "max_token_exhausted": False,
                        }
                        for lineage, value in zip(
                            module.OWNER_NAMES, correct, strict=True
                        )
                    ],
                }
            )
            index += 1
    duplicate = json.loads(json.dumps(rows[0]))
    duplicate["identity_sha256"] = f"{index:064x}"
    duplicate["split"] = "calibration_development"
    rows.append(duplicate)
    return rows


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_loads_exact_setwise_training_geometry(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    expected = _rows()
    _write(path, expected)
    assert module._load_rows(path) == expected


def test_training_plan_is_deterministic_and_pattern_balanced() -> None:
    rows = _rows()
    first, receipt = module.training_plan(rows, seed=module.SEED, presentations=240)
    second, second_receipt = module.training_plan(
        rows, seed=module.SEED, presentations=240
    )
    assert first == second
    assert receipt == second_receipt
    assert len(first) == 240
    assert set(receipt["stratum_presentations"]) == {
        f"{task}:{pattern}" for task in module.TASKS for pattern in module.PATTERNS
    }
    assert set(receipt["stratum_presentations"].values()) == {10}


def test_rejects_correctness_pattern_substitution(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = _rows()
    rows[0]["correctness_pattern"] = "111"
    _write(path, rows)
    with pytest.raises(module.Q36MTRSetwiseTrainError, match="pattern"):
        module._load_rows(path)


def test_rejects_owner_order_substitution(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = _rows()
    candidates = rows[0]["candidates"]
    assert isinstance(candidates, list)
    candidates[0], candidates[1] = candidates[1], candidates[0]
    _write(path, rows)
    with pytest.raises(module.Q36MTRSetwiseTrainError, match="row differs"):
        module._load_rows(path)
