from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from build_q36_mtr_commit_pairs import OUTCOMES, PAIR_SCHEMA
from hf_q36_mtr_train_commit import (
    Q36MTRCommitError,
    SEED,
    _balanced_strata,
    _load_development_pairs,
    _load_pairs,
)


def _pairs(path: Path) -> None:
    rows = []
    for index in range(5_824):
        outcome = OUTCOMES[index % len(OUTCOMES)]
        correct = {
            "both_correct": (True, True),
            "revision_only": (True, False),
            "both_wrong": (False, False),
            "unchanged_only": (False, True),
        }[outcome]
        rows.append(
            {
                "schema": PAIR_SCHEMA,
                "identity_sha256": hashlib.sha256(f"pair-{index}".encode()).hexdigest(),
                "split": (
                    "calibration_train" if index < 4_700 else "calibration_development"
                ),
                "task": ("math500", "bbh_logic", "mbpp")[index % 3],
                "question": f"question-{index}",
                "outcome_class": outcome,
                "candidates": [
                    {
                        "lineage": "revision",
                        "completion": "left",
                        "correct": correct[0],
                        "generated_tokens": 1,
                        "max_token_exhausted": False,
                    },
                    {
                        "lineage": "unchanged",
                        "completion": "right",
                        "correct": correct[1],
                        "generated_tokens": 1,
                        "max_token_exhausted": False,
                    },
                ],
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_q36_commit_pair_loader_and_strata_are_exact(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    _pairs(path)
    rows = _load_pairs(path)
    strata = _balanced_strata(rows, SEED)
    assert len(rows) == 5_824
    assert {outcome for _, outcome in strata} == set(OUTCOMES)
    assert {task for task, _ in strata} == {"math500", "bbh_logic", "mbpp"}


def test_q36_commit_pair_loader_rejects_label_tampering(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    _pairs(path)
    rows = path.read_text().splitlines()
    row = json.loads(rows[0])
    row["candidates"][0].pop("correct")
    rows[0] = json.dumps(row)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(Q36MTRCommitError):
        _load_pairs(path)


def test_q36_development_pair_loader_is_label_free(tmp_path: Path) -> None:
    path = tmp_path / "development.jsonl"
    rows = [
        {
            "schema": PAIR_SCHEMA,
            "identity_sha256": hashlib.sha256(f"dev-{index}".encode()).hexdigest(),
            "split": "development",
            "task": ("math500", "bbh_logic", "mbpp")[index % 3],
            "question": f"question-{index}",
            "candidates": [
                {"lineage": "revision", "completion": "left"},
                {"lineage": "unchanged", "completion": "right"},
            ],
        }
        for index in range(1_289)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert len(_load_development_pairs(path)) == 1_289
    rows[0]["candidates"][0]["correct"] = True
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(Q36MTRCommitError):
        _load_development_pairs(path)


def test_q36_commit_wrapper_is_one_h100_and_no_requeue() -> None:
    source = (
        Path(__file__).resolve().parent / "jobs/q36_mtr_train_commit.sbatch"
    ).read_text()
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in source
    assert "#SBATCH --no-requeue" in source
    assert "--updates 128" in source
    assert "DEVELOPMENT_PAIRS" in source
    assert "sbatch " not in source
