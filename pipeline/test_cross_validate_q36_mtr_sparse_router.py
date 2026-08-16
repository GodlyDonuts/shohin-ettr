from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import cross_validate_q36_mtr_sparse_router as module


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _lines(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_heuristic_is_sequential_and_stable() -> None:
    candidates = [
        {
            "task": "math500",
            "completion": "unfinished",
            "max_token_exhausted": True,
        },
        {
            "task": "math500",
            "completion": "The answer is 4.",
            "max_token_exhausted": False,
        },
        {
            "task": "math500",
            "completion": "The answer is 5.",
            "max_token_exhausted": False,
        },
    ]
    assert module._heuristic(candidates) == 1
    assert module._mcnemar(0, 0) == 1.0


def test_two_fold_cross_validation_is_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "FOLDS", 2)
    monkeypatch.setattr(module, "EPOCHS", 2)
    monkeypatch.setattr(module.router, "DEVELOPMENT_ROWS", 4)
    source_path = tmp_path / "source.jsonl"
    identities = [_sha(f"row-{index}") for index in range(4)]
    _lines(
        source_path,
        [
            {
                "schema": module.SOURCE_SCHEMA,
                "identity_sha256": identity,
                "split": "development",
                "task": "math500",
                "source_prompt": f"Compute value {index}.",
            }
            for index, identity in enumerate(identities)
        ],
    )
    candidate_groups: list[list[Path]] = [[], [], []]
    score_groups: list[list[Path]] = [[], [], []]
    for owner_index, lineage in enumerate(module.router.LINEAGES):
        for fold in range(2):
            candidates_path = tmp_path / f"{lineage}-{fold}.jsonl"
            rows = []
            outcomes = []
            for row_index in range(fold, 4, 2):
                correct = owner_index == row_index % 3
                completion = (
                    "The answer is correct."
                    if correct
                    else "Unfinished attempt without answer"
                )
                rows.append(
                    {
                        "schema": module.router.CANDIDATE_SCHEMA,
                        "identity_sha256": identities[row_index],
                        "split": "development",
                        "task": "math500",
                        "completion": completion,
                        "generated_tokens": len(completion.split()),
                        "max_token_exhausted": not correct,
                    }
                )
                outcomes.append(
                    {
                        "identity_sha256": identities[row_index],
                        "task": "math500",
                        "correct": correct,
                    }
                )
            _lines(candidates_path, rows)
            score_path = tmp_path / f"{lineage}-{fold}-score.json"
            score_path.write_text(
                json.dumps(
                    {
                        "schema": module.SCORE_SCHEMA,
                        "status": "complete",
                        "split": "development",
                        "rows": len(outcomes),
                        "correct": sum(row["correct"] for row in outcomes),
                        "candidates_sha256": module.router.sha256_file(candidates_path),
                        "outcomes": outcomes,
                    }
                ),
                encoding="utf-8",
            )
            candidate_groups[owner_index].append(candidates_path)
            score_groups[owner_index].append(score_path)
    args = type(
        "Args",
        (),
        {
            "development_source": source_path,
            "current_candidates": candidate_groups[0],
            "owner71_candidates": candidate_groups[1],
            "owner8_candidates": candidate_groups[2],
            "current_scores": score_groups[0],
            "owner71_scores": score_groups[1],
            "owner8_scores": score_groups[2],
            "output": tmp_path / "result.json",
        },
    )()
    result = module.cross_validate(args)
    assert result["status"] == "complete"
    assert result["rows"] == 4
    assert len(result["folds"]) == 2
    assert {row["identity_sha256"] for row in result["outcomes"]} == set(identities)
    assert result["training_labels_exclude_held_out_fold"] is True
