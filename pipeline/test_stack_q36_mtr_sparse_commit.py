from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import stack_q36_mtr_sparse_commit as module


def _identity(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _candidate(identity: str, task: str, completion: str, split: str) -> dict:
    return {
        "schema": module.router.CANDIDATE_SCHEMA,
        "identity_sha256": identity,
        "split": split,
        "task": task,
        "completion": completion,
        "generated_tokens": 5,
        "max_token_exhausted": False,
    }


def _lines(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_mbpp_always_retains_production_commit() -> None:
    row = {
        "task": "mbpp",
        "selected_lineage": "owner_8",
        "production_commit_lineage": "current",
        "margin_bin": 3,
    }
    assert module.choose_sparse(row, [row]) == (
        False,
        "conservative_executable_code_retention",
    )


def test_vote_uses_only_matching_discordant_rows() -> None:
    row = {
        "task": "math500",
        "selected_lineage": "current",
        "production_commit_lineage": "owner_71",
        "margin_bin": 2,
    }
    training = [
        {**row, "correct": True, "production_commit_correct": False},
        {**row, "correct": True, "production_commit_correct": True},
    ]
    assert module._vote(training, lambda item: (item["task"],), row, 1) is True


def test_logistic_stacker_excludes_each_selected_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "FOLDS", 2)
    monkeypatch.setattr(module, "LOGISTIC_STEPS", 5)
    rows = []
    for index in range(8):
        rows.append(
            {
                "identity_sha256": _identity(f"logistic-{index}"),
                "fold": index % 2,
                "task": "mbpp" if index == 0 else "math500",
                "selected_lineage": module.router.LINEAGES[index % 3],
                "production_commit_lineage": module.router.LINEAGES[(index + 1) % 3],
                "margin_bin": index % 4,
                "scores": [0.1 * index, 0.2, -0.1],
                "correct": index % 3 == 0,
                "production_commit_correct": index % 3 == 1,
            }
        )
    decisions, receipt = module.logistic_decisions(rows)
    assert len(decisions) == len(rows)
    assert decisions[rows[0]["identity_sha256"]] is False
    assert len(receipt["folds"]) == 2
    assert all(item["training_discordant_rows"] > 0 for item in receipt["folds"])


def test_stack_emits_complete_selected_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "FOLDS", 2)
    monkeypatch.setattr(module.router, "DEVELOPMENT_ROWS", 4)
    identities = [_identity(f"row-{index}") for index in range(4)]
    tasks = ["math500", "bbh_logic", "mbpp", "math500"]
    owner_paths: list[list[Path]] = [[], [], []]
    owner_rows: list[dict[str, dict]] = [{}, {}, {}]
    for owner_index, lineage in enumerate(module.router.LINEAGES):
        for shard in range(16):
            path = tmp_path / f"{lineage}-{shard}.jsonl"
            rows = []
            if shard < 2:
                for row_index in range(shard, 4, 2):
                    row = _candidate(
                        identities[row_index],
                        tasks[row_index],
                        f"owner {lineage} answer {row_index}",
                        "development",
                    )
                    rows.append(row)
                    owner_rows[owner_index][identities[row_index]] = row
            else:
                rows.append(
                    _candidate(
                        _identity(f"ignored-{owner_index}-{shard}"),
                        "math500",
                        "ignored training row",
                        "train",
                    )
                )
            _lines(path, rows)
            owner_paths[owner_index].append(path)

    production_paths: list[Path] = []
    production_scores: list[Path] = []
    production_correct = [True, False, True, False]
    for shard in range(16):
        path = tmp_path / f"production-{shard}.jsonl"
        rows = []
        outcomes = []
        if shard < 2:
            for row_index in range(shard, 4, 2):
                rows.append(owner_rows[0][identities[row_index]])
                outcomes.append(
                    {
                        "identity_sha256": identities[row_index],
                        "task": tasks[row_index],
                        "correct": production_correct[row_index],
                    }
                )
        else:
            rows.append(
                _candidate(
                    _identity(f"ignored-production-{shard}"),
                    "math500",
                    "ignored training row",
                    "train",
                )
            )
        _lines(path, rows)
        score = tmp_path / f"production-{shard}-score.json"
        score.write_text(
            json.dumps(
                {
                    "schema": module.cross_validation.SCORE_SCHEMA,
                    "status": "complete",
                    "split": "development",
                    "rows": len(outcomes),
                    "correct": sum(row["correct"] for row in outcomes),
                    "candidates_sha256": module.router.sha256_file(path),
                    "outcomes": outcomes,
                }
            ),
            encoding="utf-8",
        )
        production_paths.append(path)
        production_scores.append(score)

    cross_path = tmp_path / "cross.json"
    cross_path.write_text(
        json.dumps(
            {
                "schema": module.CV_SCHEMA,
                "status": "complete",
                "rows": 4,
                "training_labels_exclude_held_out_fold": True,
                "outcomes": [
                    {
                        "identity_sha256": identity,
                        "task": task,
                        "fold": index % 2,
                        "selected_lineage": module.router.LINEAGES[(index + 1) % 3],
                        "scores": [0.0, 0.2 + index, -0.1],
                        "correct": index in {1, 2, 3},
                    }
                    for index, (identity, task) in enumerate(zip(identities, tasks))
                ],
            }
        ),
        encoding="utf-8",
    )
    args = type(
        "Args",
        (),
        {
            "cross_validation_report": cross_path,
            "production_candidates": production_paths,
            "production_scores": production_scores,
            "current_candidates": owner_paths[0],
            "owner71_candidates": owner_paths[1],
            "owner8_candidates": owner_paths[2],
            "output": tmp_path / "selected.jsonl",
            "decisions": tmp_path / "decisions.jsonl",
            "report": tmp_path / "report.json",
        },
    )()
    result = module.stack(args)
    assert result["status"] == "complete"
    assert result["rows"] == 4
    assert result["training_excludes_selected_fold"] is True
    assert len(args.output.read_text().splitlines()) == 4
    decisions = [json.loads(line) for line in args.decisions.read_text().splitlines()]
    mbpp = next(row for row in decisions if row["task"] == "mbpp")
    assert mbpp["selected_source"] == "production_commit"
