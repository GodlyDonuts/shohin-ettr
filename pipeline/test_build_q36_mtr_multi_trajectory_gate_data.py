from __future__ import annotations

import argparse
import hashlib
import json

import pytest

import build_q36_mtr_multi_trajectory_gate_data as module


def _identity(index: int) -> str:
    return hashlib.sha256(f"multi-{index}".encode()).hexdigest()


def _write_jsonl(path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _fixture(tmp_path):
    tasks = module.TASKS
    development = []
    candidates = {arm: [] for arm in module.BRANCHES}
    correctness = {
        "revision": [True, True, False],
        "draft_hidden": [False, True, True],
    }
    for index, task in enumerate(tasks):
        identity = _identity(index)
        development.append(
            {
                "schema": module.EVAL_SCHEMA,
                "split": "development",
                "identity_sha256": identity,
                "task": task,
                "question": f"question {index}",
            }
        )
        for arm in module.BRANCHES:
            candidates[arm].append(
                {
                    "schema": module.CANDIDATE_SCHEMA,
                    "arm": arm,
                    "identity_sha256": identity,
                    "task": task,
                    "completion": f"{arm} answer {index}",
                }
            )
    development_path = tmp_path / "development.jsonl"
    _write_jsonl(development_path, development)
    candidate_paths = {}
    score_paths = {}
    for arm in module.BRANCHES:
        candidate_path = tmp_path / f"{arm}.jsonl"
        _write_jsonl(candidate_path, candidates[arm])
        candidate_paths[arm] = candidate_path
        score_path = tmp_path / f"{arm}.score.json"
        score_path.write_text(
            json.dumps(
                {
                    "schema": module.SCORE_SCHEMA,
                    "status": "complete",
                    "split": "development",
                    "evaluation_arm": arm,
                    "candidates_sha256": module.sha256_file(candidate_path),
                    "outcomes": [
                        {
                            "identity_sha256": _identity(index),
                            "correct": correctness[arm][index],
                        }
                        for index in range(3)
                    ],
                }
            )
        )
        score_paths[arm] = score_path
    return argparse.Namespace(
        development_eval=development_path,
        revision_candidates=candidate_paths["revision"],
        revision_score=score_paths["revision"],
        draft_hidden_candidates=candidate_paths["draft_hidden"],
        draft_hidden_score=score_paths["draft_hidden"],
        expected_rows=3,
        output=tmp_path / "train.jsonl",
        report=tmp_path / "report.json",
    )


def test_build_balances_exclusive_correct_branches(tmp_path) -> None:
    args = _fixture(tmp_path)
    report = module.run(args)
    assert report["unique_outcome_counts"] == {
        "both_correct": 1,
        "draft_hidden_only": 1,
        "revision_only": 1,
    }
    assert report["presentation_counts"] == {
        "both_correct": 1,
        "draft_hidden_only": 8,
        "revision_only": 2,
    }
    assert report["presentations"] == 11
    rows = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert len(rows) == 11
    assert {tuple(row["routing_target"]) for row in rows} == {
        (1.0, 0.0),
        (0.5, 0.5),
        (0.0, 1.0),
    }
    assert module.sha256_file(args.output) == report["output_sha256"]


def test_build_rejects_score_candidate_hash_tamper(tmp_path) -> None:
    args = _fixture(tmp_path)
    score = json.loads(args.revision_score.read_text())
    score["candidates_sha256"] = "0" * 64
    args.revision_score.write_text(json.dumps(score))
    with pytest.raises(module.Q36MTRMultiTrajectoryDataError):
        module.run(args)
