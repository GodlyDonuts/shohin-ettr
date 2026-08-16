from __future__ import annotations

import argparse
import hashlib
import json

import pytest

import build_q36_mtr_tri_trajectory_gate_data as module


def _write_jsonl(path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _fixture(tmp_path):
    patterns = []
    for pattern, count in module.EXPECTED_PATTERNS.items():
        patterns.extend([pattern] * count)
    development = []
    owner_candidates = []
    candidates = {branch: [] for branch in ("revision", "draft_hidden")}
    identities = []
    for index, pattern in enumerate(patterns):
        identity = hashlib.sha256(f"tri-{index}".encode()).hexdigest()
        task = module.TASKS[index % len(module.TASKS)]
        identities.append(identity)
        development.append(
            {
                "schema": "shohin-q36-mtr-eval-v1",
                "split": "development",
                "identity_sha256": identity,
                "task": task,
                "question": f"question {index}",
            }
        )
        owner_candidates.append(
            {
                "schema": module.MODEL_DRAFT_SCHEMA,
                "split": "development",
                "identity_sha256": identity,
                "task": task,
                "completion": f"owner answer {index}",
            }
        )
        for branch in candidates:
            candidates[branch].append(
                {
                    "schema": "shohin-q36-mtr-candidate-v1",
                    "arm": branch,
                    "identity_sha256": identity,
                    "task": task,
                    "completion": f"{branch} answer {index}",
                }
            )
    development_path = tmp_path / "development.jsonl"
    owner_path = tmp_path / "owner.jsonl"
    _write_jsonl(development_path, development)
    _write_jsonl(owner_path, owner_candidates)
    candidate_paths = {}
    score_paths = {}
    for branch, branch_index in (("revision", 1), ("draft_hidden", 2)):
        candidate_path = tmp_path / f"{branch}.jsonl"
        _write_jsonl(candidate_path, candidates[branch])
        score_path = tmp_path / f"{branch}.score.json"
        score_path.write_text(
            json.dumps(
                {
                    "schema": module.SCORE_SCHEMA,
                    "status": "complete",
                    "split": "development",
                    "evaluation_arm": branch,
                    "candidates_sha256": module.sha256_file(candidate_path),
                    "outcomes": [
                        {
                            "identity_sha256": identity,
                            "correct": patterns[index][branch_index],
                        }
                        for index, identity in enumerate(identities)
                    ],
                }
            )
        )
        candidate_paths[branch] = candidate_path
        score_paths[branch] = score_path
    owner_scores = []
    receipts = []
    for shard in range(16):
        candidate_sha = hashlib.sha256(f"owner-shard-{shard}".encode()).hexdigest()
        receipts.append({"candidates_sha256": candidate_sha})
        selected = [index for index in range(len(patterns)) if index % 16 == shard]
        score_path = tmp_path / f"owner_{shard:02d}.score.json"
        score_path.write_text(
            json.dumps(
                {
                    "schema": module.SCORE_SCHEMA,
                    "status": "complete",
                    "split": "development",
                    "candidates_sha256": candidate_sha,
                    "outcomes": [
                        {
                            "identity_sha256": identities[index],
                            "correct": patterns[index][0],
                        }
                        for index in selected
                    ],
                }
            )
        )
        owner_scores.append(score_path)
    merge_report = tmp_path / "owner.merge.json"
    merge_report.write_text(
        json.dumps(
            {
                "schema": module.MERGE_SCHEMA,
                "status": "complete",
                "rows": 7_113,
                "output_sha256": module.sha256_file(owner_path),
                "input_receipts": receipts,
            }
        )
    )
    return argparse.Namespace(
        development_eval=development_path,
        owner_candidates=owner_path,
        owner_merge_report=merge_report,
        owner_score=owner_scores,
        revision_candidates=candidate_paths["revision"],
        revision_score=score_paths["revision"],
        draft_hidden_candidates=candidate_paths["draft_hidden"],
        draft_hidden_score=score_paths["draft_hidden"],
        expected_rows=1_289,
        output=tmp_path / "train.jsonl",
        report=tmp_path / "report.json",
    )


def test_builds_exact_retention_aware_tri_trajectory_geometry(tmp_path) -> None:
    args = _fixture(tmp_path)
    report = module.run(args)
    assert report["unique_selected_rows"] == module.EXPECTED_UNIQUE
    assert report["presentations"] == module.EXPECTED_PRESENTATIONS
    assert report["branch_names"] == list(module.BRANCHES)
    rows = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert len(rows) == module.EXPECTED_PRESENTATIONS
    assert all(len(row["routing_target"]) == 3 for row in rows)
    assert all(abs(sum(row["routing_target"]) - 1.0) < 1.0e-8 for row in rows)


def test_rejects_missing_owner_score_shard(tmp_path) -> None:
    args = _fixture(tmp_path)
    args.owner_score.pop()
    with pytest.raises(module.Q36MTRTriTrajectoryDataError):
        module.run(args)
