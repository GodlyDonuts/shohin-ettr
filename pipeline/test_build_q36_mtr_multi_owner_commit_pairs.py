from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

import build_q36_mtr_multi_owner_commit_pairs as module


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> argparse.Namespace:
    train_ids = [f"{index:064x}" for index in range(16)]
    development_ids = [f"{100 + index:064x}" for index in range(16)]
    train_source = tmp_path / "train.jsonl"
    development_source = tmp_path / "development.jsonl"
    _write_jsonl(
        train_source,
        [
            {
                "schema": "shohin-pcf1-train-source-v1",
                "identity_sha256": identity,
                "split": "train",
                "task": "math500" if index % 2 else "bbh_logic",
                "source_prompt": f"train question {index}",
                "assessor": {"identity_sha256": identity},
            }
            for index, identity in enumerate(train_ids)
        ],
    )
    _write_jsonl(
        development_source,
        [
            {
                "schema": "shohin-pcf1-development-source-v1",
                "identity_sha256": identity,
                "split": "development",
                "task": "math500" if index % 2 else "bbh_logic",
                "source_prompt": f"development question {index}",
            }
            for index, identity in enumerate(development_ids)
        ],
    )
    owner_keys = ("current", "owner71", "owner8")
    candidates = {owner: [] for owner in owner_keys}
    train_scores = {owner: [] for owner in owner_keys}
    development_scores = {owner: [] for owner in owner_keys}
    for index, (train_id, development_id) in enumerate(
        zip(train_ids, development_ids, strict=True)
    ):
        left, right = module.PAIR_CHOICES[
            module.owner_pair_index(train_id, module.CALIBRATION_SEED)
        ]
        desired = module.OUTCOMES[index % len(module.OUTCOMES)]
        correctness = [False, False, False]
        if desired in ("both_correct", "revision_only"):
            correctness[left] = True
        if desired in ("both_correct", "unchanged_only"):
            correctness[right] = True
        for owner_index, owner in enumerate(owner_keys):
            candidate_path = tmp_path / f"{owner}_{index:02d}.jsonl"
            _write_jsonl(
                candidate_path,
                [
                    {
                        "schema": module.base.CANDIDATE_SCHEMA,
                        "identity_sha256": train_id,
                        "split": "train",
                        "task": "math500" if index % 2 else "bbh_logic",
                        "completion": f"{owner} train {index}",
                        "generated_tokens": 8,
                        "max_token_exhausted": False,
                    },
                    {
                        "schema": module.base.CANDIDATE_SCHEMA,
                        "identity_sha256": development_id,
                        "split": "development",
                        "task": "math500" if index % 2 else "bbh_logic",
                        "completion": f"{owner} development {index}",
                        "generated_tokens": 9,
                        "max_token_exhausted": False,
                    },
                ],
            )
            candidates[owner].append(candidate_path)
            for split, identity, correct, destination in (
                ("train", train_id, correctness[owner_index], train_scores),
                ("development", development_id, False, development_scores),
            ):
                score_path = tmp_path / f"{owner}_{split}_{index:02d}.json"
                score_path.write_text(
                    json.dumps(
                        {
                            "schema": module.base.SCORE_SCHEMA,
                            "status": "complete",
                            "split": split,
                            "rows": 1,
                            "candidates_sha256": _sha(candidate_path),
                            "outcomes": [
                                {
                                    "identity_sha256": identity,
                                    "task": "math500" if index % 2 else "bbh_logic",
                                    "correct": correct,
                                }
                            ],
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                destination[owner].append(score_path)
    return argparse.Namespace(
        train_source=train_source,
        development_source=development_source,
        current_candidates=candidates["current"],
        owner71_candidates=candidates["owner71"],
        owner8_candidates=candidates["owner8"],
        current_train_score=train_scores["current"],
        owner71_train_score=train_scores["owner71"],
        owner8_train_score=train_scores["owner8"],
        current_development_score=development_scores["current"],
        owner71_development_score=development_scores["owner71"],
        owner8_development_score=development_scores["owner8"],
        training_output=tmp_path / "training.jsonl",
        training_report=tmp_path / "training_report.json",
        development_output=tmp_path / "development_pairs.jsonl",
        development_report=tmp_path / "development_report.json",
        seed=module.CALIBRATION_SEED,
    )


def test_owner_pair_assignment_is_deterministic_and_distributed() -> None:
    identities = [f"{index:064x}" for index in range(100)]
    first = [
        module.owner_pair_index(identity, module.CALIBRATION_SEED)
        for identity in identities
    ]
    assert first == [
        module.owner_pair_index(identity, module.CALIBRATION_SEED)
        for identity in identities
    ]
    assert set(first) == {0, 1, 2}


def test_builds_diversified_training_and_label_free_development(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module.base, "COUNTS", {"train": 16, "development": 16})
    args = _fixture(tmp_path)
    training_report, development_report = module.build(args)
    training = [
        json.loads(line) for line in args.training_output.read_text().splitlines()
    ]
    development = [
        json.loads(line) for line in args.development_output.read_text().splitlines()
    ]
    assert len(training) == len(development) == 16
    assert set(training_report["outcomes"]) == set(module.OUTCOMES)
    assert set(training_report["owner_pair_counts"]) == {
        "current:owner_71",
        "current:owner_8",
        "owner_71:owner_8",
    }
    assert training_report["multi_owner_diversified_pairwise"] is True
    assert development_report["labels_or_correctness_fields"] == 0
    assert all(
        set(candidate) == {"lineage", "completion"}
        for row in development
        for candidate in row["candidates"]
    )


def test_rejects_candidate_score_hash_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module.base, "COUNTS", {"train": 16, "development": 16})
    args = _fixture(tmp_path)
    report = json.loads(args.current_train_score[0].read_text())
    report["candidates_sha256"] = "f" * 64
    args.current_train_score[0].write_text(json.dumps(report) + "\n", encoding="utf-8")
    with pytest.raises(module.base.Q36MTROwnerCommitPairError):
        module.build(args)
