import argparse
import hashlib
import json
from pathlib import Path

import pytest

import build_q36_mtr_owner_commit_pairs as module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> argparse.Namespace:
    train_source = tmp_path / "train.jsonl"
    development_source = tmp_path / "development.jsonl"
    train_ids = [f"{index:064x}" for index in range(16)]
    development_ids = [f"{100 + index:064x}" for index in range(16)]
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
    candidates: dict[str, list[Path]] = {"first": [], "second": []}
    scores: dict[str, dict[str, list[Path]]] = {
        "first": {"train": [], "development": []},
        "second": {"train": [], "development": []},
    }
    for owner in ("first", "second"):
        for index, (train_id, development_id) in enumerate(
            zip(train_ids, development_ids, strict=True)
        ):
            candidate_path = tmp_path / f"{owner}_{index:02d}.jsonl"
            candidate_rows = [
                {
                    "schema": module.CANDIDATE_SCHEMA,
                    "identity_sha256": train_id,
                    "split": "train",
                    "task": "math500" if index % 2 else "bbh_logic",
                    "completion": f"{owner} train completion {index}",
                    "generated_tokens": 10 + index,
                    "max_token_exhausted": index % 5 == 0,
                },
                {
                    "schema": module.CANDIDATE_SCHEMA,
                    "identity_sha256": development_id,
                    "split": "development",
                    "task": "math500" if index % 2 else "bbh_logic",
                    "completion": f"{owner} development completion {index}",
                    "generated_tokens": 20 + index,
                    "max_token_exhausted": index % 7 == 0,
                },
            ]
            _write_jsonl(candidate_path, candidate_rows)
            candidates[owner].append(candidate_path)
            for split, identity in (
                ("train", train_id),
                ("development", development_id),
            ):
                score_path = tmp_path / f"{owner}_{split}_{index:02d}.json"
                correct = index % 2 == 0 if owner == "first" else index % 3 == 0
                score_path.write_text(
                    json.dumps(
                        {
                            "schema": module.SCORE_SCHEMA,
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
                scores[owner][split].append(score_path)
    return argparse.Namespace(
        train_source=train_source,
        development_source=development_source,
        first_candidates=candidates["first"],
        second_candidates=candidates["second"],
        first_train_score=scores["first"]["train"],
        second_train_score=scores["second"]["train"],
        first_development_score=scores["first"]["development"],
        second_development_score=scores["second"]["development"],
        training_output=tmp_path / "training_pairs.jsonl",
        training_report=tmp_path / "training_report.json",
        development_output=tmp_path / "development_pairs.jsonl",
        development_report=tmp_path / "development_report.json",
        seed=module.CALIBRATION_SEED,
    )


def test_builds_labeled_training_and_label_free_development(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(module, "COUNTS", {"train": 16, "development": 16})
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
    assert training_report["owner_trajectory_pair"] is True
    assert all(row["split"].startswith("calibration_") for row in training)
    assert all(
        "correct" in candidate for row in training for candidate in row["candidates"]
    )
    assert development_report["labels_or_correctness_fields"] == 0
    assert all(
        set(candidate) == {"lineage", "completion"}
        for row in development
        for candidate in row["candidates"]
    )


def test_rejects_score_report_not_bound_to_candidate_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(module, "COUNTS", {"train": 16, "development": 16})
    args = _fixture(tmp_path)
    report = json.loads(args.first_train_score[0].read_text())
    report["candidates_sha256"] = "f" * 64
    args.first_train_score[0].write_text(json.dumps(report) + "\n", encoding="utf-8")
    with pytest.raises(
        module.Q36MTROwnerCommitPairError, match="score/candidate binding"
    ):
        module.build(args)
