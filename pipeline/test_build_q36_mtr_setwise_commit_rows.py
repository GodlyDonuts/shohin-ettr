from __future__ import annotations

import json
from pathlib import Path

import pytest

import build_q36_mtr_setwise_commit_rows as module
from test_build_q36_mtr_multi_owner_commit_pairs import _fixture as pair_fixture


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(module.base, "COUNTS", {"train": 16, "development": 16})
    args = pair_fixture(tmp_path)
    for owner_index, paths in enumerate(
        (args.current_train_score, args.owner71_train_score, args.owner8_train_score)
    ):
        for index, path in enumerate(paths):
            payload = json.loads(path.read_text(encoding="utf-8"))
            pattern = f"{index % 8:03b}"
            payload["outcomes"][0]["correct"] = pattern[owner_index] == "1"
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return args


def test_builds_complete_three_owner_setwise_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _fixture(tmp_path, monkeypatch)
    train_report, development_report = module.build(args)
    train = [json.loads(line) for line in args.training_output.read_text().splitlines()]
    development = [
        json.loads(line) for line in args.development_output.read_text().splitlines()
    ]
    assert train_report["correctness_patterns"] == {
        pattern: 2 for pattern in module.PATTERNS
    }
    assert train_report["owner_lineages"] == list(module.OWNER_NAMES)
    assert development_report["labels_or_correctness_fields"] == 0
    assert all(len(row["candidates"]) == 3 for row in train + development)
    assert all(
        [candidate["lineage"] for candidate in row["candidates"]]
        == list(module.OWNER_NAMES)
        for row in train + development
    )
    assert all(
        "correct" not in candidate
        for row in development
        for candidate in row["candidates"]
    )


def test_rejects_missing_correctness_pattern(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _fixture(tmp_path, monkeypatch)
    for path in (
        args.current_train_score + args.owner71_train_score + args.owner8_train_score
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["outcomes"][0]["correct"] = False
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(module.Q36MTRSetwiseDataError, match="correctness pattern"):
        module.build(args)


def test_rejects_score_candidate_hash_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _fixture(tmp_path, monkeypatch)
    payload = json.loads(args.owner8_train_score[0].read_text(encoding="utf-8"))
    payload["candidates_sha256"] = "f" * 64
    args.owner8_train_score[0].write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(module.base.Q36MTROwnerCommitPairError):
        module.build(args)
