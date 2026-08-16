"""Tests for the paired 120B-A12B screen reducer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import score_nemotron_super_screen as score


def _candidate(arm: str, identity: str) -> dict[str, object]:
    return {
        "schema": score.CANDIDATE_SCHEMA,
        "arm": arm,
        "identity_sha256": identity,
        "task": "math500",
        "completion": "answer",
        "generated_tokens": 1,
        "max_token_exhausted": False,
    }


def test_load_candidates_binds_arm_shards_and_identities(tmp_path: Path) -> None:
    identities = {f"{index:064x}" for index in range(4)}
    paths = []
    for index, identity in enumerate(sorted(identities)):
        path = tmp_path / f"{index}.jsonl"
        path.write_text(json.dumps(_candidate("revision", identity)) + "\n")
        paths.append(path)
    assert set(score.load_candidates("revision", paths, identities)) == identities
    row = _candidate("unchanged", sorted(identities)[0])
    paths[0].write_text(json.dumps(row) + "\n")
    with pytest.raises(score.NemotronSuperScoreError):
        score.load_candidates("revision", paths, identities)


def test_load_candidates_accepts_explicit_confirmation_shard_geometry(
    tmp_path: Path,
) -> None:
    identities = {f"{index:064x}" for index in range(16)}
    paths = []
    for index, identity in enumerate(sorted(identities)):
        path = tmp_path / f"{index:02d}.jsonl"
        path.write_text(json.dumps(_candidate("revision", identity)) + "\n")
        paths.append(path)
    assert (
        set(score.load_candidates("revision", paths, identities, shards=16))
        == identities
    )
    with pytest.raises(score.NemotronSuperScoreError):
        score.load_candidates("revision", paths[:15], identities, shards=16)


def test_paired_report_is_exact_and_directional() -> None:
    left = {"a": True, "b": True, "c": False, "d": True}
    right = {"a": False, "b": True, "c": True, "d": False}
    report = score.paired_report(left, right)
    assert report["left_only_correct"] == 2
    assert report["right_only_correct"] == 1
    assert report["net_correct"] == 1
    assert report["mcnemar_exact_two_sided_p"] == 1.0


def test_score_job_initializes_cluster_local_tmp() -> None:
    source = (
        Path(__file__)
        .with_name("jobs")
        .joinpath("nemotron_super_score.sbatch")
        .read_text()
    )
    assert "q36_init_local_tmp" in source
    assert "trap q36_cleanup_local_tmp EXIT" in source


def test_run_reduces_full_1023_row_sixteen_shard_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = 1023
    shards = 16
    identities = [f"{index:064x}" for index in range(rows)]
    assessors = {
        identity: {"task": "math500", "assessor": {"answer": "unused"}}
        for identity in identities
    }
    candidate_paths: dict[str, list[Path]] = {}
    for arm in score.ARMS:
        paths = []
        for shard in range(shards):
            start = (rows * shard) // shards
            end = (rows * (shard + 1)) // shards
            path = tmp_path / arm / f"shard_{shard:02d}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "".join(
                    json.dumps(_candidate(arm, identity)) + "\n"
                    for identity in identities[start:end]
                )
            )
            paths.append(path)
        candidate_paths[arm] = paths

    monkeypatch.setattr(score, "load_assessors", lambda _path, expected: assessors)
    monkeypatch.setattr(
        score,
        "qualify_allocation",
        lambda: {"probe_sha256": "a" * 64},
    )
    monkeypatch.setattr(score, "sandbox_atomic_json", lambda _path, _value: "b" * 64)
    monkeypatch.setattr(score, "qualify_mbpp_assessor_setups", lambda _rows: [])
    monkeypatch.setattr(
        score,
        "score_completion",
        lambda _assessor, completion: {"correct": completion == "answer"},
    )
    assessor_path = tmp_path / "assessors.jsonl"
    assessor_path.write_text("{}\n")
    arguments = argparse.Namespace(
        assessors=assessor_path,
        unchanged_candidates=candidate_paths["unchanged"],
        self_refinement_candidates=candidate_paths["self_refinement"],
        revision_candidates=candidate_paths["revision"],
        sandbox_receipt=tmp_path / "sandbox.json",
        output=tmp_path / "score.json",
    )
    report = score.run(arguments, rows=rows, shards=shards)
    assert report["rows"] == rows
    assert report["shards_per_arm"] == shards
    assert all(
        len(report["arms"][arm]["candidate_sha256s"]) == shards for arm in score.ARMS
    )
    assert report["revision_vs_unchanged"]["net_correct"] == 0
    assert json.loads(arguments.output.read_text())["rows"] == rows
