"""Tests for the paired 120B-A12B screen reducer."""

from __future__ import annotations

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
