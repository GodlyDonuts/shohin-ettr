from __future__ import annotations

import json
from pathlib import Path

import pytest

import hf_q36_mtr_apply_cross_host_commit as module


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_cross_host_loaders_bind_source_arm_and_identity(tmp_path: Path) -> None:
    contract = {
        "rows": 2,
        "shards": 1,
        "candidate_schema": "candidate-v1",
        "source_schema": "source-v1",
        "source_split": "external_validation",
    }
    identities = ("0" * 64, "1" * 64)
    source = tmp_path / "source.jsonl"
    revision = tmp_path / "revision.jsonl"
    _write(
        source,
        [
            {
                "schema": "source-v1",
                "split": "external_validation",
                "identity_sha256": identity,
                "task": "math500",
                "source_prompt": f"question {index}",
            }
            for index, identity in enumerate(identities)
        ],
    )
    _write(
        revision,
        [
            {
                "schema": "candidate-v1",
                "arm": "revision",
                "identity_sha256": identity,
                "task": "math500",
                "completion": f"answer {index}",
                "generated_tokens": 3,
                "max_token_exhausted": False,
            }
            for index, identity in enumerate(identities)
        ],
    )
    assert set(module.load_source(source, contract)) == set(identities)
    assert set(module.load_candidates([revision], "revision", contract)) == set(
        identities
    )
    rows = [json.loads(line) for line in revision.read_text().splitlines()]
    rows[0]["arm"] = "unchanged"
    _write(revision, rows)
    with pytest.raises(module.CrossHostCommitError, match="candidate differs"):
        module.load_candidates([revision], "revision", contract)


def test_source_superset_is_filtered_and_empty_completion_is_valid(
    tmp_path: Path,
) -> None:
    contract = {
        "rows": 1,
        "shards": 1,
        "candidate_schema": "candidate-v1",
        "source_schema": "source-v1",
        "source_split": "external_validation",
    }
    wanted, extra = "a" * 64, "b" * 64
    source = tmp_path / "source.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write(
        source,
        [
            {
                "schema": "source-v1",
                "split": "external_validation",
                "identity_sha256": identity,
                "task": "math500",
                "source_prompt": "question",
            }
            for identity in (wanted, extra)
        ],
    )
    _write(
        candidate,
        [
            {
                "schema": "candidate-v1",
                "arm": "revision",
                "identity_sha256": wanted,
                "task": "math500",
                "completion": "",
                "generated_tokens": 768,
                "max_token_exhausted": True,
            }
        ],
    )
    assert set(module.load_source(source, contract, {wanted})) == {wanted}
    assert set(module.load_candidates([candidate], "revision", contract)) == {wanted}


def test_cross_host_job_is_one_h100_and_excludes_failed_node() -> None:
    root = Path(__file__).resolve().parents[1]
    wrapper = (root / "train/jobs/q36_mtr_cross_host_commit.sbatch").read_text()
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in wrapper
    assert "#SBATCH --no-requeue" in wrapper
    assert "evc50" in wrapper
    assert "--batch-identities 2" in wrapper
    assert module.HOSTS["gpt_oss_120b_confirmation_1023"]["rows"] == 1_023
    assert module.HOSTS["gpt_oss_120b_confirmation_1023"]["shards"] == 16


def test_conservative_margin_threshold_is_semantic_and_order_consistent() -> None:
    candidates = [{"completion": "revision"}, {"completion": "unchanged"}]
    assert module.select_pair(0.8, -0.8, candidates, 0.703125) == (0, 1)
    assert module.select_pair(0.7, -0.7, candidates, 0.703125) == (1, 0)
    assert module.select_pair(0.703125, -0.703125, candidates, 0.703125) == (
        1,
        0,
    )


def test_revision_reliability_veto_falls_back_only_on_selected_unreliable_revision() -> (
    None
):
    reliable = {"completion": "answer", "max_token_exhausted": False}
    empty = {"completion": "", "max_token_exhausted": False}
    exhausted = {"completion": "answer", "max_token_exhausted": True}
    assert module.apply_revision_reliability_veto(
        0, 1, reliable, "empty_or_exhausted"
    ) == (0, 1, False)
    assert module.apply_revision_reliability_veto(
        0, 1, empty, "empty_or_exhausted"
    ) == (1, 0, True)
    assert module.apply_revision_reliability_veto(
        0, 1, exhausted, "empty_or_exhausted"
    ) == (1, 0, True)
    assert module.apply_revision_reliability_veto(
        1, 0, exhausted, "empty_or_exhausted"
    ) == (1, 0, False)
    assert module.apply_revision_reliability_veto(0, 1, exhausted, "none") == (
        0,
        1,
        False,
    )
