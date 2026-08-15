from __future__ import annotations

import json

import pytest

import score_q36_mtr_temporal_gate as module


def _write_jsonl(path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _candidate(identity: str, task: str = "math500") -> dict[str, object]:
    return {
        "schema": module.CANDIDATE_SCHEMA,
        "arm": module.ARM,
        "identity_sha256": identity,
        "task": task,
        "completion": "answer",
        "generated_tokens": 1,
        "max_token_exhausted": False,
    }


def test_temporal_candidate_loader_requires_exact_arm_and_coverage(tmp_path) -> None:
    identities = {"1" * 64, "2" * 64}
    paths = [tmp_path / "a.jsonl", tmp_path / "b.jsonl"]
    _write_jsonl(paths[0], [_candidate("1" * 64)])
    _write_jsonl(paths[1], [_candidate("2" * 64, "bbh_logic")])
    loaded = module.load_temporal_candidates(paths, identities, 2)
    assert set(loaded) == identities
    changed = _candidate("2" * 64)
    changed["arm"] = "revision"
    _write_jsonl(paths[1], [changed])
    with pytest.raises(module.Q36MTRTemporalGateScoreError):
        module.load_temporal_candidates(paths, identities, 2)


def test_baseline_loader_binds_identity_task_and_unchanged_outcome(tmp_path) -> None:
    assessors = {
        "1" * 64: {"task": "math500"},
        "2" * 64: {"task": "mbpp"},
    }
    report = {
        "schema": module.BASELINE_SCHEMA,
        "status": "complete",
        "rows": 2,
        "outcomes": [
            {
                "identity_sha256": identity,
                "task": row["task"],
                "correct": {"unchanged": index == 0},
            }
            for index, (identity, row) in enumerate(assessors.items())
        ],
    }
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert module.load_baseline(path, assessors, 2) == {
        "1" * 64: True,
        "2" * 64: False,
    }
    report["outcomes"][1]["correct"]["unchanged"] = 1
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(module.Q36MTRTemporalGateScoreError):
        module.load_baseline(path, assessors, 2)
