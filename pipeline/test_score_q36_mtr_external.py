from __future__ import annotations

import argparse
import hashlib
import json

import score_q36_mtr_external as module


def _identity(index: int) -> str:
    return hashlib.sha256(f"score-external-{index}".encode()).hexdigest()


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_external_score_computes_matched_results(tmp_path, monkeypatch):
    tasks = ["math500", "bbh_logic", "mbpp"]
    assessors = []
    for index in range(3):
        identity = _identity(index)
        assessors.append(
            {
                "schema": module.ASSESSOR_SCHEMA,
                "identity_sha256": identity,
                "split": "external_validation",
                "task": tasks[index],
                "assessor": {"identity_sha256": identity, "task": tasks[index]},
            }
        )
    assessor_path = tmp_path / "assessors.jsonl"
    _write_jsonl(assessor_path, assessors)
    monkeypatch.setitem(module.PARTITIONS, "external_validation_screen", (3, 4))
    candidate_groups = {}
    for arm_index, arm in enumerate(module.ARMS):
        path = tmp_path / f"{arm}.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "schema": module.CANDIDATE_SCHEMA,
                    "arm": arm,
                    "identity_sha256": row["identity_sha256"],
                    "task": row["task"],
                    "completion": "correct" if index <= arm_index % 3 else "wrong",
                    "generated_tokens": 1,
                    "max_token_exhausted": False,
                }
                for index, row in enumerate(assessors)
            ],
        )
        candidate_groups[f"{arm}_candidates"] = [path, path, path, path]

    def load_once(arm, paths, identities, expected_shards):
        assert expected_shards == 4
        return {row["identity_sha256"]: row for row in module._load_jsonl(paths[0])}

    monkeypatch.setattr(module, "load_candidates", load_once)
    monkeypatch.setattr(
        module, "qualify_allocation", lambda: {"probe_sha256": "a" * 64}
    )
    monkeypatch.setattr(module, "sandbox_atomic_json", lambda path, payload: "b" * 64)
    monkeypatch.setattr(module, "qualify_mbpp_assessor_setups", lambda rows: [{}])
    monkeypatch.setattr(
        module,
        "score_completion",
        lambda assessor, completion: {"correct": completion == "correct"},
    )
    args = argparse.Namespace(
        assessors=assessor_path,
        split="external_validation_screen",
        expected_rows=3,
        shard_count=4,
        sandbox_receipt=tmp_path / "sandbox.json",
        output=tmp_path / "score.json",
        **candidate_groups,
    )
    report = module.run(args)
    assert report["rows"] == 3
    assert report["arms"]["unchanged"]["correct"] == 1
    assert report["all_arm_oracle_correct"] == 3
    assert report["mbpp_setup_qualification_count"] == 1


def test_mcnemar_exact_is_symmetric():
    assert module._mcnemar_exact(7, 2) == module._mcnemar_exact(2, 7)
    assert module._mcnemar_exact(0, 0) == 1.0
