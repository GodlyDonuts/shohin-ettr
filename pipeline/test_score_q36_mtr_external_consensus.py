from __future__ import annotations

import argparse
import hashlib
import json

import score_q36_mtr_external_consensus as module


def _rows(answers):
    return {
        arm: {
            "identity_sha256": "a" * 64,
            "task": "math500",
            "completion": rf"\boxed{{{answer}}}",
        }
        for arm, answer in zip(module.ARMS, answers, strict=True)
    }


def test_plurality_prefers_interpolation_on_tie():
    rows = _rows(("1", "2", "2", "3", "1"))
    assert module.choose("plurality", "math500", rows) == "interpolation"


def test_conservative_unchanged_requires_three_challengers():
    assert (
        module.choose(
            "conservative_unchanged",
            "math500",
            _rows(("1", "2", "2", "2", "2")),
        )
        == "interpolation"
    )
    assert (
        module.choose(
            "conservative_unchanged",
            "math500",
            _rows(("1", "2", "2", "3", "3")),
        )
        == "unchanged"
    )


def test_aligned_agreement_and_retention():
    rows = _rows(("1", "2", "3", "3", "3"))
    assert module.choose("aligned_agreement", "math500", rows) == "interpolation"
    rows = _rows(("1", "1", "2", "3", "2"))
    assert module.choose("interpolation_retention", "math500", rows) == "unchanged"


def test_code_uses_interpolation_control():
    rows = {
        arm: {"identity_sha256": "a" * 64, "task": "mbpp", "completion": "pass"}
        for arm in module.ARMS
    }
    for rule in module.RULES:
        assert module.choose(rule, "mbpp", rows) == "interpolation"


def test_fixed_rule_validation_does_not_select_on_validation_labels(
    tmp_path, monkeypatch
):
    tasks = ("math500", "bbh_logic", "mbpp")
    assessors = []
    for index, task in enumerate(tasks):
        identity = hashlib.sha256(f"fixed-{index}".encode()).hexdigest()
        assessors.append(
            {
                "schema": "shohin-q36-mtr-external-validation-assessor-v1",
                "split": "external_validation",
                "identity_sha256": identity,
                "task": task,
                "assessor": {"identity_sha256": identity, "task": task},
            }
        )
    assessor_path = tmp_path / "assessors.jsonl"
    assessor_path.write_text("".join(json.dumps(row) + "\n" for row in assessors))
    monkeypatch.setitem(module.PARTITIONS, "external_validation", (3, 1))
    groups = {}
    for arm in module.ARMS:
        path = tmp_path / f"{arm}.jsonl"
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "schema": "shohin-q36-mtr-candidate-v1",
                        "arm": arm,
                        "identity_sha256": row["identity_sha256"],
                        "task": row["task"],
                        "completion": "correct" if arm == "interpolation" else "wrong",
                        "generated_tokens": 1,
                        "max_token_exhausted": False,
                    }
                )
                + "\n"
                for row in assessors
            )
        )
        groups[f"{arm}_candidates"] = [path]
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
    report = module.run(
        argparse.Namespace(
            assessors=assessor_path,
            split="external_validation",
            expected_rows=3,
            shard_count=1,
            fixed_rule="interpolation_retention",
            sandbox_receipt=tmp_path / "sandbox.json",
            output=tmp_path / "result.json",
            **groups,
        )
    )
    assert report["selection_mode"] == "fixed"
    assert report["fixed_rule"] == "interpolation_retention"
    assert tuple(report["rules"]) == ("interpolation_retention",)
    assert report["best_correct"] == 1
