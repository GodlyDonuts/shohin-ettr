from __future__ import annotations

import argparse
import hashlib
import json

import analyze_q36_mtr_external_outcomes as module


def _identity(index: int) -> str:
    return hashlib.sha256(f"external-outcome-{index}".encode()).hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value))


def _detailed():
    rows = [
        {
            "identity_sha256": _identity(index),
            "task": task,
            "correct": dict(zip(module.ARMS, values, strict=True)),
        }
        for index, (task, values) in enumerate(
            [
                ("math500", (False, True, True, False, True)),
                ("bbh_logic", (True, False, True, False, False)),
                ("mbpp", (False, False, False, False, True)),
                ("math500", (False, False, False, False, False)),
            ]
        )
    ]
    rows.sort(key=lambda row: row["identity_sha256"])
    arms = {}
    for arm in module.ARMS:
        arms[arm] = {
            "correct": sum(row["correct"][arm] for row in rows),
            "total": len(rows),
            "domains": {
                task: {
                    "correct": sum(
                        row["correct"][arm] for row in rows if row["task"] == task
                    ),
                    "total": sum(row["task"] == task for row in rows),
                }
                for task in module.TASKS
            },
        }
    return {
        "schema": module.REPORT_SCHEMA,
        "status": "complete",
        "split": "external_validation_screen",
        "rows": len(rows),
        "arms": arms,
        "outcomes": rows,
        "all_arm_oracle_correct": 3,
    }


def test_outcome_analysis_reconstructs_pairwise_and_selects_companion(tmp_path):
    detailed = tmp_path / "detailed.json"
    forest = tmp_path / "forest.json"
    consensus = tmp_path / "consensus.json"
    _write(detailed, _detailed())
    _write(
        forest,
        {
            "schema": module.FOREST_SCHEMA,
            "status": "complete",
            "split": "external_validation_screen",
            "rows": 4,
            "target": {"correct": 3},
        },
    )
    _write(
        consensus,
        {
            "schema": module.CONSENSUS_SCHEMA,
            "status": "complete",
            "split": "external_validation_screen",
            "rows": 4,
            "rules": {"plurality": {"correct": 2}},
        },
    )
    args = argparse.Namespace(
        detailed=detailed,
        forest=forest,
        consensus=consensus,
        output=tmp_path / "analysis.json",
    )
    report = module.run(args)
    assert report["selected_architecture"] == "learned_forest"
    assert report["selected_correct"] == 3
    assert report["best_fixed_arm"] == "revision"
    assert report["oracle_gap_over_best_fixed"] == 1
    geometry = report["failure_geometry"]
    assert geometry["all_fixed_arms_wrong"] == {
        "rows": 1,
        "by_task": {"math500": 1, "bbh_logic": 0, "mbpp": 0},
    }
    assert geometry["exclusive_correct_by_arm"]["interpolation"]["rows"] == 1
    assert geometry["correctness_patterns"]["00000"]["rows"] == 1
    assert geometry["retention_vs_unchanged"]["revision"] == {
        "unchanged_correct_retained": 1,
        "unchanged_correct_lost": 0,
        "unchanged_wrong_repaired": 1,
        "net_gain_over_unchanged": 1,
        "unchanged_correct_retention_rate": 1.0,
    }
    pair = report["pairwise"]["revision__vs__interpolation"]
    assert pair["left_only_correct"] == 1
    assert pair["right_only_correct"] == 1
    assert pair["net_left_minus_right"] == 0


def test_outcome_analysis_rejects_aggregate_tamper(tmp_path):
    detailed = _detailed()
    detailed["arms"]["revision"]["correct"] += 1
    path = tmp_path / "detailed.json"
    _write(path, detailed)
    args = argparse.Namespace(
        detailed=path,
        forest=None,
        consensus=None,
        output=tmp_path / "analysis.json",
    )
    try:
        module.run(args)
    except module.Q36MTRExternalOutcomeAnalysisError:
        return
    raise AssertionError("tampered external aggregate was accepted")
