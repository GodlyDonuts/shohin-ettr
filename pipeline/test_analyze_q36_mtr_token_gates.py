from __future__ import annotations

import argparse
import json

import pytest

import analyze_q36_mtr_token_gates as module


def _score(
    path, arm: str, correct: set[int], unchanged: set[int], rows: int = 256
) -> None:
    outcomes = [
        {
            "identity_sha256": f"{index:064x}",
            "task": module.TASKS[index % len(module.TASKS)],
            f"{arm}_correct": index in correct,
            "unchanged_correct": index in unchanged,
        }
        for index in range(rows)
    ]
    task_totals = {
        task: sum(row["task"] == task for row in outcomes) for task in module.TASKS
    }
    task_correct = {
        task: sum(
            row["task"] == task and index in correct
            for index, row in enumerate(outcomes)
        )
        for task in module.TASKS
    }
    arm_only = len(correct - unchanged)
    unchanged_only = len(unchanged - correct)
    payload = {
        "schema": module.SCORE_SCHEMA,
        "status": "complete",
        "arm": arm,
        "rows": rows,
        "shard_count": 4,
        "assessors_sha256": "a" * 64,
        "baseline_score_sha256": "b" * 64,
        "temporal_candidate_sha256s": [f"{index:064x}" for index in range(4)],
        "sandbox_receipt_sha256": "c" * 64,
        "sandbox_probe_sha256": "d" * 64,
        "mbpp_setup_qualification_count": task_totals["mbpp"],
        arm: {"correct": len(correct), "total": rows},
        "unchanged": {"correct": len(unchanged), "total": rows},
        "gain_over_unchanged_count": len(correct) - len(unchanged),
        "paired_vs_unchanged": {
            "temporal_only_correct": arm_only,
            "unchanged_only_correct": unchanged_only,
            "mcnemar_exact_two_sided_p": module._mcnemar_exact(
                arm_only, unchanged_only
            ),
        },
        "outcomes": outcomes,
    }
    payload[arm].update(
        {
            "domains": {
                task: {"correct": task_correct[task], "total": task_totals[task]}
                for task in module.TASKS
            },
            "empty_completions": 0,
            "max_token_exhausted": 0,
        }
    )
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload))


def test_analysis_selects_correct_then_retention_winner(tmp_path) -> None:
    unchanged = set(range(100))
    supervised = tmp_path / "supervised" / "score.json"
    multi = tmp_path / "multi" / "score.json"
    _score(supervised, "temporal_gate", set(range(145)), unchanged)
    _score(multi, "multi_trajectory_gate", set(range(147)), unchanged)
    output = tmp_path / "result.json"
    report = module.run(
        argparse.Namespace(
            score=[supervised, multi],
            incumbent_revision_correct=141,
            incumbent_score=None,
            output=output,
        )
    )
    assert report["winner"] == "multi_trajectory"
    assert report["winner_correct"] == 147
    assert report["winner_beats_incumbent_revision"] is True
    assert report["winner_retention_at_least_90_percent"] is True
    assert report["promotion_candidate"] == "multi_trajectory"
    assert (
        report["recommended_next_action"]
        == "interpret_already_staged_1023_validation_for_promotion_candidate"
    )
    pair = report["pairwise"]["multi_trajectory_vs_response_supervised"]
    assert pair["multi_trajectory_only_correct"] == 2
    assert pair["response_supervised_only_correct"] == 0
    assert json.loads(output.read_text()) == report


def test_analysis_rejects_mismatched_baseline(tmp_path) -> None:
    first = tmp_path / "supervised" / "score.json"
    second = tmp_path / "multi" / "score.json"
    _score(first, "temporal_gate", {1}, {1})
    _score(second, "multi_trajectory_gate", {1}, {2})
    with pytest.raises(module.Q36MTRTokenGateAnalysisError):
        module.run(
            argparse.Namespace(
                score=[first, second],
                incumbent_revision_correct=141,
                incumbent_score=None,
                output=tmp_path / "result.json",
            )
        )


def test_analysis_supports_full_validation_geometry(tmp_path) -> None:
    first = tmp_path / "q36_supervised" / "validation" / "score.json"
    second = tmp_path / "q36_multi" / "validation" / "score.json"
    unchanged = set(range(420))
    _score(first, "temporal_gate", set(range(530)), unchanged, rows=1023)
    _score(second, "multi_trajectory_gate", set(range(550)), unchanged, rows=1023)
    report = module.run(
        argparse.Namespace(
            score=[first, second],
            incumbent_revision_correct=None,
            incumbent_score=None,
            output=tmp_path / "result.json",
        )
    )
    assert report["rows"] == 1023
    assert report["winner"] == "multi_trajectory"
    assert report["winner_beats_incumbent_revision"] is None


def test_analysis_distinguishes_routing_only_multi_trajectory(tmp_path) -> None:
    unchanged = set(range(80))
    causal = tmp_path / "q36_multi" / "score.json"
    routing = tmp_path / "q36_multi_routing_only" / "score.json"
    _score(causal, "multi_trajectory_gate", set(range(110)), unchanged, rows=200)
    _score(routing, "multi_trajectory_gate", set(range(120)), unchanged, rows=200)
    report = module.run(
        argparse.Namespace(
            score=[causal, routing],
            incumbent_revision_correct=115,
            incumbent_score=None,
            output=tmp_path / "result.json",
        )
    )
    assert sorted(report["variants"]) == ["multi_routing_only", "multi_trajectory"]
    assert report["winner"] == "multi_routing_only"


def test_analysis_distinguishes_tri_geometry_router(tmp_path) -> None:
    tri = tmp_path / "q36_tri_geometry" / "score.json"
    routing = tmp_path / "q36_tri_trajectory" / "score.json"
    unchanged = set(range(100))
    _score(tri, "multi_trajectory_gate", set(range(130)), unchanged, rows=200)
    _score(routing, "multi_trajectory_gate", set(range(120)), unchanged, rows=200)
    report = module.run(
        argparse.Namespace(
            score=[tri, routing],
            incumbent_revision_correct=125,
            incumbent_score=None,
            output=tmp_path / "analysis.json",
        )
    )
    assert sorted(report["variants"]) == ["tri_geometry", "tri_trajectory"]
    assert report["winner"] == "tri_geometry"


def test_analysis_distinguishes_hierarchical_router(tmp_path) -> None:
    hierarchy = tmp_path / "q36_tri_hierarchical" / "score.json"
    set_mass = tmp_path / "q36_tri_hierarchical_set_mass" / "score.json"
    geometry = tmp_path / "q36_tri_geometry" / "score.json"
    unchanged = set(range(100))
    _score(hierarchy, "multi_trajectory_gate", set(range(140)), unchanged, rows=200)
    _score(set_mass, "multi_trajectory_gate", set(range(145)), unchanged, rows=200)
    _score(geometry, "multi_trajectory_gate", set(range(130)), unchanged, rows=200)
    report = module.run(
        argparse.Namespace(
            score=[hierarchy, set_mass, geometry],
            incumbent_revision_correct=125,
            incumbent_score=None,
            output=tmp_path / "analysis.json",
        )
    )
    assert sorted(report["variants"]) == [
        "tri_geometry",
        "tri_hierarchical",
        "tri_hierarchical_set_mass",
    ]
    assert report["winner"] == "tri_hierarchical_set_mass"


def test_analysis_loads_matched_revision_incumbent(tmp_path) -> None:
    first = tmp_path / "q36_supervised" / "validation" / "score.json"
    second = tmp_path / "q36_multi" / "validation" / "score.json"
    unchanged = set(range(80))
    _score(first, "temporal_gate", set(range(110)), unchanged, rows=200)
    _score(second, "multi_trajectory_gate", set(range(120)), unchanged, rows=200)
    incumbent = tmp_path / "detailed_score.json"
    incumbent.write_text(
        json.dumps(
            {
                "schema": module.EXTERNAL_SCORE_SCHEMA,
                "status": "complete",
                "rows": 200,
                "arms": {
                    "revision": {"correct": 115},
                    "unchanged": {"correct": 80},
                },
            }
        )
    )
    report = module.run(
        argparse.Namespace(
            score=[first, second],
            incumbent_revision_correct=None,
            incumbent_score=incumbent,
            output=tmp_path / "result.json",
        )
    )
    assert report["incumbent_revision_correct"] == 115
    assert report["winner_beats_incumbent_revision"] is True
    assert report["incumbent_score"]["sha256"] == module.sha256_file(incumbent)


def test_analysis_rejects_tampered_domain_and_paired_evidence(tmp_path) -> None:
    first = tmp_path / "q36_supervised" / "score.json"
    second = tmp_path / "q36_multi" / "score.json"
    unchanged = set(range(80))
    _score(first, "temporal_gate", set(range(110)), unchanged, rows=200)
    _score(second, "multi_trajectory_gate", set(range(120)), unchanged, rows=200)
    payload = json.loads(second.read_text())
    payload["multi_trajectory_gate"]["domains"]["math500"]["correct"] += 1
    payload["paired_vs_unchanged"]["temporal_only_correct"] += 1
    second.write_text(json.dumps(payload))
    with pytest.raises(module.Q36MTRTokenGateAnalysisError):
        module.run(
            argparse.Namespace(
                score=[first, second],
                incumbent_revision_correct=115,
                incumbent_score=None,
                output=tmp_path / "result.json",
            )
        )


def test_analysis_rejects_domain_reallocation_and_shared_receipt_drift(
    tmp_path,
) -> None:
    first = tmp_path / "q36_supervised" / "score.json"
    second = tmp_path / "q36_multi" / "score.json"
    unchanged = set(range(80))
    _score(first, "temporal_gate", set(range(110)), unchanged, rows=200)
    _score(second, "multi_trajectory_gate", set(range(120)), unchanged, rows=200)
    payload = json.loads(second.read_text())
    payload["multi_trajectory_gate"]["domains"]["math500"]["correct"] += 1
    payload["multi_trajectory_gate"]["domains"]["bbh_logic"]["correct"] -= 1
    payload["assessors_sha256"] = "e" * 64
    second.write_text(json.dumps(payload))
    with pytest.raises(module.Q36MTRTokenGateAnalysisError):
        module.run(
            argparse.Namespace(
                score=[first, second],
                incumbent_revision_correct=115,
                incumbent_score=None,
                output=tmp_path / "result.json",
            )
        )


def test_analysis_does_not_promote_empty_or_domain_zero_winner(tmp_path) -> None:
    first = tmp_path / "q36_supervised" / "score.json"
    second = tmp_path / "q36_multi" / "score.json"
    unchanged = set(range(80))
    _score(first, "temporal_gate", set(range(110)), unchanged, rows=200)
    winner_correct = {index for index in range(200) if index % len(module.TASKS) != 2}
    _score(second, "multi_trajectory_gate", winner_correct, unchanged, rows=200)
    payload = json.loads(second.read_text())
    payload["multi_trajectory_gate"]["empty_completions"] = 1
    second.write_text(json.dumps(payload))
    report = module.run(
        argparse.Namespace(
            score=[first, second],
            incumbent_revision_correct=115,
            incumbent_score=None,
            output=tmp_path / "result.json",
        )
    )
    assert report["winner"] == "multi_trajectory"
    assert report["promotion_candidate"] is None
    assert (
        report["promotion_eligibility"]["multi_trajectory"]["every_domain_nonzero"]
        is False
    )
    assert (
        report["promotion_eligibility"]["multi_trajectory"]["zero_empty_completions"]
        is False
    )
    assert (
        report["recommended_next_action"]
        == "do_not_promote_new_router_from_this_screen"
    )
