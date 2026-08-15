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
    payload = {
        "schema": module.SCORE_SCHEMA,
        "status": "complete",
        "arm": arm,
        "rows": rows,
        arm: {"correct": len(correct), "total": rows},
        "outcomes": outcomes,
    }
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
            output=output,
        )
    )
    assert report["winner"] == "multi_trajectory"
    assert report["winner_correct"] == 147
    assert report["winner_beats_incumbent_revision"] is True
    assert report["winner_retention_at_least_90_percent"] is True
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
            output=tmp_path / "result.json",
        )
    )
    assert report["rows"] == 1023
    assert report["winner"] == "multi_trajectory"
    assert report["winner_beats_incumbent_revision"] is None
