from __future__ import annotations

import json
from pathlib import Path

import pytest

from analyze_moe_termination_commit import TerminationCommitError
from confirm_moe_termination_commit import confirm


def _identity(index: int) -> str:
    return f"{index:064x}"


def _candidate(path: Path, arm: str, exhausted: list[bool]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for index, value in enumerate(exhausted, start=1):
            handle.write(
                json.dumps(
                    {
                        "schema": "candidate-v1",
                        "arm": arm,
                        "identity_sha256": _identity(index),
                        "task": ("bbh_logic", "math500", "mbpp")[index - 1],
                        "completion": f"{arm}-{index}",
                        "generated_tokens": 8,
                        "max_token_exhausted": value,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return path


def _fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, list[Path]], str]:
    host = "nemotron-fixture"
    candidates = {
        "unchanged": [_candidate(tmp_path / "u.jsonl", "unchanged", [True] * 3)],
        "self_refinement": [
            _candidate(tmp_path / "s.jsonl", "self_refinement", [True, False, True])
        ],
        "revision": [
            _candidate(tmp_path / "r.jsonl", "revision", [False, False, False])
        ],
    }
    score = tmp_path / "score.json"
    score.write_text(
        json.dumps(
            {
                "schema": "score-v1",
                "outcomes": [
                    {
                        "identity_sha256": _identity(1),
                        "task": "bbh_logic",
                        "correct": {
                            "unchanged": False,
                            "self_refinement": False,
                            "revision": True,
                        },
                    },
                    {
                        "identity_sha256": _identity(2),
                        "task": "math500",
                        "correct": {
                            "unchanged": True,
                            "self_refinement": True,
                            "revision": False,
                        },
                    },
                    {
                        "identity_sha256": _identity(3),
                        "task": "mbpp",
                        "correct": {
                            "unchanged": False,
                            "self_refinement": False,
                            "revision": True,
                        },
                    },
                ],
            }
        )
        + "\n"
    )
    predeclaration = tmp_path / "predeclaration.json"
    predeclaration.write_text(
        json.dumps(
            {
                "schema": "shohin-moe-termination-aware-commit-predeclaration-v1",
                "status": "frozen_before_mechanics_completion_candidate_generation_and_scoring",
                "rule": {
                    "derivation_commit": "aeced4169e3ee863d1996b50bb4ad489c676aab3",
                    "implementation_source": "pipeline/analyze_moe_termination_commit.py",
                    "implementation_sha256": "eb3fea20006463555abc4df5e8fbdb490ed15d21e1a576d32d6418eede78c378",
                    "decision": "select revision iff the baseline exhausted its generation limit and revision did not; otherwise retain the baseline",
                    "uses_task_label": False,
                    "uses_correctness_at_selection": False,
                    "uses_assessor_at_selection": False,
                    "uses_completion_text": False,
                    "model_visible_fields": [
                        "baseline.max_token_exhausted",
                        "revision.max_token_exhausted",
                    ],
                },
                "target": {"host": host, "score": str(score.resolve())},
                "candidate_inputs": {
                    arm: [str(path.resolve()) for path in paths]
                    for arm, paths in candidates.items()
                },
                "state_at_freeze": {
                    "mechanics_report_exists": False,
                    "training_root_exists": False,
                    "candidate_root_exists": False,
                    "score_exists": False,
                    "candidate_rows_available": 0,
                    "scientific_score_available": False,
                },
                "confirmation_contract": {
                    "primary_conservative_baseline": "self_refinement",
                    "report_both_baselines": True,
                    "required_rows": 3,
                    "required_tasks": ["bbh_logic", "math500", "mbpp"],
                    "success_requires": [
                        "committed_correct > baseline_correct",
                        "retained_baseline_correct == baseline_correct",
                        "paired_losses == 0",
                    ],
                    "result_policy": "report the exact result regardless of direction without altering the rule or live graph",
                },
            }
        )
        + "\n"
    )
    return predeclaration, score, candidates, host


def test_confirmation_binds_predeclaration_and_reports_prospective_success(
    tmp_path: Path,
) -> None:
    predeclaration, score, candidates, host = _fixture(tmp_path)
    report = confirm(
        predeclaration=predeclaration,
        host=host,
        score=score,
        candidate_paths=candidates,
    )
    assert report["schema"] == "shohin-moe-termination-aware-commit-confirmation-v1"
    assert report["status"] == "complete_prospective_confirmation"
    assert report["confirmation"]["success"] is True
    assert report["confirmation"]["success_checks"] == {
        "committed_correct_exceeds_baseline": True,
        "all_baseline_correct_retained": True,
        "zero_paired_losses": True,
    }
    assert report["selectors"]["self_refinement"]["selected_correct"] == 3
    assert report["interpretation_boundary"]["predeclared_confirmation"] is True


def test_confirmation_rejects_candidate_path_substitution(tmp_path: Path) -> None:
    predeclaration, score, candidates, host = _fixture(tmp_path)
    candidates["revision"] = [tmp_path / "different.jsonl"]
    with pytest.raises(TerminationCommitError, match="candidate paths"):
        confirm(
            predeclaration=predeclaration,
            host=host,
            score=score,
            candidate_paths=candidates,
        )


def test_confirmation_rejects_post_result_freeze_claim(tmp_path: Path) -> None:
    predeclaration, score, candidates, host = _fixture(tmp_path)
    payload = json.loads(predeclaration.read_text())
    payload["state_at_freeze"]["score_exists"] = True
    predeclaration.write_text(json.dumps(payload) + "\n")
    with pytest.raises(TerminationCommitError, match="predeclaration"):
        confirm(
            predeclaration=predeclaration,
            host=host,
            score=score,
            candidate_paths=candidates,
        )


def test_confirmation_rejects_boolean_zero_row_claim(tmp_path: Path) -> None:
    predeclaration, score, candidates, host = _fixture(tmp_path)
    payload = json.loads(predeclaration.read_text())
    payload["state_at_freeze"]["candidate_rows_available"] = False
    predeclaration.write_text(json.dumps(payload) + "\n")
    with pytest.raises(TerminationCommitError, match="predeclaration"):
        confirm(
            predeclaration=predeclaration,
            host=host,
            score=score,
            candidate_paths=candidates,
        )
