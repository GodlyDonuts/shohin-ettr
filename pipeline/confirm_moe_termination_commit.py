#!/usr/bin/env python3
"""Apply a frozen termination-aware commit as a prospective host confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import analyze_moe_termination_commit as analyzer
from analyze_moe_termination_commit import (
    ARMS,
    TerminationCommitError,
    atomic_json,
    replay,
    sha256_file,
)

PREDECLARATION_SCHEMA = "shohin-moe-termination-aware-commit-predeclaration-v1"
CONFIRMATION_SCHEMA = "shohin-moe-termination-aware-commit-confirmation-v1"
RULE_SOURCE_SHA256 = "eb3fea20006463555abc4df5e8fbdb490ed15d21e1a576d32d6418eede78c378"
RULE_NAME = "select_revision_only_when_baseline_exhausted_and_revision_not_exhausted"
TARGET_CANDIDATE_SCHEMA = "shohin-nemotron-super-fixed-draft-candidate-v1"
TARGET_SCORE_SCHEMA = "shohin-nemotron-super-fixed-draft-screen-score-v1"


def _load_predeclaration(path: Path) -> tuple[dict[str, Any], str]:
    digest = sha256_file(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TerminationCommitError("predeclaration is not JSON") from exc
    if not isinstance(payload, dict):
        raise TerminationCommitError("predeclaration differs")
    return payload, digest


def _validate_target_candidate_schemas(
    candidate_paths: dict[str, list[Path]],
) -> None:
    for arm in ARMS:
        for path in candidate_paths[arm]:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise TerminationCommitError(
                            f"target candidate {path}:{line_number} is not JSON"
                        ) from exc
                    if row.get("schema") != TARGET_CANDIDATE_SCHEMA:
                        raise TerminationCommitError("target candidate schema differs")


def confirm(
    *,
    predeclaration: Path,
    host: str,
    score: Path,
    candidate_paths: dict[str, list[Path]],
) -> dict[str, Any]:
    declaration, declaration_sha256 = _load_predeclaration(predeclaration)
    rule = declaration.get("rule")
    target = declaration.get("target")
    state = declaration.get("state_at_freeze")
    contract = declaration.get("confirmation_contract")
    candidate_rows_at_freeze = (
        state.get("candidate_rows_available") if isinstance(state, dict) else None
    )
    if (
        declaration.get("schema") != PREDECLARATION_SCHEMA
        or declaration.get("status")
        != "frozen_before_mechanics_completion_candidate_generation_and_scoring"
        or not isinstance(rule, dict)
        or rule.get("derivation_commit") != "aeced4169e3ee863d1996b50bb4ad489c676aab3"
        or rule.get("implementation_source")
        != "pipeline/analyze_moe_termination_commit.py"
        or rule.get("implementation_sha256") != RULE_SOURCE_SHA256
        or rule.get("decision")
        != "select revision iff the baseline exhausted its generation limit and revision did not; otherwise retain the baseline"
        or rule.get("uses_task_label") is not False
        or rule.get("uses_correctness_at_selection") is not False
        or rule.get("uses_assessor_at_selection") is not False
        or rule.get("uses_completion_text") is not False
        or rule.get("model_visible_fields")
        != [
            "baseline.max_token_exhausted",
            "revision.max_token_exhausted",
        ]
        or not isinstance(target, dict)
        or target.get("host") != host
        or not isinstance(state, dict)
        or any(
            state.get(name) is not False
            for name in (
                "mechanics_report_exists",
                "training_root_exists",
                "candidate_root_exists",
                "score_exists",
                "scientific_score_available",
            )
        )
        or isinstance(candidate_rows_at_freeze, bool)
        or candidate_rows_at_freeze != 0
        or not isinstance(contract, dict)
        or contract.get("primary_conservative_baseline") != "self_refinement"
        or contract.get("report_both_baselines") is not True
        or contract.get("result_policy")
        != "report the exact result regardless of direction without altering the rule or live graph"
        or contract.get("success_requires")
        != [
            "committed_correct > baseline_correct",
            "retained_baseline_correct == baseline_correct",
            "paired_losses == 0",
        ]
    ):
        raise TerminationCommitError("predeclaration differs")

    analyzer_path = Path(analyzer.__file__).resolve()
    if sha256_file(analyzer_path) != RULE_SOURCE_SHA256:
        raise TerminationCommitError("frozen rule implementation differs")
    if str(score.resolve()) != target.get("score"):
        raise TerminationCommitError("predeclared score path differs")
    declared_candidates = declaration.get("candidate_inputs")
    if not isinstance(declared_candidates, dict) or any(
        [str(path.resolve()) for path in candidate_paths.get(arm, [])]
        != declared_candidates.get(arm)
        for arm in ARMS
    ):
        raise TerminationCommitError("predeclared candidate paths differ")

    report = replay(host=host, score=score, candidate_paths=candidate_paths)
    _validate_target_candidate_schemas(candidate_paths)
    required_rows = contract.get("required_rows")
    required_tasks = contract.get("required_tasks")
    if (
        isinstance(required_rows, bool)
        or not isinstance(required_rows, int)
        or report.get("row_count") != required_rows
        or report["evidence"].get("score_schema") != TARGET_SCORE_SCHEMA
        or required_tasks != ["bbh_logic", "math500", "mbpp"]
        or set(report["selectors"]) != {"unchanged", "self_refinement"}
    ):
        raise TerminationCommitError("confirmation geometry differs")

    primary = report["selectors"]["self_refinement"]
    success_checks = {
        "committed_correct_exceeds_baseline": primary["gain_correct"] > 0,
        "all_baseline_correct_retained": primary["retained_baseline_correct"]
        == primary["baseline_correct"],
        "zero_paired_losses": primary["paired_losses"] == 0,
    }
    report["schema"] = CONFIRMATION_SCHEMA
    report["status"] = "complete_prospective_confirmation"
    report["confirmation"] = {
        "predeclaration_path": str(predeclaration.resolve()),
        "predeclaration_sha256": declaration_sha256,
        "rule_source_path": str(analyzer_path),
        "rule_source_sha256": RULE_SOURCE_SHA256,
        "rule_name": RULE_NAME,
        "primary_baseline": "self_refinement",
        "success_checks": success_checks,
        "success": all(success_checks.values()),
        "result_policy_satisfied": True,
    }
    report["interpretation_boundary"] = {
        "development_only": False,
        "predeclared_confirmation": True,
        "qualified_release": False,
        "claim": (
            "prospective_cross_family_termination_commit_confirmation_pass"
            if report["confirmation"]["success"]
            else "prospective_cross_family_termination_commit_confirmation_fail"
        ),
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predeclaration", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--score", type=Path, required=True)
    for arm in ARMS:
        parser.add_argument(
            f"--{arm.replace('_', '-')}-candidates",
            type=Path,
            action="append",
            required=True,
        )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = confirm(
        predeclaration=args.predeclaration,
        host=args.host,
        score=args.score,
        candidate_paths={arm: getattr(args, f"{arm}_candidates") for arm in ARMS},
    )
    atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
