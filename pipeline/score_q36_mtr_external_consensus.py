#!/usr/bin/env python3
"""Evaluate label-free Q36 trajectory composers on the external screen."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from typing import Any

from pcf1_code_sandbox import (
    atomic_json as sandbox_atomic_json,
    qualify_allocation,
    qualify_mbpp_assessor_setups,
    score_completion,
)
from score_q36_mtr_external import (
    ARMS,
    ROWS,
    TASKS,
    _mcnemar_exact,
    load_assessors,
    load_candidates,
    sha256_file,
)
from select_q36_mtr_consensus import normalized_answer

SCHEMA = "shohin-q36-mtr-external-consensus-score-v1"
RULES = (
    "plurality",
    "conservative_unchanged",
    "aligned_agreement",
    "interpolation_retention",
)
PREFERENCE = (
    "interpolation",
    "revision",
    "unchanged",
    "self_refinement",
    "draft_hidden",
)


class Q36MTRExternalConsensusError(RuntimeError):
    """External consensus inputs or decisions differ."""


def _answers(task: str, rows: dict[str, dict[str, Any]]) -> dict[str, str | None]:
    if tuple(rows) != ARMS:
        raise Q36MTRExternalConsensusError("external consensus arm order differs")
    if task == "mbpp":
        return {arm: None for arm in ARMS}
    return {
        arm: normalized_answer(task, row["completion"]) for arm, row in rows.items()
    }


def _first_arm_for_answer(answers: dict[str, str | None], answer: str) -> str:
    return next(arm for arm in PREFERENCE if answers[arm] == answer)


def choose(rule: str, task: str, rows: dict[str, dict[str, Any]]) -> str:
    if rule not in RULES:
        raise Q36MTRExternalConsensusError("external consensus rule differs")
    if task == "mbpp":
        return "interpolation"
    answers = _answers(task, rows)
    counts = Counter(answer for answer in answers.values() if answer is not None)
    if not counts:
        return "unchanged"
    if rule == "plurality":
        support = max(counts.values())
        winners = {answer for answer, count in counts.items() if count == support}
        return next(arm for arm in PREFERENCE if answers[arm] in winners)
    if rule == "conservative_unchanged":
        unchanged = answers["unchanged"]
        challengers = [
            (count, answer)
            for answer, count in counts.items()
            if answer != unchanged and count >= 3
        ]
        if not challengers:
            return "unchanged"
        _, answer = max(challengers, key=lambda item: (item[0], item[1]))
        return _first_arm_for_answer(answers, answer)
    if rule == "aligned_agreement":
        revision = answers["revision"]
        hidden = answers["draft_hidden"]
        if revision is not None and revision == hidden:
            return (
                "interpolation" if answers["interpolation"] == revision else "revision"
            )
        return "unchanged"
    interpolation = answers["interpolation"]
    owner = answers["unchanged"]
    refinement = answers["self_refinement"]
    if owner is not None and owner == refinement and owner != interpolation:
        return "unchanged"
    return "interpolation"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTRExternalConsensusError("external consensus output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    assessors = load_assessors(args.assessors)
    identities = set(assessors)
    path_groups = {arm: getattr(args, f"{arm}_candidates") for arm in ARMS}
    candidates = {
        arm: load_candidates(arm, paths, identities)
        for arm, paths in path_groups.items()
    }
    sandbox = qualify_allocation()
    sandbox_sha256 = sandbox_atomic_json(args.sandbox_receipt, sandbox)
    setup_receipts = qualify_mbpp_assessor_setups(
        [row["assessor"] for row in assessors.values() if row["task"] == "mbpp"]
    )

    task_rows = Counter(row["task"] for row in assessors.values())
    correctness: dict[str, dict[str, bool]] = {rule: {} for rule in RULES}
    selected_arms: dict[str, Counter[str]] = {rule: Counter() for rule in RULES}
    task_correct: dict[str, Counter[str]] = {rule: Counter() for rule in RULES}
    unchanged_correct: dict[str, bool] = {}
    for identity in sorted(identities):
        assessor_row = assessors[identity]
        rows = {arm: candidates[arm][identity] for arm in ARMS}
        task = assessor_row["task"]
        if any(row["task"] != task for row in rows.values()):
            raise Q36MTRExternalConsensusError("external consensus task differs")
        unchanged_score = score_completion(
            assessor_row["assessor"], rows["unchanged"]["completion"]
        )
        unchanged_correct[identity] = bool(unchanged_score["correct"])
        for rule in RULES:
            selected = choose(rule, task, rows)
            score = score_completion(
                assessor_row["assessor"], rows[selected]["completion"]
            )
            correct = score.get("correct")
            if not isinstance(correct, bool):
                raise Q36MTRExternalConsensusError("external consensus score differs")
            correctness[rule][identity] = correct
            selected_arms[rule][selected] += 1
            task_correct[rule][task] += int(correct)

    reports: dict[str, Any] = {}
    base_correct = sum(unchanged_correct.values())
    for rule in RULES:
        correct = sum(correctness[rule].values())
        rule_only = sum(
            correctness[rule][identity] and not unchanged_correct[identity]
            for identity in identities
        )
        unchanged_only = sum(
            unchanged_correct[identity] and not correctness[rule][identity]
            for identity in identities
        )
        reports[rule] = {
            "correct": correct,
            "total": ROWS,
            "accuracy": correct / ROWS,
            "gain_over_unchanged_count": correct - base_correct,
            "selection_counts": dict(sorted(selected_arms[rule].items())),
            "domains": {
                task: {
                    "correct": task_correct[rule][task],
                    "total": task_rows[task],
                }
                for task in TASKS
            },
            "paired_vs_unchanged": {
                "rule_only_correct": rule_only,
                "unchanged_only_correct": unchanged_only,
                "mcnemar_exact_two_sided_p": _mcnemar_exact(rule_only, unchanged_only),
            },
        }
    best = max(RULES, key=lambda rule: (reports[rule]["correct"], -RULES.index(rule)))
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "split": "external_validation_screen",
        "rows": ROWS,
        "rules": reports,
        "best_rule": best,
        "best_correct": reports[best]["correct"],
        "assessors_sha256": sha256_file(args.assessors),
        "input_candidate_sha256s": {
            arm: [sha256_file(path) for path in paths]
            for arm, paths in path_groups.items()
        },
        "sandbox_receipt_sha256": sandbox_sha256,
        "sandbox_probe_sha256": sandbox.get("probe_sha256"),
        "mbpp_setup_qualification_count": len(setup_receipts),
    }
    _atomic_json(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assessors", type=Path, required=True)
    for arm in ARMS:
        parser.add_argument(
            f"--{arm.replace('_', '-')}-candidates",
            type=Path,
            action="append",
            required=True,
        )
    parser.add_argument("--sandbox-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(
        json.dumps(
            {"best_rule": report["best_rule"], "correct": report["best_correct"]}
        )
    )


if __name__ == "__main__":
    main()
