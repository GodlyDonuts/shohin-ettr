#!/usr/bin/env python3
"""Atomically score the matched Mixtral validation and learned commit."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from pcf1_code_sandbox import (
    atomic_json as sandbox_atomic_json,
    qualify_allocation,
    qualify_mbpp_assessor_setups,
    score_completion,
)
from score_q36_mtr_external import _mcnemar_exact, load_assessors
import train_apply_mixtral_8x22b_commit as commit

SCHEMA = "shohin-mixtral-8x22b-selective-commit-validation-score-v1"
ARMS = (*commit.ARMS, "selective_commit")
BASE_ARMS = commit.ARMS
ROWS = commit.VALIDATION_ROWS
SHARDS = commit.VALIDATION_SHARDS
ASSESSOR_SHA256 = "af86c3e882c05cb336ed2231011feba4bfdeb93eb6fb8de5539abb71d04ec16e"
COMMIT_MODEL_SHA256 = "eb7512137a9d13224dbbb25d0866eeda6bcf3aa252b6a323b0902fa1d998dbc6"


class MixtralCommitScoreError(RuntimeError):
    """The validation score or learned-commit lineage differed."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise MixtralCommitScoreError("score output exists")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(encoded).hexdigest()


def _paired(left: dict[str, bool], right: dict[str, bool]) -> dict[str, Any]:
    if set(left) != set(right) or not left:
        raise MixtralCommitScoreError("paired coverage differs")
    left_only = sum(left[key] and not right[key] for key in left)
    right_only = sum(right[key] and not left[key] for key in left)
    return {
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "net_correct": left_only - right_only,
        "mcnemar_exact_two_sided_p": _mcnemar_exact(left_only, right_only),
    }


def _validate_probabilities(observed: Any, expected: list[float]) -> None:
    if (
        not isinstance(observed, list)
        or len(observed) != len(BASE_ARMS)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
            for value in observed
        )
        or any(
            not math.isclose(float(left), right, rel_tol=0.0, abs_tol=1e-15)
            for left, right in zip(observed, expected, strict=True)
        )
    ):
        raise MixtralCommitScoreError("commit probabilities differ")


def validate_application(
    args: argparse.Namespace,
    sources: list[dict[str, Any]],
    candidates: dict[str, list[dict[str, Any]]],
    candidate_receipts: dict[str, list[str]],
) -> tuple[dict[str, str], dict[str, Any]]:
    if commit.sha256_file(args.model) != COMMIT_MODEL_SHA256:
        raise MixtralCommitScoreError("commit model hash differs")
    model, weights = commit._load_model(args.model)
    application = commit._json(args.application_report)
    selected_rows = commit._jsonl(args.commit_candidates)
    selection_rows = commit._jsonl(args.selections)
    model_sha256 = commit.sha256_file(args.model)
    if (
        application.get("schema") != commit.APPLICATION_REPORT_SCHEMA
        or application.get("status") != "complete"
        or application.get("rows") != ROWS
        or application.get("source_sha256") != commit.VALIDATION_SOURCE_SHA256
        or application.get("candidate_report_sha256s") != candidate_receipts
        or application.get("model_sha256") != model_sha256
        or application.get("output") != str(args.commit_candidates.resolve())
        or application.get("output_sha256")
        != commit.sha256_file(args.commit_candidates)
        or application.get("selections") != str(args.selections.resolve())
        or application.get("selections_sha256") != commit.sha256_file(args.selections)
        or application.get("assessor_access_count") != 0
        or application.get("validation_labels_read") != 0
        or application.get("task_label_used_as_feature") is not False
        or application.get("sealed_access") != {"holdout": 0, "product": 0, "public": 0}
        or len(selected_rows) != ROWS
        or len(selection_rows) != ROWS
    ):
        raise MixtralCommitScoreError("commit application report differs")

    selections: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for index, source in enumerate(sources):
        identity = source["identity_sha256"]
        features = [
            commit.candidate_features(
                source["source_prompt"], arm, candidates[arm][index]
            )
            for arm in BASE_ARMS
        ]
        selected_index, probabilities = commit.select_arm(weights, features)
        selected_arm = BASE_ARMS[selected_index]
        selection = selection_rows[index]
        output = selected_rows[index]
        candidate = candidates[selected_arm][index]
        if (
            selection.get("schema") != commit.SELECTION_SCHEMA
            or selection.get("split") != "external_validation_confirmation"
            or selection.get("identity_sha256") != identity
            or selection.get("selected_arm") != selected_arm
            or selection.get("model_sha256") != model_sha256
        ):
            raise MixtralCommitScoreError("commit selection differs")
        _validate_probabilities(selection.get("probabilities"), probabilities)
        if (
            output.get("schema") != commit.CANDIDATE_OUTPUT_SCHEMA
            or output.get("arm") != "selective_commit"
            or output.get("selected_arm") != selected_arm
            or output.get("identity_sha256") != identity
            or output.get("task") != source["task"]
            or output.get("completion") != candidate["completion"]
            or output.get("generated_tokens") != candidate["generated_tokens"]
            or output.get("max_token_exhausted") != candidate["max_token_exhausted"]
            or output.get("model_sha256") != model_sha256
        ):
            raise MixtralCommitScoreError("commit candidate differs")
        selections[identity] = selected_arm
        counts[selected_arm] += 1
    if (
        dict(sorted(counts.items())) != application.get("selection_counts")
        or model.get("task_label_used_as_feature") is not False
        or model.get("validation_labels_read") != 0
    ):
        raise MixtralCommitScoreError("commit selection count differs")
    return selections, application


def _arm_metrics(
    arm: str,
    outcomes: dict[str, dict[str, bool]],
    tasks: dict[str, str],
    candidates: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    baseline = outcomes["unchanged"]
    values = outcomes[arm]
    correct = sum(values.values())
    baseline_correct = sum(baseline.values())
    retained = sum(baseline[key] and values[key] for key in values)
    task_rows = Counter(tasks.values())
    task_correct = Counter(tasks[key] for key, value in values.items() if value)
    return {
        "correct": correct,
        "total": ROWS,
        "accuracy": correct / ROWS,
        "gain_over_unchanged_count": correct - baseline_correct,
        "gain_over_unchanged_percentage_points": 100.0
        * (correct - baseline_correct)
        / ROWS,
        "unchanged_correct_retained": retained,
        "unchanged_correct_retention": (
            retained / baseline_correct if baseline_correct else None
        ),
        "domains": {
            task: {"correct": task_correct[task], "total": task_rows[task]}
            for task in commit.TASKS
        },
        "empty_completions": sum(
            not candidates[arm][key]["completion"].strip() for key in values
        ),
        "max_token_exhausted": sum(
            candidates[arm][key]["max_token_exhausted"] for key in values
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise MixtralCommitScoreError("score output exists")
    sources = commit.load_sources(
        args.source,
        rows=ROWS,
        expected_sha256=commit.VALIDATION_SOURCE_SHA256,
    )
    base_lists, candidate_receipts = commit.load_candidates(
        args.candidate_root,
        sources,
        shards=SHARDS,
        expected_source_sha256=commit.VALIDATION_SOURCE_SHA256,
        expected_split="external_validation_confirmation",
    )
    selections, application = validate_application(
        args, sources, base_lists, candidate_receipts
    )

    # The assessor remains unopened until all four arms and their lineage pass.
    if commit.sha256_file(args.assessors) != ASSESSOR_SHA256:
        raise MixtralCommitScoreError("validation assessor hash differs")
    assessors = load_assessors(args.assessors, ROWS)
    source_identities = {row["identity_sha256"] for row in sources}
    if set(assessors) != source_identities:
        raise MixtralCommitScoreError("source and assessor identities differ")

    sandbox = qualify_allocation()
    sandbox_sha256 = sandbox_atomic_json(args.sandbox_receipt, sandbox)
    setup_receipts = qualify_mbpp_assessor_setups(
        [row["assessor"] for row in assessors.values() if row["task"] == "mbpp"]
    )
    setup_receipts_sha256 = hashlib.sha256(
        json.dumps(setup_receipts, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    by_identity: dict[str, dict[str, dict[str, Any]]] = {
        arm: {row["identity_sha256"]: row for row in base_lists[arm]}
        for arm in BASE_ARMS
    }
    by_identity["selective_commit"] = {
        row["identity_sha256"]: row for row in commit._jsonl(args.commit_candidates)
    }
    outcomes: dict[str, dict[str, bool]] = {arm: {} for arm in ARMS}
    tasks = {identity: assessors[identity]["task"] for identity in assessors}
    for identity in sorted(source_identities):
        assessor = assessors[identity]
        for arm in BASE_ARMS:
            candidate = by_identity[arm][identity]
            if candidate["task"] != assessor["task"]:
                raise MixtralCommitScoreError("candidate task differs")
            result = score_completion(assessor["assessor"], candidate["completion"])
            correct = result.get("correct")
            if not isinstance(correct, bool):
                raise MixtralCommitScoreError("score result differs")
            outcomes[arm][identity] = correct
        outcomes["selective_commit"][identity] = outcomes[selections[identity]][
            identity
        ]

    metrics = {arm: _arm_metrics(arm, outcomes, tasks, by_identity) for arm in ARMS}
    comparisons = {
        "commit_vs_unchanged": _paired(
            outcomes["selective_commit"], outcomes["unchanged"]
        ),
        "commit_vs_self_refinement": _paired(
            outcomes["selective_commit"], outcomes["self_refinement"]
        ),
        "commit_vs_revision": _paired(
            outcomes["selective_commit"], outcomes["revision"]
        ),
        "revision_vs_unchanged": _paired(outcomes["revision"], outcomes["unchanged"]),
        "revision_vs_self_refinement": _paired(
            outcomes["revision"], outcomes["self_refinement"]
        ),
    }
    commit_metrics = metrics["selective_commit"]
    unchanged_metrics = metrics["unchanged"]
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "host": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "total_parameters": 141_000_000_000,
        "active_parameters": 39_000_000_000,
        "rows": ROWS,
        "shards_per_base_arm": SHARDS,
        "source_sha256": commit.sha256_file(args.source),
        "assessors_sha256": ASSESSOR_SHA256,
        "commit_model_sha256": COMMIT_MODEL_SHA256,
        "application_report_sha256": commit.sha256_file(args.application_report),
        "commit_candidates_sha256": commit.sha256_file(args.commit_candidates),
        "selections_sha256": commit.sha256_file(args.selections),
        "base_candidate_report_sha256s": candidate_receipts,
        "sandbox_receipt_sha256": sandbox_sha256,
        "sandbox_probe_sha256": sandbox.get("probe_sha256"),
        "mbpp_setup_qualification_count": len(setup_receipts),
        "mbpp_setup_qualifications_sha256": setup_receipts_sha256,
        "arms": metrics,
        "comparisons": comparisons,
        "selection_counts": application["selection_counts"],
        "decision": {
            "commit_improves_unchanged": comparisons["commit_vs_unchanged"][
                "net_correct"
            ]
            > 0,
            "commit_improves_self_refinement": comparisons["commit_vs_self_refinement"][
                "net_correct"
            ]
            > 0,
            "commit_improves_revision": comparisons["commit_vs_revision"]["net_correct"]
            > 0,
            "commit_retains_at_least_95_percent_unchanged_correct": (
                commit_metrics["unchanged_correct_retention"] is not None
                and commit_metrics["unchanged_correct_retention"] >= 0.95
            ),
            "commit_all_domains_nonnegative_vs_unchanged": all(
                commit_metrics["domains"][task]["correct"]
                >= unchanged_metrics["domains"][task]["correct"]
                for task in commit.TASKS
            ),
        },
        "outcomes": [
            {
                "identity_sha256": identity,
                "task": tasks[identity],
                "selected_arm": selections[identity],
                "correct": {arm: outcomes[arm][identity] for arm in ARMS},
            }
            for identity in sorted(source_identities)
        ],
        "assessor_open_phase": "after_all_four_arms_and_application_validated",
        "distinct_completions_scored_per_identity": 3,
        "selective_commit_score_derived_from_selected_scored_arm": True,
        "task_label_used_as_commit_feature": False,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
        "stop_after_result": True,
    }
    _atomic_json(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--assessors", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--commit-candidates", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--application-report", type=Path, required=True)
    parser.add_argument("--sandbox-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {arm: result["arms"][arm]["correct"] for arm in ARMS},
            sort_keys=True,
        )
    )
