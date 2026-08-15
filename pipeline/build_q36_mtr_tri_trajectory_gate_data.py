#!/usr/bin/env python3
"""Build balanced owner/revision/draft-hidden routing supervision."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from build_q36_mtr_multi_trajectory_gate_data import (
    SCORE_SCHEMA,
    TASKS,
    _atomic_json,
    _atomic_lines,
    _candidates,
    _development,
    _load_jsonl,
    _outcomes,
    sha256_file,
)

ROW_SCHEMA = "shohin-q36-mtr-tri-trajectory-gate-train-v1"
REPORT_SCHEMA = "shohin-q36-mtr-tri-trajectory-gate-data-report-v1"
MODEL_DRAFT_SCHEMA = "shohin-q36-mtr-model-draft-v1"
MERGE_SCHEMA = "shohin-q36-mtr-merged-drafts-v1"
BRANCHES = ("owner", "revision", "draft_hidden")
EXPECTED_PATTERNS = {
    (False, False, False): 598,
    (False, False, True): 29,
    (False, True, False): 55,
    (False, True, True): 114,
    (True, False, False): 16,
    (True, False, True): 17,
    (True, True, False): 115,
    (True, True, True): 345,
}
EXCLUSIVE_PRESENTATIONS = {"owner": 16, "revision": 2, "draft_hidden": 8}
EXPECTED_UNIQUE = 691
EXPECTED_PRESENTATIONS = 1_189


class Q36MTRTriTrajectoryDataError(RuntimeError):
    """Three-trajectory development inputs or outputs differ."""


def _owner_candidates(
    path: Path, merge_report: Path, identities: set[str]
) -> dict[str, dict[str, Any]]:
    try:
        report = json.loads(merge_report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Q36MTRTriTrajectoryDataError(
            "owner merge report is unreadable"
        ) from error
    if (
        report.get("schema") != MERGE_SCHEMA
        or report.get("status") != "complete"
        or report.get("output_sha256") != sha256_file(path)
        or report.get("rows") != 7_113
        or not isinstance(report.get("input_receipts"), list)
        or len(report["input_receipts"]) != 16
    ):
        raise Q36MTRTriTrajectoryDataError("owner merge receipt differs")
    rows: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(path):
        identity = row.get("identity_sha256")
        if row.get("split") != "development":
            continue
        if (
            row.get("schema") != MODEL_DRAFT_SCHEMA
            or identity not in identities
            or identity in rows
            or row.get("task") not in TASKS
            or not isinstance(row.get("completion"), str)
            or not row["completion"].strip()
        ):
            raise Q36MTRTriTrajectoryDataError("owner candidate differs")
        rows[identity] = row
    if set(rows) != identities:
        raise Q36MTRTriTrajectoryDataError("owner candidate coverage differs")
    return rows


def _owner_outcomes(
    score_paths: list[Path], merge_report: Path, identities: set[str]
) -> dict[str, bool]:
    merge = json.loads(merge_report.read_text(encoding="utf-8"))
    candidate_hashes = {
        receipt.get("candidates_sha256") for receipt in merge["input_receipts"]
    }
    outcomes: dict[str, bool] = {}
    observed_hashes: set[str] = set()
    for path in score_paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        candidate_sha = report.get("candidates_sha256")
        if (
            report.get("schema") != SCORE_SCHEMA
            or report.get("status") != "complete"
            or report.get("split") != "development"
            or candidate_sha not in candidate_hashes
            or candidate_sha in observed_hashes
            or not isinstance(report.get("outcomes"), list)
        ):
            raise Q36MTRTriTrajectoryDataError("owner score differs")
        observed_hashes.add(candidate_sha)
        for row in report["outcomes"]:
            identity = row.get("identity_sha256") if isinstance(row, dict) else None
            if (
                identity not in identities
                or identity in outcomes
                or not isinstance(row.get("correct"), bool)
            ):
                raise Q36MTRTriTrajectoryDataError("owner outcome differs")
            outcomes[identity] = row["correct"]
    if observed_hashes != candidate_hashes or set(outcomes) != identities:
        raise Q36MTRTriTrajectoryDataError("owner score coverage differs")
    return outcomes


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.expected_rows != 1_289 or len(args.owner_score) != 16:
        raise Q36MTRTriTrajectoryDataError("tri-trajectory settings differ")
    development = _development(args.development_eval, args.expected_rows)
    identities = set(development)
    candidates: dict[str, dict[str, dict[str, Any]]] = {
        "owner": _owner_candidates(
            args.owner_candidates, args.owner_merge_report, identities
        ),
        "revision": _candidates(args.revision_candidates, "revision", identities),
        "draft_hidden": _candidates(
            args.draft_hidden_candidates, "draft_hidden", identities
        ),
    }
    outcomes = {
        "owner": _owner_outcomes(args.owner_score, args.owner_merge_report, identities),
        "revision": _outcomes(
            args.revision_score,
            "revision",
            sha256_file(args.revision_candidates),
            identities,
        ),
        "draft_hidden": _outcomes(
            args.draft_hidden_score,
            "draft_hidden",
            sha256_file(args.draft_hidden_candidates),
            identities,
        ),
    }
    patterns = Counter(
        tuple(outcomes[branch][identity] for branch in BRANCHES)
        for identity in identities
    )
    if dict(patterns) != EXPECTED_PATTERNS:
        raise Q36MTRTriTrajectoryDataError("tri-trajectory outcome geometry differs")

    rows: list[dict[str, Any]] = []
    unique = Counter()
    presented = Counter()
    tasks = Counter()
    for identity in sorted(identities):
        correct = [branch for branch in BRANCHES if outcomes[branch][identity]]
        if not correct:
            continue
        outcome_class = "_".join(correct) + "_correct"
        target = [float(branch in correct) / len(correct) for branch in BRANCHES]
        selected = min(
            correct,
            key=lambda branch: (
                len(candidates[branch][identity]["completion"]),
                branch,
            ),
        )
        presentations = (
            EXCLUSIVE_PRESENTATIONS.get(correct[0], 1) if len(correct) == 1 else 1
        )
        unique[outcome_class] += 1
        for presentation in range(presentations):
            row_identity = hashlib.sha256(
                f"{ROW_SCHEMA}|{identity}|{presentation}".encode()
            ).hexdigest()
            row = {
                "schema": ROW_SCHEMA,
                "identity_sha256": row_identity,
                "source_identity_sha256": identity,
                "task": development[identity]["task"],
                "question": development[identity]["question"],
                "response": candidates[selected][identity]["completion"],
                "branch_names": list(BRANCHES),
                "routing_target": target,
                "outcome_class": outcome_class,
                "selected_response_arm": selected,
                "presentation": presentation,
            }
            rows.append(row)
            presented[outcome_class] += 1
            tasks[row["task"]] += 1
    if sum(unique.values()) != EXPECTED_UNIQUE or len(rows) != EXPECTED_PRESENTATIONS:
        raise Q36MTRTriTrajectoryDataError(
            "tri-trajectory presentation geometry differs"
        )
    rows.sort(key=lambda row: row["identity_sha256"])
    output_sha = _atomic_lines(args.output, rows)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "development_rows": args.expected_rows,
        "unique_selected_rows": sum(unique.values()),
        "presentations": len(rows),
        "branch_names": list(BRANCHES),
        "initial_branch_weights": [0.1, 0.8, 0.1],
        "outcome_patterns": {
            "".join(map(lambda value: str(int(value)), key)): value
            for key, value in sorted(patterns.items())
        },
        "unique_outcome_counts": dict(sorted(unique.items())),
        "presentation_counts": dict(sorted(presented.items())),
        "task_presentations": dict(sorted(tasks.items())),
        "exclusive_presentation_rule": EXCLUSIVE_PRESENTATIONS,
        "development_eval_sha256": sha256_file(args.development_eval),
        "owner_candidates_sha256": sha256_file(args.owner_candidates),
        "owner_merge_report_sha256": sha256_file(args.owner_merge_report),
        "owner_score_sha256s": [sha256_file(path) for path in args.owner_score],
        "revision_candidates_sha256": sha256_file(args.revision_candidates),
        "revision_score_sha256": sha256_file(args.revision_score),
        "draft_hidden_candidates_sha256": sha256_file(args.draft_hidden_candidates),
        "draft_hidden_score_sha256": sha256_file(args.draft_hidden_score),
        "output": str(args.output.resolve()),
        "output_sha256": output_sha,
    }
    _atomic_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-eval", type=Path, required=True)
    parser.add_argument("--owner-candidates", type=Path, required=True)
    parser.add_argument("--owner-merge-report", type=Path, required=True)
    parser.add_argument("--owner-score", type=Path, action="append", required=True)
    parser.add_argument("--revision-candidates", type=Path, required=True)
    parser.add_argument("--revision-score", type=Path, required=True)
    parser.add_argument("--draft-hidden-candidates", type=Path, required=True)
    parser.add_argument("--draft-hidden-score", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=1_289)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps({"presentations": report["presentations"]}, sort_keys=True))


if __name__ == "__main__":
    main()
