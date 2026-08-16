#!/usr/bin/env python3
"""Build balanced development supervision for revision/draft-hidden routing."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any

EVAL_SCHEMA = "shohin-q36-mtr-eval-v1"
CANDIDATE_SCHEMA = "shohin-q36-mtr-candidate-v1"
SCORE_SCHEMA = "shohin-q36-mtr-draft-preview-v1"
ROW_SCHEMA = "shohin-q36-mtr-multi-trajectory-gate-train-v1"
REPORT_SCHEMA = "shohin-q36-mtr-multi-trajectory-gate-data-report-v1"
BRANCHES = ("revision", "draft_hidden")
PRESENTATIONS = {"both_correct": 1, "revision_only": 2, "draft_hidden_only": 8}
TASKS = ("math500", "bbh_logic", "mbpp")


class Q36MTRMultiTrajectoryDataError(RuntimeError):
    """Multi-trajectory training inputs or output differ."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    except (OSError, json.JSONDecodeError) as error:
        raise Q36MTRMultiTrajectoryDataError(f"unreadable JSONL: {path}") from error
    if not all(isinstance(row, dict) for row in rows):
        raise Q36MTRMultiTrajectoryDataError("multi-trajectory JSONL differs")
    return rows


def _development(path: Path, expected_rows: int) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(path):
        identity = row.get("identity_sha256")
        if (
            row.get("schema") != EVAL_SCHEMA
            or row.get("split") != "development"
            or not isinstance(identity, str)
            or len(identity) != 64
            or identity in rows
            or row.get("task") not in TASKS
            or not isinstance(row.get("question"), str)
            or not row["question"].strip()
        ):
            raise Q36MTRMultiTrajectoryDataError(
                "multi-trajectory development row differs"
            )
        rows[identity] = row
    if len(rows) != expected_rows or set(row["task"] for row in rows.values()) != set(
        TASKS
    ):
        raise Q36MTRMultiTrajectoryDataError(
            "multi-trajectory development coverage differs"
        )
    return rows


def _candidates(
    path: Path, arm: str, identities: set[str]
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(path):
        identity = row.get("identity_sha256")
        if (
            row.get("schema") != CANDIDATE_SCHEMA
            or row.get("arm") != arm
            or identity not in identities
            or identity in rows
            or not isinstance(row.get("completion"), str)
            or not row["completion"].strip()
            or row.get("task") not in TASKS
        ):
            raise Q36MTRMultiTrajectoryDataError("multi-trajectory candidate differs")
        rows[identity] = row
    if set(rows) != identities:
        raise Q36MTRMultiTrajectoryDataError(
            "multi-trajectory candidate coverage differs"
        )
    return rows


def _outcomes(
    path: Path, arm: str, candidates_sha256: str, identities: set[str]
) -> dict[str, bool]:
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise Q36MTRMultiTrajectoryDataError(
            "multi-trajectory score unreadable"
        ) from error
    if (
        not isinstance(report, dict)
        or report.get("schema") != SCORE_SCHEMA
        or report.get("status") != "complete"
        or report.get("split") != "development"
        or report.get("evaluation_arm") != arm
        or report.get("candidates_sha256") != candidates_sha256
        or not isinstance(report.get("outcomes"), list)
    ):
        raise Q36MTRMultiTrajectoryDataError("multi-trajectory score differs")
    outcomes: dict[str, bool] = {}
    for row in report["outcomes"]:
        identity = row.get("identity_sha256") if isinstance(row, dict) else None
        if (
            identity not in identities
            or identity in outcomes
            or not isinstance(row.get("correct"), bool)
        ):
            raise Q36MTRMultiTrajectoryDataError("multi-trajectory outcome differs")
        outcomes[identity] = row["correct"]
    if set(outcomes) != identities:
        raise Q36MTRMultiTrajectoryDataError(
            "multi-trajectory outcome coverage differs"
        )
    return outcomes


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise Q36MTRMultiTrajectoryDataError("multi-trajectory output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            encoded = json.dumps(row, sort_keys=True) + "\n"
            handle.write(encoded)
            digest.update(encoded.encode())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTRMultiTrajectoryDataError("multi-trajectory report exists")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.expected_rows <= 0 or args.output == args.report:
        raise Q36MTRMultiTrajectoryDataError("multi-trajectory settings differ")
    development = _development(args.development_eval, args.expected_rows)
    identities = set(development)
    candidate_paths = {
        "revision": args.revision_candidates,
        "draft_hidden": args.draft_hidden_candidates,
    }
    score_paths = {
        "revision": args.revision_score,
        "draft_hidden": args.draft_hidden_score,
    }
    candidates = {
        arm: _candidates(path, arm, identities) for arm, path in candidate_paths.items()
    }
    candidate_hashes = {arm: sha256_file(path) for arm, path in candidate_paths.items()}
    outcomes = {
        arm: _outcomes(score_paths[arm], arm, candidate_hashes[arm], identities)
        for arm in BRANCHES
    }
    unique = Counter()
    presented = Counter()
    task_presentations = Counter()
    rows: list[dict[str, Any]] = []
    for identity in sorted(identities):
        revision_correct = outcomes["revision"][identity]
        hidden_correct = outcomes["draft_hidden"][identity]
        if revision_correct and hidden_correct:
            outcome_class = "both_correct"
            target = [0.5, 0.5]
            selected = min(
                BRANCHES,
                key=lambda arm: (len(candidates[arm][identity]["completion"]), arm),
            )
        elif revision_correct:
            outcome_class = "revision_only"
            target = [1.0, 0.0]
            selected = "revision"
        elif hidden_correct:
            outcome_class = "draft_hidden_only"
            target = [0.0, 1.0]
            selected = "draft_hidden"
        else:
            continue
        unique[outcome_class] += 1
        for presentation in range(PRESENTATIONS[outcome_class]):
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
            task_presentations[row["task"]] += 1
    if not rows or set(task_presentations) != set(TASKS):
        raise Q36MTRMultiTrajectoryDataError(
            "multi-trajectory selected coverage differs"
        )
    rows.sort(key=lambda row: row["identity_sha256"])
    output_sha256 = _atomic_lines(args.output, rows)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "development_rows": args.expected_rows,
        "unique_selected_rows": sum(unique.values()),
        "presentations": len(rows),
        "unique_outcome_counts": dict(sorted(unique.items())),
        "presentation_counts": dict(sorted(presented.items())),
        "task_presentations": dict(sorted(task_presentations.items())),
        "branch_names": list(BRANCHES),
        "presentation_rule": PRESENTATIONS,
        "development_eval_sha256": sha256_file(args.development_eval),
        "candidate_sha256s": candidate_hashes,
        "score_sha256s": {arm: sha256_file(path) for arm, path in score_paths.items()},
        "output": str(args.output.resolve()),
        "output_sha256": output_sha256,
    }
    _atomic_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-eval", type=Path, required=True)
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
    print(
        json.dumps(
            {
                "presentations": report["presentations"],
                "unique_selected_rows": report["unique_selected_rows"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
