#!/usr/bin/env python3
"""Stack out-of-fold sparse routing with the production Q36 commit trajectory."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import cross_validate_q36_mtr_sparse_router as cross_validation
import train_apply_q36_mtr_sparse_router as router

REPORT_SCHEMA = "shohin-q36-mtr-stacked-router-report-v1"
SELECTION_SCHEMA = "shohin-q36-mtr-stacked-router-selection-v1"
CV_SCHEMA = "shohin-q36-mtr-sparse-router-cross-validation-v1"
FOLDS = 16


class Q36MTRStackedRouterError(RuntimeError):
    """Stacked-router inputs, held-fold training, or outputs differ."""


def _margin_bin(scores: list[float]) -> int:
    if len(scores) != 3 or any(not math.isfinite(value) for value in scores):
        raise Q36MTRStackedRouterError("stacked-router sparse scores differ")
    ordered = sorted(scores, reverse=True)
    margin = ordered[0] - ordered[1]
    if margin < 0.1:
        return 0
    if margin < 0.2:
        return 1
    if margin < 0.5:
        return 2
    return 3


def _vote(
    training: list[dict[str, Any]],
    key: Callable[[dict[str, Any]], tuple[Any, ...]],
    row: dict[str, Any],
    minimum: int,
) -> bool | None:
    discordant = [
        item
        for item in training
        if key(item) == key(row)
        and item["correct"] != item["production_commit_correct"]
    ]
    if len(discordant) < minimum:
        return None
    sparse = sum(
        item["correct"] and not item["production_commit_correct"] for item in discordant
    )
    production = sum(
        item["production_commit_correct"] and not item["correct"] for item in discordant
    )
    return sparse >= production


def choose_sparse(
    row: dict[str, Any], training: list[dict[str, Any]]
) -> tuple[bool, str]:
    """Choose one router using only outcomes from the other identity folds."""

    if row["task"] == "mbpp":
        return False, "conservative_executable_code_retention"
    hierarchy: list[tuple[str, Callable[[dict[str, Any]], tuple[Any, ...]], int]] = [
        (
            "task_sparse_production_margin",
            lambda item: (
                item["task"],
                item["selected_lineage"],
                item["production_commit_lineage"],
                item["margin_bin"],
            ),
            1,
        ),
        (
            "task_sparse_production",
            lambda item: (
                item["task"],
                item["selected_lineage"],
                item["production_commit_lineage"],
            ),
            3,
        ),
        (
            "task_sparse",
            lambda item: (item["task"], item["selected_lineage"]),
            3,
        ),
        (
            "sparse_production",
            lambda item: (
                item["selected_lineage"],
                item["production_commit_lineage"],
            ),
            3,
        ),
        ("margin", lambda item: (item["margin_bin"],), 3),
    ]
    for name, key, minimum in hierarchy:
        decision = _vote(training, key, row, minimum)
        if decision is not None:
            return decision, name
    sparse = sum(
        item["correct"] and not item["production_commit_correct"] for item in training
    )
    production = sum(
        item["production_commit_correct"] and not item["correct"] for item in training
    )
    return sparse >= production, "global_discordant_prior"


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise Q36MTRStackedRouterError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    import hashlib

    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTRStackedRouterError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def stack(args: argparse.Namespace) -> dict[str, Any]:
    if (
        args.cross_validation_report.is_symlink()
        or not args.cross_validation_report.is_file()
    ):
        raise Q36MTRStackedRouterError("stacked-router cross-validation report differs")
    cross_payload = json.loads(args.cross_validation_report.read_text(encoding="utf-8"))
    raw_outcomes = cross_payload.get("outcomes")
    if (
        cross_payload.get("schema") != CV_SCHEMA
        or cross_payload.get("status") != "complete"
        or cross_payload.get("rows") != router.DEVELOPMENT_ROWS
        or cross_payload.get("training_labels_exclude_held_out_fold") is not True
        or not isinstance(raw_outcomes, list)
        or len(raw_outcomes) != router.DEVELOPMENT_ROWS
    ):
        raise Q36MTRStackedRouterError("stacked-router cross-validation differs")

    production_candidates = router.load_development_candidates(
        args.production_candidates
    )
    production_scores = cross_validation._scores(
        args.production_scores,
        production_candidates,
        args.production_candidates,
    )
    owner_paths = (
        args.current_candidates,
        args.owner71_candidates,
        args.owner8_candidates,
    )
    owners = [router.load_development_candidates(paths) for paths in owner_paths]
    expected = set(production_candidates)
    if (
        any(set(owner) != expected for owner in owners)
        or set(production_scores) != expected
    ):
        raise Q36MTRStackedRouterError("stacked-router owner coverage differs")

    enriched: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_outcomes:
        identity = raw.get("identity_sha256") if isinstance(raw, dict) else None
        if (
            not isinstance(identity, str)
            or identity in seen
            or identity not in expected
            or raw.get("task") not in router.TASKS
            or raw.get("selected_lineage") not in router.LINEAGES
            or not isinstance(raw.get("correct"), bool)
            or isinstance(raw.get("fold"), bool)
            or not isinstance(raw.get("fold"), int)
            or not 0 <= raw["fold"] < FOLDS
        ):
            raise Q36MTRStackedRouterError("stacked-router outcome differs")
        production_row = production_candidates[identity]
        matches = [
            lineage
            for lineage, owner in zip(router.LINEAGES, owners, strict=True)
            if owner[identity]["completion"] == production_row["completion"]
        ]
        production_lineage = matches[0] if len(matches) == 1 else "ambiguous"
        row = dict(raw)
        row["production_commit_correct"] = production_scores[identity]["correct"]
        row["production_commit_lineage"] = production_lineage
        row["margin_bin"] = _margin_bin(raw.get("scores"))
        enriched.append(row)
        seen.add(identity)
    if seen != expected:
        raise Q36MTRStackedRouterError("stacked-router outcome coverage differs")

    selected_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for row in enriched:
        training = [item for item in enriched if item["fold"] != row["fold"]]
        use_sparse, reason = choose_sparse(row, training)
        identity = row["identity_sha256"]
        if use_sparse:
            selected_index = router.LINEAGES.index(row["selected_lineage"])
            candidate = dict(owners[selected_index][identity])
            selected_source = "sparse_router"
            selected_correct = row["correct"]
        else:
            candidate = dict(production_candidates[identity])
            selected_source = "production_commit"
            selected_correct = row["production_commit_correct"]
        candidate["stacked_router_selection"] = {
            "schema": SELECTION_SCHEMA,
            "selected_source": selected_source,
            "reason": reason,
            "fold": row["fold"],
            "training_excludes_selected_fold": True,
            "cross_validation_report_sha256": router.sha256_file(
                args.cross_validation_report
            ),
        }
        selected_rows.append(candidate)
        decisions.append(
            {
                "schema": SELECTION_SCHEMA,
                "identity_sha256": identity,
                "task": row["task"],
                "fold": row["fold"],
                "selected_source": selected_source,
                "reason": reason,
                "correct": selected_correct,
                "production_commit_correct": row["production_commit_correct"],
                "sparse_router_correct": row["correct"],
            }
        )
        reasons[reason] += 1
    selected_rows.sort(key=lambda row: row["identity_sha256"])
    decisions.sort(key=lambda row: row["identity_sha256"])
    output_sha = _atomic_lines(args.output, selected_rows)
    decisions_sha = _atomic_lines(args.decisions, decisions)
    correct = sum(row["correct"] for row in decisions)
    production_correct = sum(row["production_commit_correct"] for row in decisions)
    meta_only = sum(
        row["correct"] and not row["production_commit_correct"] for row in decisions
    )
    production_only = sum(
        row["production_commit_correct"] and not row["correct"] for row in decisions
    )
    result = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "interpretation": "engineering_nested_out_of_fold_stacked_trajectory_commit",
        "rows": len(decisions),
        "correct": correct,
        "accuracy": correct / len(decisions),
        "gain_over_production_commit_count": correct - production_correct,
        "gain_over_production_commit_points": 100.0
        * (correct - production_correct)
        / len(decisions),
        "production_commit_correct": production_correct,
        "training_excludes_selected_fold": True,
        "conservative_executable_code_retention": True,
        "selection_counts": dict(
            sorted(Counter(row["selected_source"] for row in decisions).items())
        ),
        "reason_counts": dict(sorted(reasons.items())),
        "domains": {
            task: {
                "rows": sum(row["task"] == task for row in decisions),
                "correct": sum(
                    row["task"] == task and row["correct"] for row in decisions
                ),
            }
            for task in router.TASKS
        },
        "paired_vs_production_commit": {
            "both_correct": sum(
                row["correct"] and row["production_commit_correct"] for row in decisions
            ),
            "stacked_only_correct": meta_only,
            "production_only_correct": production_only,
            "both_wrong": sum(
                not row["correct"] and not row["production_commit_correct"]
                for row in decisions
            ),
            "mcnemar_exact_two_sided_p": cross_validation._mcnemar(
                meta_only, production_only
            ),
        },
        "cross_validation_report": str(args.cross_validation_report.resolve()),
        "cross_validation_report_sha256": router.sha256_file(
            args.cross_validation_report
        ),
        "output": str(args.output.resolve()),
        "output_sha256": output_sha,
        "decisions": str(args.decisions.resolve()),
        "decisions_sha256": decisions_sha,
        "production_candidate_sha256": [
            router.sha256_file(path) for path in args.production_candidates
        ],
        "production_score_sha256": [
            router.sha256_file(path) for path in args.production_scores
        ],
    }
    _atomic_json(args.report, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-validation-report", type=Path, required=True)
    parser.add_argument(
        "--production-candidates", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--production-scores", type=Path, action="append", required=True
    )
    for owner in ("current", "owner71", "owner8"):
        parser.add_argument(
            f"--{owner}-candidates", type=Path, action="append", required=True
        )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    payload = stack(parse_args())
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
