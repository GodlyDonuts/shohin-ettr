#!/usr/bin/env python3
"""Compare Q36 token-gate screens and nominate the measured winner."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
from typing import Any

from score_q36_mtr_external import _mcnemar_exact, sha256_file

SCHEMA = "shohin-q36-mtr-token-gate-comparison-v1"
SCORE_SCHEMA = "shohin-q36-mtr-temporal-gate-score-v1"
ARMS = ("temporal_gate", "multi_trajectory_gate")
TASKS = ("math500", "bbh_logic", "mbpp")


class Q36MTRTokenGateAnalysisError(RuntimeError):
    """Token-gate screens are incomplete, mismatched, or duplicated."""


def _load_score(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Q36MTRTokenGateAnalysisError(
            f"unreadable token-gate score: {path}"
        ) from error
    arm = payload.get("arm") if isinstance(payload, dict) else None
    result = payload.get(arm) if isinstance(arm, str) else None
    outcomes = payload.get("outcomes") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != SCORE_SCHEMA
        or payload.get("status") != "complete"
        or arm not in ARMS
        or payload.get("rows") != 256
        or not isinstance(result, dict)
        or result.get("total") != 256
        or not isinstance(result.get("correct"), int)
        or isinstance(result.get("correct"), bool)
        or not 0 <= result["correct"] <= 256
        or not isinstance(outcomes, list)
        or len(outcomes) != 256
    ):
        raise Q36MTRTokenGateAnalysisError("token-gate score geometry differs")
    parsed: dict[str, dict[str, Any]] = {}
    correct_key = f"{arm}_correct"
    for row in outcomes:
        identity = row.get("identity_sha256") if isinstance(row, dict) else None
        if (
            not isinstance(identity, str)
            or len(identity) != 64
            or identity in parsed
            or row.get("task") not in TASKS
            or not isinstance(row.get(correct_key), bool)
            or not isinstance(row.get("unchanged_correct"), bool)
        ):
            raise Q36MTRTokenGateAnalysisError("token-gate outcome differs")
        parsed[identity] = {
            "task": row["task"],
            "correct": row[correct_key],
            "unchanged_correct": row["unchanged_correct"],
        }
    if sum(row["correct"] for row in parsed.values()) != result["correct"]:
        raise Q36MTRTokenGateAnalysisError("token-gate correct count differs")
    return {"payload": payload, "arm": arm, "result": result, "outcomes": parsed}


def _variant_name(path: Path, arm: str, occupied: set[str]) -> str:
    parent = path.parent.name.casefold()
    if arm == "multi_trajectory_gate":
        candidate = "multi_trajectory"
    elif "supervised" in parent:
        candidate = "response_supervised"
    elif "temporal" in parent:
        candidate = "causal_only"
    else:
        candidate = path.stem
    if candidate in occupied or not candidate.isidentifier():
        raise Q36MTRTokenGateAnalysisError("token-gate variant identity differs")
    return candidate


def run(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.score) < 2 or args.output.exists() or args.output.is_symlink():
        raise Q36MTRTokenGateAnalysisError("token-gate analysis settings differ")
    variants: dict[str, dict[str, Any]] = {}
    identities: set[str] | None = None
    unchanged: dict[str, bool] | None = None
    for path in args.score:
        loaded = _load_score(path)
        name = _variant_name(path, loaded["arm"], set(variants))
        current_identities = set(loaded["outcomes"])
        current_unchanged = {
            identity: row["unchanged_correct"]
            for identity, row in loaded["outcomes"].items()
        }
        if identities is None:
            identities = current_identities
            unchanged = current_unchanged
        elif current_identities != identities or current_unchanged != unchanged:
            raise Q36MTRTokenGateAnalysisError("token-gate benchmark identity differs")
        variants[name] = {**loaded, "path": path}
    assert identities is not None and unchanged is not None
    unchanged_correct = sum(unchanged.values())
    summaries: dict[str, Any] = {}
    for name, loaded in variants.items():
        outcomes = loaded["outcomes"]
        correct = sum(row["correct"] for row in outcomes.values())
        retained = sum(
            row["correct"] and unchanged[identity] for identity, row in outcomes.items()
        )
        task_correct = Counter(
            row["task"] for row in outcomes.values() if row["correct"]
        )
        task_total = Counter(row["task"] for row in outcomes.values())
        summaries[name] = {
            "arm": loaded["arm"],
            "score": str(loaded["path"].resolve()),
            "score_sha256": sha256_file(loaded["path"]),
            "correct": correct,
            "total": 256,
            "accuracy": correct / 256,
            "gain_over_unchanged_count": correct - unchanged_correct,
            "retained_unchanged_correct": retained,
            "retention": retained / unchanged_correct,
            "domains": {
                task: {"correct": task_correct[task], "total": task_total[task]}
                for task in TASKS
            },
        }
    pairwise: dict[str, Any] = {}
    names = sorted(variants)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            left_outcomes = variants[left]["outcomes"]
            right_outcomes = variants[right]["outcomes"]
            left_only = sum(
                left_outcomes[identity]["correct"]
                and not right_outcomes[identity]["correct"]
                for identity in identities
            )
            right_only = sum(
                right_outcomes[identity]["correct"]
                and not left_outcomes[identity]["correct"]
                for identity in identities
            )
            pairwise[f"{left}_vs_{right}"] = {
                f"{left}_only_correct": left_only,
                f"{right}_only_correct": right_only,
                "mcnemar_exact_two_sided_p": _mcnemar_exact(left_only, right_only),
            }
    ranked = sorted(
        summaries,
        key=lambda name: (
            -summaries[name]["correct"],
            -summaries[name]["retained_unchanged_correct"],
            name,
        ),
    )
    winner = ranked[0]
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "rows": 256,
        "unchanged_correct": unchanged_correct,
        "incumbent_revision_correct": args.incumbent_revision_correct,
        "variants": summaries,
        "pairwise": pairwise,
        "ranking": ranked,
        "winner": winner,
        "winner_correct": summaries[winner]["correct"],
        "winner_beats_incumbent_revision": (
            summaries[winner]["correct"] > args.incumbent_revision_correct
        ),
        "winner_retention_at_least_90_percent": math.isfinite(
            summaries[winner]["retention"]
        )
        and summaries[winner]["retention"] >= 0.9,
        "recommended_next_action": "evaluate_winner_on_1023_validation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", type=Path, action="append", required=True)
    parser.add_argument("--incumbent-revision-correct", type=int, default=141)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps({"winner": report["winner"], "correct": report["winner_correct"]}))


if __name__ == "__main__":
    main()
