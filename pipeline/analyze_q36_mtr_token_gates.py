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
EXTERNAL_SCORE_SCHEMA = "shohin-q36-mtr-external-score-v1"
ARMS = ("temporal_gate", "multi_trajectory_gate")
TASKS = ("math500", "bbh_logic", "mbpp")
HASH_LENGTH = 64


class Q36MTRTokenGateAnalysisError(RuntimeError):
    """Token-gate screens are incomplete, mismatched, or duplicated."""


def _bounded_count(value: Any, upper: int) -> bool:
    return (
        isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= upper
    )


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == HASH_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


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
    rows = payload.get("rows") if isinstance(payload, dict) else None
    unchanged_result = payload.get("unchanged") if isinstance(payload, dict) else None
    paired = payload.get("paired_vs_unchanged") if isinstance(payload, dict) else None
    shard_count = payload.get("shard_count") if isinstance(payload, dict) else None
    candidate_sha256s = (
        payload.get("temporal_candidate_sha256s") if isinstance(payload, dict) else None
    )
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != SCORE_SCHEMA
        or payload.get("status") != "complete"
        or arm not in ARMS
        or not isinstance(rows, int)
        or isinstance(rows, bool)
        or rows <= 0
        or not isinstance(result, dict)
        or result.get("total") != rows
        or not isinstance(result.get("correct"), int)
        or isinstance(result.get("correct"), bool)
        or not 0 <= result["correct"] <= rows
        or not isinstance(outcomes, list)
        or len(outcomes) != rows
        or not isinstance(shard_count, int)
        or isinstance(shard_count, bool)
        or shard_count <= 0
        or not isinstance(candidate_sha256s, list)
        or len(candidate_sha256s) != shard_count
        or len(set(candidate_sha256s)) != shard_count
        or not all(_sha256(value) for value in candidate_sha256s)
        or not _sha256(payload.get("assessors_sha256"))
        or not _sha256(payload.get("baseline_score_sha256"))
        or not _sha256(payload.get("sandbox_receipt_sha256"))
        or not _sha256(payload.get("sandbox_probe_sha256"))
        or not isinstance(payload.get("mbpp_setup_qualification_count"), int)
        or isinstance(payload.get("mbpp_setup_qualification_count"), bool)
        or payload["mbpp_setup_qualification_count"] <= 0
        or not isinstance(unchanged_result, dict)
        or unchanged_result.get("total") != rows
        or not _bounded_count(unchanged_result.get("correct"), rows)
        or not isinstance(paired, dict)
    ):
        raise Q36MTRTokenGateAnalysisError("token-gate score geometry differs")
    domains = result.get("domains")
    empty = result.get("empty_completions")
    exhausted = result.get("max_token_exhausted")
    if (
        not isinstance(domains, dict)
        or set(domains) != set(TASKS)
        or not _bounded_count(empty, rows)
        or not _bounded_count(exhausted, rows)
    ):
        raise Q36MTRTokenGateAnalysisError("token-gate diagnostics differ")
    domain_correct = domain_total = 0
    for task in TASKS:
        bucket = domains[task]
        if (
            not isinstance(bucket, dict)
            or not _bounded_count(bucket.get("total"), rows)
            or not _bounded_count(bucket.get("correct"), bucket["total"])
        ):
            raise Q36MTRTokenGateAnalysisError("token-gate domain differs")
        domain_correct += bucket["correct"]
        domain_total += bucket["total"]
    if domain_correct != result["correct"] or domain_total != rows:
        raise Q36MTRTokenGateAnalysisError("token-gate domain totals differ")
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
    for task in TASKS:
        task_rows = [row for row in parsed.values() if row["task"] == task]
        if domains[task] != {
            "correct": sum(row["correct"] for row in task_rows),
            "total": len(task_rows),
        }:
            raise Q36MTRTokenGateAnalysisError("token-gate domain outcomes differ")
    unchanged_correct = sum(row["unchanged_correct"] for row in parsed.values())
    arm_only = sum(
        row["correct"] and not row["unchanged_correct"] for row in parsed.values()
    )
    unchanged_only = sum(
        row["unchanged_correct"] and not row["correct"] for row in parsed.values()
    )
    expected_p = _mcnemar_exact(arm_only, unchanged_only)
    observed_p = paired.get("mcnemar_exact_two_sided_p")
    if (
        unchanged_result["correct"] != unchanged_correct
        or payload.get("gain_over_unchanged_count")
        != result["correct"] - unchanged_correct
        or paired.get("temporal_only_correct") != arm_only
        or paired.get("unchanged_only_correct") != unchanged_only
        or not isinstance(observed_p, (int, float))
        or isinstance(observed_p, bool)
        or not math.isfinite(observed_p)
        or not math.isclose(observed_p, expected_p, rel_tol=0.0, abs_tol=1e-15)
    ):
        raise Q36MTRTokenGateAnalysisError("token-gate paired evidence differs")
    return {
        "payload": payload,
        "arm": arm,
        "result": result,
        "outcomes": parsed,
        "rows": rows,
        "diagnostics": {
            "assessors_sha256": payload["assessors_sha256"],
            "baseline_score_sha256": payload["baseline_score_sha256"],
            "candidate_sha256s": candidate_sha256s,
            "domains": domains,
            "empty_completions": empty,
            "max_token_exhausted": exhausted,
            "mbpp_setup_qualification_count": payload["mbpp_setup_qualification_count"],
            "paired_vs_unchanged": paired,
            "sandbox_probe_sha256": payload["sandbox_probe_sha256"],
            "sandbox_receipt_sha256": payload["sandbox_receipt_sha256"],
            "shard_count": shard_count,
        },
    }


def _variant_name(path: Path, arm: str, occupied: set[str]) -> str:
    lineage = "/".join(
        parent.name.casefold() for parent in (path.parent.parent, path.parent)
    )
    if arm == "multi_trajectory_gate" and "tri_hierarchical_set_mass" in lineage:
        candidate = "tri_hierarchical_set_mass"
    elif arm == "multi_trajectory_gate" and "tri_hierarchical" in lineage:
        candidate = "tri_hierarchical"
    elif arm == "multi_trajectory_gate" and "tri_geometry" in lineage:
        candidate = "tri_geometry"
    elif arm == "multi_trajectory_gate" and "tri_trajectory" in lineage:
        candidate = "tri_trajectory"
    elif arm == "multi_trajectory_gate" and "routing_only" in lineage:
        candidate = "multi_routing_only"
    elif arm == "multi_trajectory_gate":
        candidate = "multi_trajectory"
    elif "supervised" in lineage:
        candidate = "response_supervised"
    elif "temporal" in lineage:
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
    row_count: int | None = None
    shared_binding: tuple[str, str, str] | None = None
    for path in args.score:
        loaded = _load_score(path)
        name = _variant_name(path, loaded["arm"], set(variants))
        current_identities = set(loaded["outcomes"])
        current_unchanged = {
            identity: row["unchanged_correct"]
            for identity, row in loaded["outcomes"].items()
        }
        current_binding = (
            loaded["diagnostics"]["assessors_sha256"],
            loaded["diagnostics"]["baseline_score_sha256"],
            loaded["diagnostics"]["sandbox_probe_sha256"],
        )
        if identities is None:
            identities = current_identities
            unchanged = current_unchanged
            row_count = loaded["rows"]
            shared_binding = current_binding
        elif (
            current_identities != identities
            or current_unchanged != unchanged
            or loaded["rows"] != row_count
            or current_binding != shared_binding
        ):
            raise Q36MTRTokenGateAnalysisError("token-gate benchmark identity differs")
        variants[name] = {**loaded, "path": path}
    assert identities is not None and unchanged is not None and row_count is not None
    unchanged_correct = sum(unchanged.values())
    if unchanged_correct <= 0:
        raise Q36MTRTokenGateAnalysisError("unchanged capability is absent")
    incumbent_score_receipt = None
    incumbent_revision_correct = args.incumbent_revision_correct
    incumbent_score = getattr(args, "incumbent_score", None)
    if incumbent_score is not None:
        if incumbent_revision_correct is not None:
            raise Q36MTRTokenGateAnalysisError("incumbent revision is ambiguous")
        try:
            incumbent_payload = json.loads(incumbent_score.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise Q36MTRTokenGateAnalysisError(
                "incumbent revision score is unreadable"
            ) from error
        arms = (
            incumbent_payload.get("arms")
            if isinstance(incumbent_payload, dict)
            else None
        )
        revision = arms.get("revision") if isinstance(arms, dict) else None
        baseline = arms.get("unchanged") if isinstance(arms, dict) else None
        incumbent_revision_correct = (
            revision.get("correct") if isinstance(revision, dict) else None
        )
        if (
            incumbent_payload.get("schema") != EXTERNAL_SCORE_SCHEMA
            or incumbent_payload.get("status") != "complete"
            or incumbent_payload.get("rows") != row_count
            or not isinstance(incumbent_revision_correct, int)
            or isinstance(incumbent_revision_correct, bool)
            or not 0 <= incumbent_revision_correct <= row_count
            or not isinstance(baseline, dict)
            or baseline.get("correct") != unchanged_correct
        ):
            raise Q36MTRTokenGateAnalysisError("incumbent revision score differs")
        incumbent_score_receipt = {
            "path": str(incumbent_score.resolve()),
            "sha256": sha256_file(incumbent_score),
        }
    if (
        incumbent_revision_correct is not None
        and not 0 <= incumbent_revision_correct <= row_count
    ):
        raise Q36MTRTokenGateAnalysisError("incumbent revision count differs")
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
            "total": row_count,
            "accuracy": correct / row_count,
            "gain_over_unchanged_count": correct - unchanged_correct,
            "retained_unchanged_correct": retained,
            "retention": retained / unchanged_correct,
            "domains": {
                task: {"correct": task_correct[task], "total": task_total[task]}
                for task in TASKS
            },
            **loaded["diagnostics"],
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
    eligibility: dict[str, dict[str, bool | None]] = {}
    for name in ranked:
        summary = summaries[name]
        beats_incumbent = (
            summary["correct"] > incumbent_revision_correct
            if incumbent_revision_correct is not None
            else None
        )
        eligibility[name] = {
            "beats_incumbent_revision": beats_incumbent,
            "retention_at_least_90_percent": summary["retention"] >= 0.9,
            "every_domain_nonzero": all(
                summary["domains"][task]["correct"] > 0 for task in TASKS
            ),
            "zero_empty_completions": summary["empty_completions"] == 0,
            "zero_max_token_exhausted": summary["max_token_exhausted"] == 0,
        }
    promotion_candidates = [
        name
        for name in ranked
        if eligibility[name]["beats_incumbent_revision"] is True
        and all(
            value
            for key, value in eligibility[name].items()
            if key != "beats_incumbent_revision"
        )
    ]
    promotion_candidate = promotion_candidates[0] if promotion_candidates else None
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "rows": row_count,
        "unchanged_correct": unchanged_correct,
        "incumbent_revision_correct": incumbent_revision_correct,
        "incumbent_score": incumbent_score_receipt,
        "variants": summaries,
        "pairwise": pairwise,
        "ranking": ranked,
        "promotion_eligibility": eligibility,
        "promotion_candidate": promotion_candidate,
        "winner": winner,
        "winner_correct": summaries[winner]["correct"],
        "winner_beats_incumbent_revision": (
            summaries[winner]["correct"] > incumbent_revision_correct
            if incumbent_revision_correct is not None
            else None
        ),
        "winner_retention_at_least_90_percent": math.isfinite(
            summaries[winner]["retention"]
        )
        and summaries[winner]["retention"] >= 0.9,
        "recommended_next_action": (
            "interpret_already_staged_1023_validation_for_promotion_candidate"
            if promotion_candidate is not None
            else "do_not_promote_new_router_from_this_screen"
        ),
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
    parser.add_argument("--incumbent-revision-correct", type=int)
    parser.add_argument("--incumbent-score", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps({"winner": report["winner"], "correct": report["winner_correct"]}))


if __name__ == "__main__":
    main()
