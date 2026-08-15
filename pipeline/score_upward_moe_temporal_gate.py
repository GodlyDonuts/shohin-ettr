#!/usr/bin/env python3
"""Score all matched upward-MoE temporal arms in one assessor process."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
from typing import Any

from hf_pcf1_evaluate import shard_bounds
from hf_upward_moe_evaluate_temporal_gate import (
    ARMS,
    CANDIDATE_SCHEMA,
    REPORT_SCHEMA as EVALUATION_REPORT_SCHEMA,
    ROWS,
    SHARDS,
)
from hf_upward_moe_train_temporal_gate import host_spec
from pcf1_code_sandbox import (
    atomic_json as sandbox_atomic_json,
    qualify_allocation,
    qualify_mbpp_assessor_setups,
    score_completion,
)
from upward_moe_role_lineage import sha256_file

ASSESSOR_SCHEMA = "shohin-pcf1-confirmation-assessor-v1"
REPORT_SCHEMA = "shohin-upward-moe-temporal-score-v1"
TASKS = ("math500", "bbh_logic", "mbpp")


class UpwardMoETemporalScoreError(RuntimeError):
    """The upward-MoE candidates, assessors, or paired score differed."""


def _mcnemar_exact(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if not discordant:
        return 1.0
    tail = sum(
        math.comb(discordant, index) for index in range(min(left_only, right_only) + 1)
    )
    return min(1.0, 2.0 * tail / (2**discordant))


def load_assessors(path: Path) -> dict[str, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise UpwardMoETemporalScoreError("upward assessor board is absent")
    assessors: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        identity = row.get("identity_sha256") if isinstance(row, dict) else None
        assessor = row.get("assessor") if isinstance(row, dict) else None
        if (
            not isinstance(row, dict)
            or row.get("schema") != ASSESSOR_SCHEMA
            or row.get("split") != "confirmation"
            or row.get("task") not in TASKS
            or not isinstance(identity, str)
            or len(identity) != 64
            or identity in assessors
            or not isinstance(assessor, dict)
            or assessor.get("identity_sha256") != identity
            or assessor.get("task") != row["task"]
        ):
            raise UpwardMoETemporalScoreError("upward assessor row differs")
        assessors[identity] = row
    if len(assessors) != ROWS or {row["task"] for row in assessors.values()} != set(
        TASKS
    ):
        raise UpwardMoETemporalScoreError("upward assessor coverage differs")
    return assessors


def _group(values: list[list[str]], label: str) -> dict[str, list[Path]]:
    grouped = {arm: [] for arm in ARMS}
    for arm, raw_path in values:
        if arm not in grouped:
            raise UpwardMoETemporalScoreError(f"upward {label} arm differs")
        grouped[arm].append(Path(raw_path))
    if any(len(grouped[arm]) != SHARDS for arm in ARMS):
        raise UpwardMoETemporalScoreError(f"upward {label} geometry differs")
    return grouped


def load_arm(
    arm: str,
    candidate_paths: list[Path],
    report_paths: list[Path],
    host: str,
    identities: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    spec = host_spec(host)
    candidates: dict[str, dict[str, Any]] = {}
    reports_by_index: dict[int, dict[str, Any]] = {}
    common: dict[str, Any] | None = None
    receipts = []
    for candidate_path, report_path in zip(candidate_paths, report_paths, strict=True):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        index = report.get("shard_index")
        if (
            report.get("schema") != EVALUATION_REPORT_SCHEMA
            or report.get("status") != "complete"
            or report.get("host") != spec.host
            or report.get("host_contract") != spec.receipt()
            or report.get("arm") != arm
            or report.get("split") != "development"
            or report.get("full_row_count") != ROWS
            or report.get("shard_count") != SHARDS
            or report.get("batch_size") != 1
            or report.get("assessor_access_count") != 0
            or report.get("development_labels_read") != 0
            or report.get("sealed_access") != {"holdout": 0, "product": 0, "public": 0}
            or isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < SHARDS
            or index in reports_by_index
            or report.get("candidates_output") != str(candidate_path.resolve())
            or report.get("candidates_sha256") != sha256_file(candidate_path)
        ):
            raise UpwardMoETemporalScoreError("upward evaluation report differs")
        start, end = shard_bounds(ROWS, index, SHARDS, 1)
        if report.get("row_start") != start or report.get("row_end") != end:
            raise UpwardMoETemporalScoreError("upward evaluation range differs")
        shard_common = {
            key: report.get(key)
            for key in (
                "host",
                "host_contract",
                "model_receipt",
                "role_lineage",
                "data_sha256",
                "mechanics_report_sha256",
                "generation_mode",
                "generation_sequence_contract",
                "max_new_tokens",
                "seed",
            )
        }
        if common is None:
            common = shard_common
        elif common != shard_common:
            raise UpwardMoETemporalScoreError("upward evaluation lineage differs")
        rows = [
            json.loads(line)
            for line in candidate_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if len(rows) != end - start or report.get("counters", {}).get("rows") != len(
            rows
        ):
            raise UpwardMoETemporalScoreError("upward candidate shard differs")
        for row in rows:
            identity = row.get("identity_sha256") if isinstance(row, dict) else None
            if (
                not isinstance(row, dict)
                or row.get("schema") != CANDIDATE_SCHEMA
                or row.get("host") != spec.host
                or row.get("arm") != arm
                or row.get("task") not in TASKS
                or not isinstance(identity, str)
                or identity not in identities
                or identity in candidates
                or not isinstance(row.get("completion"), str)
                or isinstance(row.get("generated_tokens"), bool)
                or not isinstance(row.get("generated_tokens"), int)
                or row["generated_tokens"] < 0
                or not isinstance(row.get("max_token_exhausted"), bool)
            ):
                raise UpwardMoETemporalScoreError("upward candidate row differs")
            candidates[identity] = row
        reports_by_index[index] = report
        receipts.append(
            {
                "shard_index": index,
                "candidate_sha256": report["candidates_sha256"],
                "report_sha256": sha256_file(report_path),
            }
        )
    if set(candidates) != identities or set(reports_by_index) != set(range(SHARDS)):
        raise UpwardMoETemporalScoreError("upward candidate coverage differs")
    return candidates, {
        **(common or {}),
        "input_receipts": sorted(receipts, key=lambda item: item["shard_index"]),
        "exact_identity_coverage": True,
    }


def _paired(left: dict[str, bool], right: dict[str, bool]) -> dict[str, Any]:
    identities = set(left)
    if identities != set(right):
        raise UpwardMoETemporalScoreError("upward paired coverage differs")
    left_only = sum(left[i] and not right[i] for i in identities)
    right_only = sum(right[i] and not left[i] for i in identities)
    return {
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "net_correct": left_only - right_only,
        "mcnemar_exact_two_sided_p": _mcnemar_exact(left_only, right_only),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise UpwardMoETemporalScoreError("upward temporal score exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.sandbox_receipt.exists():
        raise UpwardMoETemporalScoreError("upward score output exists")
    assessors = load_assessors(args.assessors)
    identities = set(assessors)
    candidates_by_arm = _group(args.candidates, "candidate")
    reports_by_arm = _group(args.reports, "report")
    candidates = {}
    custody = {}
    for arm in ARMS:
        candidates[arm], custody[arm] = load_arm(
            arm,
            candidates_by_arm[arm],
            reports_by_arm[arm],
            args.host,
            identities,
        )
    common_keys = (
        "host",
        "host_contract",
        "model_receipt",
        "role_lineage",
        "data_sha256",
        "mechanics_report_sha256",
        "generation_mode",
        "generation_sequence_contract",
        "max_new_tokens",
        "seed",
    )
    reference = {key: custody[ARMS[0]][key] for key in common_keys}
    if any(
        {key: custody[arm][key] for key in common_keys} != reference for arm in ARMS[1:]
    ):
        raise UpwardMoETemporalScoreError("upward matched-arm custody differs")
    sandbox = qualify_allocation()
    sandbox_sha256 = sandbox_atomic_json(args.sandbox_receipt, sandbox)
    setups = qualify_mbpp_assessor_setups(
        [row["assessor"] for row in assessors.values() if row["task"] == "mbpp"]
    )
    correct = {arm: {} for arm in ARMS}
    domains = {arm: Counter() for arm in ARMS}
    empty = Counter()
    exhausted = Counter()
    policy_rejections = Counter()
    for identity in sorted(identities):
        assessor = assessors[identity]
        for arm in ARMS:
            candidate = candidates[arm][identity]
            if candidate["task"] != assessor["task"]:
                raise UpwardMoETemporalScoreError("upward task binding differs")
            result = score_completion(assessor["assessor"], candidate["completion"])
            value = result.get("correct")
            if not isinstance(value, bool):
                raise UpwardMoETemporalScoreError("upward score result differs")
            correct[arm][identity] = value
            domains[arm][candidate["task"]] += int(value)
            empty[arm] += int(not candidate["completion"].strip())
            exhausted[arm] += int(candidate["max_token_exhausted"])
            execution = result.get("execution")
            policy_rejections[arm] += int(
                isinstance(execution, dict)
                and execution.get("termination_classification")
                == "candidate_policy_rejection"
            )
    task_rows = Counter(row["task"] for row in assessors.values())
    arm_metrics = {}
    for arm in ARMS:
        count = sum(correct[arm].values())
        arm_metrics[arm] = {
            "correct": count,
            "total": ROWS,
            "accuracy": count / ROWS,
            "domains": {
                task: {"correct": domains[arm][task], "total": task_rows[task]}
                for task in TASKS
            },
            "empty_completions": empty[arm],
            "max_token_exhausted": exhausted[arm],
            "candidate_policy_rejections": policy_rejections[arm],
        }
    temporal = correct["temporal_gate"]
    unchanged_correct = sum(correct["unchanged"].values())
    retained = sum(
        temporal[identity] and correct["unchanged"][identity] for identity in identities
    )
    paired = {
        control: _paired(temporal, correct[control])
        for control in ("unchanged", "self_refinement", "owner", "aligned_revision")
    }
    domain_deltas = {
        control: {
            task: domains["temporal_gate"][task] - domains[control][task]
            for task in TASKS
        }
        for control in ("unchanged", "self_refinement", "owner", "aligned_revision")
    }
    temporal_accuracy = arm_metrics["temporal_gate"]["accuracy"]
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "host": reference["host"],
        "rows": ROWS,
        "assessors_sha256": sha256_file(args.assessors),
        "semantic_assessor_opens": 1,
        "sandbox_receipt_sha256": sandbox_sha256,
        "sandbox_probe_sha256": sandbox.get("probe_sha256"),
        "mbpp_setup_qualification_count": len(setups),
        "arm_custody": custody,
        "arms": arm_metrics,
        "paired_temporal_vs_controls": paired,
        "domain_correct_deltas_temporal_vs_controls": domain_deltas,
        "unchanged_correct_retained": retained,
        "unchanged_correct_total": unchanged_correct,
        "unchanged_correct_retention": (
            retained / unchanged_correct if unchanged_correct else 1.0
        ),
        "promotion_indicators": {
            "temporal_gain_over_unchanged_at_least_5pp": temporal_accuracy
            >= arm_metrics["unchanged"]["accuracy"] + 0.05,
            "temporal_gain_over_self_refinement_at_least_3pp": temporal_accuracy
            >= arm_metrics["self_refinement"]["accuracy"] + 0.03,
            "temporal_not_worse_than_aligned_revision": temporal_accuracy
            >= arm_metrics["aligned_revision"]["accuracy"],
            "unchanged_correct_retention_at_least_95pct": (
                retained / unchanged_correct if unchanged_correct else 1.0
            )
            >= 0.95,
            "nonnegative_domain_deltas_vs_unchanged": all(
                value >= 0 for value in domain_deltas["unchanged"].values()
            ),
            "zero_empty_temporal_completions": empty["temporal_gate"] == 0,
        },
        "outcomes": [
            {
                "identity_sha256": identity,
                "task": assessors[identity]["task"],
                "correct": {arm: correct[arm][identity] for arm in ARMS},
            }
            for identity in sorted(identities)
        ],
    }
    report["research_promotion_ready"] = all(report["promotion_indicators"].values())
    _atomic_json(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host", choices=("nemotron-super", "mixtral-8x22b"), required=True
    )
    parser.add_argument("--assessors", type=Path, required=True)
    parser.add_argument(
        "--candidate", dest="candidates", nargs=2, action="append", required=True
    )
    parser.add_argument(
        "--evaluation-report", dest="reports", nargs=2, action="append", required=True
    )
    parser.add_argument("--sandbox-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
