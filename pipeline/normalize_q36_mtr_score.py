#!/usr/bin/env python3
"""Normalize the sole Q36 score into five immutable comparator arm reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

from compare_q36_mtr import ARM_SCHEMA
from q36_mtr_contract import MODEL_REVISION, TOTAL_ROWS
from score_q36_mtr import SCORE_SCHEMA

PRECOMPUTE_SCHEMA = "shohin-q36-mtr-precompute-custody-v1"
SCORER_TO_COMPARATOR = {
    "learned_commit": "learned_commit",
    "revision": "trained_revision",
    "unchanged": "unchanged",
    "self_refinement": "self_refinement",
    "draft_hidden": "draft_hidden",
}


class Q36MTRNormalizeError(RuntimeError):
    """The sole score result cannot support normalized Q36 arm reports."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _integer(value: object, *, maximum: int | None = None) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or (maximum is not None and value > maximum)
    ):
        raise Q36MTRNormalizeError("Q36 score counter differs")
    return value


def normalize(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise Q36MTRNormalizeError("Q36 normalized output exists")
    score = json.loads(args.score_report.read_text(encoding="utf-8"))
    custody = json.loads(args.precompute_custody.read_text(encoding="utf-8"))
    if (
        score.get("schema") != SCORE_SCHEMA
        or score.get("status") != "complete"
        or score.get("model_revision") != MODEL_REVISION
        or score.get("rows") != TOTAL_ROWS
        or score.get("outcome_rows") != TOTAL_ROWS
        or score.get("assessor_semantic_reads") != 1
        or score.get("score_consumption_state") != "consumed"
        or score.get("sealed_access") != {"holdout": 0, "product": 0, "public": 0}
        or custody.get("schema") != PRECOMPUTE_SCHEMA
        or custody.get("status") != "complete"
        or custody.get("model_revision") != MODEL_REVISION
        or custody.get("run_id") != score.get("run_id")
        or custody.get("identity_order_sha256") != score.get("identity_order_sha256")
        or custody.get("data_sha256")
        != score.get("input_hashes", {}).get("development_data_sha256")
    ):
        raise Q36MTRNormalizeError("Q36 score/precompute custody differs")
    metrics = score.get("metrics")
    empty = score.get("empty_completion_counts")
    policy = score.get("capability_policy_rejection_counts")
    malformed_completions = score.get("malformed_completion_counts")
    truncation = score.get("generation_truncation_counts")
    if not all(
        isinstance(value, dict)
        for value in (metrics, empty, policy, malformed_completions, truncation)
    ):
        raise Q36MTRNormalizeError("Q36 score arm counters are absent")
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.mkdir(parents=True)
    receipts = {}
    try:
        for source_arm, arm in SCORER_TO_COMPARATOR.items():
            arm_metrics = metrics.get(source_arm)
            if not isinstance(arm_metrics, dict) or set(arm_metrics) != {
                "overall",
                "math500",
                "bbh_logic",
                "mbpp",
            }:
                raise Q36MTRNormalizeError("Q36 score metric domains differ")
            normalized_metrics = {}
            for domain, value in arm_metrics.items():
                if not isinstance(value, dict):
                    raise Q36MTRNormalizeError("Q36 score metric differs")
                total = _integer(value.get("total"), maximum=TOTAL_ROWS)
                correct = _integer(value.get("correct"), maximum=total)
                normalized_metrics[domain] = {"correct": correct, "total": total}
            empty_count = _integer(empty.get(source_arm), maximum=TOTAL_ROWS)
            policy_count = _integer(policy.get(source_arm), maximum=TOTAL_ROWS)
            malformed = _integer(
                malformed_completions.get(source_arm), maximum=TOTAL_ROWS
            )
            if malformed < max(empty_count, policy_count):
                raise Q36MTRNormalizeError("Q36 malformed union differs")
            truncated = _integer(truncation.get(source_arm), maximum=TOTAL_ROWS)
            if source_arm == "learned_commit":
                malformed += _integer(score.get("commit_malformed"), maximum=TOTAL_ROWS)
                truncated += _integer(
                    score.get("commit_prompt_truncated"), maximum=TOTAL_ROWS * 2
                )
                truncated += _integer(score.get("commit_training_prompt_truncated"))
            report = {
                "schema": ARM_SCHEMA,
                "status": "complete",
                "arm": arm,
                "split": "development",
                "run_id": score["run_id"],
                "model_revision": MODEL_REVISION,
                "full_row_count": TOTAL_ROWS,
                "candidate_count": TOTAL_ROWS,
                "identity_order_sha256": score["identity_order_sha256"],
                "data_sha256": custody["data_sha256"],
                "runtime_sha256": custody["runtime_sha256"],
                "precompute_custody_sha256": sha256_file(args.precompute_custody),
                "score_report_sha256": sha256_file(args.score_report),
                "metrics": normalized_metrics,
                "truncation_count": truncated,
                "malformed_count": malformed,
                "sealed_access": {"holdout": 0, "product": 0, "public": 0},
            }
            if source_arm == "learned_commit":
                report["retention"] = score["retention"]
                report["order_consistency"] = score["order_consistency"]
            path = temporary / f"{arm}.json"
            path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            receipts[arm] = {
                "path": str((args.output / path.name).resolve()),
                "sha256": sha256_file(path),
            }
        os.replace(temporary, args.output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "schema": "shohin-q36-mtr-normalization-receipt-v1",
        "status": "complete",
        "arms": receipts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-report", type=Path, required=True)
    parser.add_argument("--precompute-custody", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(normalize(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
