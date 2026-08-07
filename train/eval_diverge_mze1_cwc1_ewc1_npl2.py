#!/usr/bin/env python3
"""Replace the exact NPL2 transition with the qualified learned MZE1 law."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import diverge_npl2_runtime as npl2_runtime
import eval_diverge_cwc1_ewc1_npl2 as composed
import eval_diverge_npl2_development as npl2_evaluator
from diverge_mze1_runtime import load_executor, sha256_path


DEVELOPMENT_SCHEMA = "shohin-diverge-mze1-cwc1-ewc1-npl2-development-v1"
CONFIRMATION_SCHEMA = "shohin-diverge-mze1-cwc1-ewc1-npl2-confirmation-seed-v1"


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mze-checkpoint", type=Path, required=True)
    parser.add_argument("--mze-checkpoint-sha256", required=True)
    parser.add_argument("--mze-report", type=Path, required=True)
    parser.add_argument("--mze-report-sha256", required=True)
    args, remaining = parser.parse_known_args()
    if sha256_path(args.mze_report) != args.mze_report_sha256:
        raise SystemExit("MZE1 qualification report hash differs")
    report = json.loads(args.mze_report.read_text(encoding="utf-8"))
    if report.get("status") != "pass" or not report.get("gate", {}).get("passed"):
        raise SystemExit("MZE1 composition requires a passing executor gate")
    executor, checkpoint = load_executor(
        args.mze_checkpoint, args.mze_checkpoint_sha256, arm="treatment"
    )
    if not checkpoint.get("gate_passed"):
        raise SystemExit("MZE1 checkpoint is not qualified")

    learned_transition = executor.transition
    npl2_runtime.apply_operation = learned_transition
    state_sha256 = str(checkpoint["treatment_state_sha256"])
    npl2_evaluator.EXECUTOR_OWNER_RECEIPT = f"mze1:{state_sha256}"
    npl2_evaluator.EXECUTOR_OWNER_CUSTODY = {
        "owner": "model-owned-presented-z97-executor",
        "checkpoint": str(args.mze_checkpoint),
        "checkpoint_sha256": args.mze_checkpoint_sha256,
        "state_sha256": state_sha256,
        "qualification_report": str(args.mze_report),
        "qualification_report_sha256": args.mze_report_sha256,
        "hard_rows": [
            [list(row) for row in operation] for operation in executor.hard_rows()
        ],
        "operation_semantics_learned_from_outcomes": True,
        "exact_operation_import_in_candidate_runtime": False,
        "exact_verifier_unchanged": True,
    }
    composed.DEVELOPMENT_SCHEMA = DEVELOPMENT_SCHEMA
    composed.CONFIRMATION_SCHEMA = CONFIRMATION_SCHEMA
    sys.argv = [sys.argv[0], *remaining]
    composed.main()


if __name__ == "__main__":
    main()
