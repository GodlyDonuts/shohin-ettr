#!/usr/bin/env python3
"""Aggregate five confirmations of the learned-executor NPL2 composition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import aggregate_diverge_cwc1_ewc1_npl2_confirmation as composed_aggregate
import aggregate_diverge_npl2_confirmation as npl2_aggregate


SCHEMA = "shohin-diverge-mze1-cwc1-ewc1-npl2-confirmation-v1"
SEED_SCHEMA = "shohin-diverge-mze1-cwc1-ewc1-npl2-confirmation-seed-v1"
_REPORTS: list[dict[str, Any]] = []
_ORIGINAL_ATOMIC = npl2_aggregate._atomic_json


def _load(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("schema") != SEED_SCHEMA
        or report.get("status") != "complete"
        or report.get("evaluation_split") != "confirmation"
        or not report.get("integrity_gate", {}).get("passed")
    ):
        raise SystemExit(f"invalid MZE1 confirmation report: {path}")
    owner = report.get("custody", {}).get("executor_owner")
    if (
        not isinstance(owner, dict)
        or owner.get("owner") != "model-owned-presented-z97-executor"
        or owner.get("exact_operation_import_in_candidate_runtime") is not False
    ):
        raise SystemExit(f"invalid MZE1 executor custody: {path}")
    _REPORTS.append(report)
    return report


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    result = dict(payload)
    owners = [report["custody"]["executor_owner"] for report in _REPORTS]
    state_hashes = {str(owner["state_sha256"]) for owner in owners}
    checkpoint_hashes = {str(owner["checkpoint_sha256"]) for owner in owners}
    report_hashes = {str(owner["qualification_report_sha256"]) for owner in owners}
    owner_exact = len(state_hashes) == len(checkpoint_hashes) == len(report_hashes) == 1
    result["schema"] = SCHEMA
    result["learned_executor"] = {
        "state_sha256": next(iter(state_hashes)) if len(state_hashes) == 1 else None,
        "checkpoint_sha256": (
            next(iter(checkpoint_hashes)) if len(checkpoint_hashes) == 1 else None
        ),
        "qualification_report_sha256": (
            next(iter(report_hashes)) if len(report_hashes) == 1 else None
        ),
        "all_seed_custody_exact": owner_exact,
        "exact_verifier_unchanged": all(
            bool(owner["exact_verifier_unchanged"]) for owner in owners
        ),
    }
    gate = dict(result["gate"])
    conditions = dict(gate["conditions"])
    conditions["learned_executor_all_seed_custody_exact"] = owner_exact
    gate["conditions"] = conditions
    gate["passed"] = all(conditions.values())
    result["gate"] = gate
    _ORIGINAL_ATOMIC(path, result)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    args, remaining = parser.parse_known_args()
    del args
    composed_aggregate.SCHEMA = SCHEMA
    composed_aggregate.SEED_SCHEMA = SEED_SCHEMA
    composed_aggregate._REPORTS = _REPORTS
    composed_aggregate._load = _load
    npl2_aggregate._atomic_json = _atomic_json
    sys.argv = [sys.argv[0], *remaining]
    composed_aggregate.main()


if __name__ == "__main__":
    main()
