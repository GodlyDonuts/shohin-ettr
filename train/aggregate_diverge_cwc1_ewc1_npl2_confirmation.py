#!/usr/bin/env python3
"""Aggregate five CWC1/EWC1/NPL2 confirmation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import aggregate_diverge_npl2_confirmation as npl2_aggregate
from eval_diverge_pqi1 import sha256_path


SCHEMA = "shohin-diverge-cwc1-ewc1-npl2-confirmation-v1"
SEED_SCHEMA = "shohin-diverge-cwc1-ewc1-npl2-confirmation-seed-v1"
_REPORTS: list[dict[str, Any]] = []
_WRAPPER_AUDIT: dict[str, Any]
_WRAPPER_AUDIT_PATH: Path
_WRAPPER_AUDIT_SHA256: str


def _load(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    compilation = report.get("custody", {}).get("world_owner", {}).get(
        "compilation", {}
    )
    if (
        report.get("schema") != SEED_SCHEMA
        or report.get("status") != "complete"
        or report.get("evaluation_split") != "confirmation"
        or not report.get("integrity_gate", {}).get("passed")
        or not compilation.get("gate", {}).get("passed")
    ):
        raise SystemExit(f"invalid CWC1/EWC1/NPL2 confirmation report: {path}")
    _REPORTS.append(report)
    return report


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    result = dict(payload)
    per_seed = {}
    all_world_gates = True
    for report in _REPORTS:
        seed = str(report["episode_seed"])
        compilation = report["custody"]["world_owner"]["compilation"]
        per_seed[seed] = compilation
        all_world_gates &= bool(compilation["gate"]["passed"])
    result["schema"] = SCHEMA
    result["world_composition"] = {
        "per_seed": per_seed,
        "all_seed_gates_passed": all_world_gates,
    }
    result["wrapper_custody"] = {
        "aggregate_audit": str(_WRAPPER_AUDIT_PATH),
        "aggregate_audit_sha256": _WRAPPER_AUDIT_SHA256,
        "aggregate_audit_passed": _WRAPPER_AUDIT["all_conditions_passed"],
    }
    gate = dict(result["gate"])
    conditions = dict(gate["conditions"])
    conditions["world_composition_all_seeds_pass"] = all_world_gates
    conditions["wrapper_audit_passed"] = bool(
        _WRAPPER_AUDIT["all_conditions_passed"]
    )
    gate["conditions"] = conditions
    gate["passed"] = all(conditions.values())
    result["gate"] = gate
    npl2_aggregate._atomic_json_original(path, result)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--wrapper-audit", type=Path, required=True)
    parser.add_argument("--wrapper-audit-sha256", required=True)
    args, remaining = parser.parse_known_args()
    if sha256_path(args.wrapper_audit) != args.wrapper_audit_sha256:
        raise SystemExit("CWC1/EWC1/NPL2 wrapper audit hash differs")
    audit = json.loads(args.wrapper_audit.read_text(encoding="utf-8"))
    if not audit.get("all_conditions_passed"):
        raise SystemExit("CWC1/EWC1/NPL2 wrapper audit did not pass")
    global _WRAPPER_AUDIT, _WRAPPER_AUDIT_PATH, _WRAPPER_AUDIT_SHA256
    _WRAPPER_AUDIT = audit
    _WRAPPER_AUDIT_PATH = args.wrapper_audit
    _WRAPPER_AUDIT_SHA256 = args.wrapper_audit_sha256
    npl2_aggregate.SCHEMA = SCHEMA
    npl2_aggregate._load = _load
    npl2_aggregate._atomic_json_original = npl2_aggregate._atomic_json
    npl2_aggregate._atomic_json = _atomic_json
    sys.argv = [sys.argv[0], *remaining]
    npl2_aggregate.main()


if __name__ == "__main__":
    main()
