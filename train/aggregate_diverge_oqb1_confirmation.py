#!/usr/bin/env python3
"""Aggregate the five conditionally opened OQB1 confirmation reports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from diverge_oqb1_data import CONFIRMATION_SEEDS
from diverge_eal1_data import canonical_sha256
from diverge_eal1_runtime import sha256_path
from eval_diverge_oqb1 import SCHEMA as EVALUATION_SCHEMA


SCHEMA = "shohin-diverge-oqb1-confirmation-aggregate-v1"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != EVALUATION_SCHEMA:
        raise RuntimeError(f"OQB1 confirmation schema differs: {path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--development-sha256", required=True)
    parser.add_argument("--confirmation", type=Path, action="append", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing existing OQB1 confirmation aggregate")
    if sha256_path(args.development) != args.development_sha256:
        raise SystemExit("OQB1 development gate hash differs")
    development = _load(args.development)
    if (
        development.get("status") != "pass"
        or not development.get("gate", {}).get("passed")
        or development.get("source_commit") != args.source_commit
        or development.get("data", {}).get("board_label") != "development"
    ):
        raise SystemExit("OQB1 development did not admit confirmation")
    if len(args.confirmation) != len(CONFIRMATION_SEEDS):
        raise SystemExit("OQB1 confirmation report count differs")

    reports = [_load(path) for path in args.confirmation]
    seeds = [int(report["data"]["board_seed"]) for report in reports]
    labels = [str(report["data"]["board_label"]) for report in reports]
    expected_labels = [f"confirmation_{seed}" for seed in CONFIRMATION_SEEDS]
    if (
        tuple(seeds) != CONFIRMATION_SEEDS
        or labels != expected_labels
        or any(report.get("status") != "pass" for report in reports)
        or any(not report.get("gate", {}).get("passed") for report in reports)
        or any(report.get("source_commit") != args.source_commit for report in reports)
        or any(
            report.get("parents") != development.get("parents") for report in reports
        )
        or any(
            report.get("training") != development.get("training") for report in reports
        )
        or len({report["data"]["board_identity_sha256"] for report in reports})
        != len(reports)
    ):
        raise SystemExit("OQB1 confirmation custody/gate differs")

    arms = {}
    for arm in reports[0]["arms"]:
        evidence_exact = sum(
            int(report["arms"][arm]["evidence_register"]["exact"]) for report in reports
        )
        evidence_total = sum(
            int(report["arms"][arm]["evidence_register"]["total"]) for report in reports
        )
        state_exact = sum(
            int(report["arms"][arm]["execution"]["state_exact"]) for report in reports
        )
        programs = sum(
            int(report["arms"][arm]["execution"]["programs"]) for report in reports
        )
        answer_exact = sum(
            int(report["arms"][arm]["execution"]["answer_exact"]) for report in reports
        )
        queries = sum(
            int(report["arms"][arm]["execution"]["queries"]) for report in reports
        )
        arms[arm] = {
            "evidence_exact": evidence_exact,
            "evidence_total": evidence_total,
            "evidence_rate": evidence_exact / evidence_total,
            "state_exact": state_exact,
            "programs": programs,
            "state_rate": state_exact / programs,
            "answer_exact": answer_exact,
            "queries": queries,
            "answer_rate": answer_exact / queries,
        }
    aggregate = {
        "schema": SCHEMA,
        "status": "pass",
        "source_commit": args.source_commit,
        "development": {
            "path": str(args.development),
            "sha256": args.development_sha256,
        },
        "seeds": seeds,
        "reports": [
            {
                "path": str(path),
                "sha256": sha256_path(path),
                "identity_sha256": report["data"]["board_identity_sha256"],
            }
            for path, report in zip(args.confirmation, reports, strict=True)
        ],
        "arms": arms,
        "all_seed_gates_passed": True,
    }
    aggregate["identity_sha256"] = canonical_sha256(aggregate)
    _atomic_json(args.output, aggregate)
    os.chmod(args.output, 0o444)
    print(
        json.dumps(
            {
                "status": "pass",
                "output": str(args.output),
                "output_sha256": sha256_path(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
