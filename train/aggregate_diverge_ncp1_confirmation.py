#!/usr/bin/env python3
"""Aggregate the five fixed DIVERGE-NCP1 confirmation reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from diverge_ncp1_data import CONFIRMATION_SEEDS


SCHEMA = "shohin-diverge-ncp1-confirmation-aggregate-v1"
PROGRAM_ARMS = (
    "treatment",
    "renamed",
    "reverse",
    "source_scrub",
    "shuffled_table",
    "shuffled_table_model",
)
EXECUTION_ARMS = (
    "treatment",
    "renamed",
    "source_scrub",
    "shuffled_table",
    "shuffled_table_model",
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _rate(exact: int, total: int) -> float:
    return exact / max(1, total)


def _sum(reports: list[dict[str, Any]], *keys: str) -> int:
    total = 0
    for report in reports:
        value: Any = report
        for key in keys:
            value = value[key]
        total += int(value)
    return total


def _score_programs(
    reports: list[dict[str, Any]], arm: str
) -> dict[str, int | float]:
    exact = _sum(reports, "program", arm, "exact")
    total = _sum(reports, "program", arm, "total")
    token_exact = _sum(reports, "program", arm, "token_exact")
    tokens = _sum(reports, "program", arm, "tokens")
    return {
        "exact": exact,
        "total": total,
        "rate": _rate(exact, total),
        "token_exact": token_exact,
        "tokens": tokens,
        "token_rate": _rate(token_exact, tokens),
    }


def _score_execution(
    reports: list[dict[str, Any]], arm: str
) -> dict[str, int | float]:
    state_exact = _sum(reports, "execution", arm, "state_exact")
    programs = _sum(reports, "execution", arm, "programs")
    query_exact = _sum(reports, "execution", arm, "query_exact")
    queries = _sum(reports, "execution", arm, "queries")
    return {
        "state_exact": state_exact,
        "programs": programs,
        "state_rate": _rate(state_exact, programs),
        "query_exact": query_exact,
        "queries": queries,
        "query_rate": _rate(query_exact, queries),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        action="append",
        nargs=3,
        metavar=("SEED", "PATH", "SHA256"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing existing NCP1 confirmation aggregate")

    entries = sorted(
        ((int(seed), Path(path), digest) for seed, path, digest in args.result),
        key=lambda value: value[0],
    )
    if tuple(seed for seed, _, _ in entries) != CONFIRMATION_SEEDS:
        raise SystemExit("NCP1 confirmation seed set differs")

    reports: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    model_custody: dict[str, Any] | None = None
    parent_custody: dict[str, Any] | None = None
    for seed, path, expected in entries:
        actual = sha256_path(path)
        if actual != expected:
            raise SystemExit("NCP1 confirmation result hash differs")
        report = json.loads(path.read_text())
        if report.get("schema") != "shohin-diverge-ncp1-evaluation-v1":
            raise SystemExit("NCP1 confirmation schema differs")
        if report.get("source_commit") != args.source_commit:
            raise SystemExit("NCP1 confirmation source commit differs")
        if report.get("status") != "pass" or not all(
            report["gate"]["conditions"].values()
        ):
            raise SystemExit("NCP1 confirmation result does not pass")
        if f"confirmation_{seed}_public.jsonl" not in report["data"]["public"]:
            raise SystemExit("NCP1 confirmation seed/path binding differs")
        if model_custody is None:
            model_custody = report["training"]
            parent_custody = report["parent_eal2"]
        elif (
            report["training"] != model_custody
            or report["parent_eal2"] != parent_custody
        ):
            raise SystemExit("NCP1 confirmation model custody differs")
        reports.append(report)
        receipts.append({"seed": seed, "path": str(path), "sha256": actual})

    reader_exact = _sum(reports, "reader", "complete_exact")
    reader_total = _sum(reports, "reader", "total")
    law_commits = sum(int(report["law_commits"]) for report in reports)
    programs = {
        arm: _score_programs(reports, arm)
        for arm in PROGRAM_ARMS
    }
    execution = {
        arm: _score_execution(reports, arm)
        for arm in EXECUTION_ARMS
    }
    conditions = {
        "all_five_pass": len(reports) == 5,
        "all_seed_gates_pass": all(report["gate"]["passed"] for report in reports),
        "reader_exact": reader_exact == reader_total,
        "all_laws_commit": law_commits == 5 * 256,
        "normal_program_exact": programs["treatment"]["rate"] == 1.0,
        "renamed_program_exact": programs["renamed"]["rate"] == 1.0,
        "reverse_program_exact": programs["reverse"]["rate"] == 1.0,
        "normal_state_exact": execution["treatment"]["state_rate"] == 1.0,
        "normal_query_exact": execution["treatment"]["query_rate"] == 1.0,
        "renamed_state_exact": execution["renamed"]["state_rate"] == 1.0,
        "source_scrub_program_zero": programs["source_scrub"]["exact"] == 0,
        "shuffled_table_program_zero": programs["shuffled_table"]["exact"] == 0,
        "control_model_program_zero": programs["shuffled_table_model"]["exact"]
        == 0,
    }
    report = {
        "schema": SCHEMA,
        "status": "pass" if all(conditions.values()) else "fail",
        "source_commit": args.source_commit,
        "parent_eal2": parent_custody,
        "training": model_custody,
        "receipts": receipts,
        "aggregate": {
            "reader": {
                "exact": reader_exact,
                "total": reader_total,
                "rate": _rate(reader_exact, reader_total),
            },
            "law_commits": law_commits,
            "program": programs,
            "execution": execution,
        },
        "gate": {"conditions": conditions, "passed": all(conditions.values())},
    }
    _atomic_json(args.output, report)
    os.chmod(args.output, 0o444)
    print(
        json.dumps(
            {
                "status": report["status"],
                "program": programs["treatment"]["rate"],
                "renamed": programs["renamed"]["rate"],
                "reverse": programs["reverse"]["rate"],
                "state": execution["treatment"]["state_rate"],
                "query": execution["treatment"]["query_rate"],
                "control": programs["shuffled_table_model"]["rate"],
                "output": str(args.output),
                "output_sha256": sha256_path(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
