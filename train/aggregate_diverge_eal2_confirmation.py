#!/usr/bin/env python3
"""Aggregate five fixed DIVERGE-EAL2 confirmation reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from diverge_eal2_data import CONFIRMATION_SEEDS


SCHEMA = "shohin-diverge-eal2-confirmation-aggregate-v1"


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


def _sum(reports: list[dict[str, Any]], path: tuple[str, ...]) -> int:
    return sum(int(_nested(report, path)) for report in reports)


def _nested(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        value = value[key]
    return value


def _rate(exact: int, total: int) -> float:
    return exact / max(1, total)


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
        raise SystemExit("refusing existing EAL2 confirmation aggregate")
    entries = sorted(
        ((int(seed), Path(path), digest) for seed, path, digest in args.result),
        key=lambda value: value[0],
    )
    if tuple(seed for seed, _, _ in entries) != CONFIRMATION_SEEDS:
        raise SystemExit("EAL2 confirmation seed set differs")
    reports = []
    receipts = []
    checkpoint_sha256 = None
    reader_state_sha256 = None
    for seed, path, expected in entries:
        actual = sha256_path(path)
        if actual != expected:
            raise SystemExit("EAL2 confirmation result hash differs")
        report = json.loads(path.read_text())
        if report.get("status") != "pass" or not all(
            report["gate"]["conditions"].values()
        ):
            raise SystemExit("EAL2 confirmation result does not pass")
        if int(report["data"]["public"].split("seed_")[-1].split("_")[0]) != seed:
            raise SystemExit("EAL2 confirmation seed/path binding differs")
        if report.get("source_commit") != args.source_commit:
            raise SystemExit("EAL2 confirmation source commit differs")
        if checkpoint_sha256 is None:
            checkpoint_sha256 = report["checkpoint_sha256"]
            reader_state_sha256 = report["reader_state_sha256"]
        elif (
            report["checkpoint_sha256"] != checkpoint_sha256
            or report["reader_state_sha256"] != reader_state_sha256
        ):
            raise SystemExit("EAL2 confirmation model custody differs")
        reports.append(report)
        receipts.append({"seed": seed, "path": str(path), "sha256": actual})

    normal_exact = _sum(reports, ("reader", "normal", "complete_exact"))
    normal_total = _sum(reports, ("reader", "normal", "total"))
    counterfactual_exact = _sum(reports, ("reader", "counterfactual", "complete_exact"))
    counterfactual_total = _sum(reports, ("reader", "counterfactual", "total"))
    scrub_exact = _sum(reports, ("reader", "temporal_scrub", "complete_exact"))
    scrub_total = _sum(reports, ("reader", "temporal_scrub", "total"))
    state_exact = _sum(reports, ("execution", "learned", "state_exact"))
    programs = _sum(reports, ("execution", "learned", "programs"))
    query_exact = _sum(reports, ("execution", "learned", "query_exact"))
    queries = _sum(reports, ("execution", "learned", "queries"))
    shuffled_state = _sum(
        reports, ("execution", "shuffled_episode_evidence", "state_exact")
    )
    shuffled_query = _sum(
        reports, ("execution", "shuffled_episode_evidence", "query_exact")
    )
    transplant_state = _sum(
        reports, ("execution", "unrelated_law_transplant", "state_exact")
    )
    transplant_query = _sum(
        reports, ("execution", "unrelated_law_transplant", "query_exact")
    )
    aggregate = {
        "normal": {
            "exact": normal_exact,
            "total": normal_total,
            "rate": _rate(normal_exact, normal_total),
        },
        "counterfactual": {
            "exact": counterfactual_exact,
            "total": counterfactual_total,
            "rate": _rate(counterfactual_exact, counterfactual_total),
        },
        "temporal_scrub": {
            "exact": scrub_exact,
            "total": scrub_total,
            "rate": _rate(scrub_exact, scrub_total),
        },
        "learned_execution": {
            "state_exact": state_exact,
            "programs": programs,
            "state_rate": _rate(state_exact, programs),
            "query_exact": query_exact,
            "queries": queries,
            "query_rate": _rate(query_exact, queries),
        },
        "controls": {
            "shuffled_state_exact": shuffled_state,
            "shuffled_query_exact": shuffled_query,
            "transplant_state_exact": transplant_state,
            "transplant_query_exact": transplant_query,
        },
    }
    conditions = {
        "all_five_pass": len(reports) == 5,
        "normal_exact": aggregate["normal"]["rate"] == 1.0,
        "counterfactual_exact": aggregate["counterfactual"]["rate"] == 1.0,
        "scrub_at_most_30_percent": aggregate["temporal_scrub"]["rate"] <= 0.30,
        "state_exact": aggregate["learned_execution"]["state_rate"] == 1.0,
        "query_exact": aggregate["learned_execution"]["query_rate"] == 1.0,
        "all_seed_gates_pass": all(report["gate"]["passed"] for report in reports),
    }
    report = {
        "schema": SCHEMA,
        "status": "pass" if all(conditions.values()) else "fail",
        "source_commit": args.source_commit,
        "checkpoint_sha256": checkpoint_sha256,
        "reader_state_sha256": reader_state_sha256,
        "receipts": receipts,
        "aggregate": aggregate,
        "gate": {"conditions": conditions, "passed": all(conditions.values())},
    }
    _atomic_json(args.output, report)
    os.chmod(args.output, 0o444)
    print(
        json.dumps(
            {
                "status": report["status"],
                "normal": aggregate["normal"]["rate"],
                "counterfactual": aggregate["counterfactual"]["rate"],
                "scrub": aggregate["temporal_scrub"]["rate"],
                "state": aggregate["learned_execution"]["state_rate"],
                "query": aggregate["learned_execution"]["query_rate"],
                "output": str(args.output),
                "output_sha256": sha256_path(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
