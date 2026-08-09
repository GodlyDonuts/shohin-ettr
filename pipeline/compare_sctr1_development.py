#!/usr/bin/env python3
"""Apply the frozen SCTR1 development gate to complete matched reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-sctr1-development-comparison-v1"
DOMAINS = ("math500", "bbh_logic", "mbpp")
ROW_COUNT = 1289


class SCTR1ComparisonError(RuntimeError):
    """An SCTR1 report set is incomplete or not causally matched."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != "complete":
        raise SCTR1ComparisonError(f"incomplete SCTR1 report: {path}")
    return value


def accuracy(report: dict[str, Any], domain: str = "overall") -> float:
    metric = report.get("metrics", {}).get(domain)
    if not isinstance(metric, dict) or metric.get("total", 0) <= 0:
        raise SCTR1ComparisonError(f"missing SCTR1 metric: {domain}")
    return int(metric["generated_correct"]) / int(metric["total"])


def correct(report: dict[str, Any], domain: str) -> int:
    metric = report.get("metrics", {}).get(domain)
    if not isinstance(metric, dict):
        raise SCTR1ComparisonError(f"missing SCTR1 domain: {domain}")
    return int(metric["generated_correct"])


def complete_coverage(report: dict[str, Any]) -> bool:
    shard_count = report.get("shard_count")
    return report.get("full_row_count") == ROW_COUNT and (
        shard_count == 1
        or (
            report.get("merged_from_shards") is True
            and isinstance(shard_count, int)
            and shard_count >= 2
        )
    )


def identities(report: dict[str, Any]) -> list[str]:
    path = Path(str(report.get("candidates_output", "")))
    if not path.is_file() or report.get("candidates_sha256") != sha256_file(path):
        raise SCTR1ComparisonError("candidate receipt differs")
    values = [
        json.loads(line).get("identity_sha256")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(values) != ROW_COUNT or len(set(values)) != ROW_COUNT:
        raise SCTR1ComparisonError("candidate identity coverage differs")
    return values


def data_lineage(path: Path) -> dict[str, Any]:
    report = load_report(path)
    banks = report.get("banks")
    if not isinstance(banks, list):
        raise SCTR1ComparisonError("data lineage lacks banks")
    return {
        "pairs_sha256": report.get("pairs_sha256"),
        "drafts_sha256": report.get("drafts_sha256"),
        "draft_receipt_sha256": report.get("draft_receipt_sha256"),
        "split_seed": report.get("split_seed"),
        "banks": [item.get("sha256") for item in banks],
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing existing comparison: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as destination:
        json.dump(payload, destination, indent=2, sort_keys=True)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)


def compare(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "selective_commit": args.selective,
        "always_revise": args.always_revise,
        "unchanged_second_pass": args.unchanged,
        "long_single_generation": args.long,
        "independent_commitment": args.independent,
        "shuffled_commit": args.shuffled,
    }
    reports = {name: load_report(path) for name, path in paths.items()}
    selective = reports["selective_commit"]
    shuffled = reports["shuffled_commit"]
    if selective.get("schema") != "shohin-sctr1-selective-commit-evaluation-v1":
        raise SCTR1ComparisonError("selective report schema differs")
    if shuffled.get("schema") != selective.get("schema"):
        raise SCTR1ComparisonError("shuffled report schema differs")
    if reports["always_revise"].get("schema") != "shohin-idr1-revision-evaluation-v1":
        raise SCTR1ComparisonError("always-revise report schema differs")
    independent = reports["independent_commitment"]
    if (
        independent.get("schema") != selective.get("schema")
        or independent.get("mask_internal_draft") is not True
    ):
        raise SCTR1ComparisonError("independent selective report differs")
    for name in ("unchanged_second_pass", "long_single_generation"):
        report = reports[name]
        if (
            report.get("schema") != "shohin-ttr1-control-evaluation-v1"
            or report.get("control") != name
        ):
            raise SCTR1ComparisonError(f"control report differs: {name}")

    model_revision = selective.get("model_revision")
    reference_ids = identities(selective)
    for name, report in reports.items():
        if (
            report.get("split") != "development"
            or report.get("model_revision") != model_revision
            or not complete_coverage(report)
            or identities(report) != reference_ids
        ):
            raise SCTR1ComparisonError(f"SCTR1 arm is not matched: {name}")
    if data_lineage(args.sctr_data_report) != data_lineage(args.standard_data_report):
        raise SCTR1ComparisonError("selective and standard data lineage differs")

    scores = {name: accuracy(report) for name, report in reports.items()}
    unchanged = reports["unchanged_second_pass"]
    domain_deltas = {
        domain: correct(selective, domain) - correct(unchanged, domain)
        for domain in DOMAINS
    }
    strongest_nontrained = max(
        ("unchanged_second_pass", "long_single_generation"),
        key=lambda name: scores[name],
    )
    malformed = int(selective.get("commitment", {}).get("malformed", -1))
    gates = {
        "selective_beats_unchanged_by_5_points": scores["selective_commit"]
        >= scores["unchanged_second_pass"] + 0.05,
        "selective_at_least_always_revise": scores["selective_commit"]
        >= scores["always_revise"],
        "selective_beats_strongest_nontrained_by_3_points": scores["selective_commit"]
        >= scores[strongest_nontrained] + 0.03,
        "selective_beats_shuffled_by_3_points": scores["selective_commit"]
        >= scores["shuffled_commit"] + 0.03,
        "all_domain_deltas_nonnegative": all(
            delta >= 0 for delta in domain_deltas.values()
        ),
        "zero_malformed_commitments": malformed == 0,
        "complete_identity_coverage": all(complete_coverage(report) for report in reports.values()),
    }
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "split": "development",
        "model_revision": model_revision,
        "arms": {
            name: {
                "path": str(paths[name].resolve()),
                "sha256": sha256_file(paths[name]),
                "accuracy": scores[name],
            }
            for name in paths
        },
        "data_lineage": data_lineage(args.sctr_data_report),
        "strongest_nontrained_control": strongest_nontrained,
        "selective_minus_unchanged_points": 100
        * (scores["selective_commit"] - scores["unchanged_second_pass"]),
        "selective_minus_always_revise_points": 100
        * (scores["selective_commit"] - scores["always_revise"]),
        "selective_minus_strongest_nontrained_points": 100
        * (scores["selective_commit"] - scores[strongest_nontrained]),
        "domain_correct_count_deltas_vs_unchanged": domain_deltas,
        "malformed_commitments": malformed,
        "gates": gates,
        "gate_pass": all(gates.values()),
        "holdout_authorized": all(gates.values()),
    }
    atomic_json(args.output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selective", type=Path, required=True)
    parser.add_argument("--always-revise", type=Path, required=True)
    parser.add_argument("--unchanged", type=Path, required=True)
    parser.add_argument("--long", type=Path, required=True)
    parser.add_argument("--independent", type=Path, required=True)
    parser.add_argument("--shuffled", type=Path, required=True)
    parser.add_argument("--sctr-data-report", type=Path, required=True)
    parser.add_argument("--standard-data-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = compare(parser.parse_args())
    print(json.dumps({"gate_pass": result["gate_pass"], "gates": result["gates"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
