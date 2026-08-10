#!/usr/bin/env python3
"""Audit record-level edit locality between CTE1 proposals and gold ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Sequence


SCHEMA = "shohin-ltr1-record-locality-v1"
TRANSACTION = re.compile(r"<<([^<>\n]+)>>")


class LTR1AuditError(ValueError):
    """The immutable CTE1 data or report differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def records(text: str) -> list[str]:
    return [re.sub(r"\s+", "", match) for match in TRANSACTION.findall(text)]


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for index, right_item in enumerate(right, start=1):
            if left_item == right_item:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            current.append(
                min(
                    previous[right_index] + 1,
                    current[-1] + 1,
                    previous[right_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def run(args: argparse.Namespace) -> dict[str, object]:
    if sha256_file(args.data) != args.expected_data_sha256:
        raise LTR1AuditError("development data SHA-256 differs")
    if sha256_file(args.report) != args.expected_report_sha256:
        raise LTR1AuditError("CTE1 report SHA-256 differs")

    data_rows = [json.loads(line) for line in args.data.read_text().splitlines()]
    report = json.loads(args.report.read_text())
    if (
        len(data_rows) != 666
        or report.get("schema") != "shohin-cte1-development-evaluation-v1"
        or report.get("control") != "normal"
        or report.get("holdout_used") is not False
        or report.get("public_test_opened") is not False
        or len(report.get("details", [])) != 666
    ):
        raise LTR1AuditError("CTE1 inputs differ")

    gold = {row["identity_sha256"]: row for row in data_rows}
    details = report["details"]
    identities = [row.get("identity_sha256") for row in details]
    if len(set(identities)) != 666 or set(identities) != set(gold):
        raise LTR1AuditError("identity join differs")

    wrong_copy_fractions: list[float] = []
    wrong_edit_distances: list[int] = []
    counts: Counter[str] = Counter()
    per_row: list[dict[str, object]] = []
    for detail in details:
        identity = detail["identity_sha256"]
        target = gold[identity]
        proposal_records = records(str(detail.get("completion", "")))
        gold_records = records(str(target.get("response", "")))
        if not gold_records:
            raise LTR1AuditError("gold ledger has no record")
        overlap = lcs_length(proposal_records, gold_records)
        distance = edit_distance(proposal_records, gold_records)
        fraction = overlap / len(gold_records)
        correct = bool(detail.get("correct"))
        counts["rows"] += 1
        counts["correct"] += correct
        counts["proposal_with_records"] += bool(proposal_records)
        counts["exact_ledger"] += proposal_records == gold_records
        if not correct:
            counts["wrong_rows"] += 1
            counts["wrong_with_records"] += bool(proposal_records)
            counts["wrong_at_most_two_edits"] += distance <= 2
            wrong_copy_fractions.append(fraction)
            wrong_edit_distances.append(distance)
        per_row.append(
            {
                "identity_sha256": identity,
                "correct": correct,
                "proposal_records": len(proposal_records),
                "gold_records": len(gold_records),
                "lcs_records": overlap,
                "gold_copy_fraction": fraction,
                "record_edit_distance": distance,
            }
        )

    wrong_rows = counts["wrong_rows"]
    mean_copy = sum(wrong_copy_fractions) / wrong_rows
    median_copy = quantile(wrong_copy_fractions, 0.5)
    at_most_two_fraction = counts["wrong_at_most_two_edits"] / wrong_rows
    gates = {
        "all_666_identities_join_once": counts["rows"] == 666,
        "at_least_500_wrong_with_records": counts["wrong_with_records"] >= 500,
        "wrong_mean_gold_copy_at_least_35_percent": mean_copy >= 0.35,
        "wrong_median_gold_copy_at_least_25_percent": median_copy >= 0.25,
        "at_least_half_wrong_at_most_two_edits": at_most_two_fraction >= 0.50,
        "public_test_and_holdout_closed": True,
    }
    result = {
        "schema": SCHEMA,
        "status": "pass" if all(gates.values()) else "fail",
        "holdout_used": False,
        "public_test_opened": False,
        "data_sha256": args.expected_data_sha256,
        "cte1_report_sha256": args.expected_report_sha256,
        "counts": dict(sorted(counts.items())),
        "metrics": {
            "wrong_mean_gold_copy_fraction": mean_copy,
            "wrong_median_gold_copy_fraction": median_copy,
            "wrong_p90_gold_copy_fraction": quantile(wrong_copy_fractions, 0.9),
            "wrong_median_record_edit_distance": quantile(wrong_edit_distances, 0.5),
            "wrong_at_most_two_edits_fraction": at_most_two_fraction,
        },
        "gates": gates,
        "rows": per_row,
    }
    if args.output.exists():
        raise LTR1AuditError("refusing existing output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({key: result[key] for key in ("status", "counts", "metrics", "gates")}, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
