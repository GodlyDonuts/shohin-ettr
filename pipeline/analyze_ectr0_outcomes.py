#!/usr/bin/env python3
"""Attribute paired ECTR0 outcomes without reopening the closed gate."""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import json
import os
from pathlib import Path
import re
from typing import Any

from aggregate_ectr0_executor_revision import (
    exact_direct_correct,
    load_arm,
    sha256_file,
)


class ECTR0AttributionError(RuntimeError):
    """The immutable ECTR0 evidence is incomplete or inconsistent."""


_DECIMAL = re.compile(r"^-?[0-9]+(?:\.[0-9]+)?$")
_FRACTION = re.compile(r"^(-?[0-9]+)/([0-9]+)$")


def canonical_prediction(value: Any) -> tuple[str, str]:
    text = str(value).strip().replace(",", "")
    fraction = _FRACTION.fullmatch(text)
    try:
        if fraction:
            return ("number", str(Fraction(int(fraction.group(1)), int(fraction.group(2)))))
        if _DECIMAL.fullmatch(text):
            return ("number", str(Fraction(Decimal(text))))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        pass
    return ("text", " ".join(text.casefold().split()))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise ECTR0AttributionError(f"refusing to replace output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _pairwise_counts(
    first: dict[str, dict[str, Any]], second: dict[str, dict[str, Any]]
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for identity in sorted(first):
        first_correct = bool(first[identity]["correct"])
        second_correct = bool(second[identity]["correct"])
        if first_correct and not second_correct:
            counts["first_only_correct"] += 1
        elif second_correct and not first_correct:
            counts["second_only_correct"] += 1
        elif first_correct:
            counts["both_correct"] += 1
        else:
            counts["both_wrong"] += 1
    return dict(sorted(counts.items()))


def run(args: argparse.Namespace) -> dict[str, Any]:
    arms = {
        "aligned": load_arm(args.aligned_report, "aligned"),
        "receipt_absent": load_arm(args.absent_report, "receipt_absent"),
        "receipt_shuffled": load_arm(args.shuffled_report, "receipt_shuffled"),
    }
    immutable = arms["aligned"]["immutable"]
    if any(arm["immutable"] != immutable for arm in arms.values()):
        raise ECTR0AttributionError("cross-arm immutable receipt differs")
    identities = set(arms["aligned"]["details"])
    if len(identities) != 666 or any(set(arm["details"]) != identities for arm in arms.values()):
        raise ECTR0AttributionError("cross-arm identity coverage differs")

    direct_correct = exact_direct_correct(
        args.data,
        args.expected_data_sha256,
        args.ctf_report,
        args.expected_ctf_sha256,
    )
    matrix: Counter[str] = Counter()
    behavior: Counter[str] = Counter()
    aligned = arms["aligned"]["details"]
    absent = arms["receipt_absent"]["details"]
    shuffled = arms["receipt_shuffled"]["details"]

    expected_executor: dict[str, bool] = {}
    for identity in sorted(identities):
        aligned_row = aligned[identity]
        absent_row = absent[identity]
        shuffled_row = shuffled[identity]
        executor_values = {
            bool(aligned_row["executor_correct"]),
            bool(absent_row["executor_correct"]),
            bool(shuffled_row["executor_correct"]),
        }
        if len(executor_values) != 1:
            raise ECTR0AttributionError(f"executor correctness differs for {identity}")
        expected_executor[identity] = executor_values.pop()

        flags = (
            direct_correct[identity],
            expected_executor[identity],
            bool(aligned_row["correct"]),
            bool(absent_row["correct"]),
            bool(shuffled_row["correct"]),
        )
        matrix["".join("1" if flag else "0" for flag in flags)] += 1

        direct_prediction = canonical_prediction(aligned_row["direct_prediction"])
        executor_prediction = canonical_prediction(aligned_row["executor_prediction"])
        aligned_prediction = canonical_prediction(aligned_row["prediction"])
        absent_prediction = canonical_prediction(absent_row["prediction"])
        shuffled_prediction = canonical_prediction(shuffled_row["prediction"])
        behavior["aligned_matches_direct"] += aligned_prediction == direct_prediction
        behavior["aligned_matches_executor"] += aligned_prediction == executor_prediction
        behavior["absent_matches_direct"] += absent_prediction == direct_prediction
        behavior["shuffled_matches_direct"] += shuffled_prediction == direct_prediction
        if direct_prediction != executor_prediction:
            behavior["direct_executor_disagree"] += 1
            if aligned_prediction == direct_prediction:
                behavior["aligned_chooses_direct_when_disagree"] += 1
            elif aligned_prediction == executor_prediction:
                behavior["aligned_chooses_executor_when_disagree"] += 1
            else:
                behavior["aligned_chooses_other_when_disagree"] += 1

    rows = len(identities)
    if sum(matrix.values()) != rows:
        raise ECTR0AttributionError("outcome matrix coverage differs")
    direct_executor_union = sum(
        direct_correct[identity] or expected_executor[identity] for identity in identities
    )
    report = {
        "schema": "shohin-ectr0-attribution-v1",
        "status": "complete",
        "holdout_used": False,
        "public_test_opened": False,
        "rows": rows,
        "inputs": {
            "data": str(args.data.resolve()),
            "data_sha256": sha256_file(args.data),
            "ctf_report": str(args.ctf_report.resolve()),
            "ctf_report_sha256": sha256_file(args.ctf_report),
            "arm_report_sha256": {
                name: arm["report_sha256"] for name, arm in arms.items()
            },
        },
        "correct": {
            "direct": sum(direct_correct.values()),
            "executor": sum(expected_executor.values()),
            "direct_or_executor_oracle": direct_executor_union,
            "aligned": sum(bool(row["correct"]) for row in aligned.values()),
            "receipt_absent": sum(bool(row["correct"]) for row in absent.values()),
            "receipt_shuffled": sum(bool(row["correct"]) for row in shuffled.values()),
        },
        "outcome_matrix": [
            {
                "direct_correct": key[0] == "1",
                "executor_correct": key[1] == "1",
                "aligned_correct": key[2] == "1",
                "absent_correct": key[3] == "1",
                "shuffled_correct": key[4] == "1",
                "rows": count,
            }
            for key, count in sorted(matrix.items(), key=lambda item: (-item[1], item[0]))
        ],
        "paired": {
            "aligned_vs_absent": _pairwise_counts(aligned, absent),
            "aligned_vs_shuffled": _pairwise_counts(aligned, shuffled),
        },
        "prediction_behavior": dict(sorted(behavior.items())),
        "interpretation_boundary": (
            "Read-only attribution only. It may localize copying and arbitration behavior, "
            "but cannot rescue or reinterpret the closed ECTR0 capability gate."
        ),
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-report", type=Path, action="append", required=True)
    parser.add_argument("--absent-report", type=Path, action="append", required=True)
    parser.add_argument("--shuffled-report", type=Path, action="append", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--ctf-report", type=Path, required=True)
    parser.add_argument("--expected-ctf-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
