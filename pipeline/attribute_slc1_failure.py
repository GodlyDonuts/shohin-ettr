#!/usr/bin/env python3
"""Read-only attribution of the closed SLC1 autoregressive compiler failure."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from materialize_structured_ledger_sft import LedgerMaterializationError, parse_ledger


SCHEMA = "shohin-slc1-failure-attribution-v1"


class SLC1AttributionError(ValueError):
    """Raised when closed-result custody differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _value(text: str, states: list[Fraction]) -> Fraction:
    if text.startswith("@R"):
        index = int(text[2:])
        if index < 0 or index >= len(states):
            raise ValueError("reference")
        return states[index]
    return Fraction(text)


def arithmetic_consistent(parsed: dict[str, Any]) -> bool:
    states: list[Fraction] = []
    for record in parsed["records"]:
        try:
            left = _value(record["left"], states)
            right = _value(record["right"], states)
            operation = record["operation"]
            if operation == "ADD":
                result = left + right
            elif operation == "SUB":
                result = left - right
            elif operation == "MUL":
                result = left * right
            elif operation == "DIV":
                result = left / right
            elif operation == "POW" and right.denominator == 1:
                result = left ** right.numerator
            else:
                return False
            stated = Fraction(record["result"])
        except (ValueError, ZeroDivisionError, OverflowError):
            return False
        if result != stated:
            return False
        states.append(stated)
    return bool(states) and Fraction(parsed["commit"]["value"]) == states[-1]


def run(args: argparse.Namespace) -> dict[str, Any]:
    comparison = json.loads(args.comparison.read_text())
    if comparison.get("overall_pass") is not False or comparison.get("holdout_used") is not False:
        raise SLC1AttributionError("comparison is not the closed development failure")
    rows = {
        row["identity_sha256"]: row
        for row in (
            json.loads(line) for line in args.development.read_text().splitlines() if line.strip()
        )
    }
    if len(rows) != 3917:
        raise SLC1AttributionError("development population differs")
    reports = sorted(args.evaluation_root.glob("normal/shard_*.json"))
    if len(reports) != 8:
        raise SLC1AttributionError("normal shard count differs")

    counters: Counter[str] = Counter()
    predicted_depth: Counter[int] = Counter()
    depth_matrix: dict[int, Counter[int]] = defaultdict(Counter)
    syntax_failures: Counter[str] = Counter()
    identities: set[str] = set()
    for path in reports:
        report = json.loads(path.read_text())
        if report.get("control") != "normal" or report.get("holdout_used") is not False:
            raise SLC1AttributionError("evaluation shard differs")
        for detail in report["details"]:
            identity = str(detail["identity_sha256"])
            if identity in identities or identity not in rows:
                raise SLC1AttributionError("identity coverage differs")
            identities.add(identity)
            gold = parse_ledger(str(rows[identity]["response"]))
            gold_depth = len(gold["records"])
            completion = str(detail["completion"])
            rough_depth = sum(
                line.startswith("R") and "|" in line
                for line in completion.strip().splitlines()
            )
            predicted_depth[rough_depth] += 1
            depth_matrix[gold_depth][rough_depth] += 1
            counters["rows"] += 1
            counters["exhausted"] += int(detail["exhausted"])
            try:
                parsed = parse_ledger(completion)
            except LedgerMaterializationError as error:
                syntax_failures[str(error)] += 1
                continue
            counters["syntax_valid"] += 1
            counters["arithmetic_consistent"] += int(arithmetic_consistent(parsed))
            if parsed["records"]:
                counters["first_operation_exact"] += int(
                    parsed["records"][0]["operation"] == gold["records"][0]["operation"]
                )
                counters["first_operands_exact"] += int(
                    parsed["records"][0]["left"] == gold["records"][0]["left"]
                    and parsed["records"][0]["right"] == gold["records"][0]["right"]
                )
                counters["first_result_exact"] += int(
                    parsed["records"][0]["result"] == gold["records"][0]["result"]
                )
    if len(identities) != 3917:
        raise SLC1AttributionError("attribution coverage differs")
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "comparison": str(args.comparison.resolve()),
        "comparison_sha256": sha256_file(args.comparison),
        "development": str(args.development.resolve()),
        "development_sha256": sha256_file(args.development),
        "counts": dict(sorted(counters.items())),
        "rates": {
            key: value / counters["rows"]
            for key, value in sorted(counters.items())
            if key != "rows"
        },
        "rough_predicted_depth": {str(key): value for key, value in sorted(predicted_depth.items())},
        "gold_to_rough_predicted_depth": {
            str(gold_depth): {str(key): value for key, value in sorted(values.items())}
            for gold_depth, values in sorted(depth_matrix.items())
        },
        "syntax_failures": dict(sorted(syntax_failures.items())),
        "diagnosis": (
            "The autoregressive owner learned ledger markers but not reliable program "
            "unfolding or arithmetic state transition; use a fixed-slot recurrent "
            "compiler with typed heads rather than another text-decoder fit."
        ),
        "holdout_used": False,
    }
    if args.output.exists():
        raise SLC1AttributionError("refusing existing output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
