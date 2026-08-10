#!/usr/bin/env python3
"""Audit whether exact ledgers admit source-pointer fixed-slot supervision."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable


SCHEMA = "shohin-slc1-fixed-slot-geometry-audit-v3"
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])(?:\d+\.\d+|\d+|\.\d+)")


class FixedSlotGeometryError(ValueError):
    """Raised when the immutable ledger or pointer geometry differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise FixedSlotGeometryError(
                    f"invalid JSON at line {line_number}"
                ) from error
            if not isinstance(row, dict):
                raise FixedSlotGeometryError(f"non-object row at line {line_number}")
            yield row


def numeric_spans(question: str) -> list[dict[str, Any]]:
    spans = []
    for match in NUMBER_RE.finditer(question):
        surface = match.group(0)
        try:
            magnitude = Fraction(surface)
        except (ValueError, ZeroDivisionError) as error:
            raise FixedSlotGeometryError("numeric surface is not exact") from error
        spans.append(
            {
                "start": match.start(),
                "end": match.end(),
                "surface": surface,
                "magnitude": magnitude,
            }
        )
    return spans


def _fraction(value: dict[str, Any]) -> Fraction:
    numerator = value.get("numerator")
    denominator = value.get("denominator")
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        raise FixedSlotGeometryError("ledger fraction differs")
    return Fraction(numerator, denominator)


def audit_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    histograms: dict[str, Counter[int]] = {
        "candidate_count": Counter(),
        "depth": Counter(),
        "source_literal_match_count": Counter(),
    }
    maxima = {
        "question_characters": 0,
        "candidate_count": 0,
        "depth": 0,
        "numerator_digits": 0,
        "denominator_digits": 0,
    }
    failures: list[dict[str, Any]] = []
    for row in rows:
        counters["rows"] += 1
        identity = row.get("identity_sha256")
        question = row.get("question")
        records = row.get("records")
        if not isinstance(identity, str) or not isinstance(question, str):
            raise FixedSlotGeometryError("row identity or question differs")
        if not isinstance(records, list) or not 1 <= len(records) <= 5:
            raise FixedSlotGeometryError("record geometry differs")
        spans = numeric_spans(question)
        histograms["candidate_count"][len(spans)] += 1
        histograms["depth"][len(records)] += 1
        maxima["question_characters"] = max(maxima["question_characters"], len(question))
        maxima["candidate_count"] = max(maxima["candidate_count"], len(spans))
        maxima["depth"] = max(maxima["depth"], len(records))

        used: set[int] = set()
        prior_results: list[Fraction] = []
        row_complete = True
        for record_index, record in enumerate(records):
            dependencies = {
                dependency["operand_role"]: dependency["record_index"]
                for dependency in record.get("dependencies", [])
            }
            for role, value in zip(("left", "right"), record["operands"], strict=True):
                fraction = _fraction(value)
                maxima["numerator_digits"] = max(
                    maxima["numerator_digits"], len(str(abs(fraction.numerator)))
                )
                maxima["denominator_digits"] = max(
                    maxima["denominator_digits"], len(str(fraction.denominator))
                )
                if role in dependencies:
                    dependency = dependencies[role]
                    if type(dependency) is not int or not 0 <= dependency < record_index:
                        raise FixedSlotGeometryError("noncausal dependency")
                    counters["dependency_operands"] += 1
                    continue
                matches = [
                    index
                    for index, span in enumerate(spans)
                    if span["magnitude"] == abs(fraction)
                ]
                histograms["source_literal_match_count"][len(matches)] += 1
                counters["source_literal_operands"] += 1
                if not matches:
                    negated_dependencies = [
                        index
                        for index, prior_result in enumerate(prior_results)
                        if fraction == -prior_result
                    ]
                    if negated_dependencies:
                        counters["negated_dependency_operands"] += 1
                    else:
                        counters["unmatched_source_literal_operands"] += 1
                        row_complete = False
                        if len(failures) < 20:
                            failures.append(
                                {
                                    "identity_sha256": identity,
                                    "record": record_index,
                                    "role": role,
                                    "value": str(fraction),
                                    "question": question,
                                }
                            )
                    continue
                unused = [index for index in matches if index not in used]
                chosen = unused[0] if unused else matches[0]
                used.add(chosen)
                counters["reused_pointer_targets"] += int(not unused)
                counters["ambiguous_source_literal_operands"] += int(len(matches) > 1)
            result = _fraction(record["result"])
            maxima["numerator_digits"] = max(
                maxima["numerator_digits"], len(str(abs(result.numerator)))
            )
            maxima["denominator_digits"] = max(
                maxima["denominator_digits"], len(str(result.denominator))
            )
            prior_results.append(result)
        counters["fully_pointer_supervisable_rows"] += int(row_complete)
    return {
        "counts": dict(sorted(counters.items())),
        "rates": {
            "fully_pointer_supervisable": counters["fully_pointer_supervisable_rows"]
            / counters["rows"],
            "unmatched_source_literal_operand": counters[
                "unmatched_source_literal_operands"
            ]
            / max(1, counters["source_literal_operands"]),
            "ambiguous_source_literal_operand": counters[
                "ambiguous_source_literal_operands"
            ]
            / max(1, counters["source_literal_operands"]),
        },
        "histograms": {
            name: {str(key): value for key, value in sorted(histogram.items())}
            for name, histogram in histograms.items()
        },
        "maxima": maxima,
        "failures": failures,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    splits = {}
    for name, path, expected in (
        ("train", args.train, args.expected_train_sha256),
        ("development", args.development, args.expected_development_sha256),
    ):
        digest = sha256_file(path)
        if expected and digest != expected:
            raise FixedSlotGeometryError(f"{name} SHA-256 differs")
        splits[name] = {
            "path": str(path.resolve()),
            "sha256": digest,
            **audit_rows(_iter_jsonl(path)),
        }
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "splits": splits,
        "holdout_used": False,
        "admitted": all(
            split["rates"]["fully_pointer_supervisable"] == 1.0
            for split in splits.values()
        ),
    }
    if args.output.exists():
        raise FixedSlotGeometryError("refusing existing output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--expected-train-sha256")
    parser.add_argument("--expected-development-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
