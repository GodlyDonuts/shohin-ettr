#!/usr/bin/env python3
"""Read-only structural attribution of the closed FSTC1 skeleton result."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


SCHEMA = "shohin-fstc1-skeleton-attribution-v1"
EVAL_SCHEMA = "shohin-fstc1-skeleton-evaluation-v1"
UNARY_GROUP_RE = re.compile(r"-\s*\(")


class FSTC1AttributionError(ValueError):
    """Raised when closed FSTC1 evidence differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bucket(question: str, family: str) -> list[str]:
    parentheses = question.count("(")
    if parentheses == 0:
        parenthesis_bucket = "parentheses:0"
    elif parentheses <= 2:
        parenthesis_bucket = "parentheses:1-2"
    else:
        parenthesis_bucket = "parentheses:3+"
    unary = bool(UNARY_GROUP_RE.search(question))
    has_mul_div = "*" in question or "/" in question
    has_add_sub = "+" in question or "-" in question
    return [
        f"family:{family}",
        parenthesis_bucket,
        f"unary_group:{str(unary).lower()}",
        f"mixed_precedence:{str(has_mul_div and has_add_sub).lower()}",
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    evaluation = json.loads(args.evaluation.read_text())
    if (
        evaluation.get("schema") != EVAL_SCHEMA
        or evaluation.get("status") != "complete"
        or evaluation.get("holdout_used") is not False
        or evaluation.get("control") != "normal"
        or int(evaluation.get("counts", {}).get("rows", -1)) != 3917
    ):
        raise FSTC1AttributionError("normal evaluation differs")
    rows = {
        row["identity_sha256"]: row
        for row in (
            json.loads(line) for line in args.data.read_text().splitlines() if line.strip()
        )
    }
    if len(rows) != 3917:
        raise FSTC1AttributionError("development population differs")
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    intersections: Counter[str] = Counter()
    seen = set()
    for detail in evaluation["details"]:
        identity = detail["identity_sha256"]
        if identity in seen or identity not in rows:
            raise FSTC1AttributionError("evaluation identity coverage differs")
        seen.add(identity)
        row = rows[identity]
        question = str(row["question"])
        complete = bool(detail["complete_skeleton_exact"])
        for bucket in _bucket(question, str(row["family"])):
            groups[bucket]["rows"] += 1
            groups[bucket]["complete"] += int(complete)
            groups[bucket]["depth_error"] += int(not detail["depth_exact"])
            groups[bucket]["operation_error"] += int(
                not detail["operation_sequence_exact"]
            )
            groups[bucket]["operand_error"] += int(not detail["operand_value_exact"])
        if not complete:
            flags = []
            for metric in (
                "depth_exact",
                "operation_sequence_exact",
                "operand_value_exact",
                "polarity_exact",
            ):
                if not detail[metric]:
                    flags.append(metric.removesuffix("_exact"))
            intersections["+".join(flags) or "other"] += 1
    if len(seen) != 3917:
        raise FSTC1AttributionError("evaluation coverage differs")
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "holdout_used": False,
        "evaluation": str(args.evaluation.resolve()),
        "evaluation_sha256": sha256_file(args.evaluation),
        "data": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "groups": {
            name: {
                **dict(sorted(counts.items())),
                "complete_rate": counts["complete"] / counts["rows"],
            }
            for name, counts in sorted(groups.items())
        },
        "failure_intersections": dict(sorted(intersections.items())),
        "diagnosis_boundary": (
            "FSTC1 strongly solves linear/product structure but misses nested scope, "
            "unary-group binding, and mixed-precedence composition; any successor "
            "must add an explicit model-owned parse stack/tree rather than widen or "
            "extend the fixed-slot recurrence."
        ),
    }
    if args.output.exists():
        raise FSTC1AttributionError("refusing existing output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
