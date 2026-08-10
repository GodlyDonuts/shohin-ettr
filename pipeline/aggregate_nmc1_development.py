#!/usr/bin/env python3
"""Aggregate the frozen NMC1 development gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

SCHEMA = "shohin-nmc1-development-aggregate-v1"


class NMC1AggregateError(ValueError):
    """NMC1 report geometry differs."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path, arm: str, control: str) -> dict[str, object]:
    report = json.loads(path.read_text())
    if (
        report.get("schema") != "shohin-nmc1-development-evaluation-v1"
        or report.get("arm") != arm
        or report.get("control") != control
        or report.get("holdout_used") is not False
        or report.get("public_test_opened") is not False
    ):
        raise NMC1AggregateError("input report differs")
    return report


def rate(counts: dict[str, int], key: str) -> float:
    return counts.get(key, 0) / counts["rows"]


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists():
        raise NMC1AggregateError("refusing existing aggregate")
    program = load(args.program, "program", "normal")
    shuffled = load(args.shuffled, "program", "source_shuffled")
    direct = load(args.direct, "direct", "normal")
    identities = [
        {str(row["identity_sha256"]) for row in report["details"]}
        for report in (program, shuffled, direct)
    ]
    if (
        identities[0] != identities[1]
        or identities[0] != identities[2]
        or len(identities[0]) != 666
    ):
        raise NMC1AggregateError("evaluation identities differ")
    pc = program["counts"]
    sc = shuffled["counts"]
    dc = direct["counts"]
    normal = rate(pc, "normal:correct")
    direct_rate = rate(dc, "answer_correct")
    shuffled_rate = rate(sc, "normal:correct")
    normal_correct_multi_digit = pc.get("normal_correct_multi_digit_rows", 0)
    carry_multi = pc.get("carry_reset:normal_correct_multi_digit_correct", 0) / max(
        normal_correct_multi_digit, 1
    )
    normal_multi = 1.0 if normal_correct_multi_digit else 0.0
    opcode = rate(pc, "opcode_permuted:correct")
    metrics = {
        "program_syntax_rate": rate(pc, "syntax_valid"),
        "program_execution_valid_rate": rate(pc, "normal:valid"),
        "program_answer_rate": normal,
        "program_exact_rate": rate(pc, "program_exact"),
        "direct_answer_rate": direct_rate,
        "source_shuffled_answer_rate": shuffled_rate,
        "carry_reset_multi_digit_rate": carry_multi,
        "normal_multi_digit_rate": normal_multi,
        "opcode_permuted_answer_rate": opcode,
        "normal_correct_multi_digit_rows": normal_correct_multi_digit,
    }
    gates = {
        "syntax_at_least_0p90": metrics["program_syntax_rate"] >= 0.90,
        "execution_valid_at_least_0p85": metrics["program_execution_valid_rate"]
        >= 0.85,
        "answer_at_least_0p60": normal >= 0.60,
        "beats_direct_by_0p03": normal - direct_rate >= 0.03,
        "source_shuffled_at_most_0p10": shuffled_rate <= 0.10,
        "carry_reset_loses_0p10_multi_digit": normal_multi - carry_multi >= 0.10,
        "opcode_permutation_loses_0p30": normal - opcode >= 0.30,
        "zero_exhaustion": program["exhausted"] == 0
        and shuffled["exhausted"] == 0
        and direct["exhausted"] == 0,
    }
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "holdout_used": False,
        "public_test_opened": False,
        "inputs": {
            "program": {
                "path": str(args.program.resolve()),
                "sha256": sha256_file(args.program),
            },
            "shuffled": {
                "path": str(args.shuffled.resolve()),
                "sha256": sha256_file(args.shuffled),
            },
            "direct": {
                "path": str(args.direct.resolve()),
                "sha256": sha256_file(args.direct),
            },
        },
        "rows": 666,
        "metrics": metrics,
        "gates": gates,
        "overall_pass": all(gates.values()),
    }
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--shuffled", type=Path, required=True)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args())
    return 0 if result["overall_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
