#!/usr/bin/env python3
"""Reduce the frozen normal and shuffled CTF1 capability-floor reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


SCHEMA = "shohin-ctf1-capability-floor-aggregate-v1"


class CTF1AggregateError(ValueError):
    """Frozen CTF1 reports differ."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path, control: str) -> dict:
    report = json.loads(path.read_text())
    if (
        report.get("schema") != "shohin-ctf1-capability-floor-evaluation-v1"
        or report.get("status") != "complete"
        or report.get("control") != control
        or report.get("holdout_used") is not False
        or report.get("public_test_opened") is not False
        or report.get("counts", {}).get("rows") != 666
    ):
        raise CTF1AggregateError(f"{control} report differs")
    return report


def run(args: argparse.Namespace) -> dict[str, object]:
    normal = load(args.normal, "normal")
    shuffled = load(args.source_shuffled, "source_shuffled")
    for key in (
        "model_revision",
        "development_data_sha256",
        "lam_checkpoint_sha256",
        "seed",
        "max_new_tokens",
    ):
        if normal.get(key) != shuffled.get(key):
            raise CTF1AggregateError(f"report custody differs: {key}")
    n = normal["counts"]
    s = shuffled["counts"]
    correct = int(n.get("correct", 0))
    linked_correct = int(n.get("linked_correct", 0))
    reset_retained = int(n.get("state_reset_linked_correct", 0))
    state_reset_loss = 1.0 - reset_retained / linked_correct if linked_correct else 0.0
    gates = {
        "at_least_600_compiled_executable": int(n.get("executable_rows", 0)) >= 600,
        "aligned_at_least_300": correct >= 300,
        "aligned_beats_cte1_by_100": correct - 134 >= 100,
        "source_shuffle_at_most_67": int(s.get("correct", 0)) <= 67,
        "at_least_300_linked_rows": int(n.get("linked_rows", 0)) >= 300,
        "state_reset_loss_at_least_20_points": state_reset_loss >= 0.20,
        "opcode_loss_at_least_30_points": (
            correct - int(n.get("opcode_permuted_correct", 0))
        )
        / 666
        >= 0.30,
        "zero_normal_execution_invalid": int(n.get("execution_invalid", 0)) == 0,
        "public_test_closed": True,
    }
    result = {
        "schema": SCHEMA,
        "status": "pass" if all(gates.values()) else "fail",
        "holdout_used": False,
        "public_test_opened": False,
        "custody": {
            key: normal[key]
            for key in (
                "model_revision",
                "development_data_sha256",
                "lam_checkpoint_sha256",
                "seed",
                "max_new_tokens",
            )
        },
        "report_sha256": {
            "normal": sha256_file(args.normal),
            "source_shuffled": sha256_file(args.source_shuffled),
        },
        "scores": {
            "aligned_correct": correct,
            "source_shuffled_correct": int(s.get("correct", 0)),
            "trained_cte1_reference_correct": 134,
            "direct_08b_reference_correct": 267,
            "compiled_rows": int(n.get("compiled_rows", 0)),
            "executable_rows": int(n.get("executable_rows", 0)),
            "linked_rows": int(n.get("linked_rows", 0)),
            "state_reset_loss_on_linked_correct": state_reset_loss,
            "opcode_permuted_correct": int(n.get("opcode_permuted_correct", 0)),
            "normal_execution_invalid": int(n.get("execution_invalid", 0)),
            "normal_exhausted": int(normal.get("exhausted", 0)),
        },
        "gates": gates,
    }
    if args.output.exists():
        raise CTF1AggregateError("refusing existing aggregate")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--source-shuffled", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
