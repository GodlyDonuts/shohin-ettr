#!/usr/bin/env python3
"""Wrap NTA1 transactions into full source documents for DIVERGE-NTA3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing existing NTA3 board or report")
    if sha256_path(args.input) != args.input_sha256:
        raise SystemExit("NTA3 input hash differs")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line]
    if len(rows) != 279:
        raise SystemExit("NTA3 input row count differs")
    output = []
    for row in rows:
        steps = list(map(str, row["wrong_steps"]))
        document = (
            f"Problem:\n{row['question']}\n\n"
            f"Candidate reasoning:\n{' ; '.join(steps)}"
            f"\n\nCandidate final answer: {row['wrong_answer']}"
        )
        result = {
            key: value
            for key, value in row.items()
            if key not in {"wrong_steps", "correct_steps", "question"}
        }
        result["schema"] = "shohin-diverge-nta3-board-v1"
        result["document"] = document
        result["transaction_sha256s"] = [
            hashlib.sha256(step.encode()).hexdigest() for step in steps
        ]
        output.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, args.output)
    report = {
        "schema": "shohin-diverge-nta3-board-report-v1",
        "input": str(args.input),
        "input_sha256": args.input_sha256,
        "output": str(args.output),
        "output_sha256": sha256_path(args.output),
        "rows": len(output),
        "transactions": sum(int(row["depth"]) for row in output),
        "runtime_step_lists_present": any(
            "wrong_steps" in row or "correct_steps" in row for row in output
        ),
    }
    report_tmp = args.report.with_suffix(args.report.suffix + ".tmp")
    report_tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(report_tmp, args.report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
