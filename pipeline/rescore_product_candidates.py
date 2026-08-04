#!/usr/bin/env python3
"""Recompute immutable candidate labels with the current answer matcher."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from hf_product_reasoning_eval import TASKS


SCHEMA = "shohin-product-candidate-rescore-v1"


class ProductCandidateRescoreError(RuntimeError):
    """Candidate rows cannot be rescored under the evaluator contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rescore(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    transitions: Counter[str] = Counter()
    tasks: Counter[str] = Counter()
    identities: set[str] = set()
    for row in rows:
        task = str(row.get("task") or "")
        identity = str(row.get("identity_sha256") or "")
        if task not in TASKS or TASKS[task].get("kind") != "answer":
            raise ProductCandidateRescoreError("candidate task is not answer-scored")
        if not identity or "gold" not in row or "prediction" not in row:
            raise ProductCandidateRescoreError("candidate scoring fields are incomplete")
        matcher = TASKS[task]["match"]
        updated = bool(matcher(row.get("prediction"), row.get("gold")))
        previous = bool(row.get("correct"))
        rescored = dict(row)
        rescored["correct"] = updated
        rescored["rescore_schema"] = SCHEMA
        output.append(rescored)
        transitions[f"{int(previous)}->{int(updated)}"] += 1
        tasks[task] += 1
        identities.add(identity)
    if not output:
        raise ProductCandidateRescoreError("candidate source is empty")
    return output, {
        "schema": SCHEMA,
        "rows": len(output),
        "identities": len(identities),
        "task_counts": dict(sorted(tasks.items())),
        "label_transitions": dict(sorted(transitions.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output, args.report):
        if output.exists():
            raise ProductCandidateRescoreError(f"refusing existing output: {output}")
    source_bytes = args.candidates.read_bytes()
    rows = [json.loads(line) for line in source_bytes.splitlines() if line.strip()]
    rescored, report = rescore(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    output_digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for row in rescored:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            output_digest.update(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    report.update(
        {
            "source": str(args.candidates.resolve()),
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "output": str(args.output.resolve()),
            "output_sha256": output_digest.hexdigest(),
        }
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = args.report.with_suffix(args.report.suffix + ".partial")
    with temporary_report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_report, args.report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
