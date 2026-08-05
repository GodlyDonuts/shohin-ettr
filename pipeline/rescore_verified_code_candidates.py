#!/usr/bin/env python3
"""Re-execute sampled code after an evaluator program-assembly correction."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from hf_product_reasoning_eval import TASKS, _row_identity
from hf_product_reasoning_rollouts import score_completion


SCHEMA = "shohin-verified-code-candidate-rescore-v1"


class VerifiedCodeRescoreError(RuntimeError):
    """Candidate and bank rows cannot support exact code rescoring."""


def rescore(
    candidates: list[dict[str, Any]],
    bank_rows: list[dict[str, Any]],
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bank = {
        str(row.get("identity_sha256") or _row_identity(str(row["task"]), row)): row
        for row in bank_rows
    }
    output: list[dict[str, Any]] = []
    transitions: Counter[str] = Counter()
    tasks: Counter[str] = Counter()
    identities: set[str] = set()
    for candidate in candidates:
        identity = str(candidate.get("identity_sha256") or "")
        task = str(candidate.get("task") or "")
        if identity not in bank:
            raise VerifiedCodeRescoreError("candidate identity is absent from bank")
        if task not in TASKS or TASKS[task]["kind"] != "code":
            raise VerifiedCodeRescoreError("candidate task is not code-scored")
        row = bank[identity]
        if str(row["task"]) != task:
            raise VerifiedCodeRescoreError("candidate and bank tasks differ")
        score = score_completion(
            row, str(candidate.get("completion") or ""), timeout_seconds
        )
        updated = dict(candidate)
        previous = bool(updated.get("correct"))
        updated.update(score)
        updated["rescore_schema"] = SCHEMA
        output.append(updated)
        transitions[f"{int(previous)}->{int(bool(score['correct']))}"] += 1
        tasks[task] += 1
        identities.add(identity)
    if not output:
        raise VerifiedCodeRescoreError("candidate source is empty")
    return output, {
        "schema": SCHEMA,
        "rows": len(output),
        "identities": len(identities),
        "task_counts": dict(sorted(tasks.items())),
        "label_transitions": dict(sorted(transitions.items())),
        "timeout_seconds": timeout_seconds,
    }


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise VerifiedCodeRescoreError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for row in rows:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise VerifiedCodeRescoreError(f"refusing existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=3.0)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("timeout must be positive")
    candidate_bytes = args.candidates.read_bytes()
    bank_bytes = args.bank.read_bytes()
    candidates = [json.loads(line) for line in candidate_bytes.splitlines() if line]
    rows = [json.loads(line) for line in bank_bytes.splitlines() if line]
    rescored, report = rescore(candidates, rows, args.timeout_seconds)
    report.update(
        {
            "source": str(args.candidates.resolve()),
            "source_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
            "bank": str(args.bank.resolve()),
            "bank_sha256": hashlib.sha256(bank_bytes).hexdigest(),
        }
    )
    report["output_sha256"] = _atomic_lines(args.output, rescored)
    report["output"] = str(args.output.resolve())
    _atomic_json(args.report, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
