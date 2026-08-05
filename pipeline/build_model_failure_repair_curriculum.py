#!/usr/bin/env python3
"""Turn executed model failures on a training board into repair trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-model-failure-repair-curriculum-v1"
SYSTEM_TASK = (
    "Write Python code that solves the task and passes every test. Return only "
    "executable Python code, without Markdown fences."
)


class ModelFailureRepairError(RuntimeError):
    """The model evaluation cannot support a complete repair curriculum."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(row: dict[str, Any]) -> str:
    return hashlib.sha256(f"mbpp\0{row['text']}".encode()).hexdigest()


def _diagnostic(execution: dict[str, Any]) -> str:
    fields = []
    for key in ("returncode", "stdout", "stderr"):
        value = execution.get(key)
        if value not in (None, ""):
            fields.append(f"{key}: {value}")
    return ("\n".join(fields) or "The shown tests did not pass.")[:2000]


def build(
    board_rows: list[dict[str, Any]], eval_report: dict[str, Any]
) -> list[dict[str, Any]]:
    if eval_report.get("status") != "complete" or eval_report.get("task") != "mbpp":
        raise ModelFailureRepairError("evaluation report differs")
    board = {_identity(row): row for row in board_rows}
    results = {
        str(row.get("identity_sha256") or ""): row
        for row in eval_report.get("results") or ()
    }
    if (
        len(board) != len(board_rows)
        or len(results) != len(board)
        or set(results) != set(board)
        or int(eval_report.get("total", -1)) != len(board)
    ):
        raise ModelFailureRepairError("evaluation coverage differs from board")
    rows = []
    for identity in sorted(board):
        result = results[identity]
        if bool(result.get("correct")):
            continue
        source = board[identity]
        completion = str(result.get("completion") or "").strip()
        if not completion:
            completion = "# No executable solution was produced."
        execution = result.get("execution") or {}
        repair = (
            "Repair the previous Python solution. Return only the complete corrected "
            "executable Python code, without Markdown fences or explanation.\n\n"
            f"Original task:\n{source['text']}\n\nPrevious solution:\n{completion}\n\n"
            "Observed result from executing only the public tests shown below:\n"
            f"{_diagnostic(execution)}\n\n"
            "Correct every defect while preserving the requested function interface."
        )
        tests = "\n".join(str(value) for value in source.get("test_list") or ())
        rows.append(
            {
                "question": f"{SYSTEM_TASK}\n\nTask:\n{repair}\n\nTests:\n{tests}",
                "response": str(source["code"]).strip(),
                "source": "mbpp_model_failure_repair",
                "source_identity_sha256": identity,
                "task_id": int(source["task_id"]),
                "previous_completion_sha256": hashlib.sha256(
                    completion.encode()
                ).hexdigest(),
                "failure_execution": execution,
            }
        )
    return rows


def _rows(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_bytes().splitlines() if line]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelFailureRepairError(f"malformed JSONL: {path}") from exc


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise ModelFailureRepairError(f"refusing existing output: {path}")
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    evaluation = json.loads(args.evaluation.read_text())
    if evaluation.get("data_sha256") != _sha256(args.board):
        raise ModelFailureRepairError("evaluation board hash differs")
    rows = build(_rows(args.board), evaluation)
    if not rows:
        raise ModelFailureRepairError("evaluation contains no failures")
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "board": str(args.board.resolve()),
        "board_sha256": _sha256(args.board),
        "evaluation": str(args.evaluation.resolve()),
        "evaluation_sha256": _sha256(args.evaluation),
        "evaluation_total": int(evaluation["total"]),
        "evaluation_correct": int(evaluation["correct"]),
        "repair_rows": len(rows),
        "output": str(args.output.resolve()),
    }
    report["output_sha256"] = _atomic_lines(args.output, rows)
    if args.report.exists():
        raise ModelFailureRepairError(f"refusing existing report: {args.report}")
    args.report.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
