#!/usr/bin/env python3
"""Build one-round code-repair prompts from prompt-visible test failures."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-visible-code-repair-bank-v1"


class VisibleCodeRepairError(RuntimeError):
    """The frozen candidate selection cannot support a visible-only repair bank."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_bytes().splitlines() if line]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VisibleCodeRepairError(f"malformed JSONL: {path}") from exc


def _identity(row: dict[str, Any]) -> str:
    identity = str(row.get("identity_sha256") or "")
    if identity:
        return identity
    task = str(row.get("task") or "")
    question = next(
        (
            str(row[key])
            for key in ("question", "problem", "prompt", "text", "input")
            if row.get(key)
        ),
        "",
    )
    if not task or not question:
        raise VisibleCodeRepairError("row identity cannot be derived")
    return hashlib.sha256(f"{task}\0{question}".encode()).hexdigest()


def _diagnostic(execution: dict[str, Any], limit: int) -> str:
    fields = []
    for key in ("returncode", "timed_out", "stdout", "stderr"):
        value = execution.get(key)
        if value not in (None, "", False):
            fields.append(f"{key}: {value}")
    rendered = "\n".join(fields) or "The shown tests did not pass."
    return rendered[:limit]


def build_repair_rows(
    bank_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    selection: dict[str, Any],
    *,
    diagnostic_chars: int,
) -> list[dict[str, Any]]:
    if diagnostic_chars <= 0:
        raise VisibleCodeRepairError("diagnostic character limit must be positive")
    if selection.get("schema") != "shohin-visible-code-candidate-selection-v1":
        raise VisibleCodeRepairError("selection schema differs")
    bank = {_identity(row): row for row in bank_rows}
    if len(bank) != len(bank_rows):
        raise VisibleCodeRepairError("bank identities repeat")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[_identity(candidate)].append(candidate)
    if set(grouped) != set(bank):
        raise VisibleCodeRepairError("candidate and bank identities differ")
    selected = selection.get("results") or []
    if len(selected) != len(bank) or {_identity(row) for row in selected} != set(bank):
        raise VisibleCodeRepairError("selection and bank identities differ")

    repairs = []
    for decision in selected:
        if bool(decision.get("selected_correct")):
            continue
        identity = _identity(decision)
        task = str(decision.get("task") or "")
        if task != "mbpp":
            raise VisibleCodeRepairError("only MBPP visible-test repair is supported")
        sample_index = int(decision.get("selected_sample_index", -1))
        choices = [
            row for row in grouped[identity] if int(row.get("sample_index", -1)) == sample_index
        ]
        if len(choices) != 1:
            raise VisibleCodeRepairError("selected candidate is missing or repeated")
        candidate = choices[0]
        original = bank[identity]
        execution = candidate.get("execution") or {}
        if bool(execution.get("passed")):
            raise VisibleCodeRepairError("failed selection contains a passing candidate")
        completion = str(candidate.get("completion") or "").strip()
        if not completion:
            completion = "# No executable solution was produced."
        prompt = (
            "Repair the previous Python solution. Return only the complete corrected "
            "executable Python code, without Markdown fences or explanation.\n\n"
            f"Original task:\n{original['text']}\n\n"
            f"Previous solution:\n{completion}\n\n"
            "Observed result from executing only the public tests shown below:\n"
            f"{_diagnostic(execution, diagnostic_chars)}\n\n"
            "Correct every defect while preserving the requested function interface."
        )
        repaired = dict(original)
        repaired.update(
            {
                "text": prompt,
                "original_identity_sha256": identity,
                "selected_sample_index": sample_index,
                "repair_source_execution": execution,
                "repair_schema": SCHEMA,
            }
        )
        repairs.append(repaired)
    return repairs


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise VisibleCodeRepairError(f"refusing existing output: {path}")
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
        raise VisibleCodeRepairError(f"refusing existing report: {path}")
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
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--diagnostic-chars", type=int, default=2000)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    if selection.get("candidates_sha256") != _sha256(args.candidates):
        raise VisibleCodeRepairError("selection candidate hash differs")
    if selection.get("bank_sha256") != _sha256(args.bank):
        raise VisibleCodeRepairError("selection bank hash differs")
    rows = build_repair_rows(
        _rows(args.bank),
        _rows(args.candidates),
        selection,
        diagnostic_chars=args.diagnostic_chars,
    )
    if not rows:
        raise VisibleCodeRepairError("selection has no failed rows to repair")
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "bank": str(args.bank.resolve()),
        "bank_sha256": _sha256(args.bank),
        "candidates": str(args.candidates.resolve()),
        "candidates_sha256": _sha256(args.candidates),
        "selection": str(args.selection.resolve()),
        "selection_sha256": _sha256(args.selection),
        "source_total": int(selection["total"]),
        "source_selected_correct": int(selection["selected_correct"]),
        "repair_rows": len(rows),
        "diagnostic_chars": args.diagnostic_chars,
    }
    report["output"] = str(args.output.resolve())
    report["output_sha256"] = _atomic_lines(args.output, rows)
    _atomic_json(args.report, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
