#!/usr/bin/env python3
"""Select a modal answer from autonomous candidates without reading gold."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from hf_product_reasoning_eval import (
    _clean_number,
    _normalize_aime_integer,
    _normalize_math,
    _normalize_short_answer,
)


SCHEMA = "shohin-product-self-consistency-selection-v1"


class SelfConsistencySelectionError(RuntimeError):
    """Candidate rows cannot support deterministic self-consistency."""


def canonical_prediction(task: str, prediction: Any) -> str | None:
    if prediction is None:
        return None
    value = str(prediction)
    if task == "gsm8k":
        cleaned = _clean_number(value)
        if cleaned is None:
            return None
        try:
            if "/" in cleaned:
                numerator, denominator = cleaned.split("/", 1)
                number = Fraction(int(numerator), int(denominator))
            else:
                number = Fraction(Decimal(cleaned))
        except (InvalidOperation, ValueError, ZeroDivisionError):
            return None
        return f"{number.numerator}/{number.denominator}"
    if task == "aime":
        return _normalize_aime_integer(value)
    if task == "math500":
        return _normalize_math(value)
    if task in {"bbh_logic", "gpqa"}:
        return _normalize_short_answer(value)
    raise SelfConsistencySelectionError("candidate task is unsupported")


def choose_modal_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise SelfConsistencySelectionError("candidate group is empty")
    ordered = sorted(candidates, key=lambda row: int(row["sample_index"]))
    task = str(ordered[0]["task"])
    if any(str(row["task"]) != task for row in ordered):
        raise SelfConsistencySelectionError("candidate tasks differ")
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in ordered:
        canonical = canonical_prediction(task, row.get("prediction"))
        if canonical is not None:
            groups.setdefault(canonical, []).append(row)
    if not groups:
        return ordered[0]
    winner = max(groups.values(), key=lambda group: len(group))
    return winner[0]


def select(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in candidates:
        identity = str(row.get("identity_sha256") or "")
        if not identity:
            raise SelfConsistencySelectionError("candidate identity is missing")
        grouped.setdefault(identity, []).append(row)
    results: list[dict[str, Any]] = []
    first_correct = oracle_correct = selected_correct = 0
    for identity, group in grouped.items():
        ordered = sorted(group, key=lambda row: int(row["sample_index"]))
        sample_indices = [int(row["sample_index"]) for row in ordered]
        if sample_indices != list(range(len(ordered))):
            raise SelfConsistencySelectionError("sample indices differ")
        selected = choose_modal_candidate(ordered)
        first = bool(ordered[0]["correct"])
        oracle = any(bool(row["correct"]) for row in ordered)
        picked = bool(selected["correct"])
        first_correct += int(first)
        oracle_correct += int(oracle)
        selected_correct += int(picked)
        results.append(
            {
                "identity_sha256": identity,
                "task": selected["task"],
                "first_correct": first,
                "oracle_correct": oracle,
                "selected_correct": picked,
                "selected_sample_index": int(selected["sample_index"]),
                "selected_prediction": selected.get("prediction"),
                "selected_completion": selected.get("completion"),
            }
        )
    total = len(results)
    if not total:
        raise SelfConsistencySelectionError("candidate source is empty")
    return {
        "schema": SCHEMA,
        "total": total,
        "samples": len(candidates) // total,
        "first_correct": first_correct,
        "first_accuracy": first_correct / total,
        "oracle_correct": oracle_correct,
        "oracle_accuracy": oracle_correct / total,
        "selected_correct": selected_correct,
        "selected_accuracy": selected_correct / total,
        "selector_reads_gold": False,
        "selector": "canonical_answer_mode_first_seen_tie_v1",
        "results": results,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise SelfConsistencySelectionError(f"refusing to replace output: {path}")
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_bytes = args.candidates.read_bytes()
    candidates = [
        json.loads(line) for line in source_bytes.splitlines() if line.strip()
    ]
    report = select(candidates)
    report["candidates"] = str(args.candidates.resolve())
    report["candidates_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    _atomic_json(args.output, report)
    print(
        f"[self-consistency] selected={report['selected_correct']}/"
        f"{report['total']} first={report['first_correct']} "
        f"oracle={report['oracle_correct']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
