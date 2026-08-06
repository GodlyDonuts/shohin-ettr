#!/usr/bin/env python3
"""Build the frozen source-disjoint DIVERGE-NFE1 train and confirmation data."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from diverge_nfe1_data import (
    SCALAR_OPERATIONS,
    complete_verified_chain,
    corrupt_visible_operator,
    training_records,
    validate_board_row,
    validate_training_record,
)


INPUT_SHA256 = "2461d6f70b44a142854d56c24e1fb42d600065e5788a2c4e055ba47b12696549"


class NFE1BuildError(RuntimeError):
    """The frozen NFE1 board cannot be materialized exactly."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_corpus(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise NFE1BuildError("NFE1 source corpus hash differs")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def _source_identity(source_sha256: str, row: Mapping[str, Any]) -> str:
    payload = (
        source_sha256
        + "\0augmented_gsm8k\0"
        + str(row.get("question") or "")
        + "\0"
        + str(row.get("response") or "")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _confirmation_record(
    row: Mapping[str, Any],
    *,
    source_sha256: str,
    training_texts: set[str],
) -> dict[str, Any] | None:
    if row.get("source") != "augmented_gsm8k":
        return None
    equations = complete_verified_chain(row, training_texts=training_texts)
    if equations is None:
        return None
    identity = _source_identity(source_sha256, row)
    steps: list[dict[str, Any]] = []
    for index, equation in enumerate(equations):
        corrupted = corrupt_visible_operator(equation)
        step = {
            "step_index": index,
            "source_text": corrupted.text,
            "source_sha256": hashlib.sha256(corrupted.text.encode("ascii")).hexdigest(),
            "visible_operator": corrupted.operator,
            "lhs": equation.lhs,
            "argument": equation.argument,
            "rhs": equation.rhs,
            "gold_operation": equation.operation,
            "gold_source_sha256": equation.exact_identity,
            "mention_spans": [list(span) for span in corrupted.mention_spans],
        }
        steps.append(step)
    record = {
        "schema": "shohin-diverge-nfe1-board-v1",
        "split": "confirmation",
        "identity_sha256": identity,
        "source_corpus_sha256": source_sha256,
        "source_dataset": "augmented_gsm8k",
        "source_row_sha256": _canonical_sha256(row),
        "question_sha256": hashlib.sha256(
            str(row.get("question") or "").encode()
        ).hexdigest(),
        "depth": len(steps),
        "answer": int(str(row["answer"])),
        "steps": steps,
        "query": "Return the terminal scalar.",
        "selection": {
            "model_score_used": False,
            "complete_verified_chain": True,
            "exact_training_equation_overlap": False,
            "all_three_outcomes_distinct": True,
        },
    }
    validate_board_row(record)
    return record


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build(
    rows: list[dict[str, Any]],
    *,
    source_sha256: str,
    expected_training: int,
    expected_eligible: int,
    expected_board: int,
    expected_transactions: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    training = training_records(rows)
    if len(training) != expected_training:
        raise NFE1BuildError(
            f"NFE1 training equation count differs: {len(training)} != {expected_training}"
        )
    for record in training:
        validate_training_record(record)
    training_texts = {str(record["text"]) for record in training}
    eligible = [
        record
        for row in rows
        if (
            record := _confirmation_record(
                row,
                source_sha256=source_sha256,
                training_texts=training_texts,
            )
        )
        is not None
    ]
    eligible.sort(key=lambda record: str(record["identity_sha256"]))
    if len(eligible) != expected_eligible:
        raise NFE1BuildError(
            f"NFE1 eligible row count differs: {len(eligible)} != {expected_eligible}"
        )
    deep = [record for record in eligible if int(record["depth"]) >= 3]
    shallow = [record for record in eligible if int(record["depth"]) == 2]
    if len(deep) > expected_board:
        raise NFE1BuildError("NFE1 deep rows exceed board capacity")
    selected = deep + shallow[: expected_board - len(deep)]
    selected.sort(key=lambda record: str(record["identity_sha256"]))
    if len(selected) != expected_board:
        raise NFE1BuildError("NFE1 selected board count differs")
    transactions = sum(int(record["depth"]) for record in selected)
    if transactions != expected_transactions:
        raise NFE1BuildError(
            f"NFE1 transaction count differs: {transactions} != {expected_transactions}"
        )
    report = {
        "schema": "shohin-diverge-nfe1-board-report-v1",
        "source_corpus_sha256": source_sha256,
        "training_rows": len(training),
        "training_operation_counts": dict(
            sorted(Counter(str(record["operation"]) for record in training).items())
        ),
        "eligible_rows": len(eligible),
        "eligible_depth_counts": dict(
            sorted(Counter(str(record["depth"]) for record in eligible).items())
        ),
        "selected_rows": len(selected),
        "selected_transactions": transactions,
        "selected_depth_counts": dict(
            sorted(Counter(str(record["depth"]) for record in selected).items())
        ),
        "selected_operation_counts": dict(
            sorted(
                Counter(
                    str(step["gold_operation"])
                    for record in selected
                    for step in record["steps"]
                ).items()
            )
        ),
        "candidate_operations": list(SCALAR_OPERATIONS),
        "model_score_used_for_selection": False,
        "all_selected_rows_valid": True,
    }
    return training, selected, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", default=INPUT_SHA256)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--board-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-training", type=int, default=2179)
    parser.add_argument("--expected-eligible", type=int, default=109)
    parser.add_argument("--expected-board", type=int, default=96)
    parser.add_argument("--expected-transactions", type=int, default=222)
    args = parser.parse_args()
    for path in (args.training_output, args.board_output, args.report):
        if path.exists():
            raise SystemExit(f"refusing existing NFE1 artifact: {path}")
    rows = _load_corpus(args.input, args.input_sha256)
    training, board, report = build(
        rows,
        source_sha256=args.input_sha256,
        expected_training=args.expected_training,
        expected_eligible=args.expected_eligible,
        expected_board=args.expected_board,
        expected_transactions=args.expected_transactions,
    )
    _atomic_jsonl(args.training_output, training)
    _atomic_jsonl(args.board_output, board)
    report.update(
        {
            "input": str(args.input),
            "training_output": str(args.training_output),
            "training_output_sha256": sha256_path(args.training_output),
            "board_output": str(args.board_output),
            "board_output_sha256": sha256_path(args.board_output),
        }
    )
    _atomic_json(args.report, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
