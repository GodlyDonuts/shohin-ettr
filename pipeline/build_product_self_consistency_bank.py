#!/usr/bin/env python3
"""Freeze the exact held-out rows used by product self-consistency evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from hf_product_reasoning_eval import (
    TASKS,
    _row_identity,
    _task_prompt,
    select_rows,
)


SCHEMA = "shohin-product-self-consistency-bank-v1"


class SelfConsistencyBankError(RuntimeError):
    """The held-out self-consistency bank contract was violated."""


def build_rows(
    task_name: str,
    source_rows: list[dict[str, Any]],
    count: int,
    subset_seed: int,
) -> list[dict[str, Any]]:
    task = TASKS[task_name]
    if task["kind"] != "answer":
        raise SelfConsistencyBankError("self-consistency requires an answer task")
    selected = select_rows(task_name, source_rows, count, subset_seed)
    frozen: list[dict[str, Any]] = []
    for row in selected:
        gold = task["gold"](row)
        if gold is None:
            raise SelfConsistencyBankError("selected row has no scoreable answer")
        if task_name == "gsm8k":
            stored_answer = f"#### {gold}"
        elif task_name == "math500":
            stored_answer = rf"\boxed{{{gold}}}"
        else:
            stored_answer = gold
        frozen.append(
            {
                "schema": SCHEMA,
                "identity_sha256": _row_identity(task_name, row),
                "question": _task_prompt(task_name, row),
                "task": task_name,
                "training_group": "sealed_self_consistency_eval",
                "answer": stored_answer,
                "expected_answer_normalized": gold,
            }
        )
    return frozen


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise SelfConsistencyBankError(f"refusing to replace output: {path}")
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
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--subset-seed", type=int, default=20260802)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data_bytes = args.data.read_bytes()
    source_rows = [
        json.loads(line) for line in data_bytes.splitlines() if line.strip()
    ]
    rows = build_rows(args.task, source_rows, args.count, args.subset_seed)
    output_sha256 = _atomic_lines(args.output, rows)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "task": args.task,
                "source": str(args.data.resolve()),
                "source_sha256": hashlib.sha256(data_bytes).hexdigest(),
                "count": len(rows),
                "subset_seed": args.subset_seed,
                "output": str(args.output.resolve()),
                "output_sha256": output_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
