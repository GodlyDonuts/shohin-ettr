#!/usr/bin/env python3
"""Restore the tests for the admitted MBPP train/validation code anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from build_verified_code_repair_curriculum import (
    VerifiedCodeRepairError,
    _jsonl,
    _load_raw,
    _program_result,
    _sha256,
    _test_program,
)


SCHEMA = "shohin-verified-mbpp-anchor-board-v1"


def materialize(
    raw_rows: list[dict[str, Any]],
    anchor_rows: list[dict[str, Any]],
    *,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    anchors = {str(row.get("question") or ""): row for row in anchor_rows}
    raw = {str(row.get("text") or ""): row for row in raw_rows}
    if (
        len(anchors) != len(anchor_rows)
        or len(raw) != len(raw_rows)
        or set(anchors) != set(raw)
    ):
        raise VerifiedCodeRepairError("raw and admitted anchor identities differ")
    output = []
    for question in sorted(anchors, key=lambda value: int(raw[value]["task_id"])):
        source = anchors[question]
        row = raw[question]
        code = str(row.get("code") or "").strip()
        if (
            code != str(source.get("response") or "").strip()
            or str(source.get("source") or "")
            not in {"mbpp_train", "mbpp_validation"}
        ):
            raise VerifiedCodeRepairError("raw MBPP row differs from admitted anchor")
        execution = _program_result(_test_program(code, row), timeout_seconds)
        if not execution["passed"]:
            raise VerifiedCodeRepairError("admitted anchor no longer passes its tests")
        output.append(
            {
                "task": "mbpp",
                "task_id": int(row["task_id"]),
                "text": question,
                "code": code,
                "test_list": [str(value) for value in row.get("test_list") or ()],
                "test_setup_code": str(row.get("test_setup_code") or ""),
                "source": str(source["source"]),
                "reference_execution_sha256": hashlib.sha256(
                    json.dumps(execution, sort_keys=True).encode()
                ).hexdigest(),
            }
        )
    return output


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise VerifiedCodeRepairError(f"refusing existing output: {path}")
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
        raise VerifiedCodeRepairError(f"refusing existing report: {path}")
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--anchor-sha256", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=4.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if _sha256(args.anchor) != args.anchor_sha256:
        raise VerifiedCodeRepairError("anchor hash differs")
    anchors = _jsonl(args.anchor)
    admitted = {str(row["question"]): row for row in anchors}
    raw = [row for row in _load_raw(args.dataset_revision) if str(row["text"]) in admitted]
    rows = materialize(raw, anchors, timeout_seconds=args.timeout_seconds)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "anchor": str(args.anchor.resolve()),
        "anchor_sha256": _sha256(args.anchor),
        "dataset": "google-research-datasets/mbpp:full",
        "dataset_revision": args.dataset_revision,
        "rows": len(rows),
        "reference_execution_passed": len(rows),
        "train_rows": sum(row["source"] == "mbpp_train" for row in rows),
        "validation_rows": sum(row["source"] == "mbpp_validation" for row in rows),
        "output": str(args.output.resolve()),
    }
    report["output_sha256"] = _atomic_lines(args.output, rows)
    _atomic_json(args.report, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
