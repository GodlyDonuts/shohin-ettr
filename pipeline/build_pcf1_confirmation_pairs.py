#!/usr/bin/env python3
"""Materialize label-free PCF1 confirmation candidates after calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from build_pcf1_commit_pairs import load_arm, load_lines, sha256_file
from hf_pcf1_evaluate import EVAL_SCHEMA

PAIR_SCHEMA = "shohin-pcf1-confirmation-pair-v1"
REPORT_SCHEMA = "shohin-pcf1-confirmation-pair-report-v1"


class PCF1ConfirmationError(RuntimeError):
    """The one-shot PCF1 confirmation candidate custody differs."""


def reject_protected_path(path: Path) -> None:
    rendered = f"{path}\n{path.resolve(strict=False)}".casefold()
    if any(word in rendered for word in ("holdout", "product", "public")):
        raise PCF1ConfirmationError(f"protected path supplied to PCF1: {path}")


def load_confirmation(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in load_lines(path):
        identity = row.get("identity_sha256")
        if (
            row.get("schema") != EVAL_SCHEMA
            or row.get("split") != "confirmation"
            or not isinstance(identity, str)
            or len(identity) != 64
            or identity in rows
            or not isinstance(row.get("question"), str)
            or not row["question"].strip()
            or not isinstance(row.get("source_prompt"), str)
            or not row["source_prompt"].strip()
            or row.get("runtime_fields") != ["question", "source_prompt"]
            or any(
                field in row
                for field in (
                    "assessor",
                    "answer",
                    "correct",
                    "gold",
                    "response",
                    "target",
                )
            )
        ):
            raise PCF1ConfirmationError("PCF1 confirmation data differs")
        rows[identity] = row
    if len(rows) != 1289:
        raise PCF1ConfirmationError("PCF1 confirmation cardinality differs")
    return rows


def atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise PCF1ConfirmationError(f"refusing existing PCF1 output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise PCF1ConfirmationError(f"refusing existing PCF1 output: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise PCF1ConfirmationError(f"refusing existing PCF1 output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise PCF1ConfirmationError(f"refusing existing PCF1 output: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def build(args: argparse.Namespace) -> dict[str, Any]:
    for path in (
        args.confirmation_data,
        args.revision_report,
        args.revision_candidates,
        args.unchanged_report,
        args.unchanged_candidates,
        args.candidates_root,
        args.output,
        args.report,
    ):
        reject_protected_path(path)
    if any(path.exists() or path.is_symlink() for path in (args.output, args.report)):
        raise PCF1ConfirmationError("PCF1 confirmation pair output already exists")
    data = load_confirmation(args.confirmation_data)
    revision, revision_receipt = load_arm(
        args.revision_report,
        args.revision_candidates,
        "revision",
        "confirmation",
        candidates_root=args.candidates_root,
    )
    unchanged, unchanged_receipt = load_arm(
        args.unchanged_report,
        args.unchanged_candidates,
        "unchanged",
        "confirmation",
        candidates_root=args.candidates_root,
    )
    data_sha256 = sha256_file(args.confirmation_data)
    if any(
        Path(str(receipt.get("data", ""))).resolve() != args.confirmation_data.resolve()
        or receipt.get("data_sha256") != data_sha256
        for receipt in (revision_receipt, unchanged_receipt)
    ):
        raise PCF1ConfirmationError("PCF1 confirmation arm/data binding differs")
    if set(data) != set(revision) or set(data) != set(unchanged):
        raise PCF1ConfirmationError("PCF1 confirmation identity coverage differs")
    rows: list[dict[str, Any]] = []
    for identity in sorted(data):
        source, left, right = data[identity], revision[identity], unchanged[identity]
        if left.get("task") != source.get("task") or right.get("task") != source.get(
            "task"
        ):
            raise PCF1ConfirmationError("PCF1 confirmation task binding differs")
        rows.append(
            {
                "schema": PAIR_SCHEMA,
                "identity_sha256": identity,
                "split": "confirmation",
                "task": source["task"],
                "question": source["question"],
                "candidates": [
                    {"lineage": "revision", "completion": left["completion"]},
                    {"lineage": "unchanged", "completion": right["completion"]},
                ],
            }
        )
    output_sha256 = atomic_lines(args.output, rows)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "output": str(args.output.resolve()),
        "output_sha256": output_sha256,
        "rows": len(rows),
        "labels_or_correctness_fields": 0,
        "source_disjoint_from_calibration": True,
        "inputs": {
            "data": str(args.confirmation_data.resolve()),
            "data_sha256": data_sha256,
            "revision_report": str(args.revision_report.resolve()),
            "revision_report_sha256": sha256_file(args.revision_report),
            "revision_candidates_sha256": revision_receipt["candidates_sha256"],
            "unchanged_report": str(args.unchanged_report.resolve()),
            "unchanged_report_sha256": sha256_file(args.unchanged_report),
            "unchanged_candidates_sha256": unchanged_receipt["candidates_sha256"],
        },
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    atomic_json(args.report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-data", type=Path, required=True)
    parser.add_argument("--revision-report", type=Path, required=True)
    parser.add_argument("--revision-candidates", type=Path, required=True)
    parser.add_argument("--unchanged-report", type=Path, required=True)
    parser.add_argument("--unchanged-candidates", type=Path, required=True)
    parser.add_argument("--candidates-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = build(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
