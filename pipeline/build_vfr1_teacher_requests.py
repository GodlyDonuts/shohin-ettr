#!/usr/bin/env python3
"""Build unique train-only requests for verified fault-first revision traces."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from build_idr1_revision_data import REPORT_SCHEMA as IDR1_REPORT_SCHEMA
from build_idr1_revision_data import TRAIN_SCHEMA as IDR1_TRAIN_SCHEMA
from build_vcr1_revision_data import load_source_banks, sha256_file


REQUEST_SCHEMA = "shohin-vfr1-teacher-request-v1"
REPORT_SCHEMA = "shohin-vfr1-teacher-request-report-v1"
EXPECTED_IDENTITIES = 5824


class VFR1RequestError(RuntimeError):
    """VFR1 source data or request geometry differs."""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise VFR1RequestError(f"empty VFR1 source: {path}")
    return rows


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise VFR1RequestError(f"refusing existing VFR1 output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
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
        raise VFR1RequestError(f"refusing existing VFR1 report: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def teacher_prompt(question: str, reference: str, task: str) -> str:
    format_rule = (
        "Inside <REVISION>, write only executable Python code without Markdown fences."
        if task == "mbpp"
        else (
            "Inside <REVISION>, write a complete derivation and end with exactly one "
            "final answer inside \\boxed{}."
        )
    )
    return (
        "Create a verified fault-first correction trace for training a reasoning "
        "model. The first block must identify the earliest decisive error or missing "
        "justification in the internal draft. If the draft is sound, say that it was "
        "independently checked. Do not quote a boxed answer or executable code in the "
        "fault block. The second block must solve the problem completely. Treat the "
        "verified reference as a correctness constraint, never mention it as an "
        "authority, and independently derive the result. Output exactly two blocks:\n"
        "<FAULT>\nconcise diagnosis\n</FAULT>\n"
        "<REVISION>\ncomplete corrected solution\n</REVISION>\n\n"
        f"{format_rule}\n\nSOURCE AND INTERNAL DRAFT:\n{question}\n\n"
        f"VERIFIED REFERENCE OUTCOME OR SOLUTION:\n{reference}"
    )


def collect_unique_requests(
    train_rows: list[dict[str, Any]], sources: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in train_rows:
        if row.get("schema") != IDR1_TRAIN_SCHEMA:
            raise VFR1RequestError("IDR1 train schema differs")
        identity = str(row.get("source_identity_sha256", ""))
        if identity not in sources:
            raise VFR1RequestError("IDR1 train identity is absent from source banks")
        source = sources[identity]
        task = str(source.get("task", ""))
        candidate = {
            "schema": REQUEST_SCHEMA,
            "identity_sha256": identity,
            "task": task,
            "outcome_class": row.get("outcome_class"),
            "target_kind": row.get("target_kind"),
            "teacher_prompt": teacher_prompt(
                str(row.get("question", "")), str(row.get("response", "")), task
            ),
            "reference_solution": str(row.get("response", "")),
            "assessor": source,
            "runtime_fields": ["teacher_prompt"],
        }
        previous = grouped.get(identity)
        if previous is not None and previous != candidate:
            raise VFR1RequestError("duplicate IDR1 presentations disagree")
        grouped[identity] = candidate
    if not set(grouped).issubset(sources):
        raise VFR1RequestError("VFR1 train identities escape the source banks")
    return [grouped[identity] for identity in sorted(grouped)]


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.report.exists():
        raise VFR1RequestError("VFR1 request output already exists")
    idr_report = json.loads(args.idr_report.read_text(encoding="utf-8"))
    expected = idr_report.get("outputs", {}).get("train", {})
    if (
        idr_report.get("schema") != IDR1_REPORT_SCHEMA
        or idr_report.get("status") != "complete"
        or Path(expected.get("path", "")).resolve() != args.idr_train.resolve()
        or expected.get("sha256") != sha256_file(args.idr_train)
    ):
        raise VFR1RequestError("IDR1 train report binding differs")
    sources = load_source_banks(args.banks)
    requests = collect_unique_requests(_load_jsonl(args.idr_train), sources)
    if len(requests) != EXPECTED_IDENTITIES:
        raise VFR1RequestError("VFR1 identity count differs")
    output_sha256 = _atomic_lines(args.output, requests)
    target_kind_counts = Counter(str(row["target_kind"]) for row in requests)
    short_reference_counts = Counter(
        str(row["target_kind"])
        for row in requests
        if len(str(row["reference_solution"])) < 80
    )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "idr_train": str(args.idr_train.resolve()),
        "idr_train_sha256": sha256_file(args.idr_train),
        "idr_report": str(args.idr_report.resolve()),
        "idr_report_sha256": sha256_file(args.idr_report),
        "banks": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.banks
        ],
        "identities": len(requests),
        "task_counts": dict(Counter(str(row["task"]) for row in requests)),
        "target_kind_counts": dict(target_kind_counts),
        "short_reference_counts": dict(short_reference_counts),
        "output": str(args.output.resolve()),
        "output_sha256": output_sha256,
        "teacher_runtime_fields": ["teacher_prompt"],
        "assessor_fields_visible_to_teacher": False,
    }
    _atomic_json(args.report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idr-train", type=Path, required=True)
    parser.add_argument("--idr-report", type=Path, required=True)
    parser.add_argument("--bank", dest="banks", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    report = build(parser.parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
