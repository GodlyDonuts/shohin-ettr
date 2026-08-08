#!/usr/bin/env python3
"""Build and merge the conditional AQC1 product-evaluation stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from build_idr1_revision_data import internal_revision_prompt
from hf_product_reasoning_eval import _task_prompt

VCR_SCHEMA = "shohin-vcr1-product-eval-v1"
SOURCE_SCHEMA = "shohin-aqc1-product-source-v1"
REVISION_SCHEMA = "shohin-aqc1-product-revision-v1"
MERGED_SCHEMA = "shohin-aqc1-product-candidates-v1"
PAIR_SCHEMA = "shohin-aqc1-whole-trajectory-pair-v1"
REPORT_SCHEMA = "shohin-aqc1-product-data-report-v1"


class ProductDataError(RuntimeError):
    """The product board or generated lineage is incomplete."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise ProductDataError(f"refusing existing output: {path}")
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


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise ProductDataError(f"refusing existing report: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_report(
    args: argparse.Namespace, stage: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    output_hash = atomic_lines(args.output, rows)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "stage": stage,
        "output": str(args.output.resolve()),
        "output_sha256": output_hash,
        "rows": len(rows),
        "tasks": sorted({str(row["task"]) for row in rows}),
    }
    atomic_json(args.report, report)
    return report


def source(args: argparse.Namespace) -> dict[str, Any]:
    receipt = json.loads(args.input_report.read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != "shohin-vcr1-product-data-report-v1"
        or receipt.get("status") != "complete"
        or Path(receipt.get("output", "")).resolve() != args.input.resolve()
        or receipt.get("output_sha256") != sha256_file(args.input)
    ):
        raise ProductDataError("VCR1 product receipt differs")
    output = []
    for row in load_jsonl(args.input):
        if row.get("schema") != VCR_SCHEMA:
            raise ProductDataError("VCR1 product row schema differs")
        assessor = dict(row["assessor"])
        assessor.update(
            schema=SOURCE_SCHEMA,
            identity_sha256=row["identity_sha256"],
            task=row["task"],
        )
        output.append(assessor)
    if len(output) != 568 or len({row["identity_sha256"] for row in output}) != 568:
        raise ProductDataError("product source coverage differs")
    return write_report(args, "source", output)


def load_bound(
    path: Path, report_path: Path, stage: str | None = None
) -> list[dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise ProductDataError("input report is incomplete")
    if stage is not None and report.get("stage") != stage:
        raise ProductDataError("input stage differs")
    expected_path = report.get("output") or report.get("candidates_output")
    expected_hash = report.get("output_sha256") or report.get("candidates_sha256")
    if Path(
        str(expected_path)
    ).resolve() != path.resolve() or expected_hash != sha256_file(path):
        raise ProductDataError("input path or hash differs")
    return load_jsonl(path)


def merge(args: argparse.Namespace) -> dict[str, Any]:
    sources = load_bound(args.source, args.source_report, "source")
    expected_ids = [row["identity_sha256"] for row in sources]
    candidates: dict[str, dict[str, Any]] = {}
    intervals: list[tuple[int, int]] = []
    for path, report_path in zip(args.candidate, args.candidate_report, strict=True):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "complete":
            raise ProductDataError("candidate shard is incomplete")
        if report.get("data_sha256") != sha256_file(args.source):
            raise ProductDataError("candidate shard source differs")
        if Path(report.get("candidates_output", "")).resolve() != path.resolve():
            raise ProductDataError("candidate shard path differs")
        if report.get("candidates_sha256") != sha256_file(path):
            raise ProductDataError("candidate shard hash differs")
        skip, count = int(report["skip"]), int(report["count"])
        intervals.append((skip, skip + count))
        for row in load_jsonl(path):
            identity = row["identity_sha256"]
            if identity in candidates:
                raise ProductDataError("candidate identity is duplicated")
            candidates[identity] = row
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor or end <= start:
            raise ProductDataError("candidate shard coverage is not contiguous")
        cursor = end
    if cursor != len(sources) or set(candidates) != set(expected_ids):
        raise ProductDataError("candidate coverage differs")
    rows = []
    for identity in expected_ids:
        candidate = candidates[identity]
        rows.append(
            {
                "schema": MERGED_SCHEMA,
                "identity_sha256": identity,
                "task": candidate["task"],
                "lineage": args.lineage,
                "completion": candidate["completion"],
                "correct": bool(candidate["correct"]),
                "prediction": candidate.get("prediction"),
                "generated_tokens": candidate.get("generated_tokens"),
                "max_token_exhausted": bool(candidate.get("max_token_exhausted")),
            }
        )
    return write_report(args, f"merged-{args.lineage}", rows)


def revision(args: argparse.Namespace) -> dict[str, Any]:
    sources = load_bound(args.source, args.source_report, "source")
    drafts = load_bound(args.drafts, args.drafts_report, "merged-draft")
    draft_by_id = {row["identity_sha256"]: row for row in drafts}
    if set(draft_by_id) != {row["identity_sha256"] for row in sources}:
        raise ProductDataError("draft coverage differs")
    rows = []
    for source_row in sources:
        identity = source_row["identity_sha256"]
        task = str(source_row["task"])
        draft = draft_by_id[identity]
        prompt = internal_revision_prompt(
            _task_prompt(task, source_row),
            str(draft["completion"]),
            "mbpp" if task in ("humaneval", "mbpp") else task,
        )
        rows.append(
            {
                "schema": REVISION_SCHEMA,
                "identity_sha256": identity,
                "task": task,
                "question": prompt,
                "assessor": source_row,
                "internal_draft": draft,
                "runtime_fields": ["question"],
            }
        )
    return write_report(args, "revision", rows)


def pairs(args: argparse.Namespace) -> dict[str, Any]:
    sources = load_bound(args.source, args.source_report, "source")
    idr = load_bound(args.idr, args.idr_report, "merged-idr1")
    control = load_bound(args.control, args.control_report, "merged-control")
    maps = (
        {row["identity_sha256"]: row for row in idr},
        {row["identity_sha256"]: row for row in control},
    )
    expected = {row["identity_sha256"] for row in sources}
    if any(set(mapped) != expected for mapped in maps):
        raise ProductDataError("revision lineage coverage differs")
    rows = []
    for source_row in sources:
        identity = source_row["identity_sha256"]
        candidates = []
        for lineage, mapped in zip(("idr1", "control"), maps, strict=True):
            item = mapped[identity]
            candidates.append(
                {
                    "lineage": lineage,
                    "completion": item["completion"],
                    "correct": bool(item["correct"]),
                    "prediction": item.get("prediction"),
                }
            )
        outcome = (
            "both_correct"
            if all(item["correct"] for item in candidates)
            else (
                "idr1_only"
                if candidates[0]["correct"]
                else "control_only" if candidates[1]["correct"] else "both_wrong"
            )
        )
        rows.append(
            {
                "schema": PAIR_SCHEMA,
                "identity_sha256": identity,
                "split": "product",
                "source_split": "product",
                "task": source_row["task"],
                "question": _task_prompt(str(source_row["task"]), source_row),
                "outcome_class": outcome,
                "candidates": candidates,
            }
        )
    return write_report(args, "pairs", rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output", type=Path, required=True)
    common.add_argument("--report", type=Path, required=True)

    source_parser = subparsers.add_parser("source", parents=[common])
    source_parser.add_argument("--input", type=Path, required=True)
    source_parser.add_argument("--input-report", type=Path, required=True)

    merge_parser = subparsers.add_parser("merge", parents=[common])
    merge_parser.add_argument("--source", type=Path, required=True)
    merge_parser.add_argument("--source-report", type=Path, required=True)
    merge_parser.add_argument("--candidate", type=Path, action="append", required=True)
    merge_parser.add_argument(
        "--candidate-report", type=Path, action="append", required=True
    )
    merge_parser.add_argument(
        "--lineage", choices=("draft", "idr1", "control"), required=True
    )

    revision_parser = subparsers.add_parser("revision", parents=[common])
    revision_parser.add_argument("--source", type=Path, required=True)
    revision_parser.add_argument("--source-report", type=Path, required=True)
    revision_parser.add_argument("--drafts", type=Path, required=True)
    revision_parser.add_argument("--drafts-report", type=Path, required=True)

    pairs_parser = subparsers.add_parser("pairs", parents=[common])
    pairs_parser.add_argument("--source", type=Path, required=True)
    pairs_parser.add_argument("--source-report", type=Path, required=True)
    pairs_parser.add_argument("--idr", type=Path, required=True)
    pairs_parser.add_argument("--idr-report", type=Path, required=True)
    pairs_parser.add_argument("--control", type=Path, required=True)
    pairs_parser.add_argument("--control-report", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "source":
        report = source(args)
    elif args.command == "merge":
        if len(args.candidate) != len(args.candidate_report):
            raise ProductDataError("candidate/report count differs")
        report = merge(args)
    elif args.command == "revision":
        report = revision(args)
    else:
        report = pairs(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
