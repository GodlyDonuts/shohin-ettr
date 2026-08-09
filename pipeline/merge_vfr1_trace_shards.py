#!/usr/bin/env python3
"""Merge complete VFR1 generation shards and reapply the quality gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from build_vcr1_revision_data import sha256_file
from build_vfr1_teacher_requests import REPORT_SCHEMA as REQUEST_REPORT_SCHEMA
from build_vfr1_teacher_requests import REQUEST_SCHEMA
from score_vfr1_trace_quality import (
    REPORT_SCHEMA as QUALITY_REPORT_SCHEMA,
    TRACE_REPORT_SCHEMA,
    VFR1QualityError,
    load_traces,
    summarize,
)


REPORT_SCHEMA = "shohin-vfr1-merged-trace-report-v1"


class VFR1MergeError(VFR1QualityError):
    """VFR1 shard coverage or provenance differs."""


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise VFR1MergeError(f"refusing existing VFR1 merge: {path}")
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
        raise VFR1MergeError(f"refusing existing VFR1 merge report: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _request_identities(path: Path) -> list[str]:
    identities: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            identity = str(row.get("identity_sha256", ""))
            if row.get("schema") != REQUEST_SCHEMA or len(identity) != 64:
                raise VFR1MergeError("VFR1 request schema differs")
            identities.append(identity)
    if len(identities) != len(set(identities)) or not identities:
        raise VFR1MergeError("VFR1 request identities differ")
    return identities


def merge(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.report.exists() or args.quality_report.exists():
        raise VFR1MergeError("VFR1 merged output already exists")
    if len(args.traces) != len(args.trace_reports) or len(args.traces) < 2:
        raise VFR1MergeError("VFR1 trace/report shard count differs")
    request_report = json.loads(args.request_report.read_text(encoding="utf-8"))
    if (
        request_report.get("schema") != REQUEST_REPORT_SCHEMA
        or request_report.get("status") != "complete"
        or Path(request_report.get("output", "")).resolve() != args.requests.resolve()
        or request_report.get("output_sha256") != sha256_file(args.requests)
    ):
        raise VFR1MergeError("VFR1 request report binding differs")
    request_ids = _request_identities(args.requests)
    shard_count = len(args.traces)
    reports: dict[int, dict[str, Any]] = {}
    rows_by_identity: dict[str, dict[str, Any]] = {}
    common_receipts: dict[str, Any] | None = None
    for trace_path, report_path in zip(args.traces, args.trace_reports, strict=True):
        shard_report = json.loads(report_path.read_text(encoding="utf-8"))
        index = int(shard_report.get("shard_index", -1))
        if (
            shard_report.get("schema") != TRACE_REPORT_SCHEMA
            or shard_report.get("status") != "complete"
            or int(shard_report.get("shard_count", -1)) != shard_count
            or not 0 <= index < shard_count
            or index in reports
            or Path(shard_report.get("output", "")).resolve() != trace_path.resolve()
            or shard_report.get("output_sha256") != sha256_file(trace_path)
            or int(shard_report.get("rows", -1))
            != int(shard_report.get("row_end", -1))
            - int(shard_report.get("row_start", -1))
        ):
            raise VFR1MergeError("VFR1 shard report differs")
        receipts = {
            key: shard_report.get(key)
            for key in (
                "model_root",
                "model_revision",
                "model_loader",
                "adapter_checkpoint_sha256",
                "requests_sha256",
                "request_report_sha256",
            )
        }
        if common_receipts is None:
            common_receipts = receipts
        elif receipts != common_receipts:
            raise VFR1MergeError("VFR1 shard settings differ")
        reports[index] = shard_report
        for row in load_traces(trace_path):
            identity = str(row["identity_sha256"])
            if identity in rows_by_identity:
                raise VFR1MergeError("VFR1 merged identity is duplicated")
            rows_by_identity[identity] = row
    if set(reports) != set(range(shard_count)):
        raise VFR1MergeError("VFR1 shard index coverage differs")
    expected_bounds = [
        (len(request_ids) * index // shard_count, len(request_ids) * (index + 1) // shard_count)
        for index in range(shard_count)
    ]
    actual_bounds = [
        (int(reports[index]["row_start"]), int(reports[index]["row_end"]))
        for index in range(shard_count)
    ]
    if actual_bounds != expected_bounds or set(rows_by_identity) != set(request_ids):
        raise VFR1MergeError("VFR1 full identity coverage differs")
    merged = [rows_by_identity[identity] for identity in request_ids]
    output_sha256 = _atomic_lines(args.output, merged)
    summary = summarize(merged, len(request_ids))
    quality = {
        "schema": QUALITY_REPORT_SCHEMA,
        "status": "complete",
        "traces": str(args.output.resolve()),
        "traces_sha256": output_sha256,
        **summary,
        "decision": (
            "allow_matched_vfr1_capability_fit"
            if summary["gate_pass"]
            else "close_vfr1_before_capability_fit"
        ),
    }
    _atomic_json(args.quality_report, quality)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "requests": str(args.requests.resolve()),
        "requests_sha256": sha256_file(args.requests),
        "request_report": str(args.request_report.resolve()),
        "request_report_sha256": sha256_file(args.request_report),
        "shard_count": shard_count,
        "shards": [
            {
                "trace": str(args.traces[index].resolve()),
                "trace_sha256": sha256_file(args.traces[index]),
                "report": str(args.trace_reports[index].resolve()),
                "report_sha256": sha256_file(args.trace_reports[index]),
                "row_start": reports[index]["row_start"],
                "row_end": reports[index]["row_end"],
            }
            for index in range(shard_count)
        ],
        "common_receipts": common_receipts,
        "rows": len(merged),
        "output": str(args.output.resolve()),
        "output_sha256": output_sha256,
        "quality_report": str(args.quality_report.resolve()),
        "quality_report_sha256": sha256_file(args.quality_report),
        **summary,
    }
    _atomic_json(args.report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--request-report", type=Path, required=True)
    parser.add_argument("--trace", dest="traces", type=Path, action="append", required=True)
    parser.add_argument("--trace-report", dest="trace_reports", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path, required=True)
    report = merge(parser.parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
