#!/usr/bin/env python3
"""Apply the frozen VFR1 train-only trace-quality gate."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from build_vcr1_revision_data import sha256_file


TRACE_SCHEMA = "shohin-vfr1-generated-trace-v1"
TRACE_REPORT_SCHEMA = "shohin-vfr1-generated-trace-report-v1"
REPORT_SCHEMA = "shohin-vfr1-trace-quality-report-v1"


class VFR1QualityError(RuntimeError):
    """VFR1 trace evidence is incomplete or inconsistent."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise VFR1QualityError(f"refusing existing VFR1 quality report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_traces(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            identity = str(row.get("identity_sha256", ""))
            if row.get("schema") != TRACE_SCHEMA or len(identity) != 64:
                raise VFR1QualityError("VFR1 trace schema differs")
            if identity in identities:
                raise VFR1QualityError("VFR1 trace identity is duplicated")
            identities.add(identity)
            rows.append(row)
    if not rows:
        raise VFR1QualityError("VFR1 trace file is empty")
    return rows


def summarize(rows: list[dict[str, Any]], expected_rows: int) -> dict[str, Any]:
    if len(rows) != expected_rows:
        raise VFR1QualityError("VFR1 quality row count differs")
    parsed = sum(row.get("parse_error") is None for row in rows)
    verified = sum(bool(row.get("score", {}).get("correct")) for row in rows)
    leaked = sum(bool(row.get("reference_leak")) for row in rows)
    exhausted = sum(bool(row.get("max_token_exhausted")) for row in rows)
    fault_boxed = sum(r"\boxed" in str(row.get("fault", "")) for row in rows)
    malformed_scores = sum(not isinstance(row.get("score"), dict) for row in rows)
    metrics = {
        "rows": len(rows),
        "parsed": parsed,
        "parse_fraction": parsed / len(rows),
        "verified": verified,
        "verified_fraction": verified / len(rows),
        "reference_leaks": leaked,
        "reference_leak_fraction": leaked / len(rows),
        "max_token_exhausted": exhausted,
        "exhaustion_fraction": exhausted / len(rows),
        "fault_blocks_with_boxed_answer": fault_boxed,
        "malformed_scores": malformed_scores,
        "task_counts": dict(Counter(str(row.get("task")) for row in rows)),
        "target_kind_counts": dict(
            Counter(str(row.get("target_kind")) for row in rows)
        ),
        "verified_by_target_kind": dict(
            Counter(
                str(row.get("target_kind"))
                for row in rows
                if bool(row.get("score", {}).get("correct"))
            )
        ),
    }
    gates = {
        "parse_at_least_0_95": metrics["parse_fraction"] >= 0.95,
        "verified_at_least_0_90": metrics["verified_fraction"] >= 0.90,
        "reference_leak_at_most_0_02": metrics["reference_leak_fraction"] <= 0.02,
        "zero_boxed_answers_in_fault": fault_boxed == 0,
        "exhaustion_at_most_0_10": metrics["exhaustion_fraction"] <= 0.10,
        "all_scores_structured": malformed_scores == 0,
    }
    return {"metrics": metrics, "gates": gates, "gate_pass": all(gates.values())}


def score(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise VFR1QualityError("VFR1 quality output already exists")
    trace_report = json.loads(args.trace_report.read_text(encoding="utf-8"))
    if (
        trace_report.get("schema") != TRACE_REPORT_SCHEMA
        or trace_report.get("status") != "complete"
        or Path(trace_report.get("output", "")).resolve() != args.traces.resolve()
        or trace_report.get("output_sha256") != sha256_file(args.traces)
        or int(trace_report.get("rows", -1)) != args.expected_rows
    ):
        raise VFR1QualityError("VFR1 trace report binding differs")
    rows = load_traces(args.traces)
    summary = summarize(rows, args.expected_rows)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "traces": str(args.traces.resolve()),
        "traces_sha256": sha256_file(args.traces),
        "trace_report": str(args.trace_report.resolve()),
        "trace_report_sha256": sha256_file(args.trace_report),
        "teacher_receipts": {
            key: trace_report.get(key)
            for key in (
                "model_root",
                "model_revision",
                "adapter_checkpoint_sha256",
                "requests_sha256",
                "generated_tokens",
                "elapsed_seconds",
                "generated_tokens_per_second",
                "peak_gpu_memory_bytes",
            )
        },
        **summary,
        "decision": (
            "allow_full_train_only_trace_generation"
            if summary["gate_pass"]
            else "close_exact_vfr1_teacher_format_before_full_generation"
        ),
    }
    _atomic_json(args.output, report)
    report["output_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--trace-report", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    report = score(parser.parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
