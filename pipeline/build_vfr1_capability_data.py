#!/usr/bin/env python3
"""Build matched fault-first and shuffled-fault revision curricula."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from build_idr1_revision_data import (
    EVAL_SCHEMA as IDR1_EVAL_SCHEMA,
    REPORT_SCHEMA as IDR1_REPORT_SCHEMA,
    TRAIN_SCHEMA as IDR1_TRAIN_SCHEMA,
)
from build_vcr1_revision_data import sha256_file
from merge_vfr1_trace_shards import REPORT_SCHEMA as MERGED_TRACE_REPORT_SCHEMA
from score_vfr1_trace_quality import REPORT_SCHEMA as QUALITY_REPORT_SCHEMA
from hf_vfr1_generate_traces import TRACE_SCHEMA


TRAIN_SCHEMA = "shohin-vfr1-capability-train-v1"
EVAL_SCHEMA = "shohin-vfr1-capability-eval-v1"
REPORT_SCHEMA = "shohin-vfr1-capability-data-report-v1"
EXPECTED_TRAIN_ROWS = 9655
EXPECTED_TRAIN_IDENTITIES = 5824
EXPECTED_DEVELOPMENT_ROWS = 1289
FALLBACK_FAULT = (
    "The internal draft requires a fresh constraint-by-constraint check before "
    "emitting the corrected solution."
)


class VFR1CapabilityDataError(RuntimeError):
    """VFR1 capability data or provenance differs."""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise VFR1CapabilityDataError(f"empty VFR1 source: {path}")
    return rows


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise VFR1CapabilityDataError(f"refusing existing VFR1 output: {path}")
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
        raise VFR1CapabilityDataError(f"refusing existing VFR1 report: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def format_trace(fault: str, revision: str) -> str:
    if not fault.strip() or not revision.strip() or r"\boxed" in fault:
        raise VFR1CapabilityDataError("VFR1 target blocks differ")
    return (
        f"<FAULT>\n{fault.strip()}\n</FAULT>\n"
        f"<REVISION>\n{revision.strip()}\n</REVISION>"
    )


def fault_first_prompt(question: str) -> str:
    return (
        f"{question}\n\nBefore the replacement solution, emit exactly one concise "
        "diagnosis inside <FAULT>...</FAULT>. Do not put a boxed answer or "
        "executable code in that block. Then emit exactly one complete corrected "
        "solution inside <REVISION>...</REVISION>."
    )


def shuffled_faults(
    faults: dict[str, tuple[str, str]], block_size: int = 32
) -> dict[str, str]:
    """Rotate faults among nearby-length same-task identities."""

    if block_size < 2:
        raise VFR1CapabilityDataError("fault shuffle block is too small")
    groups: dict[str, list[tuple[str, str]]] = {}
    for identity, (task, fault) in faults.items():
        groups.setdefault(task, []).append((identity, fault))
    assigned: dict[str, str] = {}
    for task_rows in groups.values():
        ordered = sorted(
            task_rows,
            key=lambda item: (
                len(item[1]),
                hashlib.sha256(item[0].encode()).hexdigest(),
            ),
        )
        for start in range(0, len(ordered), block_size):
            block = ordered[start : start + block_size]
            if len(block) == 1:
                raise VFR1CapabilityDataError("fault shuffle produced a singleton")
            donor_faults = [fault for _, fault in block[1:] + block[:1]]
            for (identity, _), donor in zip(block, donor_faults, strict=True):
                assigned[identity] = donor
    if set(assigned) != set(faults):
        raise VFR1CapabilityDataError("fault shuffle coverage differs")
    return assigned


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise VFR1CapabilityDataError("VFR1 capability output already exists")
    idr_report = json.loads(args.idr_report.read_text(encoding="utf-8"))
    for split, path in (("train", args.idr_train), ("development", args.idr_development)):
        expected = idr_report.get("outputs", {}).get(split, {})
        if (
            idr_report.get("schema") != IDR1_REPORT_SCHEMA
            or idr_report.get("status") != "complete"
            or Path(expected.get("path", "")).resolve() != path.resolve()
            or expected.get("sha256") != sha256_file(path)
        ):
            raise VFR1CapabilityDataError("IDR1 capability source binding differs")
    merged_report = json.loads(args.trace_report.read_text(encoding="utf-8"))
    quality_report = json.loads(args.quality_report.read_text(encoding="utf-8"))
    if (
        merged_report.get("schema") != MERGED_TRACE_REPORT_SCHEMA
        or merged_report.get("status") != "complete"
        or Path(merged_report.get("output", "")).resolve() != args.traces.resolve()
        or merged_report.get("output_sha256") != sha256_file(args.traces)
        or Path(merged_report.get("quality_report", "")).resolve()
        != args.quality_report.resolve()
        or merged_report.get("quality_report_sha256") != sha256_file(args.quality_report)
        or quality_report.get("schema") != QUALITY_REPORT_SCHEMA
        or quality_report.get("status") != "complete"
        or quality_report.get("gate_pass") is not True
    ):
        raise VFR1CapabilityDataError("VFR1 trace quality did not authorize capability data")

    traces = _load_jsonl(args.traces)
    trace_map = {str(row.get("identity_sha256")): row for row in traces}
    if len(trace_map) != EXPECTED_TRAIN_IDENTITIES:
        raise VFR1CapabilityDataError("VFR1 trace identity count differs")
    original_train = _load_jsonl(args.idr_train)
    original_development = _load_jsonl(args.idr_development)
    identities = {str(row.get("source_identity_sha256")) for row in original_train}
    if identities != set(trace_map):
        raise VFR1CapabilityDataError("VFR1 train/trace identity coverage differs")

    canonical: dict[str, dict[str, str]] = {}
    fallback_count = 0
    for row in original_train:
        if row.get("schema") != IDR1_TRAIN_SCHEMA:
            raise VFR1CapabilityDataError("IDR1 train schema differs")
        identity = str(row["source_identity_sha256"])
        trace = trace_map[identity]
        use_generated = (
            trace.get("schema") == TRACE_SCHEMA
            and trace.get("parse_error") is None
            and bool(trace.get("score", {}).get("correct"))
            and not bool(trace.get("reference_leak"))
            and not bool(trace.get("max_token_exhausted"))
        )
        fault = str(trace.get("fault", "")) if use_generated else FALLBACK_FAULT
        revision = str(trace.get("revision", "")) if use_generated else str(row["response"])
        candidate = {"task": str(trace["task"]), "fault": fault, "revision": revision}
        previous = canonical.get(identity)
        if previous is not None and previous != candidate:
            raise VFR1CapabilityDataError("VFR1 duplicate presentations disagree")
        if previous is None and not use_generated:
            fallback_count += 1
        canonical[identity] = candidate
    fault_map = {
        identity: (row["task"], row["fault"]) for identity, row in canonical.items()
    }
    shuffled = shuffled_faults(fault_map)
    changed = sum(shuffled[identity] != canonical[identity]["fault"] for identity in canonical)

    treatment_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    for row in original_train:
        identity = str(row["source_identity_sha256"])
        base = {
            "schema": TRAIN_SCHEMA,
            "source_identity_sha256": identity,
            "outcome_class": row["outcome_class"],
            "presentation": row["presentation"],
            "question": fault_first_prompt(str(row["question"])),
            "target_kind": row["target_kind"],
            "internal_draft_visible": True,
            "external_candidate_text_visible": False,
        }
        treatment_rows.append(
            {
                **base,
                "identity_sha256": hashlib.sha256(
                    f"vfr1-treatment\0{row['identity_sha256']}".encode()
                ).hexdigest(),
                "response": format_trace(
                    canonical[identity]["fault"], canonical[identity]["revision"]
                ),
                "fault_control": "matched",
            }
        )
        control_rows.append(
            {
                **base,
                "identity_sha256": hashlib.sha256(
                    f"vfr1-shuffled\0{row['identity_sha256']}".encode()
                ).hexdigest(),
                "response": format_trace(shuffled[identity], canonical[identity]["revision"]),
                "fault_control": "within_task_near_length_shuffle",
            }
        )
    development_rows: list[dict[str, Any]] = []
    for row in original_development:
        if row.get("schema") != IDR1_EVAL_SCHEMA:
            raise VFR1CapabilityDataError("IDR1 development schema differs")
        development_rows.append(
            {
                **row,
                "schema": EVAL_SCHEMA,
                "question": fault_first_prompt(str(row["question"])),
                "strict_trace_format": True,
            }
        )
    if (
        len(treatment_rows) != EXPECTED_TRAIN_ROWS
        or len(control_rows) != EXPECTED_TRAIN_ROWS
        or len(development_rows) != EXPECTED_DEVELOPMENT_ROWS
        or changed < int(0.95 * EXPECTED_TRAIN_IDENTITIES)
    ):
        raise VFR1CapabilityDataError("VFR1 capability geometry differs")

    args.output.mkdir(parents=True)
    paths = {
        "train_treatment": args.output / "train_treatment.jsonl",
        "train_shuffled": args.output / "train_shuffled.jsonl",
        "development": args.output / "development.jsonl",
    }
    hashes = {
        "train_treatment": _atomic_lines(paths["train_treatment"], treatment_rows),
        "train_shuffled": _atomic_lines(paths["train_shuffled"], control_rows),
        "development": _atomic_lines(paths["development"], development_rows),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "idr_report": str(args.idr_report.resolve()),
        "idr_report_sha256": sha256_file(args.idr_report),
        "trace_report": str(args.trace_report.resolve()),
        "trace_report_sha256": sha256_file(args.trace_report),
        "quality_report": str(args.quality_report.resolve()),
        "quality_report_sha256": sha256_file(args.quality_report),
        "traces": str(args.traces.resolve()),
        "traces_sha256": sha256_file(args.traces),
        "train_identities": len(canonical),
        "generated_trace_fallbacks": fallback_count,
        "shuffled_fault_text_changed": changed,
        "outputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": hashes[name],
                "rows": len(treatment_rows) if name.startswith("train") else len(development_rows),
            }
            for name, path in paths.items()
        },
        "runtime_fields": ["question"],
        "internal_draft_visible": True,
        "assessor_fields_visible_to_model": False,
    }
    _atomic_json(args.output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idr-train", type=Path, required=True)
    parser.add_argument("--idr-development", type=Path, required=True)
    parser.add_argument("--idr-report", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--trace-report", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    report = build(parser.parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
