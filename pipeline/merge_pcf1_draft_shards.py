#!/usr/bin/env python3
"""Merge the exact source-only PCF1 draft shards into one write-once bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from hf_pcf1_generate_drafts import (
    DRAFT_SCHEMA,
    REPORT_SCHEMA,
    load_sources,
    reject_protected_path,
)

MERGED_REPORT_SCHEMA = "shohin-pcf1-merged-drafts-v1"


class PCF1DraftMergeError(RuntimeError):
    """PCF1 draft shards do not form one complete immutable collection."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise PCF1DraftMergeError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    try:
        with temporary.open("xb") as handle:
            for row in rows:
                encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
                handle.write(encoded)
                digest.update(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except FileExistsError as error:
        raise PCF1DraftMergeError(f"refusing existing output: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise PCF1DraftMergeError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except FileExistsError as error:
        raise PCF1DraftMergeError(f"refusing existing output: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def merge(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.shard_reports) != len(args.shard_candidates):
        raise PCF1DraftMergeError("PCF1 report/candidate shard count differs")
    for path in (
        args.source_root,
        *args.shard_reports,
        *args.shard_candidates,
        args.output,
        args.report,
    ):
        reject_protected_path(path)
    if args.output.exists() or args.report.exists():
        raise PCF1DraftMergeError("PCF1 merged draft output already exists")
    source_rows, _ = load_sources(args.source_root)
    source_report_sha256 = sha256_file(args.source_root / "report.json")
    common: dict[str, Any] | None = None
    ranges: list[tuple[int, int]] = []
    by_start: dict[int, list[dict[str, Any]]] = {}
    receipts: list[dict[str, Any]] = []
    elapsed = peak = prompt_tokens = generated_tokens = exhausted = 0
    for report_path, candidates in zip(
        args.shard_reports, args.shard_candidates, strict=True
    ):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("schema") != REPORT_SCHEMA
            or report.get("status") != "complete"
            or report.get("supervisor_fields_visible_to_model") is not False
            or report.get("sealed_access") != {"holdout": 0, "product": 0, "public": 0}
        ):
            raise PCF1DraftMergeError("PCF1 draft shard report differs")
        shard_common = {
            key: report.get(key)
            for key in (
                "model_root",
                "model_revision",
                "model_loader",
                "adapter_checkpoint_sha256",
                "environment_receipt_sha256",
                "environment_tree_sha256",
                "source_report_sha256",
                "source_counts",
                "generation_mode",
                "thinking_enabled",
                "max_new_tokens",
                "seed",
                "batch_size",
                "shard_count",
                "full_row_count",
            )
        }
        if common is None:
            common = shard_common
        elif shard_common != common:
            raise PCF1DraftMergeError("PCF1 draft shard settings differ")
        if (
            report.get("source_root") != str(args.source_root.resolve())
            or report.get("source_report_sha256") != source_report_sha256
            or report.get("full_row_count") != len(source_rows)
        ):
            raise PCF1DraftMergeError("PCF1 draft/source truth binding differs")
        start, end = report.get("row_start"), report.get("row_end")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start < end
        ):
            raise PCF1DraftMergeError("PCF1 draft shard range differs")
        if (
            Path(str(report.get("candidates_output", ""))).resolve()
            != candidates.resolve()
        ):
            raise PCF1DraftMergeError("PCF1 explicit shard candidate path differs")
        reject_protected_path(candidates)
        if sha256_file(candidates) != report.get("candidates_sha256"):
            raise PCF1DraftMergeError("PCF1 draft shard hash differs")
        rows = [
            json.loads(line)
            for line in candidates.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != end - start:
            raise PCF1DraftMergeError("PCF1 draft shard content differs")
        allowed = {
            "schema",
            "identity_sha256",
            "split",
            "task",
            "completion",
            "generated_tokens",
            "max_token_exhausted",
            "prompt_sha256",
            "adapter_checkpoint_sha256",
            "model_revision",
            "finish_reason",
            "wall_seconds",
        }
        for source, row in zip(source_rows[start:end], rows, strict=True):
            if (
                set(row) != allowed
                or row.get("schema") != DRAFT_SCHEMA
                or row.get("identity_sha256") != source["identity_sha256"]
                or row.get("split") != source["split"]
                or row.get("task") != source["task"]
                or row.get("model_revision") != report.get("model_revision")
                or row.get("adapter_checkpoint_sha256")
                != report.get("adapter_checkpoint_sha256")
                or row.get("prompt_sha256")
                != hashlib.sha256(source["source_prompt"].encode()).hexdigest()
                or not isinstance(row.get("completion"), str)
                or not row["completion"].strip()
                or isinstance(row.get("generated_tokens"), bool)
                or not isinstance(row.get("generated_tokens"), int)
                or row["generated_tokens"] <= 0
                or not isinstance(row.get("max_token_exhausted"), bool)
                or row.get("finish_reason")
                != ("length" if row["max_token_exhausted"] else "stop")
                or isinstance(row.get("wall_seconds"), bool)
                or not isinstance(row.get("wall_seconds"), (int, float))
                or row["wall_seconds"] < 0
            ):
                raise PCF1DraftMergeError("PCF1 draft/source row binding differs")
        ranges.append((start, end))
        if start in by_start:
            raise PCF1DraftMergeError("duplicate PCF1 draft shard")
        by_start[start] = rows
        elapsed += float(report.get("elapsed_seconds", 0.0))
        peak = max(peak, int(report.get("peak_gpu_memory_bytes", 0)))
        prompt_tokens += int(report.get("prompt_tokens", 0))
        generated_tokens += int(report.get("generated_tokens", 0))
        exhausted += int(report.get("max_token_exhausted", 0))
        receipts.append(
            {
                "report": str(report_path.resolve()),
                "report_sha256": sha256_file(report_path),
                "candidates": str(candidates.resolve()),
                "candidates_sha256": report["candidates_sha256"],
                "row_start": start,
                "row_end": end,
            }
        )
    if (
        common is None
        or common.get("shard_count") != 16
        or common.get("batch_size") != 2
        or common.get("seed") != 2026080818
        or common.get("full_row_count") != 7113
        or len(args.shard_reports) != common.get("shard_count")
    ):
        raise PCF1DraftMergeError("PCF1 draft shard count differs")
    ordered_ranges = sorted(ranges)
    cursor = 0
    for start, end in ordered_ranges:
        if start != cursor:
            raise PCF1DraftMergeError("PCF1 draft shard ranges are not contiguous")
        cursor = end
    if cursor != common.get("full_row_count"):
        raise PCF1DraftMergeError("PCF1 draft coverage is incomplete")
    rows = [row for start, _ in ordered_ranges for row in by_start[start]]
    identities = [row.get("identity_sha256") for row in rows]
    if len(set(identities)) != len(rows):
        raise PCF1DraftMergeError("PCF1 draft identities are duplicated")
    output_sha256 = atomic_lines(args.output, rows)
    report = {
        "schema": MERGED_REPORT_SCHEMA,
        "status": "complete",
        **common,
        "output": str(args.output.resolve()),
        "output_sha256": output_sha256,
        "rows": len(rows),
        "inputs": receipts,
        "aggregate_gpu_seconds": elapsed,
        "maximum_peak_gpu_memory_bytes": peak,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "max_token_exhausted": exhausted,
        "exact_identity_coverage": True,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    atomic_json(args.report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--shard-report",
        dest="shard_reports",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--shard-candidates",
        dest="shard_candidates",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    report = merge(parser.parse_args())
    print(
        json.dumps(
            {"rows": report["rows"], "output_sha256": report["output_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
