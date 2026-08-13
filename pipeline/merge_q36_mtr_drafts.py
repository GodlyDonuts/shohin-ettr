#!/usr/bin/env python3
"""Merge the 16 exact Q36 owner-draft shards with full identity custody."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from hf_q36_mtr_generate_drafts import (
    DRAFT_IDENTITIES,
    DRAFT_MAX_NEW_TOKENS,
    DRAFT_SEED,
    DRAFT_SHARDS,
    MODEL_REVISION,
    REPORT_SCHEMA,
    SCHEMA as DRAFT_SCHEMA,
    load_sources,
    sha256_file,
)

SCHEMA = "shohin-q36-mtr-merged-drafts-v1"


class Q36MTRDraftMergeError(RuntimeError):
    """Q36-MTR draft shards differ from the exact source partition."""


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise Q36MTRDraftMergeError(f"refusing existing Q36 draft merge: {path}")
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
    os.replace(temporary, path)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTRDraftMergeError(f"refusing existing Q36 draft report: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def merge(args: argparse.Namespace) -> dict[str, Any]:
    if (
        len(args.shard_reports) != DRAFT_SHARDS
        or len(args.shard_candidates) != DRAFT_SHARDS
    ):
        raise Q36MTRDraftMergeError("Q36-MTR requires exactly 16 draft shards")
    if args.output.exists() or args.report.exists():
        raise Q36MTRDraftMergeError("Q36-MTR draft merge output exists")
    sources, freeze_report = load_sources(
        args.train_source, args.development_source, args.freeze_report
    )
    common: dict[str, Any] | None = None
    by_start: dict[int, list[dict[str, Any]]] = {}
    ranges: list[tuple[int, int]] = []
    receipts: list[dict[str, Any]] = []
    total_elapsed = maximum_peak = prompt_tokens = generated_tokens = exhausted = 0
    seen_indices: set[int] = set()
    allowed = {
        "schema",
        "identity_sha256",
        "split",
        "task",
        "prompt_sha256",
        "owner_checkpoint_sha256",
        "model_revision",
        "completion",
        "generated_tokens",
        "max_token_exhausted",
        "finish_reason",
        "wall_seconds",
    }
    for report_path, candidates_path in zip(
        args.shard_reports, args.shard_candidates, strict=True
    ):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("schema") != REPORT_SCHEMA
            or report.get("status") != "complete"
            or report.get("capability_scored") is not False
            or report.get("sealed_access") != {"holdout": 0, "product": 0, "public": 0}
            or report.get("model_revision") != MODEL_REVISION
            or report.get("generation_mode") != "greedy"
            or report.get("rendered_chat_tokenization") != "add_special_tokens_false"
            or report.get("max_new_tokens") != DRAFT_MAX_NEW_TOKENS
            or report.get("seed") != DRAFT_SEED
            or report.get("shard_count") != DRAFT_SHARDS
            or report.get("full_rows") != DRAFT_IDENTITIES
        ):
            raise Q36MTRDraftMergeError("Q36-MTR draft shard report differs")
        shard_index = report.get("shard_index")
        start = report.get("row_start")
        end = report.get("row_end")
        if (
            isinstance(shard_index, bool)
            or not isinstance(shard_index, int)
            or not 0 <= shard_index < DRAFT_SHARDS
            or shard_index in seen_indices
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start != DRAFT_IDENTITIES * shard_index // DRAFT_SHARDS
            or end != DRAFT_IDENTITIES * (shard_index + 1) // DRAFT_SHARDS
        ):
            raise Q36MTRDraftMergeError("Q36-MTR draft shard range differs")
        seen_indices.add(shard_index)
        shard_common = {
            key: report.get(key)
            for key in (
                "model_revision",
                "model_loader",
                "owner_checkpoint_sha256",
                "owner_update",
                "owner_role",
                "freeze_report_sha256",
                "freeze_identity_receipts",
                "train_source_sha256",
                "development_source_sha256",
                "generation_mode",
                "rendered_chat_tokenization",
                "max_new_tokens",
                "seed",
                "shard_count",
                "full_rows",
            )
        }
        if common is None:
            common = shard_common
        elif common != shard_common:
            raise Q36MTRDraftMergeError("Q36-MTR draft shard settings differ")
        if Path(
            str(report.get("output", ""))
        ).resolve() != candidates_path.resolve() or report.get(
            "output_sha256"
        ) != sha256_file(
            candidates_path
        ):
            raise Q36MTRDraftMergeError("Q36-MTR draft candidate hash differs")
        rows = [
            json.loads(line)
            for line in candidates_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if len(rows) != end - start or report.get("rows") != len(rows):
            raise Q36MTRDraftMergeError("Q36-MTR draft shard cardinality differs")
        for source, row in zip(sources[start:end], rows, strict=True):
            tokens = row.get("generated_tokens")
            exhausted_row = row.get("max_token_exhausted")
            wall = row.get("wall_seconds")
            if (
                set(row) != allowed
                or row.get("schema") != DRAFT_SCHEMA
                or row.get("identity_sha256") != source["identity_sha256"]
                or row.get("split") != source["split"]
                or row.get("task") != source["task"]
                or row.get("prompt_sha256")
                != hashlib.sha256(source["source_prompt"].encode()).hexdigest()
                or row.get("owner_checkpoint_sha256")
                != report.get("owner_checkpoint_sha256")
                or row.get("model_revision") != MODEL_REVISION
                or not isinstance(row.get("completion"), str)
                or not row["completion"].strip()
                or isinstance(tokens, bool)
                or not isinstance(tokens, int)
                or tokens <= 0
                or not isinstance(exhausted_row, bool)
                or row.get("finish_reason") != ("length" if exhausted_row else "stop")
                or isinstance(wall, bool)
                or not isinstance(wall, (int, float))
                or not math.isfinite(float(wall))
                or wall < 0
            ):
                raise Q36MTRDraftMergeError("Q36-MTR draft/source row binding differs")
        by_start[start] = rows
        ranges.append((start, end))
        total_elapsed += float(report.get("elapsed_seconds", 0.0))
        maximum_peak = max(maximum_peak, int(report.get("peak_gpu_memory_bytes", 0)))
        generated_tokens += int(report.get("generated_tokens", 0))
        exhausted += int(report.get("max_token_exhausted", 0))
        prompt_tokens += int(report.get("prompt_tokens", 0))
        receipts.append(
            {
                "shard_index": shard_index,
                "report": str(report_path.resolve()),
                "report_sha256": sha256_file(report_path),
                "candidates": str(candidates_path.resolve()),
                "candidates_sha256": report["output_sha256"],
                "row_start": start,
                "row_end": end,
            }
        )
    ordered_ranges = sorted(ranges)
    if ordered_ranges != [
        (
            DRAFT_IDENTITIES * index // DRAFT_SHARDS,
            DRAFT_IDENTITIES * (index + 1) // DRAFT_SHARDS,
        )
        for index in range(DRAFT_SHARDS)
    ]:
        raise Q36MTRDraftMergeError("Q36-MTR draft coverage is not exact")
    rows = [row for start, _ in ordered_ranges for row in by_start[start]]
    identities = [row["identity_sha256"] for row in rows]
    if len(rows) != DRAFT_IDENTITIES or len(set(identities)) != DRAFT_IDENTITIES:
        raise Q36MTRDraftMergeError("Q36-MTR merged draft identities differ")
    output_sha256 = _atomic_lines(args.output, rows)
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        **(common or {}),
        "freeze_report_identity_receipts": freeze_report["identity_receipts"],
        "output": str(args.output.resolve()),
        "output_sha256": output_sha256,
        "rows": len(rows),
        "ordered_identity_sha256": hashlib.sha256(
            ("\n".join(identities) + "\n").encode()
        ).hexdigest(),
        "input_receipts": sorted(receipts, key=lambda item: item["shard_index"]),
        "aggregate_gpu_seconds": total_elapsed,
        "maximum_peak_gpu_memory_bytes": maximum_peak,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "max_token_exhausted": exhausted,
        "exact_identity_coverage": True,
        "duplicate_identities": 0,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    _atomic_json(args.report, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--development-source", type=Path, required=True)
    parser.add_argument("--freeze-report", type=Path, required=True)
    parser.add_argument(
        "--shard-report",
        dest="shard_reports",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--shard-candidates",
        dest="shard_candidates",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(merge(parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
