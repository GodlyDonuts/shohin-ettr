#!/usr/bin/env python3
"""Merge sixteen exact host-owned upward-MoE draft shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from hf_product_reasoning_eval import GENERATED_ONLY_SEQUENCE_CONTRACT
from hf_q36_mtr_generate_drafts import (
    _atomic_json,
    _atomic_lines,
    load_sources,
)
from hf_upward_moe_generate_drafts import (
    REPORT_SCHEMA as SHARD_REPORT_SCHEMA,
    SCHEMA as DRAFT_SCHEMA,
    host_spec,
)
from q36_mtr_roles import (
    DRAFT_IDENTITIES,
    DRAFT_MAX_NEW_TOKENS,
    DRAFT_SEED,
    DRAFT_SHARDS,
)
from upward_moe_role_lineage import sha256_file

SCHEMA = "shohin-upward-moe-merged-drafts-v1"


class UpwardMoEDraftMergeError(RuntimeError):
    """The upward-MoE draft shards differed from their frozen source partition."""


def merge(args: argparse.Namespace) -> dict[str, Any]:
    spec = host_spec(args.host)
    if (
        len(args.shard_reports) != DRAFT_SHARDS
        or len(args.shard_candidates) != DRAFT_SHARDS
        or args.output.exists()
        or args.report.exists()
    ):
        raise UpwardMoEDraftMergeError("upward-MoE draft merge settings differ")
    sources, freeze_report = load_sources(
        args.train_source, args.development_source, args.freeze_report
    )
    by_start: dict[int, list[dict[str, Any]]] = {}
    common: dict[str, Any] | None = None
    receipts = []
    seen_indices: set[int] = set()
    total_elapsed = prompt_tokens = generated_tokens = exhausted = 0.0
    maximum_peak = {"0": 0, "1": 0}
    allowed = {
        "schema",
        "host",
        "identity_sha256",
        "split",
        "task",
        "prompt_sha256",
        "owner_checkpoint_sha256",
        "owner_state_sha256",
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
        index = report.get("shard_index")
        start = report.get("row_start")
        end = report.get("row_end")
        if (
            report.get("schema") != SHARD_REPORT_SCHEMA
            or report.get("status") != "complete"
            or report.get("host") != spec.host
            or report.get("model_revision") != spec.model_revision
            or report.get("host_contract") != spec.receipt()
            or report.get("owner_role") != "owner"
            or report.get("owner_update") != 256
            or report.get("owner_restore_exact") is not True
            or report.get("generation_mode") != "greedy"
            or report.get("generation_sequence_contract")
            != GENERATED_ONLY_SEQUENCE_CONTRACT
            or report.get("rendered_chat_tokenization") != "add_special_tokens_false"
            or report.get("max_new_tokens") != DRAFT_MAX_NEW_TOKENS
            or report.get("seed") != DRAFT_SEED
            or report.get("shard_count") != DRAFT_SHARDS
            or report.get("full_rows") != DRAFT_IDENTITIES
            or report.get("capability_scored") is not False
            or report.get("sealed_access") != {"holdout": 0, "product": 0, "public": 0}
            or isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < DRAFT_SHARDS
            or index in seen_indices
            or start != DRAFT_IDENTITIES * index // DRAFT_SHARDS
            or end != DRAFT_IDENTITIES * (index + 1) // DRAFT_SHARDS
        ):
            raise UpwardMoEDraftMergeError("upward-MoE shard report differs")
        seen_indices.add(index)
        shard_common = {
            key: report.get(key)
            for key in (
                "host",
                "model_revision",
                "host_contract",
                "model_receipt",
                "owner_checkpoint_sha256",
                "owner_state_sha256",
                "owner_role",
                "owner_update",
                "mechanics_report_sha256",
                "freeze_report_sha256",
                "freeze_identity_receipts",
                "train_source_sha256",
                "development_source_sha256",
                "generation_mode",
                "generation_sequence_contract",
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
            raise UpwardMoEDraftMergeError("upward-MoE shard lineage differs")
        if Path(
            str(report.get("output", ""))
        ).resolve() != candidates_path.resolve() or report.get(
            "output_sha256"
        ) != sha256_file(
            candidates_path
        ):
            raise UpwardMoEDraftMergeError("upward-MoE shard bytes differ")
        rows = [
            json.loads(line)
            for line in candidates_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if len(rows) != end - start or report.get("rows") != len(rows):
            raise UpwardMoEDraftMergeError("upward-MoE shard cardinality differs")
        for source, row in zip(sources[start:end], rows, strict=True):
            tokens = row.get("generated_tokens")
            wall = row.get("wall_seconds")
            hit_limit = row.get("max_token_exhausted")
            if (
                set(row) != allowed
                or row.get("schema") != DRAFT_SCHEMA
                or row.get("host") != spec.host
                or row.get("identity_sha256") != source["identity_sha256"]
                or row.get("split") != source["split"]
                or row.get("task") != source["task"]
                or row.get("prompt_sha256")
                != hashlib.sha256(str(source["source_prompt"]).encode()).hexdigest()
                or row.get("owner_checkpoint_sha256")
                != report.get("owner_checkpoint_sha256")
                or row.get("owner_state_sha256") != report.get("owner_state_sha256")
                or row.get("model_revision") != spec.model_revision
                or not isinstance(row.get("completion"), str)
                or not row["completion"].strip()
                or isinstance(tokens, bool)
                or not isinstance(tokens, int)
                or tokens <= 0
                or not isinstance(hit_limit, bool)
                or row.get("finish_reason") != ("length" if hit_limit else "stop")
                or isinstance(wall, bool)
                or not isinstance(wall, (int, float))
                or not math.isfinite(float(wall))
                or wall < 0
            ):
                raise UpwardMoEDraftMergeError(
                    "upward-MoE draft/source binding differs"
                )
        by_start[start] = rows
        peak = report.get("peak_gpu_memory_bytes")
        if not isinstance(peak, dict) or set(peak) != {"0", "1"}:
            raise UpwardMoEDraftMergeError("upward-MoE peak-memory receipt differs")
        for device in maximum_peak:
            maximum_peak[device] = max(maximum_peak[device], int(peak[device]))
        total_elapsed += float(report.get("elapsed_seconds", 0.0))
        prompt_tokens += int(report.get("prompt_tokens", 0))
        generated_tokens += int(report.get("generated_tokens", 0))
        exhausted += int(report.get("max_token_exhausted", 0))
        receipts.append(
            {
                "shard_index": index,
                "report": str(report_path.resolve()),
                "report_sha256": sha256_file(report_path),
                "candidates": str(candidates_path.resolve()),
                "candidates_sha256": report["output_sha256"],
                "row_start": start,
                "row_end": end,
            }
        )
    expected_starts = {
        DRAFT_IDENTITIES * index // DRAFT_SHARDS for index in range(DRAFT_SHARDS)
    }
    if set(by_start) != expected_starts:
        raise UpwardMoEDraftMergeError("upward-MoE shard coverage differs")
    rows = [row for start in sorted(by_start) for row in by_start[start]]
    identities = [row["identity_sha256"] for row in rows]
    if len(rows) != DRAFT_IDENTITIES or len(set(identities)) != DRAFT_IDENTITIES:
        raise UpwardMoEDraftMergeError("upward-MoE merged identities differ")
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
        "prompt_tokens": int(prompt_tokens),
        "generated_tokens": int(generated_tokens),
        "max_token_exhausted": int(exhausted),
        "exact_identity_coverage": True,
        "duplicate_identities": 0,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    _atomic_json(args.report, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        choices=("nemotron-super", "mixtral-8x22b", "nemotron-ultra"),
        required=True,
    )
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


if __name__ == "__main__":
    print(json.dumps(merge(parse_args()), sort_keys=True))
