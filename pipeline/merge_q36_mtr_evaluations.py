#!/usr/bin/env python3
"""Merge exact Q36-MTR evaluation shards without opening development labels."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from hf_q36_mtr_evaluate import (
    ARMS,
    CANDIDATE_SCHEMA,
    DATA_REPORT_SCHEMA,
    EVALUATION_SEED,
    EXPECTED_FULL_ROWS,
    EXPECTED_SHARDS,
    MODEL_REVISION,
    REPORT_SCHEMA,
    SPLITS,
    TASKS,
    load_rows,
    model_visible_runtime_fields,
    sha256_file,
)
from pcf1_code_sandbox import (
    PCF1SandboxError,
    mbpp_allocation_setup_receipts_sha256,
    validate_sandbox_receipt_payload,
)
from q36_mtr_roles import TRAINABLE_PARAMETERS

SCHEMA = "shohin-q36-mtr-merged-evaluation-v1"


class Q36MTREvaluationMergeError(RuntimeError):
    """The Q36-MTR evaluation shards differ or overlap."""


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise Q36MTREvaluationMergeError(f"refusing existing Q36 merge: {path}")
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
        raise Q36MTREvaluationMergeError(f"refusing existing Q36 report: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _metrics(
    sources: list[dict[str, Any]], candidates: list[dict[str, Any]], split: str
) -> dict[str, dict[str, int]] | None:
    if split == "development":
        return None
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for source, candidate in zip(sources, candidates, strict=True):
        if not isinstance(candidate.get("correct"), bool):
            raise Q36MTREvaluationMergeError("Q36-MTR calibration score differs")
        for domain in ("overall", source["task"]):
            buckets[domain]["total"] += 1
            buckets[domain]["generated_correct"] += int(candidate["correct"])
    if set(buckets) != {"overall", *TASKS}:
        raise Q36MTREvaluationMergeError("Q36-MTR metric domains differ")
    return {domain: dict(counter) for domain, counter in sorted(buckets.items())}


def merge(args: argparse.Namespace) -> dict[str, Any]:
    expected_shards = EXPECTED_SHARDS.get(args.split)
    if (
        args.arm not in ARMS
        or args.split not in SPLITS
        or expected_shards is None
        or len(args.shard_reports) != expected_shards
        or len(args.shard_candidates) != expected_shards
        or (
            args.split == "calibration"
            and len(args.shard_sandbox_probes) != expected_shards
        )
        or (args.split == "development" and args.shard_sandbox_probes)
    ):
        raise Q36MTREvaluationMergeError("Q36-MTR merge shard geometry differs")
    if args.output.exists() or args.report.exists():
        raise Q36MTREvaluationMergeError("Q36-MTR merged evaluation exists")
    data_report = json.loads(args.data_report.read_text(encoding="utf-8"))
    expected = data_report.get("outputs", {}).get(args.split)
    if (
        data_report.get("schema") != DATA_REPORT_SCHEMA
        or data_report.get("status") != "complete"
        or data_report.get("sealed_access") != {"holdout": 0, "product": 0, "public": 0}
        or not isinstance(expected, dict)
        or expected.get("sha256") != sha256_file(args.data)
        or Path(str(expected.get("path", ""))).resolve() != args.data.resolve()
    ):
        raise Q36MTREvaluationMergeError("Q36-MTR merge data receipt differs")
    sources = load_rows(args.data, args.split)
    by_start: dict[int, list[dict[str, Any]]] = {}
    ranges: list[tuple[int, int]] = []
    receipts: list[dict[str, Any]] = []
    common: dict[str, Any] | None = None
    seen_indices: set[int] = set()
    elapsed = peak = prompt = generated = exhausted = empty = correct = 0
    sandbox_executions = 0
    setup_receipt_count = 0
    for position, (report_path, candidates_path) in enumerate(
        zip(args.shard_reports, args.shard_candidates, strict=True)
    ):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("schema") != REPORT_SCHEMA
            or report.get("status") != "complete"
            or report.get("arm") != args.arm
            or report.get("split") != args.split
            or report.get("model_revision") != MODEL_REVISION
            or report.get("generation_mode") != "greedy"
            or report.get("rendered_chat_tokenization") != "add_special_tokens_false"
            or report.get("max_new_tokens") != 768
            or report.get("seed") != EVALUATION_SEED
            or report.get("batch_size") != 1
            or report.get("shard_count") != expected_shards
            or report.get("full_row_count") != EXPECTED_FULL_ROWS[args.split]
            or report.get("runtime_fields") != model_visible_runtime_fields(args.arm)
            or report.get("assessor_fields_visible_to_model") is not False
            or report.get("assessor_board_access_count") != 0
            or report.get("sealed_access") != {"holdout": 0, "product": 0, "public": 0}
            or report.get("trainable_parameters") != TRAINABLE_PARAMETERS
        ):
            raise Q36MTREvaluationMergeError("Q36-MTR shard report differs")
        index = report.get("shard_index")
        start = report.get("row_start")
        end = report.get("row_end")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < expected_shards
            or index in seen_indices
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or start >= end
            or end > len(sources)
        ):
            raise Q36MTREvaluationMergeError("Q36-MTR shard range differs")
        seen_indices.add(index)
        shard_common = {
            key: report.get(key)
            for key in (
                "model_revision",
                "model_loader",
                "adapter_checkpoint_sha256",
                "adapter_metadata_sha256",
                "trainable_parameters",
                "trainable_parameter_name_sha256",
                "controlled_layer_indices",
                "role",
                "data_sha256",
                "data_report_sha256",
                "runtime_fields",
                "generation_mode",
                "rendered_chat_tokenization",
                "max_new_tokens",
                "seed",
                "batch_size",
                "shard_count",
                "full_row_count",
                "environment_receipt_sha256",
                "environment_tree_sha256",
                "code_sandbox_config_sha256",
                "code_sandbox_binary_sha256",
            )
        }
        if common is None:
            common = shard_common
        elif common != shard_common:
            raise Q36MTREvaluationMergeError("Q36-MTR shard settings differ")
        if Path(
            str(report.get("candidates_output", ""))
        ).resolve() != candidates_path.resolve() or report.get(
            "candidates_sha256"
        ) != sha256_file(
            candidates_path
        ):
            raise Q36MTREvaluationMergeError("Q36-MTR candidate hash differs")
        candidates = [
            json.loads(line)
            for line in candidates_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if len(candidates) != end - start:
            raise Q36MTREvaluationMergeError("Q36-MTR candidate cardinality differs")
        label_free_fields = {
            "schema",
            "arm",
            "identity_sha256",
            "task",
            "completion",
            "generated_tokens",
            "max_token_exhausted",
        }
        for source, candidate in zip(sources[start:end], candidates, strict=True):
            if (
                candidate.get("schema") != CANDIDATE_SCHEMA
                or candidate.get("arm") != args.arm
                or candidate.get("identity_sha256") != source["identity_sha256"]
                or candidate.get("task") != source["task"]
                or not isinstance(candidate.get("completion"), str)
                or isinstance(candidate.get("generated_tokens"), bool)
                or not isinstance(candidate.get("generated_tokens"), int)
                or not isinstance(candidate.get("max_token_exhausted"), bool)
                or (args.split == "development" and set(candidate) != label_free_fields)
                or (
                    args.split == "calibration"
                    and not isinstance(candidate.get("correct"), bool)
                )
            ):
                raise Q36MTREvaluationMergeError(
                    "Q36-MTR candidate/source binding differs"
                )
        if args.split == "calibration":
            probe_path = args.shard_sandbox_probes[position]
            probe = json.loads(probe_path.read_text(encoding="utf-8"))
            try:
                validate_sandbox_receipt_payload(probe)
            except PCF1SandboxError as error:
                raise Q36MTREvaluationMergeError(
                    "Q36-MTR sandbox receipt differs"
                ) from error
            if (
                report.get("sandbox_status") != "passed"
                or report.get("sandbox_receipt_sha256") != sha256_file(probe_path)
                or report.get("sandbox_probe_sha256") != probe.get("probe_sha256")
                or report.get("mbpp_setup_receipts_sha256")
                != mbpp_allocation_setup_receipts_sha256(
                    report.get("mbpp_setup_receipts", [])
                )
            ):
                raise Q36MTREvaluationMergeError("Q36-MTR sandbox lineage differs")
            setup_receipt_count += len(report.get("mbpp_setup_receipts", []))
        elif (
            report.get("sandbox_status") != "not_applicable_no_scoring"
            or report.get("sandbox_receipt_sha256") is not None
            or report.get("sandbox_probe_sha256") is not None
            or report.get("mbpp_setup_receipts") != []
            or report.get("mbpp_setup_receipts_sha256") is not None
        ):
            raise Q36MTREvaluationMergeError(
                "Q36-MTR label-free sandbox status differs"
            )
        counters = report.get("counters", {})
        if (
            counters.get("rows") != len(candidates)
            or counters.get("generated_tokens")
            != sum(candidate["generated_tokens"] for candidate in candidates)
            or counters.get("max_token_exhausted")
            != sum(int(candidate["max_token_exhausted"]) for candidate in candidates)
            or counters.get("empty_completions")
            != sum(int(not candidate["completion"].strip()) for candidate in candidates)
        ):
            raise Q36MTREvaluationMergeError("Q36-MTR shard counters differ")
        by_start[start] = candidates
        ranges.append((start, end))
        prompt += int(counters.get("prompt_tokens", 0))
        generated += int(counters.get("generated_tokens", 0))
        exhausted += int(counters.get("max_token_exhausted", 0))
        empty += int(counters.get("empty_completions", 0))
        correct += int(counters.get("correct", 0))
        sandbox_executions += int(counters.get("sandbox_executions", 0))
        elapsed += float(report.get("elapsed_seconds", 0.0))
        peak = max(peak, int(report.get("peak_gpu_memory_bytes", 0)))
        receipts.append(
            {
                "shard_index": index,
                "report": str(report_path.resolve()),
                "report_sha256": sha256_file(report_path),
                "candidates": str(candidates_path.resolve()),
                "candidates_sha256": report["candidates_sha256"],
                "sandbox_probe_sha256": report.get("sandbox_receipt_sha256"),
                "row_start": start,
                "row_end": end,
            }
        )
    ordered_ranges = sorted(ranges)
    cursor = 0
    for start, end in ordered_ranges:
        if start != cursor:
            raise Q36MTREvaluationMergeError("Q36-MTR shard ranges overlap or gap")
        cursor = end
    if cursor != len(sources):
        raise Q36MTREvaluationMergeError("Q36-MTR evaluation coverage is incomplete")
    candidates = [row for start, _ in ordered_ranges for row in by_start[start]]
    identities = [candidate["identity_sha256"] for candidate in candidates]
    if len(set(identities)) != len(sources):
        raise Q36MTREvaluationMergeError("Q36-MTR merged identities duplicate")
    output_sha256 = _atomic_lines(args.output, candidates)
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "arm": args.arm,
        "split": args.split,
        **(common or {}),
        "rows": len(candidates),
        "ordered_identity_sha256": hashlib.sha256(
            ("\n".join(identities) + "\n").encode()
        ).hexdigest(),
        "output": str(args.output.resolve()),
        "output_sha256": output_sha256,
        "metrics": _metrics(sources, candidates, args.split),
        "counters": {
            "rows": len(candidates),
            "prompt_tokens": prompt,
            "generated_tokens": generated,
            "max_token_exhausted": exhausted,
            "empty_completions": empty,
            "correct": correct,
            "sandbox_executions": sandbox_executions,
        },
        "aggregate_gpu_seconds": elapsed,
        "maximum_peak_gpu_memory_bytes": peak,
        "setup_receipt_count": setup_receipt_count,
        "input_receipts": sorted(receipts, key=lambda item: item["shard_index"]),
        "exact_identity_coverage": True,
        "duplicate_identities": 0,
        "assessor_board_access_count": 0,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    _atomic_json(args.report, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-report", type=Path, required=True)
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
    parser.add_argument(
        "--shard-sandbox-probe",
        dest="shard_sandbox_probes",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(merge(parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
