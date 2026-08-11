#!/usr/bin/env python3
"""Merge exact PCF1 evaluation shards and independently rescore coverage."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from hf_pcf1_evaluate import (
    ARMS,
    CANDIDATE_SCHEMA,
    BWRAP_SHA256,
    DATA_REPORT_SCHEMA,
    REPORT_SCHEMA,
    SANDBOX_CONFIG_SHA256,
    SPLITS,
    load_rows,
)
from pcf1_code_sandbox import (
    PCF1SandboxError,
    mbpp_allocation_setup_receipts_sha256,
    validate_mbpp_setup_qualification_receipt,
    validate_sandbox_receipt_payload,
)

MERGED_REPORT_SCHEMA = "shohin-pcf1-merged-evaluation-v1"
EXPECTED_LORA_LAYER_INDICES = [30, 31, 32, 33]


class PCF1MergeError(RuntimeError):
    """PCF1 shard coverage, lineage, or immutable input differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_sealed_path(path: Path) -> None:
    rendered = f"{path}\n{path.resolve(strict=False)}".casefold()
    if any(word in rendered for word in ("holdout", "product", "public")):
        raise PCF1MergeError(f"sealed path supplied to PCF1 merge: {path}")


def explicit_shard_root(path: Path) -> Path:
    reject_sealed_path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise PCF1MergeError("PCF1 shard root is not an explicit directory")
    resolved = path.resolve(strict=True)
    if resolved in {Path("/"), Path.home().resolve()}:
        raise PCF1MergeError("PCF1 shard root is too broad")
    return resolved


def explicit_shard_file(path: Path, root: Path, label: str) -> Path:
    reject_sealed_path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise PCF1MergeError(f"PCF1 {label} is not an explicit regular file")
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise PCF1MergeError(f"PCF1 {label} escapes the shard root") from error
    if not relative.parts:
        raise PCF1MergeError(f"PCF1 {label} equals the shard root")
    current = root
    for part in relative.parts[:-1]:
        current /= part
        if current.is_symlink():
            raise PCF1MergeError(f"PCF1 {label} traverses a symbolic directory")
    return resolved


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise PCF1MergeError(f"empty PCF1 candidate shard: {path}")
    return rows


def load_sandbox_probe(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise PCF1MergeError("PCF1 shard sandbox probe is not an explicit file")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PCF1MergeError("PCF1 shard sandbox probe is unreadable") from error
    if not isinstance(payload, dict):
        raise PCF1MergeError("PCF1 shard sandbox probe differs")
    try:
        validate_sandbox_receipt_payload(payload)
    except PCF1SandboxError as error:
        raise PCF1MergeError("PCF1 shard sandbox probe differs") from error
    return payload, hashlib.sha256(raw).hexdigest()


def _expected_setup_sha256s(rows: list[dict[str, Any]]) -> list[str]:
    expected: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if row.get("task") != "mbpp":
            continue
        assessor = row.get("assessor")
        setup = (
            assessor.get("test_setup_code", "") if isinstance(assessor, dict) else None
        )
        if not isinstance(setup, str):
            raise PCF1MergeError("PCF1 MBPP calibration setup differs")
        digest = hashlib.sha256(setup.encode()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            expected.append(digest)
    return expected


def validate_shard_setup_receipts(
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    split: str,
    sandbox_probe: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    receipts = report.get("mbpp_allocation_setup_receipts")
    count = report.get("mbpp_allocation_setup_receipt_count")
    digest = report.get("mbpp_allocation_setup_receipts_sha256")
    if split == "confirmation":
        if (
            report.get("mbpp_allocation_setup_status")
            != "not_applicable_no_code_scoring"
            or receipts != []
            or count != 0
            or digest is not None
        ):
            raise PCF1MergeError("PCF1 confirmation setup qualification differs")
        return []
    if (
        report.get("mbpp_allocation_setup_status") != "passed"
        or not isinstance(receipts, list)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count != len(receipts)
        or not isinstance(sandbox_probe, dict)
        or not isinstance(sandbox_probe.get("probe_sha256"), str)
    ):
        raise PCF1MergeError("PCF1 calibration setup qualification differs")
    expected = _expected_setup_sha256s(rows)
    if len(receipts) != len(expected):
        raise PCF1MergeError("PCF1 calibration setup qualification coverage differs")
    for receipt, setup_sha256 in zip(receipts, expected, strict=True):
        if not isinstance(receipt, dict):
            raise PCF1MergeError("PCF1 calibration setup receipt differs")
        try:
            validate_mbpp_setup_qualification_receipt(
                receipt,
                allocation_probe_sha256=str(sandbox_probe["probe_sha256"]),
                setup_source_sha256=setup_sha256,
            )
        except PCF1SandboxError as error:
            raise PCF1MergeError("PCF1 calibration setup receipt differs") from error
    if digest != mbpp_allocation_setup_receipts_sha256(receipts):
        raise PCF1MergeError("PCF1 calibration setup receipt hash differs")
    return receipts


def atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise PCF1MergeError(f"refusing existing PCF1 output: {path}")
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
        raise PCF1MergeError(f"refusing existing PCF1 output: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise PCF1MergeError(f"refusing existing PCF1 output: {path}")
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
        raise PCF1MergeError(f"refusing existing PCF1 output: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def summarize_candidates(
    sources: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    arm: str,
    split: str,
) -> dict[str, dict[str, Any]] | None:
    """Score only the generated PCF1 candidate; source rows have no proposal arms."""

    if len(sources) != len(candidates):
        raise PCF1MergeError("PCF1 metric coverage differs")
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for source, candidate in zip(sources, candidates, strict=True):
        tokens = candidate.get("generated_tokens")
        exhausted = candidate.get("max_token_exhausted")
        completion = candidate.get("completion")
        correct = candidate.get("correct")
        label_free_fields = {
            "schema",
            "arm",
            "identity_sha256",
            "task",
            "completion",
            "generated_tokens",
            "max_token_exhausted",
        }
        if (
            candidate.get("schema") != CANDIDATE_SCHEMA
            or candidate.get("arm") != arm
            or candidate.get("identity_sha256") != source.get("identity_sha256")
            or candidate.get("task") != source.get("task")
            or not isinstance(completion, str)
            or isinstance(tokens, bool)
            or not isinstance(tokens, int)
            or tokens < 0
            or not isinstance(exhausted, bool)
        ):
            raise PCF1MergeError("PCF1 candidate content differs")
        if split == "confirmation":
            if set(candidate) != label_free_fields:
                raise PCF1MergeError("PCF1 confirmation candidate exposes assessment")
        elif not isinstance(correct, bool):
            raise PCF1MergeError("PCF1 calibration candidate lacks assessment")
        if split == "confirmation":
            continue
        for domain in ("overall", str(source["task"])):
            buckets[domain]["total"] += 1
            buckets[domain]["generated_correct"] += int(correct)
    if split == "confirmation":
        return None
    if set(buckets) != {"overall", "math500", "bbh_logic", "mbpp"}:
        raise PCF1MergeError("PCF1 metric domain coverage differs")
    return {domain: dict(counter) for domain, counter in sorted(buckets.items())}


def merge(args: argparse.Namespace) -> dict[str, Any]:
    shard_sandbox_probes = list(getattr(args, "shard_sandbox_probes", []))
    if len(args.shard_reports) != len(args.shard_candidates):
        raise PCF1MergeError("PCF1 shard report/candidate arity differs")
    if args.arm not in ARMS or args.split not in SPLITS:
        raise PCF1MergeError("PCF1 arm/split differs")
    if (
        args.split == "calibration"
        and len(shard_sandbox_probes) != len(args.shard_reports)
    ) or (args.split == "confirmation" and shard_sandbox_probes):
        raise PCF1MergeError("PCF1 shard sandbox-probe arity differs")
    shard_root = explicit_shard_root(args.shard_root)
    args.shard_reports = [
        explicit_shard_file(path, shard_root, f"shard {index} report")
        for index, path in enumerate(args.shard_reports)
    ]
    args.shard_candidates = [
        explicit_shard_file(path, shard_root, f"shard {index} candidates")
        for index, path in enumerate(args.shard_candidates)
    ]
    shard_sandbox_probes = [
        explicit_shard_file(path, shard_root, f"shard {index} sandbox probe")
        for index, path in enumerate(shard_sandbox_probes)
    ]
    input_inodes: set[tuple[int, int]] = set()
    for path in (*args.shard_reports, *args.shard_candidates, *shard_sandbox_probes):
        identity = (path.stat().st_dev, path.stat().st_ino)
        if identity in input_inodes:
            raise PCF1MergeError("PCF1 shard input identity is duplicated")
        input_inodes.add(identity)
    for path in (
        args.data,
        args.data_report,
        args.candidates_output,
        args.report,
        *args.shard_reports,
        *args.shard_candidates,
        *shard_sandbox_probes,
    ):
        reject_sealed_path(path)
    if any(
        path.exists() or path.is_symlink()
        for path in (args.candidates_output, args.report)
    ):
        raise PCF1MergeError("PCF1 merged evaluation output already exists")
    if len(args.shard_reports) != len(args.shard_candidates):
        raise PCF1MergeError("PCF1 shard report/candidate arity differs")
    if args.arm not in ARMS or args.split not in SPLITS:
        raise PCF1MergeError("PCF1 arm/split differs")
    if (
        args.split == "calibration"
        and len(shard_sandbox_probes) != len(args.shard_reports)
    ) or (args.split == "confirmation" and shard_sandbox_probes):
        raise PCF1MergeError("PCF1 shard sandbox-probe arity differs")
    data_report = json.loads(args.data_report.read_text(encoding="utf-8"))
    expected = data_report.get("outputs", {}).get(args.split, {})
    if (
        data_report.get("schema") != DATA_REPORT_SCHEMA
        or data_report.get("status") != "complete"
        or data_report.get("sealed_access") != {"holdout": 0, "product": 0, "public": 0}
        or Path(str(expected.get("path", ""))).resolve() != args.data.resolve()
        or expected.get("sha256") != sha256_file(args.data)
    ):
        raise PCF1MergeError("PCF1 data receipt differs")
    source_rows = load_rows(args.data, args.split)
    candidates_by_identity: dict[str, dict[str, Any]] = {}
    ranges: list[tuple[int, int]] = []
    input_receipts: list[dict[str, Any]] = []
    setup_receipt_shards: list[dict[str, Any]] = []
    shard_indices: set[int] = set()
    common: dict[str, Any] | None = None
    elapsed = peak = prompt = generated = exhausted = empty = sandbox_executions = 0
    capability_policy_rejections = 0
    for position, (report_path, candidates_path) in enumerate(
        zip(args.shard_reports, args.shard_candidates, strict=True)
    ):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        sandbox_probe_path = (
            shard_sandbox_probes[position] if args.split == "calibration" else None
        )
        sandbox_probe = sandbox_probe_sha256 = None
        if sandbox_probe_path is not None:
            sandbox_probe, sandbox_probe_sha256 = load_sandbox_probe(sandbox_probe_path)
        if (
            report.get("schema") != REPORT_SCHEMA
            or report.get("status") != "complete"
            or report.get("arm") != args.arm
            or report.get("split") != args.split
            or report.get("metrics") is not None
            or report.get("assessment_mode")
            != (
                "calibration_immediate"
                if args.split == "calibration"
                else "confirmation_deferred"
            )
            or report.get("assessor_board_access_count") != 0
            or report.get("sealed_access") != {"holdout": 0, "product": 0, "public": 0}
        ):
            raise PCF1MergeError("PCF1 shard report differs")
        adapter_metadata = report.get("adapter_metadata")
        try:
            adapter_metadata_sha256 = hashlib.sha256(
                json.dumps(
                    adapter_metadata, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
        except (TypeError, ValueError) as error:
            raise PCF1MergeError("PCF1 shard adapter metadata differs") from error
        trainable_parameters = report.get("trainable_parameters")
        trainable_name_sha256 = report.get("trainable_parameter_name_sha256")
        if (
            not isinstance(adapter_metadata, dict)
            or adapter_metadata.get("arm") != "baseline"
            or adapter_metadata.get("model_revision")
            != "81eaece1948f3875421d9a45bc55487d10e2d894"
            or adapter_metadata.get("model_loader") != "multimodal"
            or adapter_metadata.get("lora_layers") != 4
            or adapter_metadata.get("lora_rank") != 8
            or adapter_metadata.get("lora_alpha") != 16
            or adapter_metadata.get("lora_scope") != "token_mixer"
            or adapter_metadata.get("trainable_parameters") != trainable_parameters
            or adapter_metadata.get("trainable_parameter_name_sha256")
            != trainable_name_sha256
            or report.get("adapter_metadata_sha256") != adapter_metadata_sha256
            or isinstance(trainable_parameters, bool)
            or not isinstance(trainable_parameters, int)
            or trainable_parameters <= 0
            or not isinstance(trainable_name_sha256, str)
            or len(trainable_name_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in trainable_name_sha256
            )
            or report.get("lora_layer_indices") != EXPECTED_LORA_LAYER_INDICES
            or report.get("code_sandbox_config_sha256") != SANDBOX_CONFIG_SHA256
            or report.get("code_sandbox_binary_sha256") != BWRAP_SHA256
            or (
                args.split == "calibration"
                and (
                    report.get("code_sandbox_status") != "passed"
                    or report.get("code_sandbox_probe_passed") is not True
                    or not isinstance(report.get("code_sandbox_probe_sha256"), str)
                    or len(report["code_sandbox_probe_sha256"]) != 64
                    or not isinstance(
                        report.get("code_sandbox_probe_result_sha256"), str
                    )
                    or len(report["code_sandbox_probe_result_sha256"]) != 64
                    or not isinstance(report.get("sandbox_receipt_sha256"), str)
                    or len(report["sandbox_receipt_sha256"]) != 64
                    or report.get("code_sandbox_probe_sha256")
                    != report.get("sandbox_receipt_sha256")
                )
            )
            or (
                args.split == "confirmation"
                and (
                    report.get("code_sandbox_status")
                    != "not_applicable_no_code_scoring"
                    or report.get("code_sandbox_probe_passed") is not None
                    or report.get("code_sandbox_probe_sha256") is not None
                    or report.get("code_sandbox_probe_result_sha256") is not None
                    or report.get("sandbox_receipt_sha256") is not None
                )
            )
            or report.get("environment_verified") is not True
            or not isinstance(report.get("environment_receipt_sha256"), str)
            or len(report["environment_receipt_sha256"]) != 64
            or not isinstance(report.get("environment_tree_sha256"), str)
            or len(report["environment_tree_sha256"]) != 64
            or (
                args.split == "calibration"
                and (
                    sandbox_probe_sha256 != report.get("code_sandbox_probe_sha256")
                    or sandbox_probe_sha256 != report.get("sandbox_receipt_sha256")
                    or sandbox_probe.get("probe_sha256")
                    != report.get("code_sandbox_probe_result_sha256")
                )
            )
        ):
            raise PCF1MergeError("PCF1 shard adapter trainables differ")
        shard_common = {
            key: report.get(key)
            for key in (
                "model_root",
                "model_revision",
                "model_loader",
                "adapter_checkpoint_sha256",
                "adapter_metadata_sha256",
                "trainable_parameters",
                "trainable_parameter_name_sha256",
                "lora_layer_indices",
                "code_sandbox_config_sha256",
                "code_sandbox_binary_sha256",
                "code_sandbox_probe_sha256",
                "code_sandbox_probe_result_sha256",
                "sandbox_receipt_sha256",
                "code_sandbox_status",
                "code_sandbox_probe_passed",
                "environment_verified",
                "environment_receipt_sha256",
                "environment_tree_sha256",
                "data_sha256",
                "data_report_sha256",
                "generation_mode",
                "max_new_tokens",
                "seed",
                "batch_size",
                "shard_count",
                "full_row_count",
                "assessment_mode",
                "assessor_board_access_count",
                "runtime_fields",
            )
        }
        if common is None:
            common = shard_common
        elif shard_common != common:
            raise PCF1MergeError("PCF1 shard settings differ")
        start, end = report.get("row_start"), report.get("row_end")
        shard_index = report.get("shard_index")
        shard_count = report.get("shard_count")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start < end <= len(source_rows)
            or isinstance(shard_index, bool)
            or not isinstance(shard_index, int)
            or isinstance(shard_count, bool)
            or not isinstance(shard_count, int)
            or not 0 <= shard_index < shard_count
            or shard_index in shard_indices
        ):
            raise PCF1MergeError("PCF1 shard range differs")
        setup_receipts = validate_shard_setup_receipts(
            report, source_rows[start:end], args.split, sandbox_probe
        )
        if args.split == "calibration":
            setup_receipt_shards.append(
                {
                    "shard_index": shard_index,
                    "row_start": start,
                    "row_end": end,
                    "receipts": setup_receipts,
                    "receipt_count": len(setup_receipts),
                    "receipts_sha256": mbpp_allocation_setup_receipts_sha256(
                        setup_receipts
                    ),
                }
            )
        shard_indices.add(shard_index)
        ranges.append((start, end))
        reject_sealed_path(candidates_path)
        if Path(
            str(report.get("candidates_output", ""))
        ).resolve() != candidates_path.resolve() or sha256_file(
            candidates_path
        ) != report.get(
            "candidates_sha256"
        ):
            raise PCF1MergeError("PCF1 candidate shard hash differs")
        candidates = load_jsonl(candidates_path)
        if len(candidates) != end - start:
            raise PCF1MergeError("PCF1 candidate shard cardinality differs")
        summarize_candidates(source_rows[start:end], candidates, args.arm, args.split)
        for source, candidate in zip(source_rows[start:end], candidates, strict=True):
            identity = str(source["identity_sha256"])
            if (
                candidate.get("schema") != CANDIDATE_SCHEMA
                or candidate.get("arm") != args.arm
                or candidate.get("identity_sha256") != identity
                or candidate.get("task") != source.get("task")
                or identity in candidates_by_identity
            ):
                raise PCF1MergeError("PCF1 candidate/source binding differs")
            candidates_by_identity[identity] = candidate
        counters = report.get("counters", {})
        if (
            counters.get("rows") != len(candidates)
            or isinstance(counters.get("prompt_tokens"), bool)
            or not isinstance(counters.get("prompt_tokens"), int)
            or counters.get("prompt_tokens", 0) <= 0
            or counters.get("generated_tokens")
            != sum(int(candidate["generated_tokens"]) for candidate in candidates)
            or counters.get("max_token_exhausted")
            != sum(int(candidate["max_token_exhausted"]) for candidate in candidates)
            or counters.get("empty_completions")
            != sum(
                int(not str(candidate["completion"]).strip())
                for candidate in candidates
            )
            or isinstance(counters.get("sandbox_executions"), bool)
            or not isinstance(counters.get("sandbox_executions"), int)
            or counters.get("sandbox_executions", -1) < 0
            or isinstance(counters.get("capability_policy_rejections"), bool)
            or not isinstance(counters.get("capability_policy_rejections"), int)
            or counters.get("capability_policy_rejections", -1) < 0
            or (
                args.split == "confirmation" and counters.get("sandbox_executions") != 0
            )
            or (
                args.split == "confirmation"
                and counters.get("capability_policy_rejections") != 0
            )
            or (
                args.split == "calibration"
                and counters.get("sandbox_executions", 0) <= 0
            )
        ):
            raise PCF1MergeError("PCF1 shard counters differ")
        elapsed_seconds = report.get("elapsed_seconds")
        peak_bytes = report.get("peak_gpu_memory_bytes")
        if (
            isinstance(elapsed_seconds, bool)
            or not isinstance(elapsed_seconds, (int, float))
            or not math.isfinite(float(elapsed_seconds))
            or elapsed_seconds < 0
            or isinstance(peak_bytes, bool)
            or not isinstance(peak_bytes, int)
            or peak_bytes < 0
        ):
            raise PCF1MergeError("PCF1 shard resource accounting differs")
        prompt += int(counters["prompt_tokens"])
        generated += int(counters.get("generated_tokens", 0))
        exhausted += int(counters.get("max_token_exhausted", 0))
        empty += int(counters.get("empty_completions", 0))
        sandbox_executions += int(counters.get("sandbox_executions", 0))
        capability_policy_rejections += int(
            counters.get("capability_policy_rejections", 0)
        )
        elapsed += float(elapsed_seconds)
        peak = max(peak, int(peak_bytes))
        input_receipts.append(
            {
                "report": str(report_path.resolve()),
                "report_sha256": sha256_file(report_path),
                "candidates": str(candidates_path.resolve()),
                "candidates_sha256": report["candidates_sha256"],
                "row_start": start,
                "row_end": end,
                "shard_index": shard_index,
                "sandbox_probe": (
                    str(sandbox_probe_path.resolve())
                    if sandbox_probe_path is not None
                    else None
                ),
                "sandbox_probe_sha256": sandbox_probe_sha256,
                "mbpp_allocation_setup_receipts_sha256": report.get(
                    "mbpp_allocation_setup_receipts_sha256"
                ),
            }
        )
    if common is None:
        raise PCF1MergeError("PCF1 has no shard reports")
    if (
        len(args.shard_reports) != common.get("shard_count")
        or len(shard_indices) != common.get("shard_count")
        or common.get("full_row_count") != len(source_rows)
        or common.get("data_sha256") != sha256_file(args.data)
        or common.get("data_report_sha256") != sha256_file(args.data_report)
    ):
        raise PCF1MergeError("PCF1 shard custody binding differs")
    if sorted(ranges) != [
        (start, end)
        for start, end in zip(
            [0, *[end for _, end in sorted(ranges)[:-1]]],
            [end for _, end in sorted(ranges)],
            strict=True,
        )
    ]:
        raise PCF1MergeError("PCF1 shard ranges are not contiguous")
    if sorted(ranges)[-1][1] != len(source_rows):
        raise PCF1MergeError("PCF1 shard coverage is incomplete")
    ordered = [
        candidates_by_identity[str(row["identity_sha256"])] for row in source_rows
    ]
    candidates_sha256 = atomic_lines(args.candidates_output, ordered)
    metrics = summarize_candidates(source_rows, ordered, args.arm, args.split)
    setup_receipt_shards.sort(key=lambda receipt: receipt["shard_index"])
    setup_receipt_shards_sha256 = (
        hashlib.sha256(
            b"".join(
                (
                    json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
                ).encode()
                for receipt in setup_receipt_shards
            )
        ).hexdigest()
        if args.split == "calibration"
        else None
    )
    report = {
        "schema": MERGED_REPORT_SCHEMA,
        "status": "complete",
        "arm": args.arm,
        "split": args.split,
        **common,
        "data": str(args.data.resolve()),
        "data_report": str(args.data_report.resolve()),
        "candidates_output": str(args.candidates_output.resolve()),
        "candidates_sha256": candidates_sha256,
        "metrics": metrics,
        "counters": {
            "rows": len(ordered),
            "prompt_tokens": prompt,
            "generated_tokens": generated,
            "max_token_exhausted": exhausted,
            "empty_completions": empty,
            "sandbox_executions": sandbox_executions,
            "capability_policy_rejections": capability_policy_rejections,
        },
        "aggregate_gpu_seconds": elapsed,
        "aggregate_wall_seconds": elapsed,
        "aggregate_prompt_tokens": prompt,
        "maximum_peak_gpu_memory_bytes": peak,
        "inputs": input_receipts,
        "shard_sandbox_probe_sha256s": [
            receipt["sandbox_probe_sha256"]
            for receipt in input_receipts
            if receipt["sandbox_probe_sha256"] is not None
        ],
        "mbpp_allocation_setup_status": (
            "passed"
            if args.split == "calibration"
            else "not_applicable_no_code_scoring"
        ),
        "mbpp_allocation_setup_receipt_shards": setup_receipt_shards,
        "mbpp_allocation_setup_receipt_count": sum(
            receipt["receipt_count"] for receipt in setup_receipt_shards
        ),
        "mbpp_allocation_setup_receipt_shards_sha256": (setup_receipt_shards_sha256),
        "exact_identity_coverage": len(ordered) == len(source_rows),
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    atomic_json(args.report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
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
    parser.add_argument(
        "--shard-sandbox-probe",
        dest="shard_sandbox_probes",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument("--candidates-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = merge(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
