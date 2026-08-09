#!/usr/bin/env python3
"""Generate and verify fault-first revision traces from frozen train requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from build_vfr1_teacher_requests import REPORT_SCHEMA as REQUEST_REPORT_SCHEMA
from build_vfr1_teacher_requests import REQUEST_SCHEMA
from hf_product_reasoning_eval import (
    _generate_completions,
    _generation_stop_token_ids,
    _load_model,
    _render_prompt,
)
from hf_product_reasoning_rollouts import score_completion
from build_vcr1_revision_data import sha256_file


TRACE_SCHEMA = "shohin-vfr1-generated-trace-v1"
REPORT_SCHEMA = "shohin-vfr1-generated-trace-report-v1"
FAULT_OPEN = "<FAULT>"
FAULT_CLOSE = "</FAULT>"
REVISION_OPEN = "<REVISION>"
REVISION_CLOSE = "</REVISION>"


class VFR1GenerationError(RuntimeError):
    """VFR1 trace generation or parsing differs from the frozen contract."""


def parse_trace(completion: str) -> tuple[str, str]:
    starts = [
        completion.count(FAULT_OPEN),
        completion.count(FAULT_CLOSE),
        completion.count(REVISION_OPEN),
        completion.count(REVISION_CLOSE),
    ]
    if starts != [1, 1, 1, 1]:
        raise VFR1GenerationError("trace tag cardinality differs")
    fault_start = completion.index(FAULT_OPEN) + len(FAULT_OPEN)
    fault_end = completion.index(FAULT_CLOSE)
    revision_start = completion.index(REVISION_OPEN) + len(REVISION_OPEN)
    revision_end = completion.index(REVISION_CLOSE)
    if not (fault_start < fault_end < revision_start < revision_end):
        raise VFR1GenerationError("trace tag order differs")
    if completion[revision_end + len(REVISION_CLOSE) :].strip():
        raise VFR1GenerationError("trace has text after revision close")
    fault = completion[fault_start:fault_end].strip()
    revision = completion[revision_start:revision_end].strip()
    if not fault or not revision:
        raise VFR1GenerationError("trace block is empty")
    if r"\boxed" in fault:
        raise VFR1GenerationError("fault block quotes a boxed answer")
    return fault, revision


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise VFR1GenerationError(f"refusing existing VFR1 output: {path}")
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
        raise VFR1GenerationError(f"refusing existing VFR1 report: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_requests(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            identity = str(row.get("identity_sha256", ""))
            if row.get("schema") != REQUEST_SCHEMA or len(identity) != 64:
                raise VFR1GenerationError("teacher request schema differs")
            if identity in identities:
                raise VFR1GenerationError("teacher request identity is duplicated")
            if row.get("runtime_fields") != ["teacher_prompt"]:
                raise VFR1GenerationError("teacher request runtime fields differ")
            identities.add(identity)
            rows.append(row)
    if not rows:
        raise VFR1GenerationError("teacher request file is empty")
    return rows


def shard_bounds(total: int, index: int, count: int) -> tuple[int, int]:
    if total <= 0 or count <= 0 or not 0 <= index < count:
        raise VFR1GenerationError("teacher shard geometry differs")
    start = total * index // count
    end = total * (index + 1) // count
    if start >= end:
        raise VFR1GenerationError("teacher shard is empty")
    return start, end


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    if args.output.exists() or args.report.exists():
        raise VFR1GenerationError("VFR1 trace output already exists")
    request_report = json.loads(args.request_report.read_text(encoding="utf-8"))
    if (
        request_report.get("schema") != REQUEST_REPORT_SCHEMA
        or request_report.get("status") != "complete"
        or Path(request_report.get("output", "")).resolve() != args.requests.resolve()
        or request_report.get("output_sha256") != sha256_file(args.requests)
    ):
        raise VFR1GenerationError("teacher request report binding differs")
    all_rows = load_requests(args.requests)
    start, end = shard_bounds(len(all_rows), args.shard_index, args.shard_count)
    rows = all_rows[start:end]
    if args.max_rows:
        rows = rows[: args.max_rows]

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, adapter_metadata, model_loader = _load_model(
        args.model_root, args.adapter_checkpoint, args.model_loader
    )
    stop_ids = _generation_stop_token_ids(tokenizer)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()

    generated: list[dict[str, Any]] = []
    parsed = verified = exhausted = generated_tokens = 0
    reference_leaks = 0
    started = time.monotonic()
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset : offset + args.batch_size]
        rendered = [
            _render_prompt(tokenizer, str(row["teacher_prompt"]), True, False)
            for row in batch
        ]
        completions, usage = _generate_completions(
            model,
            tokenizer,
            rendered,
            True,
            "greedy",
            args.max_new_tokens,
            stop_ids,
        )
        for row, completion, (tokens, was_exhausted) in zip(
            batch, completions, usage, strict=True
        ):
            fault = revision = ""
            parse_error = None
            try:
                fault, revision = parse_trace(completion)
                parsed += 1
            except VFR1GenerationError as exc:
                parse_error = str(exc)
            score = (
                score_completion(
                    row["assessor"], revision, code_timeout=args.code_timeout
                )
                if revision
                else {"correct": False}
            )
            verified += int(bool(score.get("correct")))
            leak = bool(
                re.search(
                    r"\b(?:provided|given|verified)\s+reference\b|"
                    r"\breference\s+(?:answer|solution|outcome)\b",
                    fault,
                    flags=re.IGNORECASE,
                )
            )
            reference_leaks += int(leak)
            generated.append(
                {
                    "schema": TRACE_SCHEMA,
                    "identity_sha256": row["identity_sha256"],
                    "task": row["task"],
                    "outcome_class": row["outcome_class"],
                    "target_kind": row["target_kind"],
                    "fault": fault,
                    "revision": revision,
                    "raw_completion": completion,
                    "parse_error": parse_error,
                    "reference_leak": leak,
                    "generated_tokens": tokens,
                    "max_token_exhausted": was_exhausted,
                    "score": score,
                }
            )
            exhausted += int(was_exhausted)
            generated_tokens += tokens
        print(f"[vfr1-teacher] {min(offset + len(batch), len(rows))}/{len(rows)}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    output_sha256 = _atomic_lines(args.output, generated)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "model_root": str(args.model_source_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": model_loader,
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()),
        "adapter_checkpoint_sha256": sha256_file(args.adapter_checkpoint),
        "adapter_metadata": adapter_metadata,
        "requests": str(args.requests.resolve()),
        "requests_sha256": sha256_file(args.requests),
        "request_report": str(args.request_report.resolve()),
        "request_report_sha256": sha256_file(args.request_report),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "row_start": start,
        "row_end": end,
        "rows": len(rows),
        "parsed": parsed,
        "verified": verified,
        "reference_leaks": reference_leaks,
        "max_token_exhausted": exhausted,
        "generated_tokens": generated_tokens,
        "elapsed_seconds": elapsed,
        "generated_tokens_per_second": generated_tokens / elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "output": str(args.output.resolve()),
        "output_sha256": output_sha256,
    }
    _atomic_json(args.report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-loader", choices=("auto", "causal", "multimodal"), default="auto")
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--request-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--code-timeout", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=2026080911)
    report = run(parser.parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
