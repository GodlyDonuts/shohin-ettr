#!/usr/bin/env python3
"""Evaluate a strict fault-first VFR1 revision owner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from build_vcr1_revision_data import sha256_file
from hf_idr1_evaluate_reviser import shard_bounds
from hf_product_reasoning_eval import (
    _generate_completions,
    _generation_stop_token_ids,
    _load_model,
    _render_prompt,
)
from hf_product_reasoning_rollouts import score_completion
from hf_vcr1_evaluate_reviser import _atomic_json, _atomic_lines, summarize as source_summary
from hf_vfr1_generate_traces import VFR1GenerationError, parse_trace


EVAL_SCHEMA = "shohin-vfr1-capability-eval-v1"
DATA_REPORT_SCHEMA = "shohin-vfr1-capability-data-report-v1"
REPORT_SCHEMA = "shohin-vfr1-capability-evaluation-v1"
TASKS = ("math500", "bbh_logic", "mbpp")
FROZEN_FLOORS = {"overall": 603, "math500": 223, "bbh_logic": 349, "mbpp": 17}


class VFR1EvaluationError(RuntimeError):
    """VFR1 evaluation data, parsing, or provenance differs."""


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            identity = str(row.get("identity_sha256", ""))
            if row.get("schema") != EVAL_SCHEMA or len(identity) != 64:
                raise VFR1EvaluationError("VFR1 evaluation schema differs")
            if identity in identities:
                raise VFR1EvaluationError("VFR1 evaluation identity is duplicated")
            if row.get("split") != "development" or row.get("task") not in TASKS:
                raise VFR1EvaluationError("VFR1 development split differs")
            if row.get("runtime_fields") != ["question"]:
                raise VFR1EvaluationError("VFR1 runtime fields differ")
            if row.get("strict_trace_format") is not True:
                raise VFR1EvaluationError("VFR1 strict trace contract differs")
            if not isinstance(row.get("assessor"), dict):
                raise VFR1EvaluationError("VFR1 assessor is missing")
            identities.add(identity)
            rows.append(row)
    if not rows or {str(row["task"]) for row in rows} != set(TASKS):
        raise VFR1EvaluationError("VFR1 task coverage differs")
    return rows


def summarize(
    rows: list[dict[str, Any]], results: list[dict[str, Any]]
) -> dict[str, Any]:
    summary = source_summary(rows, results)
    metrics = summary["metrics"]
    parsed = sum(result.get("parse_error") is None for result in results)
    gates = {
        "overall_at_least_603": metrics["overall"]["generated_correct"] >= 603,
        "math_at_least_223": metrics["math500"]["generated_correct"] >= 223,
        "logic_at_least_349": metrics["bbh_logic"]["generated_correct"] >= 349,
        "code_at_least_17": metrics["mbpp"]["generated_correct"] >= 17,
        "strict_parse_at_least_0_95": parsed / len(results) >= 0.95,
    }
    return {
        "metrics": metrics,
        "parsed": parsed,
        "parse_fraction": parsed / len(results),
        "gates": gates,
        "absolute_gate_pass": all(gates.values()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    if args.report.exists() or args.candidates_output.exists():
        raise VFR1EvaluationError("VFR1 evaluation output already exists")
    data_report = json.loads(args.data_report.read_text(encoding="utf-8"))
    expected = data_report.get("outputs", {}).get("development", {})
    if (
        data_report.get("schema") != DATA_REPORT_SCHEMA
        or data_report.get("status") != "complete"
        or Path(expected.get("path", "")).resolve() != args.data.resolve()
        or expected.get("sha256") != sha256_file(args.data)
        or data_report.get("internal_draft_visible") is not True
        or data_report.get("assessor_fields_visible_to_model") is not False
    ):
        raise VFR1EvaluationError("VFR1 data report binding differs")
    all_rows = load_rows(args.data)
    row_start, row_end = shard_bounds(
        len(all_rows), args.shard_index, args.shard_count, args.batch_size
    )
    rows = all_rows[row_start:row_end]

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

    results: list[dict[str, Any]] = []
    generated_tokens = exhausted = 0
    started = time.monotonic()
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset : offset + args.batch_size]
        rendered = [
            _render_prompt(tokenizer, str(row["question"]), True, False)
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
            except VFR1GenerationError as exc:
                parse_error = str(exc)
            score = (
                score_completion(row["assessor"], revision, code_timeout=args.code_timeout)
                if revision
                else {"correct": False}
            )
            results.append(
                {
                    "schema": "shohin-vfr1-capability-candidate-v1",
                    "identity_sha256": row["identity_sha256"],
                    "task": row["task"],
                    "completion": completion,
                    "fault": fault,
                    "revision": revision,
                    "parse_error": parse_error,
                    "generated_tokens": tokens,
                    "max_token_exhausted": was_exhausted,
                    **score,
                }
            )
            generated_tokens += tokens
            exhausted += int(was_exhausted)
        print(f"[vfr1-eval] {min(offset + len(batch), len(rows))}/{len(rows)}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    candidates_sha256 = _atomic_lines(args.candidates_output, results)
    summary = (
        summarize(rows, results)
        if args.shard_count == 1
        else {
            "metrics": None,
            "parsed": None,
            "parse_fraction": None,
            "gates": None,
            "absolute_gate_pass": False,
        }
    )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "model_root": str(args.model_source_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": model_loader,
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()),
        "adapter_checkpoint_sha256": sha256_file(args.adapter_checkpoint),
        "adapter_metadata": adapter_metadata,
        "data": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "data_report": str(args.data_report.resolve()),
        "data_report_sha256": sha256_file(args.data_report),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "full_row_count": len(all_rows),
        "row_start": row_start,
        "row_end": row_end,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "elapsed_seconds": elapsed,
        "generated_tokens": generated_tokens,
        "generated_tokens_per_second": generated_tokens / elapsed,
        "max_token_exhausted": exhausted,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "candidates_output": str(args.candidates_output.resolve()),
        "candidates_sha256": candidates_sha256,
        **summary,
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
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--candidates-output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--code-timeout", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=2026080913)
    report = run(parser.parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
