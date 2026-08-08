#!/usr/bin/env python3
"""Generate one sharded same-family revision lineage on the product board."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from build_aqc1_product import REPORT_SCHEMA, REVISION_SCHEMA, sha256_file
from hf_product_reasoning_eval import (
    _generate_completions,
    _generation_stop_token_ids,
    _load_model,
    _render_prompt,
)
from hf_product_reasoning_rollouts import score_completion
from hf_vcr1_evaluate_reviser import _atomic_json, _atomic_lines

RESULT_SCHEMA = "shohin-aqc1-product-revision-candidate-v1"
EVAL_REPORT_SCHEMA = "shohin-aqc1-product-revision-report-v1"


def run(args: argparse.Namespace) -> dict:
    import torch
    from transformers import AutoTokenizer

    receipt = json.loads(args.data_report.read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != REPORT_SCHEMA
        or receipt.get("stage") != "revision"
        or receipt.get("output_sha256") != sha256_file(args.data)
    ):
        raise RuntimeError("AQC1 product revision receipt differs")
    all_rows = [
        json.loads(line)
        for line in args.data.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if any(row.get("schema") != REVISION_SCHEMA for row in all_rows):
        raise RuntimeError("AQC1 product revision row differs")
    if args.skip < 0 or args.count <= 0 or args.skip + args.count > len(all_rows):
        raise RuntimeError("AQC1 product revision shard differs")
    rows = all_rows[args.skip : args.skip + args.count]

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, adapter_metadata, model_loader = _load_model(
        args.model_root, args.adapter_checkpoint, "multimodal"
    )
    stops = _generation_stop_token_ids(tokenizer)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    results = []
    generated = exhausted = 0
    started = time.monotonic()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        rendered = [
            _render_prompt(tokenizer, row["question"], True, False) for row in batch
        ]
        completions, usage = _generate_completions(
            model, tokenizer, rendered, True, "greedy", args.max_new_tokens, stops
        )
        for row, completion, (count, hit_limit) in zip(
            batch, completions, usage, strict=True
        ):
            score = score_completion(
                row["assessor"], completion, code_timeout=args.code_timeout
            )
            results.append(
                {
                    "schema": RESULT_SCHEMA,
                    "identity_sha256": row["identity_sha256"],
                    "task": row["task"],
                    "completion": completion,
                    "generated_tokens": count,
                    "max_token_exhausted": hit_limit,
                    **score,
                }
            )
            generated += count
            exhausted += int(hit_limit)
        if len(results) % 16 == 0 or len(results) == len(rows):
            print(f"[aqc1-product-revision] {len(results)}/{len(rows)}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    candidates_hash = _atomic_lines(args.candidates_output, results)
    report = {
        "schema": EVAL_REPORT_SCHEMA,
        "status": "complete",
        "model_revision": args.model_revision,
        "model_loader": model_loader,
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()),
        "adapter_checkpoint_sha256": sha256_file(args.adapter_checkpoint),
        "adapter_metadata": adapter_metadata,
        "data": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "data_report": str(args.data_report.resolve()),
        "data_report_sha256": sha256_file(args.data_report),
        "skip": args.skip,
        "count": args.count,
        "seed": args.seed,
        "generated_tokens": generated,
        "generated_tokens_per_second": generated / elapsed,
        "max_token_exhausted": exhausted,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "candidates_output": str(args.candidates_output.resolve()),
        "candidates_sha256": candidates_hash,
    }
    _atomic_json(args.report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument("--candidates-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--skip", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--code-timeout", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=2026080816)
    report = run(parser.parse_args())
    print(json.dumps({"status": report["status"], "count": report["count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
