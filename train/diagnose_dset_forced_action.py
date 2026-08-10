#!/usr/bin/env python3
"""Force DSET's action prefix to localize gate versus value-generation failure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time

import torch

from eval_dset1_span_edit import evaluate_completion, load_pairs
from hf_product_reasoning_eval import (
    _generate_completions,
    _generation_stop_token_ids,
    _load_model,
    _render_prompt,
)
from train_dset1_span_edit import sha256_file


SCHEMA = "shohin-dset-forced-action-attribution-v1"


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def run(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    if args.output.exists() or not 0 <= args.shard_index < args.shard_count:
        raise RuntimeError("forced-action output exists or shard differs")
    pairs = load_pairs(args.data, args.data_report)
    rows = [
        row
        for pair in pairs
        for row in pair
        if row["pair_member"] == "fault" and row["corruption_family"] == "choice_final"
    ]
    selected = [row for index, row in enumerate(rows) if index % args.shard_count == args.shard_index]
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, metadata, loader = _load_model(args.model_root, args.checkpoint, "auto")
    if (
        metadata.get("architecture") != "shohin-shared-post-mlp-revision-v1"
        or metadata.get("dset1_arm") != "aligned"
    ):
        raise RuntimeError("forced-action checkpoint differs")
    stop_ids = _generation_stop_token_ids(tokenizer)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    results = []
    for row in selected:
        prefix = "<REPLACE_LAST>\n"
        rendered = _render_prompt(tokenizer, row["question"], True, False) + prefix
        completions, usages = _generate_completions(
            model, tokenizer, [rendered], True, "greedy", 31, stop_ids
        )
        completion = prefix + completions[0]
        metrics = evaluate_completion(row, completion)
        results.append(
            {
                "identity_sha256": row["identity_sha256"],
                "pair_identity_sha256": row["pair_identity_sha256"],
                "completion": completion,
                "generated_tokens": usages[0][0],
                "max_token_exhausted": usages[0][1],
                **metrics,
            }
        )
        print(
            f"[forced-action] shard={args.shard_index} rows={len(results)}/{len(selected)}",
            flush=True,
        )
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "diagnostic_only": True,
        "holdout_used": False,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "row_count": len(results),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "data_sha256": sha256_file(args.data),
        "model_loader": loader,
        "script_exact": sum(row["script_exact"] for row in results),
        "execution_correct": sum(row["execution_correct"] for row in results),
        "execution_errors": sum(row["execution_error"] is not None for row in results),
        "max_token_exhausted": sum(row["max_token_exhausted"] for row in results),
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026080921)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"script_exact": report["script_exact"], "execution_correct": report["execution_correct"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
