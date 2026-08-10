#!/usr/bin/env python3
"""Evaluate the frozen DSET host as an always-rewrite edit transducer."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import time

import torch

from dset1_edit_transducer import DSET1Error, REPLACE_LAST, execute_script, parse_script
from eval_dset1_span_edit import load_pairs
from hf_product_reasoning_eval import (
    _generate_completions,
    _generation_stop_token_ids,
    _load_model,
    _render_prompt,
)
from train_dset1_span_edit import sha256_file


SCHEMA = "shohin-fret1-always-rewrite-evaluation-v1"


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def score_rewrite(row: dict, completion: str) -> dict:
    error = None
    executed = ""
    old = new = None
    pointer_exact = replacement_exact = False
    copy_characters = 0
    try:
        script = parse_script(completion)
        if script.action != REPLACE_LAST or script.old is None or script.new is None:
            raise DSET1Error("FRET1 action differs")
        old, new = script.old, script.new
        start, end = map(int, row["changed_character_span"])
        pointer_exact = row["draft"][start:end] == old and row["draft"].rfind(old) == start
        replacement_exact = new == row["gold_answer"]
        executed = execute_script(str(row["draft"]), script)
        copy_characters = max(0, len(str(row["draft"])) - len(old))
    except DSET1Error as exc:
        error = str(exc)
    return {
        "old_surface": old,
        "new_surface": new,
        "pointer_exact": pointer_exact,
        "replacement_exact": replacement_exact,
        "program_exact": pointer_exact and replacement_exact,
        "execution_correct": bool(executed) and executed == row["final_response"],
        "executed_trajectory": executed,
        "execution_error": error,
        "copy_characters": copy_characters,
        "draft_characters": len(str(row["draft"])),
    }


def run(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    if args.output.exists() or not 0 <= args.shard_index < args.shard_count:
        raise RuntimeError("FRET1 output exists or shard differs")
    pairs = load_pairs(args.data, args.data_report)
    selected_pairs = [
        pair for index, pair in enumerate(pairs) if index % args.shard_count == args.shard_index
    ]
    if not selected_pairs:
        raise RuntimeError("FRET1 shard is empty")
    rows = [row for pair in selected_pairs for row in pair]
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, metadata, loader = _load_model(args.model_root, args.checkpoint, "auto")
    if (
        metadata.get("architecture") != "shohin-shared-post-mlp-revision-v1"
        or metadata.get("dset1_arm") != args.arm
    ):
        raise RuntimeError("FRET1 checkpoint differs")
    stop_ids = _generation_stop_token_ids(tokenizer)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    results = []
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset : offset + args.batch_size]
        prefix = "<REPLACE_LAST>\n"
        rendered = [
            _render_prompt(tokenizer, str(row["question"]), True, False) + prefix
            for row in batch
        ]
        completions, usages = _generate_completions(
            model, tokenizer, rendered, True, "greedy", args.max_new_tokens, stop_ids
        )
        for row, suffix, (used, exhausted) in zip(batch, completions, usages, strict=True):
            completion = prefix + suffix
            results.append(
                {
                    "identity_sha256": row["identity_sha256"],
                    "pair_identity_sha256": row["pair_identity_sha256"],
                    "pair_member": row["pair_member"],
                    "corruption_family": row["corruption_family"],
                    "completion": completion,
                    "generated_tokens": used,
                    "max_token_exhausted": exhausted,
                    **score_rewrite(row, completion),
                }
            )
        print(
            f"[fret1] arm={args.arm} shard={args.shard_index} "
            f"rows={min(offset + len(batch), len(rows))}/{len(rows)}",
            flush=True,
        )
    groups = {"family": defaultdict(Counter), "member": defaultdict(Counter)}
    for result in results:
        for kind, name in (
            ("family", result["corruption_family"]),
            ("member", result["pair_member"]),
        ):
            groups[kind][name]["rows"] += 1
            for metric in ("pointer_exact", "replacement_exact", "program_exact", "execution_correct"):
                groups[kind][name][metric] += int(result[metric])
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "diagnostic_only": True,
        "holdout_used": False,
        "arm": args.arm,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "row_count": len(results),
        "pair_count": len(selected_pairs),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "data_sha256": sha256_file(args.data),
        "data_report_sha256": sha256_file(args.data_report),
        "model_loader": loader,
        "pointer_exact": sum(row["pointer_exact"] for row in results),
        "replacement_exact": sum(row["replacement_exact"] for row in results),
        "program_exact": sum(row["program_exact"] for row in results),
        "execution_correct": sum(row["execution_correct"] for row in results),
        "execution_errors": sum(row["execution_error"] is not None for row in results),
        "max_token_exhausted": sum(row["max_token_exhausted"] for row in results),
        "copy_characters": sum(row["copy_characters"] for row in results),
        "draft_characters": sum(row["draft_characters"] for row in results),
        "groups": {
            kind: {name: dict(counts) for name, counts in values.items()}
            for kind, values in groups.items()
        },
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
    parser.add_argument("--arm", choices=["aligned", "swapped", "hidden"], required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=31)
    parser.add_argument("--seed", type=int, default=2026081010)
    report = run(parser.parse_args())
    print(
        json.dumps(
            {
                "arm": report["arm"],
                "program_exact": report["program_exact"],
                "execution_correct": report["execution_correct"],
                "row_count": report["row_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
