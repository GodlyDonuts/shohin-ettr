#!/usr/bin/env python3
"""Evaluate exact DSET1 scripts and their deterministic executed trajectories."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import torch

from dset1_edit_transducer import DSET1Error, KEEP, execute_script, parse_script
from hf_product_reasoning_eval import (
    _generate_completions,
    _generation_stop_token_ids,
    _load_model,
    _render_prompt,
)
from train_dset1_span_edit import DATA_REPORT_SCHEMA, DATA_SCHEMA, sha256_file


REPORT_SCHEMA = "shohin-dset1-span-edit-evaluation-v1"


class DSET1EvalError(RuntimeError):
    """The DSET1 checkpoint, data, or evaluation geometry differs."""


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_pairs(data: Path, report_path: Path) -> list[list[dict[str, Any]]]:
    report = json.loads(report_path.read_text())
    expected = report.get("outputs", {}).get("diagnostic", {})
    if (
        report.get("schema") != DATA_REPORT_SCHEMA
        or report.get("status") != "complete"
        or report.get("holdout_used") is not False
        or int(report.get("max_script_tokens", 0)) != 32
        or Path(str(expected.get("path", ""))).resolve() != data.resolve()
        or expected.get("sha256") != sha256_file(data)
    ):
        raise DSET1EvalError("DSET1 diagnostic report differs")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in data.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row.get("schema") != DATA_SCHEMA:
            raise DSET1EvalError("DSET1 diagnostic row differs")
        grouped[str(row["pair_identity_sha256"])].append(row)
    pairs = []
    for pair_id in sorted(grouped):
        pair = sorted(grouped[pair_id], key=lambda row: row["pair_member"])
        if len(pair) != 2 or {row["pair_member"] for row in pair} != {"clean", "fault"}:
            raise DSET1EvalError("DSET1 diagnostic pair differs")
        pairs.append(pair)
    if len(pairs) != int(expected.get("sources", -1)):
        raise DSET1EvalError("DSET1 diagnostic pair count differs")
    return pairs


def evaluate_completion(row: dict[str, Any], completion: str) -> dict[str, Any]:
    expected = str(row["script"]).strip()
    predicted_action = None
    executed = ""
    error = None
    try:
        script = parse_script(completion)
        predicted_action = script.action
        executed = execute_script(str(row["draft"]), script)
    except DSET1Error as exc:
        error = str(exc)
    return {
        "predicted_action": predicted_action,
        "action_correct": predicted_action == row["action"],
        "script_exact": completion.strip() == expected,
        "execution_correct": bool(executed) and executed == row["final_response"],
        "trajectory_exact": executed == row["final_response"],
        "execution_error": error,
        "executed_trajectory": executed,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if args.output.exists() or not 0 <= args.shard_index < args.shard_count:
        raise DSET1EvalError("DSET1 output exists or shard differs")
    pairs = load_pairs(args.data, args.data_report)
    selected = [pair for index, pair in enumerate(pairs) if index % args.shard_count == args.shard_index]
    if not selected:
        raise DSET1EvalError("DSET1 evaluation shard is empty")
    rows = [row for pair in selected for row in pair]
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, metadata, resolved_loader = _load_model(args.model_root, args.adapter_checkpoint, "causal")
    if metadata.get("architecture") != "shohin-rme1-moe-revision-v1" or metadata.get("dset1_arm") != args.arm:
        raise DSET1EvalError("DSET1 checkpoint metadata differs")
    if hasattr(model, "reset_routing_receipt"):
        model.reset_routing_receipt()
    stop_ids = _generation_stop_token_ids(tokenizer)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    results = []
    generated_tokens = exhausted = 0
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset : offset + args.batch_size]
        rendered = [_render_prompt(tokenizer, str(row["question"]), True, False) for row in batch]
        completions, usages = _generate_completions(
            model, tokenizer, rendered, True, "greedy", args.max_new_tokens, stop_ids
        )
        for row, completion, (used, cap) in zip(batch, completions, usages, strict=True):
            metrics = evaluate_completion(row, completion)
            generated_tokens += used
            exhausted += int(cap)
            results.append(
                {
                    "identity_sha256": row["identity_sha256"],
                    "source_dseo1_identity_sha256": row["source_dseo1_identity_sha256"],
                    "pair_identity_sha256": row["pair_identity_sha256"],
                    "pair_member": row["pair_member"],
                    "corruption_family": row["corruption_family"],
                    "gold_action": row["action"],
                    "gold_script": row["script"],
                    "gold_answer": row["gold_answer"],
                    "completion": completion,
                    "generated_tokens": used,
                    "max_token_exhausted": cap,
                    **metrics,
                }
            )
        print(f"[dset1-eval] shard={args.shard_index} rows={min(offset + len(batch), len(rows))}/{len(rows)}", flush=True)
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    family = defaultdict(Counter)
    member = defaultdict(Counter)
    for result in results:
        by_pair[result["pair_identity_sha256"]].append(result)
        for groups, name in ((family, result["corruption_family"]), (member, result["pair_member"])):
            groups[name]["rows"] += 1
            for key in ("action_correct", "script_exact", "execution_correct", "trajectory_exact"):
                groups[name][key] += int(result[key])
    consistent = sum(
        len(pair) == 2 and all(row["script_exact"] for row in pair) for pair in by_pair.values()
    )
    elapsed = time.monotonic() - started
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "arm": args.arm,
        "model_root": str((args.model_source_root or args.model_root).resolve()),
        "loaded_model_root": str(args.model_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": resolved_loader,
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()),
        "adapter_checkpoint_sha256": sha256_file(args.adapter_checkpoint),
        "adapter_metadata": metadata,
        "data": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "data_report_sha256": sha256_file(args.data_report),
        "holdout_used": False,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "pair_count": len(selected),
        "row_count": len(results),
        "action_correct": sum(row["action_correct"] for row in results),
        "script_exact": sum(row["script_exact"] for row in results),
        "execution_correct": sum(row["execution_correct"] for row in results),
        "trajectory_exact": sum(row["trajectory_exact"] for row in results),
        "counterfactual_consistent_pairs": consistent,
        "execution_errors": Counter(row["execution_error"] for row in results if row["execution_error"]),
        "family_counts": {name: dict(counts) for name, counts in family.items()},
        "member_counts": {name: dict(counts) for name, counts in member.items()},
        "max_new_tokens": args.max_new_tokens,
        "generated_tokens": generated_tokens,
        "max_token_exhausted": exhausted,
        "elapsed_seconds": elapsed,
        "generated_tokens_per_second": generated_tokens / elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "routing_receipt": model.routing_receipt(),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=["aligned", "swapped", "hidden"], required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026080916)
    return parser.parse_args()


def main() -> int:
    report = run(parse_args())
    print(
        f"[dset1-eval] arm={report['arm']} scripts={report['script_exact']}/{report['row_count']} "
        f"execution={report['execution_correct']}/{report['row_count']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
