#!/usr/bin/env python3
"""Evaluate DSEO1 action binding and verifier-correct paired repair."""

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

from hf_product_reasoning_eval import (
    _generate_completions,
    _generation_stop_token_ids,
    _load_model,
    _render_prompt,
    extract_short_answer,
    match_gsm8k,
    match_short_answer,
)
from train_dseo1_paired_action import DATA_REPORT_SCHEMA, DATA_SCHEMA, sha256_file


REPORT_SCHEMA = "shohin-dseo1-paired-evaluation-v1"
ACTIONS = {"<KEEP>", "<FIX_FINAL>", "<FIX_STEP>", "<FIX_CODE>", "<REWRITE>"}


class DSEO1EvalError(RuntimeError):
    """The paired evaluator or checkpoint contract differs."""


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_action_completion(completion: str) -> tuple[str | None, str]:
    """Parse only a registered action occupying the first nonempty line."""

    lines = completion.splitlines()
    first_index = next((index for index, line in enumerate(lines) if line.strip()), None)
    if first_index is None:
        return None, ""
    action = lines[first_index].strip()
    if action not in ACTIONS:
        return None, completion.strip()
    return action, "\n".join(lines[first_index + 1 :]).strip()


def answer_correct(row: dict[str, Any], trajectory: str) -> bool:
    prediction = extract_short_answer(trajectory)
    gold = str(row["gold_answer"])
    if row["corruption_family"] == "choice_final":
        return match_short_answer(prediction, gold)
    return match_gsm8k(prediction, gold)


def load_diagnostic(data: Path, report_path: Path) -> list[list[dict[str, Any]]]:
    report = json.loads(report_path.read_text())
    expected = report.get("outputs", {}).get("diagnostic", {})
    if (
        report.get("schema") != DATA_REPORT_SCHEMA
        or report.get("status") != "complete"
        or report.get("holdout_used") is not False
        or Path(str(expected.get("path", ""))).resolve() != data.resolve()
        or expected.get("sha256") != sha256_file(data)
    ):
        raise DSEO1EvalError("DSEO1 diagnostic report differs")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in data.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row.get("schema") != DATA_SCHEMA:
            raise DSEO1EvalError("DSEO1 diagnostic row schema differs")
        grouped[str(row["pair_identity_sha256"])].append(row)
    pairs = []
    for pair_id in sorted(grouped):
        pair = sorted(grouped[pair_id], key=lambda row: row["pair_member"])
        if len(pair) != 2 or {row["pair_member"] for row in pair} != {"clean", "fault"}:
            raise DSEO1EvalError("DSEO1 diagnostic pair differs")
        pairs.append(pair)
    if len(pairs) != int(expected.get("sources", -1)):
        raise DSEO1EvalError("DSEO1 diagnostic source count differs")
    return pairs


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if args.output.exists() or not 0 <= args.shard_index < args.shard_count:
        raise DSEO1EvalError("DSEO1 output exists or shard differs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pairs = load_diagnostic(args.data, args.data_report)
    selected_pairs = [
        pair for index, pair in enumerate(pairs) if index % args.shard_count == args.shard_index
    ]
    if not selected_pairs:
        raise DSEO1EvalError("DSEO1 evaluation shard is empty")
    rows = [row for pair in selected_pairs for row in pair]

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, metadata, resolved_loader = _load_model(
        args.model_root, args.adapter_checkpoint, "causal"
    )
    if metadata.get("architecture") != "shohin-rme1-moe-revision-v1":
        raise DSEO1EvalError("DSEO1 checkpoint architecture differs")
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
        rendered = [
            _render_prompt(tokenizer, str(row["question"]), True, False)
            for row in batch
        ]
        completions, usages = _generate_completions(
            model,
            tokenizer,
            rendered,
            True,
            "greedy",
            args.max_new_tokens,
            stop_ids,
        )
        for row, completion, (used, cap) in zip(batch, completions, usages, strict=True):
            action, trajectory = parse_action_completion(completion)
            is_action_correct = action == row["action"]
            is_answer_correct = answer_correct(row, trajectory)
            generated_tokens += used
            exhausted += int(cap)
            results.append(
                {
                    "identity_sha256": row["identity_sha256"],
                    "pair_identity_sha256": row["pair_identity_sha256"],
                    "pair_member": row["pair_member"],
                    "corruption_family": row["corruption_family"],
                    "gold_action": row["action"],
                    "predicted_action": action,
                    "action_correct": is_action_correct,
                    "gold_answer": row["gold_answer"],
                    "answer_correct": is_answer_correct,
                    "completion": completion,
                    "trajectory": trajectory,
                    "generated_tokens": used,
                    "max_token_exhausted": cap,
                }
            )
        print(
            f"[dseo1-eval] shard={args.shard_index} "
            f"rows={min(offset + len(batch), len(rows))}/{len(rows)}",
            flush=True,
        )
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_pair[result["pair_identity_sha256"]].append(result)
    consistent = sum(
        len(pair) == 2
        and all(row["action_correct"] for row in pair)
        and len({row["predicted_action"] for row in pair}) == 2
        for pair in by_pair.values()
    )
    family = defaultdict(Counter)
    member = defaultdict(Counter)
    for result in results:
        family[result["corruption_family"]]["rows"] += 1
        family[result["corruption_family"]]["action_correct"] += int(
            result["action_correct"]
        )
        family[result["corruption_family"]]["answer_correct"] += int(
            result["answer_correct"]
        )
        member[result["pair_member"]]["rows"] += 1
        member[result["pair_member"]]["action_correct"] += int(
            result["action_correct"]
        )
        member[result["pair_member"]]["answer_correct"] += int(
            result["answer_correct"]
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
        "pair_count": len(selected_pairs),
        "row_count": len(results),
        "action_correct": sum(row["action_correct"] for row in results),
        "action_accuracy": sum(row["action_correct"] for row in results) / len(results),
        "answer_correct": sum(row["answer_correct"] for row in results),
        "answer_accuracy": sum(row["answer_correct"] for row in results) / len(results),
        "counterfactual_consistent_pairs": consistent,
        "counterfactual_consistency": consistent / len(selected_pairs),
        "family_metrics": {name: dict(counts) for name, counts in family.items()},
        "member_metrics": {name: dict(counts) for name, counts in member.items()},
        "max_new_tokens": args.max_new_tokens,
        "generated_tokens": generated_tokens,
        "max_token_exhausted": exhausted,
        "elapsed_seconds": elapsed,
        "generated_tokens_per_second": generated_tokens / elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "routing_receipt": model.routing_receipt(),
        "results": results,
    }
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
    parser.add_argument("--arm", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=2026080915)
    args = parser.parse_args()
    if min(args.shard_count, args.batch_size, args.max_new_tokens) <= 0:
        parser.error("DSEO1 evaluation dimensions differ")
    return args


def main() -> int:
    report = run(parse_args())
    print(
        f"[dseo1-eval] arm={report['arm']} action={report['action_accuracy']:.4f} "
        f"answer={report['answer_accuracy']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
