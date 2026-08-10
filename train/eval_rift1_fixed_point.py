#!/usr/bin/env python3
"""Evaluate recurrent fixed-point editing with one frozen DSET host."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import time

import torch

from dset1_edit_transducer import DSET1Error, execute_script, parse_script
from eval_dset1_span_edit import load_pairs
from hf_product_reasoning_eval import (
    _generate_completions,
    _generation_stop_token_ids,
    _load_model,
    _render_prompt,
)
from train_dset1_span_edit import sha256_file


SCHEMA = "shohin-rift1-fixed-point-evaluation-v1"
FRET_SCHEMA = "shohin-fret1-always-rewrite-evaluation-v1"


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_fret(paths: list[Path], arm: str) -> tuple[dict[str, dict], list[dict]]:
    if len(paths) != 8:
        raise RuntimeError("RIFT1 FRET shard count differs")
    rows = {}
    receipts = []
    shards = set()
    for path in paths:
        report = json.loads(path.read_text())
        if (
            report.get("schema") != FRET_SCHEMA
            or report.get("status") != "complete"
            or report.get("holdout_used") is not False
            or report.get("arm") != arm
            or int(report.get("shard_count", -1)) != 8
        ):
            raise RuntimeError("RIFT1 FRET report differs")
        shard = int(report["shard_index"])
        if shard in shards:
            raise RuntimeError("RIFT1 duplicate FRET shard")
        shards.add(shard)
        receipts.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
        for row in report["results"]:
            identity = str(row["identity_sha256"])
            if identity in rows:
                raise RuntimeError("RIFT1 duplicate FRET identity")
            rows[identity] = row
    if shards != set(range(8)) or len(rows) != 1908:
        raise RuntimeError("RIFT1 FRET coverage differs")
    return rows, receipts


def replace_draft(question: str, old: str, new: str) -> str:
    if question.count(old) != 1:
        raise RuntimeError("RIFT1 source/draft boundary differs")
    return question.replace(old, new, 1)


def execute_commit(candidate: str, completion: str) -> tuple[str, str | None, str | None]:
    try:
        script = parse_script(completion)
        return execute_script(candidate, script), script.action, None
    except DSET1Error as exc:
        return candidate, None, str(exc)


def run(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    if args.output.exists() or not 0 <= args.shard_index < args.shard_count:
        raise RuntimeError("RIFT1 output exists or shard differs")
    pairs = load_pairs(args.data, args.data_report)
    fret, fret_receipts = load_fret(args.fret_shards, args.arm)
    selected_pairs = [
        pair for index, pair in enumerate(pairs) if index % args.shard_count == args.shard_index
    ]
    if not selected_pairs:
        raise RuntimeError("RIFT1 shard is empty")
    rows = [row for pair in selected_pairs for row in pair]
    if any(str(row["identity_sha256"]) not in fret for row in rows):
        raise RuntimeError("RIFT1 proposal identity is missing")
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, metadata, loader = _load_model(args.model_root, args.checkpoint, "auto")
    if (
        metadata.get("architecture") != "shohin-shared-post-mlp-revision-v1"
        or metadata.get("dset1_arm") != args.arm
    ):
        raise RuntimeError("RIFT1 checkpoint differs")
    stop_ids = _generation_stop_token_ids(tokenizer)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    results = []
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset : offset + args.batch_size]
        candidates = []
        rendered = []
        for row in batch:
            proposal = fret[str(row["identity_sha256"])]
            candidate = str(proposal.get("executed_trajectory") or row["draft"])
            candidates.append(candidate)
            question = replace_draft(str(row["question"]), str(row["draft"]), candidate)
            rendered.append(_render_prompt(tokenizer, question, True, False))
        completions, usages = _generate_completions(
            model, tokenizer, rendered, True, "greedy", args.max_new_tokens, stop_ids
        )
        for row, candidate, completion, (used, exhausted) in zip(
            batch, candidates, completions, usages, strict=True
        ):
            final, action, error = execute_commit(candidate, completion)
            proposal = fret[str(row["identity_sha256"])]
            results.append(
                {
                    "identity_sha256": row["identity_sha256"],
                    "pair_identity_sha256": row["pair_identity_sha256"],
                    "pair_member": row["pair_member"],
                    "corruption_family": row["corruption_family"],
                    "proposal_correct": candidate == row["final_response"],
                    "proposal_changed": candidate != row["draft"],
                    "proposal_execution_error": proposal.get("execution_error"),
                    "commit_completion": completion,
                    "commit_action": action,
                    "commit_valid": error is None,
                    "commit_error": error,
                    "final_correct": final == row["final_response"],
                    "final_trajectory": final,
                    "generated_tokens": used,
                    "max_token_exhausted": exhausted,
                }
            )
        print(
            f"[rift1] arm={args.arm} shard={args.shard_index} "
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
            for metric in ("proposal_correct", "commit_valid", "final_correct"):
                groups[kind][name][metric] += int(result[metric])
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "holdout_used": False,
        "arm": args.arm,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "row_count": len(results),
        "pair_count": len(selected_pairs),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "data_sha256": sha256_file(args.data),
        "data_report_sha256": sha256_file(args.data_report),
        "fret_receipts": fret_receipts,
        "model_loader": loader,
        "proposal_correct": sum(row["proposal_correct"] for row in results),
        "commit_valid": sum(row["commit_valid"] for row in results),
        "final_correct": sum(row["final_correct"] for row in results),
        "commit_errors": sum(row["commit_error"] is not None for row in results),
        "max_token_exhausted": sum(row["max_token_exhausted"] for row in results),
        "groups": {
            kind: {name: dict(values) for name, values in groups.items()}
            for kind, groups in groups.items()
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
    parser.add_argument("--fret-shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=["aligned", "swapped", "hidden"], required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026081011)
    report = run(parser.parse_args())
    print(json.dumps({"arm": report["arm"], "final_correct": report["final_correct"], "row_count": report["row_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
