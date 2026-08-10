#!/usr/bin/env python3
"""Generate immutable model-owned DTMC1 training drafts in parallel shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from hf_product_reasoning_eval import (
    _generate_completions,
    _generation_stop_token_ids,
    _load_model,
    extract_gsm8k,
    match_gsm8k,
)
from hf_product_reasoning_train import PRODUCT_SYSTEM_PROMPT, render_reasoning_messages

SCHEMA = "shohin-dtmc1-model-owned-draft-shard-v1"


class DTMC1DraftError(ValueError):
    """Frozen DTMC1 draft generation custody differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path, expected_sha256: str) -> list[dict[str, object]]:
    if sha256_file(path) != expected_sha256:
        raise DTMC1DraftError("training data SHA-256 differs")
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != 6333 or len({row["identity_sha256"] for row in rows}) != 6333:
        raise DTMC1DraftError("training population differs")
    return rows


def atomic_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    if path.exists():
        raise DTMC1DraftError("refusing existing draft shard")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, object]:
    from transformers import AutoTokenizer

    if (
        args.shard_count != 8
        or not 0 <= args.shard_index < args.shard_count
        or args.batch_size != 8
        or args.max_new_tokens != 512
        or args.seed != 2026081053
    ):
        raise DTMC1DraftError("draft generation geometry differs")
    if sha256_file(args.checkpoint) != args.expected_checkpoint_sha256:
        raise DTMC1DraftError("direct checkpoint SHA-256 differs")
    rows = load_rows(args.data, args.expected_data_sha256)
    selected = [
        row
        for index, row in enumerate(rows)
        if index % args.shard_count == args.shard_index
    ]
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, metadata, loader = _load_model(args.model_root, args.checkpoint, "auto")
    if (
        metadata is None
        or metadata.get("update") != 1024
        or metadata.get("model_revision") != args.model_revision
        or metadata.get("data_sha256") != args.expected_data_sha256
    ):
        raise DTMC1DraftError("direct checkpoint metadata differs")
    stop_ids = _generation_stop_token_ids(tokenizer)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    started = time.time()
    outputs = []
    generated_tokens = 0
    exhausted = 0
    correct = 0
    for offset in range(0, len(selected), args.batch_size):
        batch = selected[offset : offset + args.batch_size]
        prompts = [
            render_reasoning_messages(
                tokenizer,
                [
                    {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
                    {"role": "user", "content": str(row["original_question"])},
                ],
                enable_thinking=False,
            )
            for row in batch
        ]
        completions, usage = _generate_completions(
            model,
            tokenizer,
            prompts,
            True,
            "greedy",
            args.max_new_tokens,
            stop_ids,
        )
        for row, completion, (tokens, row_exhausted) in zip(
            batch, completions, usage, strict=True
        ):
            prediction = extract_gsm8k(completion)
            row_correct = match_gsm8k(prediction, str(row["gold_answer"]))
            generated_tokens += tokens
            exhausted += int(row_exhausted)
            correct += int(row_correct)
            outputs.append(
                {
                    "schema": SCHEMA,
                    "identity_sha256": row["identity_sha256"],
                    "original_question": row["original_question"],
                    "gold_answer": row["gold_answer"],
                    "register_depth": row["register_depth"],
                    "draft": completion,
                    "draft_prediction": prediction,
                    "draft_correct": row_correct,
                    "generated_tokens": tokens,
                    "exhausted": row_exhausted,
                    "shard_index": args.shard_index,
                    "shard_count": args.shard_count,
                }
            )
    atomic_jsonl(args.output, outputs)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "holdout_used": False,
        "public_test_opened": False,
        "model_revision": args.model_revision,
        "model_loader": loader,
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "data_sha256": args.expected_data_sha256,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "rows": len(outputs),
        "generated_tokens": generated_tokens,
        "exhausted": exhausted,
        "draft_correct": correct,
        "elapsed_seconds": time.time() - started,
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
    }
    report_path = args.output.with_suffix(".report.json")
    temporary = report_path.with_name(f".{report_path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, report_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026081053)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
