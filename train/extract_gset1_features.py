#!/usr/bin/env python3
"""Extract immutable source/draft decision states from the frozen DSET model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch

from hf_product_reasoning_eval import _load_model, _render_prompt
from train_dset1_span_edit import load_pairs, sha256_file
from ttr1_revision import tokenize_with_draft_mask


SCHEMA = "shohin-gset1-feature-shard-v1"


def atomic_torch(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    if args.output.exists() or args.control not in {"aligned", "hidden"}:
        raise RuntimeError("GSET1 feature output exists or control differs")
    pairs, _ = load_pairs(args.data, args.data_report)
    selected = [pair for index, pair in enumerate(pairs) if index % args.shard_count == args.shard_index]
    if not selected:
        raise RuntimeError("GSET1 feature shard is empty")
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, metadata, loader = _load_model(args.model_root, args.checkpoint, "auto")
    if (
        metadata.get("architecture") != "shohin-shared-post-mlp-revision-v1"
        or metadata.get("dset1_arm") != "aligned"
    ):
        raise RuntimeError("GSET1 source checkpoint differs")
    model.eval()
    rows = [row for pair in selected for row in pair]
    records = []
    states = []
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset : offset + args.batch_size]
        token_rows = []
        mask_rows = []
        for row in batch:
            rendered = _render_prompt(tokenizer, str(row["question"]), True, False)
            tokens, draft_mask, _ = tokenize_with_draft_mask(tokenizer, rendered)
            if len(tokens) > args.max_sequence_length:
                raise RuntimeError("GSET1 complete prompt overflows")
            token_rows.append(tokens)
            mask_rows.append(draft_mask if args.control == "hidden" else [1] * len(tokens))
            records.append(
                {
                    "identity_sha256": row["identity_sha256"],
                    "pair_identity_sha256": row["pair_identity_sha256"],
                    "pair_member": row["pair_member"],
                    "corruption_family": row["corruption_family"],
                    "label": int(row["pair_member"] == "fault"),
                    "prompt_tokens": len(tokens),
                    "draft_tokens": sum(1 - value for value in draft_mask),
                }
            )
        width = max(map(len, token_rows))
        input_ids = torch.full(
            (len(batch), width), tokenizer.pad_token_id, dtype=torch.long, device="cuda:0"
        )
        attention = torch.zeros((len(batch), width), dtype=torch.long, device="cuda:0")
        lengths = []
        for index, (tokens, mask) in enumerate(zip(token_rows, mask_rows, strict=True)):
            length = len(tokens)
            lengths.append(length)
            input_ids[index, :length] = torch.tensor(tokens, dtype=torch.long, device="cuda:0")
            attention[index, :length] = torch.tensor(mask, dtype=torch.long, device="cuda:0")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            embeddings = model.text_model.embed_tokens(input_ids)
            outputs = model.text_model(
                inputs_embeds=embeddings, attention_mask=attention, use_cache=False
            )
            hidden = outputs.last_hidden_state
            selected_hidden = torch.stack(
                [hidden[index, length - 1] for index, length in enumerate(lengths)]
            )
        states.append(selected_hidden.to(dtype=torch.bfloat16, device="cpu"))
        print(
            f"[gset1-features] control={args.control} shard={args.shard_index} "
            f"rows={min(offset + len(batch), len(rows))}/{len(rows)}",
            flush=True,
        )
        del input_ids, attention, embeddings, outputs, hidden, selected_hidden
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "control": args.control,
        "holdout_used": False,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "row_count": len(records),
        "hidden_size": int(states[0].shape[-1]),
        "model_loader": loader,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "data_sha256": sha256_file(args.data),
        "data_report_sha256": sha256_file(args.data_report),
        "records": records,
        "states": torch.cat(states, dim=0),
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch(args.output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", choices=["aligned", "hidden"], required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count or args.batch_size <= 0:
        parser.error("GSET1 shard geometry differs")
    report = run(args)
    print(json.dumps({"control": report["control"], "rows": report["row_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
