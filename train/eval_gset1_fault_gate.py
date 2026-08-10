#!/usr/bin/env python3
"""Evaluate the causal GSET1 fault gate driving the frozen DSET generator."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import time

import torch

from dset1_edit_transducer import KEEP, REPLACE_LAST
from eval_dset1_span_edit import evaluate_completion, load_pairs
from gset1_fault_gate import load_gate_checkpoint
from hf_product_reasoning_eval import (
    _generate_completions,
    _generation_stop_token_ids,
    _load_model,
    _render_prompt,
)
from train_dset1_span_edit import sha256_file
from ttr1_revision import tokenize_with_draft_mask


SCHEMA = "shohin-gset1-causal-gate-evaluation-v1"


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def prompt_states(model, tokenizer, rows: list[dict], control: str, batch_size: int) -> torch.Tensor:
    states = []
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        token_rows, mask_rows = [], []
        for row in batch:
            rendered = _render_prompt(tokenizer, str(row["question"]), True, False)
            tokens, draft_mask, _ = tokenize_with_draft_mask(tokenizer, rendered)
            token_rows.append(tokens)
            mask_rows.append(draft_mask if control == "hidden" else [1] * len(tokens))
        width = max(map(len, token_rows))
        ids = torch.full(
            (len(batch), width), tokenizer.pad_token_id, dtype=torch.long, device="cuda:0"
        )
        attention = torch.zeros((len(batch), width), dtype=torch.long, device="cuda:0")
        lengths = []
        for index, (tokens, mask) in enumerate(zip(token_rows, mask_rows, strict=True)):
            lengths.append(len(tokens))
            ids[index, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device="cuda:0")
            attention[index, : len(tokens)] = torch.tensor(mask, dtype=torch.long, device="cuda:0")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.text_model(
                inputs_embeds=model.text_model.embed_tokens(ids),
                attention_mask=attention,
                use_cache=False,
            )
            states.append(
                torch.stack(
                    [outputs.last_hidden_state[index, length - 1] for index, length in enumerate(lengths)]
                ).float().cpu()
            )
    return torch.cat(states, dim=0)


def run(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    if args.output.exists() or not 0 <= args.shard_index < args.shard_count:
        raise RuntimeError("GSET1 output exists or shard differs")
    pairs = load_pairs(args.data, args.data_report)
    selected = [pair for index, pair in enumerate(pairs) if index % args.shard_count == args.shard_index]
    rows = [row for pair in selected for row in pair]
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, metadata, loader = _load_model(args.model_root, args.dset_checkpoint, "auto")
    if (
        metadata.get("architecture") != "shohin-shared-post-mlp-revision-v1"
        or metadata.get("dset1_arm") != "aligned"
    ):
        raise RuntimeError("GSET1 DSET checkpoint differs")
    gate, gate_metadata = load_gate_checkpoint(args.gate_checkpoint, device="cuda:0")
    if (
        gate_metadata.get("architecture") != "shohin-gset1-paired-fault-gate-v1"
        or gate_metadata.get("base_dset_checkpoint_sha256") != sha256_file(args.dset_checkpoint)
        or gate_metadata.get("arm") != args.arm
    ):
        raise RuntimeError("GSET1 gate checkpoint differs")
    control = str(gate_metadata["feature_control"])
    if control == "hidden":
        model.draft_control = "draft_unavailable"
    stop_ids = _generation_stop_token_ids(tokenizer)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    hidden = prompt_states(model, tokenizer, rows, control, args.batch_size)
    with torch.inference_mode():
        predicted = gate(hidden.to("cuda:0")).argmax(dim=-1).cpu().tolist()
    if args.intervention == "gold":
        actions = [int(row["pair_member"] == "fault") for row in rows]
    elif args.intervention == "inverted":
        actions = [1 - int(value) for value in predicted]
    else:
        actions = [int(value) for value in predicted]
    completions = [f"{KEEP}\n" if action == 0 else None for action in actions]
    generated = [0] * len(rows)
    exhausted = [False] * len(rows)
    replace_indices = [index for index, action in enumerate(actions) if action == 1]
    for offset in range(0, len(replace_indices), args.batch_size):
        indices = replace_indices[offset : offset + args.batch_size]
        prefix = f"{REPLACE_LAST}\n"
        rendered = [
            _render_prompt(tokenizer, str(rows[index]["question"]), True, False) + prefix
            for index in indices
        ]
        suffixes, usages = _generate_completions(
            model, tokenizer, rendered, True, "greedy", args.max_new_tokens - 1, stop_ids
        )
        for index, suffix, usage in zip(indices, suffixes, usages, strict=True):
            completions[index] = prefix + suffix
            generated[index], exhausted[index] = usage
    results = []
    family = defaultdict(Counter)
    member = defaultdict(Counter)
    by_pair = defaultdict(list)
    for index, (row, completion) in enumerate(zip(rows, completions, strict=True)):
        assert completion is not None
        metrics = evaluate_completion(row, completion)
        result = {
            "identity_sha256": row["identity_sha256"],
            "pair_identity_sha256": row["pair_identity_sha256"],
            "pair_member": row["pair_member"],
            "corruption_family": row["corruption_family"],
            "gold_action": row["action"],
            "gate_action": REPLACE_LAST if predicted[index] else KEEP,
            "executed_action": REPLACE_LAST if actions[index] else KEEP,
            "gate_action_correct": int(predicted[index]) == int(row["pair_member"] == "fault"),
            "completion": completion,
            "generated_tokens": generated[index],
            "max_token_exhausted": exhausted[index],
            **metrics,
        }
        results.append(result)
        by_pair[result["pair_identity_sha256"]].append(result)
        for grouping, key in ((family, result["corruption_family"]), (member, result["pair_member"])):
            grouping[key]["rows"] += 1
            for metric in ("gate_action_correct", "script_exact", "execution_correct"):
                grouping[key][metric] += int(result[metric])
    consistent = sum(
        len(pair) == 2 and all(item["gate_action_correct"] for item in pair)
        for pair in by_pair.values()
    )
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "arm": args.arm,
        "intervention": args.intervention,
        "feature_control": control,
        "holdout_used": False,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "pair_count": len(selected),
        "row_count": len(results),
        "dset_checkpoint_sha256": sha256_file(args.dset_checkpoint),
        "gate_checkpoint_sha256": sha256_file(args.gate_checkpoint),
        "data_sha256": sha256_file(args.data),
        "model_loader": loader,
        "gate_action_correct": sum(item["gate_action_correct"] for item in results),
        "script_exact": sum(item["script_exact"] for item in results),
        "execution_correct": sum(item["execution_correct"] for item in results),
        "counterfactual_consistent_pairs": consistent,
        "execution_errors": dict(Counter(item["execution_error"] for item in results if item["execution_error"])),
        "family_counts": {key: dict(value) for key, value in family.items()},
        "member_counts": {key: dict(value) for key, value in member.items()},
        "generated_tokens": sum(generated),
        "max_token_exhausted": sum(exhausted),
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
    parser.add_argument("--dset-checkpoint", type=Path, required=True)
    parser.add_argument("--gate-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=["aligned", "swapped", "hidden"], required=True)
    parser.add_argument("--intervention", choices=["predicted", "gold", "inverted"], default="predicted")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026080923)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"arm": report["arm"], "execution": report["execution_correct"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
