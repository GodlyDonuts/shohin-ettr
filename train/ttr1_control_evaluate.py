#!/usr/bin/env python3
"""Evaluate frozen matched controls for transferable temporal revision."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
from typing import Any

from hf_idr1_evaluate_reviser import load_rows, shard_bounds
from hf_product_reasoning_eval import (
    TASKS,
    _completion_usage,
    _generate_adapter,
    _generate_completions,
    _generation_arguments,
    _generation_stop_token_ids,
    _load_model,
    _render_prompt,
    _task_prompt,
)
from hf_product_reasoning_rollouts import score_completion
from hf_vcr1_evaluate_reviser import (
    _atomic_json,
    _atomic_lines,
    sha256_file,
    summarize,
)
from ttr1_revision import tokenize_with_draft_mask


REPORT_SCHEMA = "shohin-ttr1-control-evaluation-v1"
CONTROLS = (
    "unchanged_second_pass",
    "self_refinement",
    "long_single_generation",
    "best_of_two",
    "independent_commitment",
)


class TTR1ControlError(RuntimeError):
    """A matched TTR1 control differs from its frozen contract."""


def self_refinement_prompt(row: dict[str, Any]) -> str:
    source = _task_prompt(str(row["task"]), row["assessor"])
    draft = str(row["internal_draft"]["completion"])
    format_instruction = (
        "Return only executable Python code, without Markdown fences."
        if row["task"] == "mbpp"
        else "Return a complete solution and put the exact final answer in \\boxed{}."
    )
    return (
        "Review the attempted solution below for mistakes, then solve the original "
        "problem correctly. Do not only critique the attempt.\n\n"
        f"Original problem:\n{source}\n\nAttempt:\n{draft}\n\n{format_instruction}"
    )


def _masked_completions(
    model: Any,
    tokenizer: Any,
    rendered: list[str],
    max_new_tokens: int,
    stop_token_ids: list[int],
) -> tuple[list[str], list[tuple[int, bool]], int]:
    import torch

    token_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    masked_tokens = 0
    for prompt in rendered:
        token_ids, attention, _ = tokenize_with_draft_mask(tokenizer, prompt)
        token_rows.append(token_ids)
        mask_rows.append(attention)
        masked_tokens += attention.count(0)
    width = max(map(len, token_rows))
    pad_id = int(tokenizer.pad_token_id)
    input_ids = torch.full(
        (len(token_rows), width), pad_id, device="cuda:0", dtype=torch.long
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, (tokens, attention) in enumerate(
        zip(token_rows, mask_rows, strict=True)
    ):
        offset = width - len(tokens)
        input_ids[index, offset:] = torch.tensor(tokens, device="cuda:0")
        attention_mask[index, offset:] = torch.tensor(attention, device="cuda:0")
    arguments = _generation_arguments("greedy", max_new_tokens)
    arguments["eos_token_id"] = (
        stop_token_ids[0] if len(stop_token_ids) == 1 else stop_token_ids
    )
    with torch.inference_mode():
        output = _generate_adapter(
            model,
            {"input_ids": input_ids, "attention_mask": attention_mask},
            arguments,
            pad_id,
        )
    completions = tokenizer.batch_decode(output, skip_special_tokens=True)
    usage = [
        _completion_usage(tokens.tolist(), stop_token_ids, max_new_tokens)
        for tokens in output
    ]
    return completions, usage, masked_tokens


def _select_best_of_two(
    row: dict[str, Any], completions: list[str]
) -> tuple[str, bool]:
    if len(completions) != 2:
        raise TTR1ControlError("best-of-two candidate count differs")
    task_name = str(row["task"])
    if TASKS[task_name]["kind"] == "code":
        predictions = [completion.strip() for completion in completions]
    else:
        predictions = [TASKS[task_name]["extract"](completion) for completion in completions]
    return completions[0], predictions[0] == predictions[1]


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    data_report = json.loads(args.data_report.read_text(encoding="utf-8"))
    expected = data_report.get("outputs", {}).get(args.split, {})
    if (
        data_report.get("schema") != "shohin-idr1-revision-data-report-v1"
        or Path(expected.get("path", "")).resolve() != args.data.resolve()
        or expected.get("sha256") != sha256_file(args.data)
    ):
        raise TTR1ControlError("TTR1 control data receipt differs")
    all_rows = load_rows(args.data, args.split)
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
    if args.control == "independent_commitment" and not bool(
        adapter_metadata and adapter_metadata.get("mask_internal_draft")
    ):
        raise TTR1ControlError("independent checkpoint lacks draft masking")
    if args.control != "independent_commitment" and bool(
        adapter_metadata and adapter_metadata.get("mask_internal_draft")
    ):
        raise TTR1ControlError("standard control unexpectedly uses draft masking")
    stop_ids = _generation_stop_token_ids(tokenizer)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()

    results: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    started = time.monotonic()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        if args.control in ("unchanged_second_pass", "independent_commitment"):
            questions = [str(row["question"]) for row in batch]
            max_new_tokens = 768
        elif args.control == "self_refinement":
            questions = [self_refinement_prompt(row) for row in batch]
            max_new_tokens = 768
        else:
            questions = [
                _task_prompt(str(row["task"]), row["assessor"]) for row in batch
            ]
            max_new_tokens = 1536 if args.control == "long_single_generation" else 768
        rendered = [
            _render_prompt(tokenizer, question, True, False) for question in questions
        ]
        if args.control == "independent_commitment":
            completions, usage, hidden = _masked_completions(
                model, tokenizer, rendered, max_new_tokens, stop_ids
            )
            counters["masked_draft_tokens"] += hidden
            selected = completions
            selected_usage = usage
        elif args.control == "best_of_two":
            repeated = [prompt for prompt in rendered for _ in range(2)]
            completions, usage = _generate_completions(
                model,
                tokenizer,
                repeated,
                True,
                "qwen-thinking",
                max_new_tokens,
                stop_ids,
            )
            selected = []
            selected_usage = []
            for index, row in enumerate(batch):
                completion, agreement = _select_best_of_two(
                    row, completions[index * 2 : index * 2 + 2]
                )
                selected.append(completion)
                selected_usage.append(usage[index * 2])
                counters["candidate_agreements"] += int(agreement)
            counters["generated_tokens"] += sum(count for count, _ in usage)
        else:
            selected, selected_usage = _generate_completions(
                model,
                tokenizer,
                rendered,
                True,
                "greedy",
                max_new_tokens,
                stop_ids,
            )
        if args.control != "best_of_two":
            counters["generated_tokens"] += sum(count for count, _ in selected_usage)
        for row, completion, (token_count, exhausted) in zip(
            batch, selected, selected_usage, strict=True
        ):
            score = score_completion(row["assessor"], completion)
            results.append(
                {
                    "schema": "shohin-ttr1-control-candidate-v1",
                    "control": args.control,
                    "identity_sha256": row["identity_sha256"],
                    "task": row["task"],
                    "completion": completion,
                    "generated_tokens": token_count,
                    "max_token_exhausted": exhausted,
                    **score,
                }
            )
            counters["correct"] += int(score["correct"])
            counters["rows"] += 1
        processed = min(start + len(batch), len(rows))
        if processed % 32 == 0 or processed == len(rows):
            print(f"[ttr1-{args.control}] {processed}/{len(rows)}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    candidates_sha256 = _atomic_lines(args.candidates_output, results)
    metrics = summarize(rows, results)["metrics"] if args.shard_count == 1 else None
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "control": args.control,
        "split": args.split,
        "model_root": str(args.model_source_root.resolve()),
        "loaded_model_root": str(args.model_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": model_loader,
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()),
        "adapter_checkpoint_sha256": sha256_file(args.adapter_checkpoint),
        "adapter_metadata": adapter_metadata,
        "data": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "data_report_sha256": sha256_file(args.data_report),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "row_start": row_start,
        "row_end": row_end,
        "full_row_count": len(all_rows),
        "batch_size": args.batch_size,
        "max_new_tokens_per_attempt": max_new_tokens,
        "attempts_per_identity": 2 if args.control == "best_of_two" else 1,
        "counters": dict(sorted(counters.items())),
        "elapsed_seconds": elapsed,
        "generated_tokens_per_second": counters["generated_tokens"] / elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "metrics": metrics,
        "candidates_output": str(args.candidates_output.resolve()),
        "candidates_sha256": candidates_sha256,
        "seed": args.seed,
    }
    _atomic_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", choices=CONTROLS, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-loader", choices=("auto", "causal"), default="auto")
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "holdout"), required=True)
    parser.add_argument("--candidates-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026080821)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch size must be positive")
    return args


def main() -> int:
    report = run(parse_args())
    print(json.dumps({"control": report["control"], "metrics": report["metrics"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
