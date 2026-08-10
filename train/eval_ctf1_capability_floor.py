#!/usr/bin/env python3
"""Evaluate an unadapted capability-floor owner through the CTE1 interface."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import time

import torch

from draft_transaction_compiler import (
    DraftTransactionError,
    compile_draft_transactions,
    reset_state_reads,
)
from eval_cte1_development import (
    CTE1EvaluationError,
    atomic_json,
    candidate_fraction,
    load_microcode,
    load_rows,
    sha256_file,
    source_shuffle,
)
from hf_product_reasoning_eval import (
    _generate_completions,
    _generation_stop_token_ids,
    _load_model,
)
from hf_product_reasoning_train import PRODUCT_SYSTEM_PROMPT, render_reasoning_messages
from learned_arithmetic_microcode import LearnedArithmeticError
from typed_microcode_graph import TypedMicrocodeGraphError, execute_learned


SCHEMA = "shohin-ctf1-capability-floor-evaluation-v1"


def run(args: argparse.Namespace) -> dict[str, object]:
    from transformers import AutoTokenizer

    if args.control not in {"normal", "source_shuffled"}:
        raise CTE1EvaluationError("control differs")
    if args.max_new_tokens != 512 or args.seed != 2026081053:
        raise CTE1EvaluationError("decoding geometry differs")
    rows = load_rows(args.data, args.expected_data_sha256)
    donors = source_shuffle(rows) if args.control == "source_shuffled" else {}
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_root, trust_remote_code=True, use_fast=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, metadata, loader = _load_model(args.model_root, None, "auto")
    if metadata is not None:
        raise CTE1EvaluationError("unadapted owner unexpectedly has metadata")
    microcode = load_microcode(args.lam_checkpoint)
    stop_ids = _generation_stop_token_ids(tokenizer)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    counts: Counter[str] = Counter()
    details = []
    generated_tokens = 0
    exhausted = 0
    started = time.time()
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset : offset + args.batch_size]
        source_rows = [donors.get(str(row["identity_sha256"]), row) for row in batch]
        prompts = [
            render_reasoning_messages(
                tokenizer,
                [
                    {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
                    {"role": "user", "content": str(source["question"])},
                ],
                enable_thinking=False,
            )
            for source in source_rows
        ]
        completions, usage = _generate_completions(
            model,
            tokenizer,
            prompts,
            False,
            "greedy",
            args.max_new_tokens,
            stop_ids,
        )
        for row, source, completion, (tokens, row_exhausted) in zip(
            batch, source_rows, completions, usage, strict=True
        ):
            expected = Fraction(str(row["gold_answer"]))
            generated_tokens += tokens
            exhausted += int(row_exhausted)
            counts["rows"] += 1
            detail: dict[str, object] = {
                "identity_sha256": row["identity_sha256"],
                "source_identity_sha256": source["identity_sha256"],
                "completion": completion,
                "generated_tokens": tokens,
                "exhausted": row_exhausted,
            }
            try:
                graph, receipt = compile_draft_transactions(
                    str(source["original_question"]), completion
                )
                if receipt.rejected:
                    raise DraftTransactionError("generated transaction was rejected")
            except DraftTransactionError as error:
                counts["compile_invalid"] += 1
                counts[f"compile_invalid:{error}"] += 1
                detail["compile_error"] = str(error)
                detail["correct"] = False
                details.append(detail)
                continue
            counts["compiled_rows"] += 1
            counts["transactions"] += receipt.accepted
            counts["state_reads"] += receipt.state_reads
            counts["source_reads"] += receipt.source_reads
            counts["literal_reads"] += receipt.literal_reads
            counts["linked_rows"] += int(receipt.state_reads > 0)
            detail.update(
                {
                    "transactions": receipt.accepted,
                    "state_reads": receipt.state_reads,
                    "source_reads": receipt.source_reads,
                    "literal_reads": receipt.literal_reads,
                }
            )
            try:
                prediction = candidate_fraction(execute_learned(microcode, graph))
            except (LearnedArithmeticError, TypedMicrocodeGraphError, ZeroDivisionError) as error:
                counts["execution_invalid"] += 1
                detail["execution_error"] = type(error).__name__
                detail["correct"] = False
                details.append(detail)
                continue
            correct = prediction == expected
            counts["executable_rows"] += 1
            counts["correct"] += int(correct)
            detail["prediction"] = str(prediction)
            detail["correct"] = correct
            if args.control == "normal":
                if receipt.state_reads:
                    counts["linked_correct"] += int(correct)
                    try:
                        reset_prediction = candidate_fraction(
                            execute_learned(microcode, reset_state_reads(graph))
                        )
                        reset_correct = reset_prediction == expected
                    except (
                        LearnedArithmeticError,
                        TypedMicrocodeGraphError,
                        ZeroDivisionError,
                    ):
                        reset_correct = False
                    counts["state_reset_linked_correct"] += int(reset_correct)
                    detail["state_reset_correct"] = reset_correct
                try:
                    opcode_prediction = candidate_fraction(
                        execute_learned(microcode, graph, intervention="opcode_permuted")
                    )
                    opcode_correct = opcode_prediction == expected
                except (
                    LearnedArithmeticError,
                    TypedMicrocodeGraphError,
                    ZeroDivisionError,
                ):
                    opcode_correct = False
                counts["opcode_permuted_correct"] += int(opcode_correct)
                detail["opcode_permuted_correct"] = opcode_correct
            details.append(detail)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "holdout_used": False,
        "public_test_opened": False,
        "control": args.control,
        "model_root": str(args.model_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": loader,
        "adapter_checkpoint": None,
        "development_data_sha256": args.expected_data_sha256,
        "lam_checkpoint_sha256": sha256_file(args.lam_checkpoint),
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "generated_tokens": generated_tokens,
        "exhausted": exhausted,
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "counts": dict(sorted(counts.items())),
        "details": details,
    }
    atomic_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", choices=("normal", "source_shuffled"), required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--lam-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
