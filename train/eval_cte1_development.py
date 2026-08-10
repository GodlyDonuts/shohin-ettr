#!/usr/bin/env python3
"""Evaluate one frozen CTE1 canonical-transaction checkpoint."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from draft_transaction_compiler import (
    DraftTransactionError,
    compile_draft_transactions,
    reset_state_reads,
)
from hf_product_reasoning_eval import (
    _generate_completions,
    _generation_stop_token_ids,
    _load_model,
)
from hf_product_reasoning_train import PRODUCT_SYSTEM_PROMPT, render_reasoning_messages
from learned_arithmetic_microcode import LearnedArithmeticError, LearnedDigitMicrocode
from typed_microcode_graph import TypedMicrocodeGraphError, execute_learned


SCHEMA = "shohin-cte1-development-evaluation-v1"
DATA_SCHEMA = "shohin-cte1-canonical-transaction-data-v1"


class CTE1EvaluationError(ValueError):
    """Frozen CTE1 evaluation custody or execution differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path, expected_sha256: str) -> list[dict[str, object]]:
    if sha256_file(path) != expected_sha256:
        raise CTE1EvaluationError("development data SHA-256 differs")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if (
        len(rows) != 666
        or len({row["identity_sha256"] for row in rows}) != 666
        or any(row.get("schema") != DATA_SCHEMA for row in rows)
    ):
        raise CTE1EvaluationError("development population differs")
    return rows


def source_shuffle(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[int(row["register_depth"])].append(row)
    mapping = {}
    for depth, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: str(row["identity_sha256"]))
        if len(ordered) < 2:
            raise CTE1EvaluationError(f"source-shuffle singleton depth {depth}")
        for target, donor in zip(ordered, ordered[1:] + ordered[:1], strict=True):
            mapping[str(target["identity_sha256"])] = donor
    return mapping


def load_microcode(path: Path) -> LearnedDigitMicrocode:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "shohin-lam1-learned-arithmetic-microcode-v1":
        raise CTE1EvaluationError("LAM checkpoint differs")
    model = LearnedDigitMicrocode()
    model.load_state_dict(payload["state_dict"], strict=True)
    if model.transition_exact() != (1400, 1400):
        raise CTE1EvaluationError("LAM transition receipt differs")
    model.freeze_discrete()
    return model


def candidate_fraction(value) -> Fraction:
    numerator = int("".join(str(digit) for digit in reversed(value.numerator)))
    denominator = int("".join(str(digit) for digit in reversed(value.denominator)))
    result = Fraction(numerator, denominator)
    return -result if value.negative else result


def load_checkpoint_receipt(path: Path) -> tuple[int, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "shohin-hf-product-reasoning-checkpoint-v1":
        raise CTE1EvaluationError("fit checkpoint schema differs")
    update = payload.get("update")
    metadata = payload.get("metadata")
    if not isinstance(update, int) or not isinstance(metadata, dict):
        raise CTE1EvaluationError("fit checkpoint receipt is incomplete")
    return update, metadata


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise CTE1EvaluationError("refusing existing output")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


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
    checkpoint_update, checkpoint_metadata = load_checkpoint_receipt(args.checkpoint)
    model, metadata, loader = _load_model(args.model_root, args.checkpoint, "auto")
    restored_metadata = (
        {key: value for key, value in metadata.items() if key != "update"}
        if metadata is not None
        else None
    )
    if (
        metadata is None
        or restored_metadata != checkpoint_metadata
        or metadata.get("update") != checkpoint_update
        or checkpoint_update != 1024
        or metadata.get("model_revision") != args.model_revision
        or metadata.get("data_sha256") != args.expected_train_sha256
    ):
        raise CTE1EvaluationError("fit checkpoint metadata differs")
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
        source_rows = [
            donors.get(str(row["identity_sha256"]), row) for row in batch
        ]
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
            True,
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
            except (
                LearnedArithmeticError,
                TypedMicrocodeGraphError,
                ZeroDivisionError,
            ) as error:
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
                        reset_prediction = None
                        reset_correct = False
                    counts["state_reset_linked_correct"] += int(reset_correct)
                    detail["state_reset_prediction"] = (
                        None if reset_prediction is None else str(reset_prediction)
                    )
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
                    opcode_prediction = None
                    opcode_correct = False
                counts["opcode_permuted_correct"] += int(opcode_correct)
                detail["opcode_permuted_prediction"] = (
                    None if opcode_prediction is None else str(opcode_prediction)
                )
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
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_update": checkpoint_update,
        "training_data_sha256": metadata["data_sha256"],
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
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "details"},
            indent=2,
            sort_keys=True,
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", choices=("normal", "source_shuffled"), required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--lam-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026081053)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

