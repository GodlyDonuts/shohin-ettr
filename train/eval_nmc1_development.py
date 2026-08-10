#!/usr/bin/env python3
"""Evaluate matched NMC1 and direct-control fits on frozen development rows."""

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

from hf_product_reasoning_eval import (
    _generate_completions,
    _generation_stop_token_ids,
    _load_model,
    extract_gsm8k,
    match_gsm8k,
)
from hf_product_reasoning_train import PRODUCT_SYSTEM_PROMPT, render_reasoning_messages
from learned_arithmetic_microcode import LearnedArithmeticError, LearnedDigitMicrocode
from natural_microcode_program import (
    CLOSE,
    OPEN,
    NaturalMicrocodeError,
    RegisterProgram,
    all_actions,
    execute_learned,
    parse_program,
)
from train_lam1_microcode import candidate_fraction

SCHEMA = "shohin-nmc1-development-evaluation-v1"


class NMC1EvaluationError(ValueError):
    """Frozen NMC1 evaluation custody differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path, expected_sha256: str) -> list[dict[str, object]]:
    if sha256_file(path) != expected_sha256:
        raise NMC1EvaluationError("development data SHA-256 differs")
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    identities = [str(row.get("identity_sha256")) for row in rows]
    if len(rows) != 666 or len(set(identities)) != len(rows):
        raise NMC1EvaluationError("development population differs")
    return rows


def source_shuffle(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[int(row["register_depth"])].append(row)
    mapping = {}
    for depth, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: str(row["identity_sha256"]))
        if len(ordered) < 2:
            raise NMC1EvaluationError(f"source-shuffle singleton depth {depth}")
        for target, donor in zip(ordered, ordered[1:] + ordered[:1], strict=True):
            if target["identity_sha256"] == donor["identity_sha256"]:
                raise NMC1EvaluationError("source shuffle retained identity")
            mapping[str(target["identity_sha256"])] = donor
    return mapping


def extract_program(text: str) -> RegisterProgram:
    start = text.find(OPEN)
    end = text.find(CLOSE, start + len(OPEN)) if start >= 0 else -1
    if start < 0 or end < 0:
        raise NaturalMicrocodeError("completion lacks program envelope")
    return parse_program(text[start : end + len(CLOSE)])


def is_multi_digit(program: RegisterProgram) -> bool:
    return any(
        action.get("action") == "PUSH"
        and len(str(action.get("surface", "")).replace("/", "").lstrip("-")) > 1
        for action in all_actions(program)
    )


def load_microcode(path: Path) -> LearnedDigitMicrocode:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "shohin-lam1-learned-arithmetic-microcode-v1":
        raise NMC1EvaluationError("LAM checkpoint differs")
    model = LearnedDigitMicrocode()
    model.load_state_dict(payload["state_dict"], strict=True)
    if model.transition_exact() != (1400, 1400):
        raise NMC1EvaluationError("LAM transition receipt differs")
    model.freeze_discrete()
    return model


def load_checkpoint_receipt(path: Path) -> tuple[int, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "shohin-hf-product-reasoning-checkpoint-v1":
        raise NMC1EvaluationError("fit checkpoint schema differs")
    update = payload.get("update")
    metadata = payload.get("metadata")
    if not isinstance(update, int) or not isinstance(metadata, dict):
        raise NMC1EvaluationError("fit checkpoint receipt is incomplete")
    return update, metadata


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise NMC1EvaluationError("refusing existing evaluation output")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, object]:
    from transformers import AutoTokenizer

    if args.arm == "direct" and args.control != "normal":
        raise NMC1EvaluationError("direct control geometry differs")
    if args.max_new_tokens != 512 or args.seed != 2026081053:
        raise NMC1EvaluationError("decoding geometry differs")
    rows = load_rows(args.data, args.expected_data_sha256)
    donors = source_shuffle(rows) if args.control == "source_shuffled" else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
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
        raise NMC1EvaluationError("fit checkpoint metadata differs")
    microcode = load_microcode(args.lam_checkpoint) if args.arm == "program" else None
    stop_ids = _generation_stop_token_ids(tokenizer)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    counts: Counter[str] = Counter()
    details = []
    generated_tokens = 0
    exhausted = 0
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset : offset + args.batch_size]
        prompts = []
        for row in batch:
            source = donors.get(str(row["identity_sha256"]), row)
            question = str(source["original_question"])
            if args.arm == "program":
                question = (
                    "Compile the word problem into MICROCODE_V1. Emit only the "
                    "program. Do not emit calculated results.\n\nPROBLEM:\n" + question
                )
            prompts.append(
                render_reasoning_messages(
                    tokenizer,
                    [
                        {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
                        {"role": "user", "content": question},
                    ],
                    enable_thinking=False,
                )
            )
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
            counts["rows"] += 1
            generated_tokens += tokens
            exhausted += int(row_exhausted)
            detail: dict[str, object] = {
                "identity_sha256": row["identity_sha256"],
                "donor_identity_sha256": (
                    donors[str(row["identity_sha256"])]["identity_sha256"]
                    if donors
                    else None
                ),
                "completion": completion,
                "generated_tokens": tokens,
                "exhausted": row_exhausted,
            }
            expected = Fraction(str(row["gold_answer"]))
            if args.arm == "direct":
                prediction = extract_gsm8k(completion)
                correct = match_gsm8k(prediction, str(expected))
                counts["answer_correct"] += int(correct)
                detail.update({"prediction": prediction, "answer_correct": correct})
            else:
                assert microcode is not None
                try:
                    program = extract_program(completion)
                    counts["syntax_valid"] += 1
                    detail["syntax_valid"] = True
                    detail["program_exact"] = program == parse_program(
                        str(row["gold_program"])
                    )
                    counts["program_exact"] += int(bool(detail["program_exact"]))
                    multi_digit = is_multi_digit(program)
                    detail["multi_digit"] = multi_digit
                    intervention_correct: dict[str, bool] = {}
                    for intervention in (
                        "normal",
                        "carry_reset",
                        "opcode_permuted",
                    ):
                        try:
                            prediction = candidate_fraction(
                                execute_learned(
                                    microcode, program, intervention=intervention
                                )
                            )
                            correct = prediction == expected
                            intervention_correct[intervention] = correct
                            counts[f"{intervention}:valid"] += 1
                            counts[f"{intervention}:correct"] += int(correct)
                            if multi_digit:
                                counts[f"{intervention}:multi_digit_rows"] += 1
                                counts[f"{intervention}:multi_digit_correct"] += int(
                                    correct
                                )
                            detail[f"{intervention}_prediction"] = str(prediction)
                            detail[f"{intervention}_correct"] = correct
                        except (LearnedArithmeticError, ZeroDivisionError):
                            intervention_correct[intervention] = False
                            counts[f"{intervention}:invalid"] += 1
                            detail[f"{intervention}_correct"] = False
                    if multi_digit and intervention_correct.get("normal", False):
                        counts["normal_correct_multi_digit_rows"] += 1
                        counts["carry_reset:normal_correct_multi_digit_correct"] += int(
                            intervention_correct.get("carry_reset", False)
                        )
                except NaturalMicrocodeError:
                    counts["syntax_invalid"] += 1
                    detail["syntax_valid"] = False
                    detail["program_exact"] = False
            details.append(detail)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "holdout_used": False,
        "public_test_opened": False,
        "arm": args.arm,
        "control": args.control,
        "model_root": str(args.model_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": loader,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_update": checkpoint_update,
        "training_data_sha256": metadata["data_sha256"],
        "development_data_sha256": args.expected_data_sha256,
        "lam_checkpoint_sha256": (
            sha256_file(args.lam_checkpoint) if microcode is not None else None
        ),
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "generated_tokens": generated_tokens,
        "exhausted": exhausted,
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "counts": dict(sorted(counts.items())),
        "details": details,
    }
    _atomic_json(args.output, report)
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
    parser.add_argument("--arm", choices=("program", "direct"), required=True)
    parser.add_argument(
        "--control", choices=("normal", "source_shuffled"), required=True
    )
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
