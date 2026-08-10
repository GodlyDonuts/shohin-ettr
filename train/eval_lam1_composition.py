#!/usr/bin/env python3
"""Compose frozen BTT/WGP source compilation with learned LAM1 execution."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from byte_tape_compiler import ByteTapeCompiler, ROLES, byte_batch
from eval_btt1_byte import compile_roles, source_shuffle_indices
from learned_arithmetic_microcode import (
    LearnedArithmeticError,
    LearnedDigitMicrocode,
    execute_microcode,
)
from train_btt1_byte import load_programs
from train_lam1_microcode import assessor, candidate_fraction
from weighted_grammar_projection import project_role_logits

SCHEMA = "shohin-lam1-btt-wgp-composition-v1"


class LAM1CompositionError(RuntimeError):
    """The frozen LAM1 composition contract differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def evaluate_actions(
    microcode: LearnedDigitMicrocode,
    actions: list[dict[str, object]],
    expected,
    intervention: str,
) -> tuple[bool, bool]:
    try:
        predicted = candidate_fraction(
            execute_microcode(microcode, actions, intervention=intervention)
        )
        return predicted == expected, True
    except (LearnedArithmeticError, ZeroDivisionError):
        return False, False


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists() or args.expected_rows != 3917 or args.beam_width != 64:
        raise LAM1CompositionError("LAM1 composition geometry differs")
    btt_payload = torch.load(
        args.btt_checkpoint, map_location="cpu", weights_only=False
    )
    if (
        btt_payload.get("schema") != "shohin-btt1-training-v1"
        or btt_payload.get("data_sha256") != args.expected_train_sha256
    ):
        raise LAM1CompositionError("BTT checkpoint differs")
    lam_payload = torch.load(
        args.lam_checkpoint, map_location="cpu", weights_only=False
    )
    if (
        lam_payload.get("schema") != "shohin-lam1-learned-arithmetic-microcode-v1"
        or lam_payload.get("steps") != 32
        or lam_payload.get("seed") != 2026081041
    ):
        raise LAM1CompositionError("LAM1 checkpoint differs")
    rows = load_programs(args.data, args.expected_data_sha256, args.expected_rows)
    mapping = source_shuffle_indices(rows)
    device = torch.device("cuda")
    config = btt_payload["config"]
    compiler = ByteTapeCompiler(
        width=int(config["width"]),
        encoder_layers=int(config["encoder_layers"]),
        heads=int(config["heads"]),
    ).to(device=device, dtype=torch.bfloat16)
    compiler.load_state_dict(btt_payload["state_dict"], strict=True)
    compiler.eval()
    microcode = LearnedDigitMicrocode()
    microcode.load_state_dict(lam_payload["state_dict"], strict=True)
    microcode.eval()
    if microcode.transition_exact() != (1400, 1400):
        raise LAM1CompositionError("LAM1 local transition receipt differs")
    microcode.freeze_discrete()

    controls = (
        "normal",
        "carry_reset",
        "opcode_permuted",
        "source_shuffled",
        "zero_bytes",
    )
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    projection_cache: dict[tuple[object, ...], list[int] | None] = {}
    started = time.time()
    with torch.inference_mode():
        for compiler_control in ("normal", "source_shuffled", "zero_bytes"):
            for start in range(0, len(rows), args.batch_size):
                targets = rows[start : start + args.batch_size]
                indices = range(start, min(start + args.batch_size, len(rows)))
                sources = (
                    [rows[mapping[index]] for index in indices]
                    if compiler_control == "source_shuffled"
                    else targets
                )
                batch = byte_batch(sources, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = compiler(
                        batch["byte_ids"],
                        batch["mask"],
                        zero_bytes=compiler_control == "zero_bytes",
                    )
                logits = output.role_logits
                for row_index, (target, source) in enumerate(
                    zip(targets, sources, strict=True)
                ):
                    source_bytes = list(source.question.encode("ascii"))
                    cache_key = None
                    projected = None
                    if compiler_control == "zero_bytes":
                        shape = tuple(
                            2 if byte == 46 else 1 if 48 <= byte <= 57 else 0
                            for byte in source_bytes
                        )
                        cache_key = (len(source_bytes), shape)
                        projected = projection_cache.get(cache_key)
                    if cache_key is None or cache_key not in projection_cache:
                        projected = project_role_logits(
                            logits[row_index, : len(source.question)],
                            source_bytes,
                            beam_width=args.beam_width,
                        )
                        if cache_key is not None:
                            projection_cache[cache_key] = projected
                    if projected is None:
                        actions, valid = [{"action": "STOP"}], False
                    else:
                        role_names = [ROLES[index] for index in projected]
                        actions, valid = compile_roles(
                            source.question, role_names, flat=False
                        )
                    expected = assessor(list(target.gold_actions))
                    evaluated_controls = (
                        ("normal", "carry_reset", "opcode_permuted")
                        if compiler_control == "normal"
                        else (compiler_control,)
                    )
                    for control in evaluated_controls:
                        counts[control]["rows"] += 1
                        counts[control][f"family:{target.family}:rows"] += 1
                        if projected is None:
                            counts[control]["search_exhausted"] += 1
                        if not valid:
                            counts[control]["compiler_invalid"] += 1
                        intervention = (
                            control
                            if control in {"carry_reset", "opcode_permuted"}
                            else "normal"
                        )
                        correct, execution_valid = evaluate_actions(
                            microcode, actions, expected, intervention
                        )
                        counts[control]["execution_invalid"] += int(not execution_valid)
                        counts[control]["correct"] += int(correct and valid)
                        counts[control][f"family:{target.family}:correct"] += int(
                            correct and valid
                        )

    metrics = {
        control: {
            "counts": dict(sorted(counts[control].items())),
            "exact_rate": counts[control]["correct"] / counts[control]["rows"],
        }
        for control in controls
    }
    gates = {
        "normal_exact": counts["normal"]["correct"] == args.expected_rows,
        "zero_normal_invalid": counts["normal"].get("compiler_invalid", 0) == 0
        and counts["normal"].get("execution_invalid", 0) == 0,
        "zero_normal_exhaustion": counts["normal"].get("search_exhausted", 0) == 0,
        "source_shuffled_at_most_0p25": metrics["source_shuffled"]["exact_rate"]
        <= 0.25,
        "zero_bytes_at_most_0p25": metrics["zero_bytes"]["exact_rate"] <= 0.25,
        "carry_reset_loses_20_points": metrics["normal"]["exact_rate"]
        - metrics["carry_reset"]["exact_rate"]
        >= 0.20,
        "opcode_permutation_loses_50_points": metrics["normal"]["exact_rate"]
        - metrics["opcode_permuted"]["exact_rate"]
        >= 0.50,
    }
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "holdout_used": False,
        "rows": len(rows),
        "beam_width": args.beam_width,
        "metrics": metrics,
        "gates": gates,
        "overall_pass": all(gates.values()),
        "elapsed_seconds": time.time() - started,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "btt_checkpoint_sha256": sha256_file(args.btt_checkpoint),
        "lam_checkpoint_sha256": sha256_file(args.lam_checkpoint),
        "data_sha256": args.expected_data_sha256,
        "train_data_sha256": args.expected_train_sha256,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--btt-checkpoint", type=Path, required=True)
    parser.add_argument("--lam-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--expected-rows", type=int, default=3917)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args())
    return 0 if result["overall_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
