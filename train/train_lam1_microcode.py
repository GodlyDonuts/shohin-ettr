#!/usr/bin/env python3
"""Fit and evaluate the frozen LAM1 finite arithmetic microcode."""

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

from learned_arithmetic_microcode import (
    LearnedArithmeticError,
    LearnedDigitMicrocode,
    execute_microcode,
)

SCHEMA = "shohin-lam1-learned-arithmetic-microcode-v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def assessor(actions: list[dict[str, object]]) -> Fraction:
    stack: list[Fraction] = []
    for action in actions:
        name = action["action"]
        if name == "PUSH":
            stack.append(Fraction(str(action["surface"])))
        elif name == "NEGATE":
            stack[-1] = -stack[-1]
        elif str(name).startswith("APPLY_"):
            right, left = stack.pop(), stack.pop()
            operation = str(name).removeprefix("APPLY_")
            if operation == "ADD":
                stack.append(left + right)
            elif operation == "SUB":
                stack.append(left - right)
            elif operation == "MUL":
                stack.append(left * right)
            elif operation == "DIV" and right:
                stack.append(left / right)
            else:
                raise LearnedArithmeticError("assessor operation differs")
        elif name == "STOP":
            break
    if len(stack) != 1:
        raise LearnedArithmeticError("assessor stack differs")
    return stack[0]


def candidate_fraction(value) -> Fraction:
    numerator = int("".join(str(digit) for digit in reversed(value.numerator)))
    denominator = int("".join(str(digit) for digit in reversed(value.denominator)))
    result = Fraction(numerator, denominator)
    return -result if value.negative else result


def evaluate(
    model: LearnedDigitMicrocode, path: Path, expected_sha256: str, expected_rows: int
) -> dict[str, object]:
    if sha256_file(path) != expected_sha256:
        raise LearnedArithmeticError("program data SHA-256 differs")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != expected_rows:
        raise LearnedArithmeticError("program population differs")
    metrics: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        family = row["family"]
        actions = row["gold_actions"]
        expected = assessor(actions)
        multi_digit = any(
            action.get("action") == "PUSH"
            and len(str(action.get("surface", "")).replace(".", "").lstrip("-")) > 1
            for action in actions
        )
        for intervention in ("normal", "carry_reset", "opcode_permuted"):
            metrics[intervention]["rows"] += 1
            metrics[intervention][f"family:{family}:rows"] += 1
            if multi_digit:
                metrics[intervention]["multi_digit_rows"] += 1
            try:
                predicted = candidate_fraction(
                    execute_microcode(model, actions, intervention=intervention)
                )
                correct = predicted == expected
            except (LearnedArithmeticError, ZeroDivisionError):
                metrics[intervention]["invalid"] += 1
                correct = False
            metrics[intervention]["correct"] += int(correct)
            metrics[intervention][f"family:{family}:correct"] += int(correct)
            if multi_digit:
                metrics[intervention]["multi_digit_correct"] += int(correct)
    output = {}
    for name, counts in metrics.items():
        output[name] = {
            "counts": dict(sorted(counts.items())),
            "exact_rate": counts["correct"] / counts["rows"],
            "multi_digit_exact_rate": counts["multi_digit_correct"]
            / counts["multi_digit_rows"],
        }
    return output


def run(args: argparse.Namespace) -> dict[str, object]:
    if (
        args.output.exists()
        or args.steps != 32
        or args.learning_rate != 1.0
        or args.seed != 2026081041
    ):
        raise LearnedArithmeticError("LAM1 frozen geometry differs")
    args.output.mkdir(parents=True)
    torch.manual_seed(args.seed)
    model = LearnedDigitMicrocode()
    if sum(parameter.numel() for parameter in model.parameters()) != 108_000:
        raise LearnedArithmeticError("LAM1 parameter count differs")
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    losses = []
    started = time.time()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model.transition_loss()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    local_exact, local_total = model.transition_exact()
    model.freeze_discrete()
    train = evaluate(model, args.train, args.expected_train_sha256, 75935)
    development = evaluate(
        model, args.development, args.expected_development_sha256, 3917
    )
    normal = development["normal"]
    carry = development["carry_reset"]
    permuted = development["opcode_permuted"]
    gates = {
        "all_local_transitions_exact": local_exact == local_total == 1400,
        "train_programs_exact": train["normal"]["counts"]["correct"] == 75935,
        "development_programs_exact": normal["counts"]["correct"] == 3917,
        "zero_normal_invalid": normal["counts"].get("invalid", 0) == 0,
        "carry_reset_loses_20_points_multi_digit": normal["multi_digit_exact_rate"]
        - carry["multi_digit_exact_rate"]
        >= 0.20,
        "opcode_permutation_loses_50_points": normal["exact_rate"]
        - permuted["exact_rate"]
        >= 0.50,
    }
    checkpoint = args.output / "microcode.pt"
    torch.save(
        {
            "schema": SCHEMA,
            "state_dict": model.state_dict(),
            "steps": args.steps,
            "seed": args.seed,
        },
        checkpoint,
    )
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "holdout_used": False,
        "trainable_parameters": 108000,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "local_transition_exact": local_exact,
        "local_transition_total": local_total,
        "train": train,
        "development": development,
        "gates": gates,
        "overall_pass": all(gates.values()),
        "elapsed_seconds": time.time() - started,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "data": {
            "train_sha256": args.expected_train_sha256,
            "development_sha256": args.expected_development_sha256,
        },
    }
    atomic_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--expected-development-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026081041)
    result = run(parser.parse_args())
    return 0 if result["overall_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
