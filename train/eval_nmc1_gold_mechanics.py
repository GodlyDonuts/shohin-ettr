#!/usr/bin/env python3
"""Verify NMC1 gold programs through exact and learned executors."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path

import torch

from learned_arithmetic_microcode import LearnedArithmeticError, LearnedDigitMicrocode
from natural_microcode_program import (
    NaturalMicrocodeError,
    execute_fraction,
    execute_learned,
    parse_program,
)
from train_lam1_microcode import candidate_fraction

SCHEMA = "shohin-nmc1-gold-mechanics-v1"


class NMC1MechanicsError(ValueError):
    """NMC1 mechanics custody differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(
    path: Path, expected_sha256: str, model: LearnedDigitMicrocode
) -> dict[str, object]:
    if sha256_file(path) != expected_sha256:
        raise NMC1MechanicsError("data SHA-256 differs")
    counts: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            counts["rows"] += 1
            try:
                program = parse_program(str(row["gold_program"]))
                expected = Fraction(str(row["gold_answer"]))
                exact = execute_fraction(program)
                if exact != expected:
                    raise NaturalMicrocodeError("exact result differs")
                counts["fraction_exact"] += 1
                for intervention in ("normal", "carry_reset", "opcode_permuted"):
                    try:
                        predicted = candidate_fraction(
                            execute_learned(model, program, intervention=intervention)
                        )
                        counts[f"{intervention}:valid"] += 1
                        counts[f"{intervention}:correct"] += int(predicted == expected)
                    except (LearnedArithmeticError, ZeroDivisionError):
                        counts[f"{intervention}:invalid"] += 1
            except (NaturalMicrocodeError, KeyError, ValueError):
                counts["program_invalid"] += 1
    return {"counts": dict(sorted(counts.items()))}


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists():
        raise NMC1MechanicsError("refusing existing output")
    payload = torch.load(args.lam_checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "shohin-lam1-learned-arithmetic-microcode-v1":
        raise NMC1MechanicsError("LAM checkpoint schema differs")
    model = LearnedDigitMicrocode()
    model.load_state_dict(payload["state_dict"], strict=True)
    if model.transition_exact() != (1400, 1400):
        raise NMC1MechanicsError("LAM transitions differ")
    model.freeze_discrete()
    train = evaluate(args.train, args.expected_train_sha256, model)
    development = evaluate(args.development, args.expected_development_sha256, model)
    gates = {}
    for split, metrics in (("train", train), ("development", development)):
        counts = metrics["counts"]
        rows = counts["rows"]
        gates[f"{split}_fraction_exact"] = counts.get("fraction_exact", 0) == rows
        gates[f"{split}_learned_exact"] = counts.get("normal:correct", 0) == rows
        gates[f"{split}_zero_invalid"] = (
            counts.get("normal:invalid", 0) == 0
            and counts.get("program_invalid", 0) == 0
        )
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "holdout_used": False,
        "lam_checkpoint_sha256": sha256_file(args.lam_checkpoint),
        "train": train,
        "development": development,
        "gates": gates,
        "overall_pass": all(gates.values()),
    }
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lam-checkpoint", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--expected-development-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args())
    return 0 if result["overall_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
