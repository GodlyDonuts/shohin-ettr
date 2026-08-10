#!/usr/bin/env python3
"""Audit exact TMC1 graph coverage before any neural fit."""

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
from natural_microcode_program import parse_program
from train_lam1_microcode import candidate_fraction
from typed_microcode_graph import (
    LITERAL,
    SOURCE,
    STATE,
    TypedMicrocodeGraph,
    compile_typed_graph,
    execute_fraction,
    execute_learned,
    operand_count,
)

SCHEMA = "shohin-tmc1-graph-geometry-audit-v1"


class TMC1AuditError(ValueError):
    """The frozen graph mechanics or input custody differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(
    path: Path, expected_sha256: str, expected_rows: int
) -> list[dict[str, object]]:
    if sha256_file(path) != expected_sha256:
        raise TMC1AuditError("input data SHA-256 differs")
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != expected_rows:
        raise TMC1AuditError("input population differs")
    return rows


def load_microcode(path: Path) -> LearnedDigitMicrocode:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "shohin-lam1-learned-arithmetic-microcode-v1":
        raise TMC1AuditError("LAM checkpoint differs")
    model = LearnedDigitMicrocode()
    model.load_state_dict(payload["state_dict"], strict=True)
    if model.transition_exact() != (1400, 1400):
        raise TMC1AuditError("LAM transition receipt differs")
    model.freeze_discrete()
    return model


def literal_width(graph: TypedMicrocodeGraph) -> tuple[int, int]:
    values = [
        operand.literal
        for instruction in graph.instructions
        for operand in (instruction.left, instruction.right)
        if operand is not None and operand.kind == LITERAL
    ]
    values.extend([graph.final.literal] if graph.final.kind == LITERAL else [])
    fractions = [value for value in values if value is not None]
    return (
        max((len(str(abs(value.numerator))) for value in fractions), default=0),
        max((len(str(value.denominator)) for value in fractions), default=0),
    )


def audit_split(
    rows: list[dict[str, object]], microcode: LearnedDigitMicrocode
) -> tuple[dict[str, int], list[TypedMicrocodeGraph]]:
    counts: Counter[str] = Counter()
    graphs = []
    for row in rows:
        graph = compile_typed_graph(
            str(row["original_question"]), parse_program(str(row["gold_program"]))
        )
        expected = Fraction(str(row["gold_answer"]))
        if execute_fraction(graph) != expected:
            raise TMC1AuditError("exact graph execution differs")
        learned = candidate_fraction(execute_learned(microcode, graph))
        if learned != expected:
            raise TMC1AuditError("learned graph execution differs")
        counts["rows"] += 1
        counts["instructions"] += len(graph.instructions)
        counts["source_operands"] += operand_count(graph, SOURCE)
        counts["state_operands"] += operand_count(graph, STATE)
        counts["literal_operands"] += operand_count(graph, LITERAL)
        counts["ambiguous_source_operands"] += sum(
            operand.kind == SOURCE and len(operand.indices) > 1
            for instruction in graph.instructions
            for operand in (instruction.left, instruction.right)
            if operand is not None
        )
        counts["final_is_state"] += int(graph.final.kind == STATE)
        graphs.append(graph)
    return dict(counts), graphs


def maxima(graphs: list[TypedMicrocodeGraph]) -> dict[str, int]:
    widths = [literal_width(graph) for graph in graphs]
    return {
        "source_spans": max(len(graph.number_spans) for graph in graphs),
        "instructions": max(len(graph.instructions) for graph in graphs),
        "literal_numerator_digits": max(width[0] for width in widths),
        "literal_denominator_digits": max(width[1] for width in widths),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists():
        raise TMC1AuditError("refusing existing output")
    microcode = load_microcode(args.lam_checkpoint)
    train_rows = load_rows(args.train, args.expected_train_sha256, 6333)
    development_rows = load_rows(
        args.development, args.expected_development_sha256, 666
    )
    train_counts, train_graphs = audit_split(train_rows, microcode)
    development_counts, development_graphs = audit_split(development_rows, microcode)
    train_maxima = maxima(train_graphs)
    development_maxima = maxima(development_graphs)
    development_admitted = all(
        development_maxima[key] <= train_maxima[key] for key in train_maxima
    )
    if not development_admitted:
        raise TMC1AuditError("development exceeds train-derived graph geometry")
    carry_correct = 0
    opcode_correct = 0
    carry_invalid = 0
    opcode_invalid = 0
    for row, graph in zip(development_rows, development_graphs, strict=True):
        expected = Fraction(str(row["gold_answer"]))
        try:
            carry_correct += (
                candidate_fraction(
                    execute_learned(microcode, graph, intervention="carry_reset")
                )
                == expected
            )
        except (LearnedArithmeticError, ZeroDivisionError):
            carry_invalid += 1
        try:
            opcode_correct += (
                candidate_fraction(
                    execute_learned(microcode, graph, intervention="opcode_permuted")
                )
                == expected
            )
        except (LearnedArithmeticError, ZeroDivisionError):
            opcode_invalid += 1
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "holdout_used": False,
        "public_test_opened": False,
        "inputs": {
            "train": {
                "path": str(args.train.resolve()),
                "sha256": sha256_file(args.train),
            },
            "development": {
                "path": str(args.development.resolve()),
                "sha256": sha256_file(args.development),
            },
            "lam_checkpoint": {
                "path": str(args.lam_checkpoint.resolve()),
                "sha256": sha256_file(args.lam_checkpoint),
            },
        },
        "train": {"counts": train_counts, "maxima": train_maxima},
        "development": {
            "counts": development_counts,
            "maxima": development_maxima,
            "normal_exact": len(development_rows),
            "carry_reset_exact": carry_correct,
            "carry_reset_invalid": carry_invalid,
            "opcode_permuted_exact": opcode_correct,
            "opcode_permuted_invalid": opcode_invalid,
        },
        "gates": {
            "train_exact_all": train_counts["rows"] == 6333,
            "development_exact_all": development_counts["rows"] == 666,
            "development_admitted_by_train_geometry": development_admitted,
            "all_final_owners_are_state": (
                train_counts["final_is_state"] == 6333
                and development_counts["final_is_state"] == 666
            ),
        },
    }
    result["overall_pass"] = all(result["gates"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--expected-development-sha256", required=True)
    parser.add_argument("--lam-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args())
    return 0 if result["overall_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
