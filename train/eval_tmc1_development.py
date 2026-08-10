#!/usr/bin/env python3
"""Evaluate the frozen TMC1 typed compiler on source-disjoint development."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Sequence

import torch

from hf_product_reasoning_eval import _load_model
from learned_arithmetic_microcode import LearnedArithmeticError, LearnedDigitMicrocode
from natural_microcode_program import parse_program
from train_lam1_microcode import candidate_fraction
from train_tmc1_compiler import tokenize_sources
from typed_microcode_compiler import TypedMicrocodeCompiler, decode_graphs
from typed_microcode_graph import (
    LITERAL,
    SOURCE,
    STATE,
    Operand,
    TypedMicrocodeGraph,
    TypedMicrocodeGraphError,
    compile_typed_graph,
    execute_learned,
)

SCHEMA = "shohin-tmc1-development-evaluation-v1"


class TMC1EvaluationError(ValueError):
    """Frozen TMC1 evaluation custody differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path, expected_sha256: str) -> list[dict[str, object]]:
    if sha256_file(path) != expected_sha256:
        raise TMC1EvaluationError("development data SHA-256 differs")
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != 666 or len({row["identity_sha256"] for row in rows}) != 666:
        raise TMC1EvaluationError("development population differs")
    return rows


def row_graph(row: dict[str, object]) -> TypedMicrocodeGraph:
    return compile_typed_graph(
        str(row["original_question"]), parse_program(str(row["gold_program"]))
    )


def source_shuffle(rows: Sequence[dict[str, object]]) -> dict[str, dict[str, object]]:
    groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[int(row["register_depth"])].append(row)
    mapping = {}
    for depth, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: str(row["identity_sha256"]))
        if len(ordered) < 2:
            raise TMC1EvaluationError(f"source-shuffle singleton depth {depth}")
        for target, donor in zip(ordered, ordered[1:] + ordered[:1], strict=True):
            mapping[str(target["identity_sha256"])] = donor
    return mapping


def load_microcode(path: Path) -> LearnedDigitMicrocode:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "shohin-lam1-learned-arithmetic-microcode-v1":
        raise TMC1EvaluationError("LAM checkpoint differs")
    model = LearnedDigitMicrocode()
    model.load_state_dict(payload["state_dict"], strict=True)
    if model.transition_exact() != (1400, 1400):
        raise TMC1EvaluationError("LAM transition receipt differs")
    model.freeze_discrete()
    return model


def load_compiler(path: Path) -> tuple[TypedMicrocodeCompiler, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema") != "shohin-tmc1-typed-compiler-training-v1"
        or payload.get("updates") != 4096
    ):
        raise TMC1EvaluationError("compiler checkpoint differs")
    config = payload.get("config")
    if config != {
        "source_width": 1024,
        "width": 512,
        "source_layers": 2,
        "decoder_layers": 4,
        "heads": 8,
    }:
        raise TMC1EvaluationError("compiler geometry differs")
    model = TypedMicrocodeCompiler(**config)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device="cuda:0", dtype=torch.bfloat16).eval(), payload


def operand_equivalent(
    predicted: Operand,
    gold: Operand,
    predicted_graph: TypedMicrocodeGraph,
    gold_graph: TypedMicrocodeGraph,
) -> bool:
    if predicted.kind != gold.kind:
        return False
    if predicted.kind == STATE:
        return predicted.indices == gold.indices
    if predicted.kind == LITERAL:
        return predicted.literal == gold.literal
    if predicted.kind == SOURCE:
        predicted_values = {
            predicted_graph.number_spans[index].value for index in predicted.indices
        }
        gold_values = {gold_graph.number_spans[index].value for index in gold.indices}
        return len(predicted_values) == 1 and predicted_values == gold_values
    return False


def structural_counts(
    predicted: TypedMicrocodeGraph, gold: TypedMicrocodeGraph
) -> Counter[str]:
    counts: Counter[str] = Counter()
    counts["instruction_count_exact"] = int(
        len(predicted.instructions) == len(gold.instructions)
    )
    maximum = max(len(predicted.instructions), len(gold.instructions))
    for step in range(maximum):
        predicted_instruction = (
            predicted.instructions[step] if step < len(predicted.instructions) else None
        )
        gold_instruction = (
            gold.instructions[step] if step < len(gold.instructions) else None
        )
        counts["operation_fields"] += 1
        counts["operation_correct"] += int(
            predicted_instruction is not None
            and gold_instruction is not None
            and predicted_instruction.operation == gold_instruction.operation
        )
        for side in ("left", "right"):
            predicted_operand = (
                getattr(predicted_instruction, side)
                if predicted_instruction is not None
                else None
            )
            gold_operand = (
                getattr(gold_instruction, side)
                if gold_instruction is not None
                else None
            )
            if predicted_operand is None and gold_operand is None:
                continue
            counts["operand_fields"] += 1
            counts["operand_correct"] += int(
                predicted_operand is not None
                and gold_operand is not None
                and operand_equivalent(predicted_operand, gold_operand, predicted, gold)
            )
    counts["graph_exact"] = int(
        counts["instruction_count_exact"]
        and counts["operation_correct"] == counts["operation_fields"]
        and counts["operand_correct"] == counts["operand_fields"]
    )
    return counts


def is_multi_digit(graph: TypedMicrocodeGraph) -> bool:
    for instruction in graph.instructions:
        for operand in (instruction.left, instruction.right):
            if operand is None:
                continue
            if operand.kind == SOURCE:
                values = [graph.number_spans[index].value for index in operand.indices]
            elif operand.kind == LITERAL and operand.literal is not None:
                values = [operand.literal]
            else:
                values = []
            if any(
                len(str(abs(value.numerator))) > 1 or len(str(value.denominator)) > 1
                for value in values
            ):
                return True
    return False


def graph_payload(graph: TypedMicrocodeGraph) -> dict[str, object]:
    def operand_payload(operand: Operand | None) -> dict[str, object] | None:
        if operand is None:
            return None
        return {
            "kind": operand.kind,
            "indices": list(operand.indices),
            "literal": str(operand.literal) if operand.literal is not None else None,
        }

    return {
        "instructions": [
            {
                "operation": instruction.operation,
                "left": operand_payload(instruction.left),
                "right": operand_payload(instruction.right),
            }
            for instruction in graph.instructions
        ],
        "final": operand_payload(graph.final),
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise TMC1EvaluationError("refusing existing evaluation output")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, object]:
    from transformers import AutoTokenizer

    if args.batch_size != 32 or args.control not in {"normal", "source_shuffled"}:
        raise TMC1EvaluationError("evaluation geometry differs")
    if sha256_file(args.compiler_checkpoint) != args.expected_compiler_sha256:
        raise TMC1EvaluationError("compiler checkpoint SHA-256 differs")
    if sha256_file(args.owner_checkpoint) != args.expected_owner_sha256:
        raise TMC1EvaluationError("owner checkpoint SHA-256 differs")
    rows = load_rows(args.data, args.expected_data_sha256)
    gold_graphs = [row_graph(row) for row in rows]
    donors = source_shuffle(rows) if args.control == "source_shuffled" else {}
    source_graphs = [
        row_graph(donors.get(str(row["identity_sha256"]), row)) for row in rows
    ]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_root, trust_remote_code=True, use_fast=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    owner, metadata, loader = _load_model(
        args.model_root, args.owner_checkpoint, "auto"
    )
    if (
        metadata is None
        or metadata.get("update") != 1024
        or metadata.get("model_revision") != args.model_revision
        or metadata.get("data_sha256") != args.expected_owner_data_sha256
    ):
        raise TMC1EvaluationError("owner metadata differs")
    owner.eval().requires_grad_(False)
    compiler, compiler_receipt = load_compiler(args.compiler_checkpoint)
    microcode = load_microcode(args.lam_checkpoint)
    counts: Counter[str] = Counter()
    details = []
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for offset in range(0, len(rows), args.batch_size):
        batch_rows = rows[offset : offset + args.batch_size]
        batch_source = source_graphs[offset : offset + args.batch_size]
        batch_gold = gold_graphs[offset : offset + args.batch_size]
        encoded, candidate_mask, _ = tokenize_sources(
            tokenizer, batch_source, torch.device("cuda:0"), 512
        )
        source_count = torch.tensor(
            [len(graph.number_spans) for graph in batch_source],
            dtype=torch.long,
            device="cuda:0",
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            source_states = owner.text_model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
            output = compiler(
                source_states,
                encoded["attention_mask"].bool(),
                candidate_mask,
                source_count,
            )
        try:
            predicted_graphs = decode_graphs(
                output, [graph.source for graph in batch_source]
            )
        except TypedMicrocodeGraphError as error:
            raise TMC1EvaluationError("batch graph decoding failed") from error
        for row, predicted, gold in zip(
            batch_rows, predicted_graphs, batch_gold, strict=True
        ):
            counts["rows"] += 1
            structure = structural_counts(predicted, gold)
            counts.update(structure)
            expected = Fraction(str(row["gold_answer"]))
            detail: dict[str, object] = {
                "identity_sha256": row["identity_sha256"],
                "donor_identity_sha256": (
                    donors[str(row["identity_sha256"])]["identity_sha256"]
                    if donors
                    else None
                ),
                "predicted_graph": graph_payload(predicted),
                "instruction_count_exact": bool(structure["instruction_count_exact"]),
                "graph_exact": bool(structure["graph_exact"]),
            }
            intervention_correct = {}
            for intervention in ("normal", "carry_reset", "opcode_permuted"):
                try:
                    prediction = candidate_fraction(
                        execute_learned(microcode, predicted, intervention=intervention)
                    )
                    correct = prediction == expected
                    intervention_correct[intervention] = correct
                    counts[f"{intervention}:valid"] += 1
                    counts[f"{intervention}:correct"] += int(correct)
                    detail[f"{intervention}_prediction"] = str(prediction)
                    detail[f"{intervention}_correct"] = correct
                except (
                    LearnedArithmeticError,
                    TypedMicrocodeGraphError,
                    ZeroDivisionError,
                ):
                    intervention_correct[intervention] = False
                    counts[f"{intervention}:invalid"] += 1
                    detail[f"{intervention}_correct"] = False
            if intervention_correct.get("normal", False) and is_multi_digit(gold):
                counts["normal_correct_multi_digit_rows"] += 1
                counts["carry_reset:normal_correct_multi_digit_correct"] += int(
                    intervention_correct.get("carry_reset", False)
                )
            details.append(detail)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "holdout_used": False,
        "public_test_opened": False,
        "control": args.control,
        "model_revision": args.model_revision,
        "model_loader": loader,
        "owner_checkpoint_sha256": args.expected_owner_sha256,
        "compiler_checkpoint_sha256": args.expected_compiler_sha256,
        "compiler_updates": compiler_receipt["updates"],
        "development_data_sha256": args.expected_data_sha256,
        "lam_checkpoint_sha256": sha256_file(args.lam_checkpoint),
        "counts": dict(sorted(counts.items())),
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
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
    parser.add_argument(
        "--control", choices=("normal", "source_shuffled"), required=True
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--owner-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-owner-sha256", required=True)
    parser.add_argument("--expected-owner-data-sha256", required=True)
    parser.add_argument("--compiler-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-compiler-sha256", required=True)
    parser.add_argument("--lam-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
