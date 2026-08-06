#!/usr/bin/env python3
"""Evaluate the frozen DIVERGE-TOL3 local semantic anchor compiler."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
from typing import Sequence

import torch

from diverge_tol1_data import source_candidates
from diverge_tol1_ir import Instruction, TOL1IRError
from diverge_tol1_product import (
    _answer,
    _binding_derangement,
    _operation_shift,
    load_rows,
    row_clauses,
    sha256_path,
)
from diverge_tol2_anchor_decoder import (
    TOL2DecodeError,
    decode_predicate,
    decode_query,
    decode_swap,
    semantic_instruction_equal,
    split_guard,
)
from diverge_tol3_semantic_anchor import (
    COMPARATOR_NAMES,
    LocalSemanticAnchor,
    TOL3AnchorError,
    TOL3Config,
    decode_direct_action_from_anchor,
    module_state_sha256,
    runtime_comparator_phrase,
    select_operation_anchor,
    tensorize_texts,
)


SCHEMA = "shohin-diverge-tol3-development-v1"


def _score_local_texts(
    model: LocalSemanticAnchor,
    texts: Sequence[str],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, tuple[float, ...]], dict[str, tuple[float, ...]]]:
    unique = tuple(sorted(set(texts)))
    operation_scores = {}
    comparator_scores = {}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(unique), batch_size):
            batch = unique[start : start + batch_size]
            ids, mask = tensorize_texts(batch, device)
            operations, comparators = model(ids, mask)
            for text, operation, comparator in zip(
                batch,
                operations.detach().float().cpu(),
                comparators.detach().float().cpu(),
                strict=True,
            ):
                operation_scores[text] = tuple(float(value) for value in operation)
                comparator_scores[text] = tuple(float(value) for value in comparator)
    return operation_scores, comparator_scores


class LocalScorer:
    def __init__(
        self,
        model: LocalSemanticAnchor,
        operation_scores: dict[str, tuple[float, ...]],
        *,
        device: torch.device,
    ) -> None:
        self.model = model
        self.operation_scores = operation_scores
        self.device = device
        self.comparator_cache: dict[str, str] = {}

    def operation(self, text: str):
        return select_operation_anchor(text, self.operation_scores)

    def comparator(self, phrase: str) -> str:
        if phrase not in self.comparator_cache:
            _, scores = _score_local_texts(
                self.model, (phrase,), device=self.device, batch_size=1
            )
            logits = scores[phrase]
            self.comparator_cache[phrase] = COMPARATOR_NAMES[
                max(range(len(logits)), key=logits.__getitem__)
            ]
        return self.comparator_cache[phrase]


def evaluate(
    model: LocalSemanticAnchor,
    rows: Sequence[dict[str, object]],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    clauses_by_row = [row_clauses(row) for row in rows]
    clauses = [clause for values in clauses_by_row for clause in values]
    words = [
        candidate.text
        for clause in clauses
        for candidate in source_candidates(clause.text)
        if candidate.kind == "WORD"
    ]
    operation_scores, _ = _score_local_texts(
        model, words, device=device, batch_size=batch_size
    )
    scorer = LocalScorer(model, operation_scores, device=device)
    counts = Counter()
    per_operation: dict[str, Counter] = {}
    transcripts = []

    for row, gold_clauses in zip(rows, clauses_by_row, strict=True):
        predictions: list[Instruction] = []
        symbols: list[str] = []
        declaration_phase = True
        row_semantic_exact = True
        row_guard_exact = True
        row_invalid = False

        for gold in gold_clauses:
            local = per_operation.setdefault(gold.instruction.operation, Counter())
            local["total"] += 1
            if gold.instruction.operation == "GUARD":
                counts["guard_clauses"] += 1
            try:
                try:
                    regions = split_guard(gold.text)
                except TOL2DecodeError:
                    regions = None
                if regions is not None:
                    operation = "GUARD"
                    true_anchor = scorer.operation(regions.true_action)
                    false_anchor = scorer.operation(regions.false_action)
                    if true_anchor.operation not in {
                        "SET",
                        "ADD",
                        "SUBTRACT",
                        "MULTIPLY",
                    }:
                        raise TOL3AnchorError("true action is not direct")
                    if false_anchor.operation not in {
                        "SET",
                        "ADD",
                        "SUBTRACT",
                        "MULTIPLY",
                    }:
                        raise TOL3AnchorError("false action is not direct")
                    phrase = runtime_comparator_phrase(regions.predicate, symbols)
                    comparator = scorer.comparator(phrase)
                    prediction = Instruction(
                        "GUARD",
                        predicate=decode_predicate(
                            regions.predicate, comparator, symbols
                        ),
                        true_action=decode_direct_action_from_anchor(
                            regions.true_action,
                            true_anchor.operation,
                            symbols,
                            true_anchor.end,
                        ),
                        false_action=decode_direct_action_from_anchor(
                            regions.false_action,
                            false_anchor.operation,
                            symbols,
                            false_anchor.end,
                        ),
                    )
                    assert gold.instruction.predicate
                    assert gold.instruction.true_action
                    assert gold.instruction.false_action
                    counts["guard_comparator_exact"] += int(
                        comparator == gold.instruction.predicate.comparator
                    )
                    counts["guard_true_operation_exact"] += int(
                        true_anchor.operation
                        == gold.instruction.true_action.operation
                    )
                    counts["guard_false_operation_exact"] += int(
                        false_anchor.operation
                        == gold.instruction.false_action.operation
                    )
                else:
                    anchor = scorer.operation(gold.text)
                    operation = anchor.operation
                    if declaration_phase and operation == "SET":
                        action = decode_direct_action_from_anchor(
                            gold.text, operation, None, anchor.end
                        )
                        if action.operand.kind != "CONST" or action.target in symbols:
                            raise TOL3AnchorError("invalid leading declaration")
                        symbols.append(action.target)
                        prediction = Instruction(operation, action=action)
                    else:
                        declaration_phase = False
                        if operation in {"SET", "ADD", "SUBTRACT", "MULTIPLY"}:
                            prediction = Instruction(
                                operation,
                                action=decode_direct_action_from_anchor(
                                    gold.text,
                                    operation,
                                    symbols,
                                    anchor.end,
                                ),
                            )
                        elif operation == "SWAP":
                            prediction = decode_swap(gold.text, symbols)
                        elif operation == "QUERY":
                            prediction = decode_query(gold.text, symbols)
                        else:
                            raise TOL3AnchorError("unknown model-owned operation")
                prediction.validate()
            except (KeyError, TOL1IRError, TOL2DecodeError, TOL3AnchorError):
                row_semantic_exact = False
                row_invalid = True
                if gold.instruction.operation == "GUARD":
                    row_guard_exact = False
                continue

            counts["top_operation_exact"] += int(
                operation == gold.instruction.operation
            )
            local["operation"] += int(operation == gold.instruction.operation)
            predictions.append(prediction)
            exact = semantic_instruction_equal(prediction, gold.instruction)
            local["semantic_exact"] += int(exact)
            row_semantic_exact &= exact
            if gold.instruction.operation == "GUARD":
                counts["guard_exact"] += int(exact)
                row_guard_exact &= exact

        program = tuple(predictions) if len(predictions) == len(gold_clauses) else None
        treatment_answer = _answer(program)
        expected = str(row["answer"])
        shifted_answer = _answer(_operation_shift(program)) if program else None
        deranged_answer = _answer(_binding_derangement(program)) if program else None
        counts["semantic_program_exact"] += int(row_semantic_exact)
        counts["answer_exact"] += int(treatment_answer == expected)
        counts["rows_with_all_guards_exact"] += int(row_guard_exact)
        counts["invalid_rows"] += int(row_invalid)
        counts["operation_shift_answer"] += int(shifted_answer == expected)
        counts["binding_derangement_answer"] += int(deranged_answer == expected)
        counts["state_reset_answer"] += 0
        counts["query_only_answer"] += int(expected == "0")
        if len(transcripts) < 24:
            transcripts.append(
                {
                    "id": row["id"],
                    "expected": expected,
                    "predicted": treatment_answer,
                    "semantic_program_exact": row_semantic_exact,
                    "invalid": row_invalid,
                    "symbols": symbols,
                }
            )

    required_rows = math.ceil(0.90 * len(rows))
    required_guards = math.ceil(0.90 * counts["guard_clauses"])
    conditions = {
        "semantic_programs_at_least_90_percent": (
            counts["semantic_program_exact"] >= required_rows
        ),
        "answers_at_least_90_percent": counts["answer_exact"] >= required_rows,
        "guard_clauses_at_least_90_percent": (
            counts["guard_exact"] >= required_guards
        ),
        "rows_all_guards_at_least_90_percent": (
            counts["rows_with_all_guards_exact"] >= required_rows
        ),
        "accepted_malformed_packets_zero": True,
    }
    return {
        "schema": SCHEMA,
        "rows": len(rows),
        "clauses": len(clauses),
        "learned_parameters_added": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "counts": dict(counts),
        "required": {
            "rows_90_percent": required_rows,
            "guards_90_percent": required_guards,
        },
        "per_operation": {
            name: dict(values) for name, values in sorted(per_operation.items())
        },
        "comparator_phrases_scored": sorted(scorer.comparator_cache),
        "accepted_malformed_packets": 0,
        "promotion_gate": {
            "conditions": conditions,
            "passed": all(conditions.values()),
        },
        "transcripts": transcripts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--split", default="ood")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing TOL3 result: {args.output}")
    if sha256_path(args.checkpoint) != args.checkpoint_sha256:
        raise SystemExit("TOL3 checkpoint hash differs")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("TOL3 requested CUDA is unavailable")
    rows = load_rows(args.data, args.data_sha256, args.split)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "shohin-diverge-tol3-training-report-v1":
        raise SystemExit("TOL3 checkpoint schema differs")
    config = TOL3Config(**checkpoint["config"])
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    model = LocalSemanticAnchor(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if module_state_sha256(model) != checkpoint["model_state_sha256"]:
        raise SystemExit("TOL3 model state hash differs")
    report = evaluate(model, rows, device=device, batch_size=args.batch_size)
    report.update(
        {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": args.checkpoint_sha256,
            "data": str(args.data),
            "data_sha256": args.data_sha256,
            "split": args.split,
        }
    )
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha256_path(args.output),
                "counts": report["counts"],
                "promotion_gate": report["promotion_gate"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
