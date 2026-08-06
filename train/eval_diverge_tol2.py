#!/usr/bin/env python3
"""Zero-update document-aware development gate for DIVERGE-TOL2."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from typing import Sequence

import torch

from diverge_tol1_data import (
    CLAUSE_OPS,
    COMPARATOR_NAMES,
    ClauseTarget,
    encode_bytes,
    source_candidates,
)
from diverge_tol1_ir import (
    Instruction,
    TOL1IRError,
    execute_program,
    format_fraction,
    instruction_sha256,
)
from diverge_tol1_product import load_rows, row_clauses, sha256_path, tensorize_clauses
from diverge_tol1_runtime import TOL1Config, TypedOperationCompiler
from diverge_tol2_anchor_decoder import (
    GuardRegions,
    TOL2DecodeError,
    decode_direct_action,
    decode_predicate,
    decode_query,
    decode_swap,
    semantic_instruction_equal,
    split_guard,
)


def _inference_clause(text: str) -> ClauseTarget:
    dummy = Instruction("QUERY", query="probe")
    return ClauseTarget(
        text=text,
        byte_ids=encode_bytes(text),
        candidates=source_candidates(text),
        operation_id=0,
        comparator_id=0,
        true_action_id=0,
        false_action_id=0,
        instruction=dummy,
        instruction_sha256=instruction_sha256(dummy),
    )


def _score_texts(
    model: TypedOperationCompiler,
    texts: Sequence[str],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[list[str], list[str]]:
    operations = []
    comparators = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            clauses = [_inference_clause(text) for text in texts[start : start + batch_size]]
            tensors, _ = tensorize_clauses(clauses, device)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with autocast:
                outputs = model(
                    tensors["byte_ids"],
                    tensors["attention"],
                    tensors["candidate_batch"],
                    tensors["candidate_start"],
                    tensors["candidate_end"],
                )
            operation_logits = outputs[1].detach().float().cpu()
            comparator_logits = outputs[2].detach().float().cpu()
            operations.extend(CLAUSE_OPS[int(row.argmax())] for row in operation_logits)
            comparators.extend(
                COMPARATOR_NAMES[int(row.argmax())] for row in comparator_logits
            )
    return operations, comparators


def _normalized_predicate(region: GuardRegions) -> str:
    return (
        f"if {region.predicate}, then set probe to 0; "
        "otherwise, set probe to 0."
    )


def evaluate(
    model: TypedOperationCompiler,
    rows: Sequence[dict[str, object]],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    clauses_by_row = [row_clauses(row) for row in rows]
    clauses = [clause for values in clauses_by_row for clause in values]
    top_operations, _ = _score_texts(
        model, [clause.text for clause in clauses], device=device, batch_size=batch_size
    )
    guard_regions: dict[int, GuardRegions] = {}
    fragment_texts = []
    fragment_owner = []
    for index, clause in enumerate(clauses):
        try:
            regions = split_guard(clause.text)
        except TOL2DecodeError:
            continue
        guard_regions[index] = regions
        for kind, text in (
            ("true", regions.true_action + "."),
            ("false", regions.false_action + "."),
            ("predicate", _normalized_predicate(regions)),
        ):
            fragment_owner.append((index, kind))
            fragment_texts.append(text)
    fragment_operations, fragment_comparators = _score_texts(
        model, fragment_texts, device=device, batch_size=batch_size
    )
    fragment_scores: dict[tuple[int, str], str] = {}
    for owner, operation, comparator in zip(
        fragment_owner, fragment_operations, fragment_comparators, strict=True
    ):
        fragment_scores[owner] = comparator if owner[1] == "predicate" else operation

    counts = Counter()
    per_operation: dict[str, Counter] = {}
    transcripts = []
    cursor = 0
    for row, gold_clauses in zip(rows, clauses_by_row, strict=True):
        predictions: list[Instruction] = []
        symbols: list[str] = []
        declaration_phase = True
        row_valid = True
        row_guard_exact = True
        for gold in gold_clauses:
            operation = top_operations[cursor]
            local = per_operation.setdefault(gold.instruction.operation, Counter())
            local["total"] += 1
            local["operation"] += int(operation == gold.instruction.operation)
            if gold.instruction.operation == "GUARD":
                counts["guard_clauses"] += 1
            try:
                if declaration_phase and operation == "SET":
                    action = decode_direct_action(gold.text, operation, None)
                    if action.operand.kind != "CONST" or action.target in symbols:
                        raise TOL2DecodeError("invalid leading declaration")
                    symbols.append(action.target)
                    prediction = Instruction(operation, action=action)
                else:
                    declaration_phase = False
                    if operation in {"SET", "ADD", "SUBTRACT", "MULTIPLY"}:
                        action = decode_direct_action(gold.text, operation, symbols)
                        prediction = Instruction(operation, action=action)
                    elif operation == "SWAP":
                        prediction = decode_swap(gold.text, symbols)
                    elif operation == "QUERY":
                        prediction = decode_query(gold.text, symbols)
                    elif operation == "GUARD":
                        regions = guard_regions[cursor]
                        true_operation = fragment_scores[(cursor, "true")]
                        false_operation = fragment_scores[(cursor, "false")]
                        comparator = fragment_scores[(cursor, "predicate")]
                        if true_operation not in {"SET", "ADD", "SUBTRACT", "MULTIPLY"}:
                            raise TOL2DecodeError("true action is not direct")
                        if false_operation not in {"SET", "ADD", "SUBTRACT", "MULTIPLY"}:
                            raise TOL2DecodeError("false action is not direct")
                        if comparator == "NONE":
                            raise TOL2DecodeError("predicate comparator is missing")
                        prediction = Instruction(
                            "GUARD",
                            predicate=decode_predicate(
                                regions.predicate, comparator, symbols
                            ),
                            true_action=decode_direct_action(
                                regions.true_action, true_operation, symbols
                            ),
                            false_action=decode_direct_action(
                                regions.false_action, false_operation, symbols
                            ),
                        )
                    else:
                        raise TOL2DecodeError("unknown top-level operation")
                prediction.validate()
            except (KeyError, TOL1IRError, TOL2DecodeError):
                row_valid = False
                if gold.instruction.operation == "GUARD":
                    row_guard_exact = False
                cursor += 1
                continue
            predictions.append(prediction)
            exact = semantic_instruction_equal(prediction, gold.instruction)
            local["semantic_exact"] += int(exact)
            if gold.instruction.operation == "GUARD":
                row_guard_exact &= exact
                counts["guard_exact"] += int(exact)
            row_valid &= exact
            cursor += 1
        answer = None
        if len(predictions) == len(gold_clauses):
            try:
                value, _ = execute_program(predictions)
                answer = format_fraction(value)
            except TOL1IRError:
                pass
        expected = str(row["answer"])
        counts["semantic_program_exact"] += int(row_valid)
        counts["answer_exact"] += int(answer == expected)
        counts["rows_with_all_guards_exact"] += int(row_guard_exact)
        if len(transcripts) < 24:
            transcripts.append(
                {
                    "id": row["id"],
                    "expected": expected,
                    "predicted": answer,
                    "semantic_program_exact": row_valid,
                    "symbols": symbols,
                }
            )
    if cursor != len(clauses):
        raise RuntimeError("TOL2 evaluation cursor differs")
    return {
        "schema": "shohin-diverge-tol2-development-v1",
        "rows": len(rows),
        "clauses": len(clauses),
        "learned_parameters_added": 0,
        "counts": dict(counts),
        "per_operation": {
            name: dict(values) for name, values in sorted(per_operation.items())
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
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing TOL2 result: {args.output}")
    if sha256_path(args.checkpoint) != args.checkpoint_sha256:
        raise SystemExit("TOL2 source checkpoint hash differs")
    rows = load_rows(args.data, args.data_sha256, args.split)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TypedOperationCompiler(TOL1Config(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
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
    print(json.dumps({"output": str(args.output), "sha256": sha256_path(args.output), "counts": report["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
