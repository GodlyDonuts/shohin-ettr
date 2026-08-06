"""Tensorization, training loss, and end-to-end evaluation for DIVERGE-TOL1."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from diverge_tol1_data import (
    CLAUSE_OPS,
    MAX_CLAUSE_BYTES,
    PAD_ID,
    ROLE_NAMES,
    ROLE_TO_ID,
    ClauseTarget,
    clause_from_record,
    validate_row,
)
from diverge_tol1_ir import (
    Action,
    COMPARATORS,
    DIRECT_OPS,
    Instruction,
    TOL1IRError,
    execute_program,
    format_fraction,
    instruction_sha256,
)
from diverge_tol1_runtime import TOL1RuntimeError, decode_instruction


class TOL1ProductError(RuntimeError):
    """A TOL1 artifact or evaluation contract was violated."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path, expected_sha256: str, split: str) -> list[dict[str, object]]:
    if sha256_path(path) != expected_sha256:
        raise TOL1ProductError(f"TOL1 {split} board hash differs")
    rows = []
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                validate_row(row, split)
            except Exception as error:
                raise TOL1ProductError(
                    f"invalid TOL1 {split} row {line_number}"
                ) from error
            rows.append(row)
    if not rows:
        raise TOL1ProductError(f"empty TOL1 {split} board")
    return rows


def row_clauses(row: dict[str, object]) -> tuple[ClauseTarget, ...]:
    return tuple(clause_from_record(value) for value in row["clauses"])


def flatten_clauses(rows: Sequence[dict[str, object]]) -> list[ClauseTarget]:
    return [clause for row in rows for clause in row_clauses(row)]


def tensorize_clauses(clauses: Sequence[ClauseTarget], device: torch.device):
    if not clauses:
        raise TOL1ProductError("cannot tensorize an empty clause batch")
    byte_ids = torch.full(
        (len(clauses), MAX_CLAUSE_BYTES), PAD_ID, dtype=torch.long, device=device
    )
    attention = torch.zeros_like(byte_ids, dtype=torch.bool)
    candidate_batch = []
    candidate_start = []
    candidate_end = []
    role_targets = []
    candidate_counts = []
    for row, clause in enumerate(clauses):
        width = len(clause.byte_ids)
        byte_ids[row, :width] = torch.tensor(
            clause.byte_ids, dtype=torch.long, device=device
        )
        attention[row, :width] = True
        candidate_counts.append(len(clause.candidates))
        for candidate in clause.candidates:
            candidate_batch.append(row)
            candidate_start.append(candidate.start)
            candidate_end.append(candidate.end)
            role_targets.append(candidate.role_id)
    tensors = {
        "byte_ids": byte_ids,
        "attention": attention,
        "candidate_batch": torch.tensor(candidate_batch, dtype=torch.long, device=device),
        "candidate_start": torch.tensor(candidate_start, dtype=torch.long, device=device),
        "candidate_end": torch.tensor(candidate_end, dtype=torch.long, device=device),
        "role_targets": torch.tensor(role_targets, dtype=torch.long, device=device),
        "operation_targets": torch.tensor(
            [value.operation_id for value in clauses], dtype=torch.long, device=device
        ),
        "comparator_targets": torch.tensor(
            [value.comparator_id for value in clauses], dtype=torch.long, device=device
        ),
        "true_action_targets": torch.tensor(
            [value.true_action_id for value in clauses], dtype=torch.long, device=device
        ),
        "false_action_targets": torch.tensor(
            [value.false_action_id for value in clauses], dtype=torch.long, device=device
        ),
    }
    return tensors, tuple(candidate_counts)


def compiler_loss(outputs, tensors) -> tuple[torch.Tensor, dict[str, float]]:
    role, operation, comparator, true_action, false_action = outputs
    role_weights = torch.ones(len(ROLE_NAMES), dtype=torch.float32, device=role.device)
    role_weights[ROLE_TO_ID["NONE"]] = 0.15
    role_loss = F.cross_entropy(role, tensors["role_targets"], weight=role_weights)
    operation_loss = F.cross_entropy(operation, tensors["operation_targets"])
    guard_id = CLAUSE_OPS.index("GUARD")
    guard = tensors["operation_targets"].eq(guard_id)
    if guard.any():
        comparator_loss = F.cross_entropy(
            comparator[guard], tensors["comparator_targets"][guard]
        )
        true_loss = F.cross_entropy(
            true_action[guard], tensors["true_action_targets"][guard]
        )
        false_loss = F.cross_entropy(
            false_action[guard], tensors["false_action_targets"][guard]
        )
    else:
        comparator_loss = comparator.sum() * 0.0
        true_loss = true_action.sum() * 0.0
        false_loss = false_action.sum() * 0.0
    loss = role_loss + operation_loss + comparator_loss + true_loss + false_loss
    with torch.no_grad():
        metrics = {
            "loss": float(loss.detach()),
            "role_loss": float(role_loss.detach()),
            "operation_loss": float(operation_loss.detach()),
            "role_accuracy": float(
                role.argmax(dim=-1).eq(tensors["role_targets"]).float().mean()
            ),
            "operation_accuracy": float(
                operation.argmax(dim=-1)
                .eq(tensors["operation_targets"])
                .float()
                .mean()
            ),
            "guard_subtype_accuracy": float(
                (
                    comparator[guard].argmax(dim=-1).eq(tensors["comparator_targets"][guard])
                    & true_action[guard].argmax(dim=-1).eq(tensors["true_action_targets"][guard])
                    & false_action[guard].argmax(dim=-1).eq(tensors["false_action_targets"][guard])
                ).float().mean()
            ) if guard.any() else 1.0,
        }
    return loss, metrics


def predict_clauses(
    model,
    clauses: Sequence[ClauseTarget],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    structured: list[Instruction | None] = []
    raw: list[Instruction | None] = []
    operation_exact = 0
    comparator_exact = 0
    guard_count = 0
    cursor = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(clauses), batch_size):
            batch = clauses[start : start + batch_size]
            tensors, counts = tensorize_clauses(batch, device)
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
            role, operation, comparator, true_action, false_action = [
                value.detach().float().cpu() for value in outputs
            ]
            role_cursor = 0
            for local, (clause, count) in enumerate(zip(batch, counts, strict=True)):
                operation_exact += int(int(operation[local].argmax()) == clause.operation_id)
                if clause.instruction.operation == "GUARD":
                    guard_count += 1
                    comparator_exact += int(
                        int(comparator[local].argmax()) == clause.comparator_id
                        and int(true_action[local].argmax()) == clause.true_action_id
                        and int(false_action[local].argmax()) == clause.false_action_id
                    )
                sliced = role[role_cursor : role_cursor + count]
                arguments = (
                    clause.candidates,
                    sliced,
                    operation[local],
                    comparator[local],
                    true_action[local],
                    false_action[local],
                )
                for destination, is_structured in ((structured, True), (raw, False)):
                    try:
                        destination.append(
                            decode_instruction(*arguments, structured=is_structured)
                        )
                    except (TOL1RuntimeError, TOL1IRError):
                        destination.append(None)
                role_cursor += count
                cursor += 1
            if role_cursor != len(role):
                raise TOL1ProductError("candidate prediction cursor differs")
    if cursor != len(clauses):
        raise TOL1ProductError("clause prediction cursor differs")
    return {
        "structured": structured,
        "raw": raw,
        "operation_exact": operation_exact,
        "guard_subtype_exact": comparator_exact,
        "guard_count": guard_count,
    }


def _cycle_action(action: Action) -> Action:
    index = DIRECT_OPS.index(action.operation)
    operation = DIRECT_OPS[(index + 1) % len(DIRECT_OPS)]
    return replace(action, operation=operation)


def _operation_shift(program: Sequence[Instruction | None]) -> tuple[Instruction, ...] | None:
    output = []
    for instruction in program:
        if instruction is None:
            return None
        if instruction.operation in DIRECT_OPS:
            assert instruction.action is not None
            action = _cycle_action(instruction.action)
            output.append(Instruction(action.operation, action=action))
        elif instruction.operation == "GUARD":
            assert instruction.predicate and instruction.true_action and instruction.false_action
            comparator_index = COMPARATORS.index(instruction.predicate.comparator)
            output.append(
                replace(
                    instruction,
                    predicate=replace(
                        instruction.predicate,
                        comparator=COMPARATORS[(comparator_index + 1) % len(COMPARATORS)],
                    ),
                    true_action=_cycle_action(instruction.true_action),
                    false_action=_cycle_action(instruction.false_action),
                )
            )
        elif instruction.operation == "SWAP":
            # A dropped swap is the type-preserving opcode intervention.
            continue
        else:
            output.append(instruction)
    return tuple(output)


def _binding_derangement(program: Sequence[Instruction | None]) -> tuple[Instruction, ...] | None:
    valid = [value for value in program if value is not None]
    if len(valid) != len(program):
        return None
    names = sorted(
        {
            name
            for instruction in valid
            for name in (
                instruction.action.target if instruction.action else None,
                instruction.swap_left,
                instruction.swap_right,
                instruction.true_action.target if instruction.true_action else None,
                instruction.false_action.target if instruction.false_action else None,
                instruction.query,
            )
            if name is not None
        }
    )
    if len(names) < 2:
        return None
    mapping = {name: names[(index + 1) % len(names)] for index, name in enumerate(names)}
    output = []
    for instruction in valid:
        if instruction.action is not None:
            action = replace(instruction.action, target=mapping[instruction.action.target])
            output.append(replace(instruction, action=action))
        elif instruction.operation == "SWAP":
            assert instruction.swap_left and instruction.swap_right
            output.append(
                replace(
                    instruction,
                    swap_left=mapping[instruction.swap_left],
                    swap_right=mapping[instruction.swap_right],
                )
            )
        elif instruction.operation == "GUARD":
            assert instruction.true_action and instruction.false_action
            output.append(
                replace(
                    instruction,
                    true_action=replace(
                        instruction.true_action,
                        target=mapping[instruction.true_action.target],
                    ),
                    false_action=replace(
                        instruction.false_action,
                        target=mapping[instruction.false_action.target],
                    ),
                )
            )
        else:
            output.append(instruction)
    return tuple(output)


def _answer(program: Sequence[Instruction | None] | None) -> str | None:
    if program is None or any(value is None for value in program):
        return None
    try:
        answer, _ = execute_program(tuple(value for value in program if value is not None))
    except TOL1IRError:
        return None
    return format_fraction(answer)


def evaluate_programs(
    model,
    rows: Sequence[dict[str, object]],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    clauses = flatten_clauses(rows)
    prediction = predict_clauses(model, clauses, device=device, batch_size=batch_size)
    structured = prediction["structured"]
    raw = prediction["raw"]
    structured_instruction_exact = sum(
        value is not None and instruction_sha256(value) == gold.instruction_sha256
        for value, gold in zip(structured, clauses, strict=True)
    )
    raw_instruction_exact = sum(
        value is not None and instruction_sha256(value) == gold.instruction_sha256
        for value, gold in zip(raw, clauses, strict=True)
    )
    counts = Counter()
    feature_counts = Counter()
    feature_correct = Counter()
    transcripts = []
    cursor = 0
    for row in rows:
        width = len(row["clauses"])
        treatment_program = structured[cursor : cursor + width]
        raw_program = raw[cursor : cursor + width]
        cursor += width
        gold_hashes = [str(value["instruction_sha256"]) for value in row["clauses"]]
        program_exact = all(
            value is not None and instruction_sha256(value) == digest
            for value, digest in zip(treatment_program, gold_hashes, strict=True)
        )
        treatment_answer = _answer(treatment_program)
        raw_answer = _answer(raw_program)
        shifted_answer = _answer(_operation_shift(treatment_program))
        deranged_answer = _answer(_binding_derangement(treatment_program))
        expected = str(row["answer"])
        correct = treatment_answer == expected
        counts["program_exact"] += int(program_exact)
        counts["treatment_answer"] += int(correct)
        counts["raw_answer"] += int(raw_answer == expected)
        counts["operation_shift_answer"] += int(shifted_answer == expected)
        counts["binding_derangement_answer"] += int(deranged_answer == expected)
        counts["state_reset_answer"] += 0
        counts["query_only_answer"] += int(expected == "0")
        for feature, present in row["features"].items():
            if present:
                feature_counts[feature] += 1
                feature_correct[feature] += int(correct)
        if len(transcripts) < 24:
            transcripts.append(
                {
                    "id": row["id"],
                    "expected": expected,
                    "treatment": treatment_answer,
                    "raw": raw_answer,
                    "program_exact": program_exact,
                }
            )
    if cursor != len(clauses):
        raise TOL1ProductError("program evaluation cursor differs")
    return {
        "schema": "shohin-diverge-tol1-evaluation-v1",
        "rows": len(rows),
        "clauses": len(clauses),
        "operation_exact": prediction["operation_exact"],
        "guard_subtype_exact": prediction["guard_subtype_exact"],
        "guard_count": prediction["guard_count"],
        "structured_valid": sum(value is not None for value in structured),
        "raw_valid": sum(value is not None for value in raw),
        "structured_instruction_exact": structured_instruction_exact,
        "raw_instruction_exact": raw_instruction_exact,
        "counts": dict(counts),
        "feature_counts": dict(feature_counts),
        "feature_correct": dict(feature_correct),
        "transcripts": transcripts,
    }


__all__ = [
    "TOL1ProductError",
    "compiler_loss",
    "evaluate_programs",
    "flatten_clauses",
    "load_rows",
    "predict_clauses",
    "row_clauses",
    "sha256_path",
    "tensorize_clauses",
]
