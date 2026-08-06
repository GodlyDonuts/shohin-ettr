#!/usr/bin/env python3
"""Shape, gradient, and exact typed-decoder checks for DIVERGE-TOL1."""

import torch

from diverge_tol1_data import (
    ACTION_NAMES,
    CLAUSE_OPS,
    COMPARATOR_NAMES,
    MAX_CLAUSE_BYTES,
    ROLE_NAMES,
    ROLE_TO_ID,
    render_instruction,
)
from diverge_tol1_ir import Action, Atom, Instruction, Predicate
from diverge_tol1_runtime import TOL1Config, TypedOperationCompiler, decode_instruction


def _perfect_logits(clause):
    role = torch.full((len(clause.candidates), len(ROLE_NAMES)), -10.0)
    for index, candidate in enumerate(clause.candidates):
        role[index, candidate.role_id] = 10.0
    operation = torch.full((len(CLAUSE_OPS),), -10.0)
    operation[clause.operation_id] = 10.0
    comparator = torch.full((len(COMPARATOR_NAMES),), -10.0)
    comparator[clause.comparator_id] = 10.0
    true_action = torch.full((len(ACTION_NAMES),), -10.0)
    true_action[clause.true_action_id] = 10.0
    false_action = torch.full((len(ACTION_NAMES),), -10.0)
    false_action[clause.false_action_id] = 10.0
    return role, operation, comparator, true_action, false_action


def main() -> None:
    config = TOL1Config(width=32, layers=1)
    model = TypedOperationCompiler(config)
    ids = torch.zeros((2, MAX_CLAUSE_BYTES), dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    ids[:, 0] = 1
    ids[:, 1:8] = torch.tensor([99, 102, 118, 118, 104, 117, 34])
    mask[:, :8] = True
    batch = torch.tensor([0, 1])
    starts = torch.tensor([0, 0])
    ends = torch.tensor([6, 6])
    outputs = model(ids, mask, batch, starts, ends)
    assert outputs[0].shape == (2, len(ROLE_NAMES))
    assert outputs[1].shape == (2, len(CLAUSE_OPS))
    sum(value.sum() for value in outputs).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())

    instruction = Instruction(
        "GUARD",
        predicate=Predicate("GE", "amber", Atom("CONST", "-3/2")),
        true_action=Action("ADD", "birch", Atom("REF", "cedar")),
        false_action=Action("MULTIPLY", "delta", Atom("CONST", "2")),
    )
    clause = render_instruction(instruction, renderer=1, ood=False)
    logits = _perfect_logits(clause)
    assert decode_instruction(clause.candidates, *logits, structured=True) == instruction
    assert decode_instruction(clause.candidates, *logits, structured=False) == instruction
    assert sum(candidate.role_id != ROLE_TO_ID["NONE"] for candidate in clause.candidates) == 6
    print("diverge TOL1 runtime tests passed")


if __name__ == "__main__":
    main()
