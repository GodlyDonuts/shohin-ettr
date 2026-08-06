#!/usr/bin/env python3
"""Focused data and gradient checks for DIVERGE-TOL3 anchors."""

import torch

from diverge_tol1_data import generate_split
from diverge_tol3_semantic_anchor import (
    COMPARATOR_NAMES,
    OPERATION_NAMES,
    LocalSemanticAnchor,
    TOL3Config,
    build_anchor_examples,
    comparator_phrase,
    decode_direct_action_from_anchor,
    runtime_comparator_phrase,
    select_operation_anchor,
    tensorize_texts,
)


def main() -> None:
    examples = build_anchor_examples(generate_split("train", 24, 2026080501))
    assert {value.label for value in examples if value.task == "operation"} == set(
        range(len(OPERATION_NAMES))
    )
    assert {value.label for value in examples if value.task == "comparator"} == set(
        range(len(COMPARATOR_NAMES))
    )
    assert comparator_phrase("atlas is not below 3/2", "atlas", "3/2") == "is not below"
    assert runtime_comparator_phrase("atlas is not below 3/2", ("atlas",)) == "is not below"
    operation_scores = {
        "to": (4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "atlas": (3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "add": (0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0),
        "blaze": (3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    }
    anchor = select_operation_anchor("to atlas, add blaze", operation_scores)
    assert (anchor.operation, anchor.text) == ("ADD", "add")
    action = decode_direct_action_from_anchor(
        "to atlas, add blaze", "ADD", ("atlas", "blaze"), anchor.end
    )
    assert (action.target, action.operand.kind, action.operand.value) == (
        "atlas",
        "REF",
        "blaze",
    )
    ids, mask = tensorize_texts(["increase", "is not below"], torch.device("cpu"))
    model = LocalSemanticAnchor(TOL3Config(width=32))
    operation, comparator = model(ids, mask)
    assert operation.shape == (2, len(OPERATION_NAMES))
    assert comparator.shape == (2, len(COMPARATOR_NAMES))
    (operation.sum() + comparator.sum()).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    print("diverge TOL3 semantic anchor tests passed")


if __name__ == "__main__":
    main()
