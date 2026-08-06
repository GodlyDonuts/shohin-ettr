#!/usr/bin/env python3
"""Focused tensor/loss checks for the DIVERGE-TOL1 product path."""

import torch

from diverge_tol1_data import generate_split
from diverge_tol1_product import compiler_loss, flatten_clauses, tensorize_clauses
from diverge_tol1_runtime import TOL1Config, TypedOperationCompiler


def main() -> None:
    clauses = flatten_clauses(generate_split("train", 4, 2026080501))[:16]
    device = torch.device("cpu")
    tensors, counts = tensorize_clauses(clauses, device)
    model = TypedOperationCompiler(TOL1Config(width=32, layers=1))
    outputs = model(
        tensors["byte_ids"],
        tensors["attention"],
        tensors["candidate_batch"],
        tensors["candidate_start"],
        tensors["candidate_end"],
    )
    loss, metrics = compiler_loss(outputs, tensors)
    assert torch.isfinite(loss)
    assert sum(counts) == len(tensors["role_targets"])
    assert 0.0 <= metrics["role_accuracy"] <= 1.0
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    print("diverge TOL1 product tests passed")


if __name__ == "__main__":
    main()
