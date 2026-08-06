#!/usr/bin/env python3
"""Focused checks for the FTA1 finite-state source compiler."""

import torch

from diverge_fta1_runtime import FTA1Config, FiniteStateSourceCompiler


def test_shapes_gradients_and_padding_invariance() -> None:
    config = FTA1Config(width=32, layers=1)
    model = FiniteStateSourceCompiler(config)
    ids = torch.zeros(2, config.max_bytes, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    ids[:, 0] = 1
    ids[:, 1:5] = torch.tensor([67, 68, 69, 70])
    mask[:, :5] = True
    role, operation = model(ids, mask)
    assert role.shape == (2, config.max_bytes, 9)
    assert operation.shape == (2, 11)
    changed = ids.clone()
    changed[:, 20:] = 99
    role_changed, operation_changed = model(changed, mask)
    assert torch.equal(role[:, :5], role_changed[:, :5])
    assert torch.equal(operation, operation_changed)
    (role.sum() + operation.sum()).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def main() -> None:
    test_shapes_gradients_and_padding_invariance()
    print("diverge FTA1 runtime tests passed")


if __name__ == "__main__":
    main()
