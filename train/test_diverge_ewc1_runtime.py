from __future__ import annotations

from dataclasses import replace
import random

import torch

from diverge_ewc1_data import TRAIN_PAIRS, _record
from diverge_ewc1_runtime import (
    EquivariantWorldCompiler,
    WorldCompilerConfig,
    hard_numeric_assignment,
    tensorize_worlds,
)


def _rows(count: int = 3):
    return [
        _record(
            split="train",
            seed=29,
            serial=index,
            pair=TRAIN_PAIRS[index % len(TRAIN_PAIRS)],
            rng=random.Random(index + 31),
        )
        for index in range(count)
    ]


def test_tensorized_world_shapes_and_forward():
    rows = _rows()
    device = torch.device("cpu")
    batch = tensorize_worlds(rows, device)
    model = EquivariantWorldCompiler(WorldCompilerConfig())
    numeric, operations = model(batch)
    assert numeric.shape == (3, 2, 8)
    assert operations.shape == (3, 40)
    assert torch.isfinite(numeric[:, :, :2]).all()


def test_register_order_action_is_exact_in_equivariant_forward():
    rows = _rows(1)
    swapped = dict(rows[0])
    swapped["registers"] = list(reversed(swapped["registers"]))
    device = torch.device("cpu")
    torch.manual_seed(7)
    model = EquivariantWorldCompiler(WorldCompilerConfig()).eval()
    first, first_ops = model(tensorize_worlds(rows, device))
    second, second_ops = model(tensorize_worlds([swapped], device))
    assert torch.equal(first[:, 0], second[:, 1])
    assert torch.equal(first[:, 1], second[:, 0])
    assert torch.equal(first_ops, second_ops)


def test_absolute_control_does_not_follow_register_order():
    row = _rows(1)[0]
    swapped = dict(row)
    swapped["registers"] = list(reversed(swapped["registers"]))
    device = torch.device("cpu")
    torch.manual_seed(7)
    model = EquivariantWorldCompiler(
        replace(WorldCompilerConfig(), mode="absolute")
    ).eval()
    first, _ = model(tensorize_worlds([row], device))
    second, _ = model(tensorize_worlds([swapped], device))
    assert torch.equal(first, second)


def test_hard_numeric_assignment_is_one_to_one():
    logits = torch.tensor([[9.0, 8.0, 0.0], [9.0, 1.0, 7.0]])
    assert hard_numeric_assignment(logits, 3) == (1, 0)
