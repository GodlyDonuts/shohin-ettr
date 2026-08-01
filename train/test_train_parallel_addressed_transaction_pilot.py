from __future__ import annotations

from types import SimpleNamespace

import torch

from train_parallel_addressed_transaction_pilot import (
    _balanced_categorical_loss,
    _schedule_counts,
    _schedule_loss,
)


def _schedule() -> SimpleNamespace:
    values = {
        "opcode": torch.tensor(
            [
                [
                    [0.9, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02],
                    [0.01, 0.01, 0.01, 0.9, 0.02, 0.01, 0.01, 0.01, 0.02],
                ]
            ]
        ),
        "source": torch.tensor([[[0.7, 0.3], [0.4, 0.6]]]),
        "target": torch.tensor([[[0.6, 0.4], [0.3, 0.7]]]),
        "relation": torch.tensor([[[0.8, 0.2], [0.1, 0.9]]]),
        "type_index": torch.tensor([[[0.9, 0.1], [0.2, 0.8]]]),
        "value_code": torch.tensor([[[0.9, 0.1], [0.2, 0.8]]]),
    }
    return SimpleNamespace(
        **values,
        **{
            f"applied_{name}": torch.nn.functional.one_hot(
                value.argmax(-1),
                value.shape[-1],
            ).float()
            for name, value in values.items()
        },
    )


def _targets() -> SimpleNamespace:
    return SimpleNamespace(
        opcode=torch.tensor([[0, 3]]),
        source=torch.tensor([[0, 1]]),
        target=torch.tensor([[0, 1]]),
        relation=torch.tensor([[0, 1]]),
        type_index=torch.tensor([[0, 1]]),
        value_code=torch.tensor([[0, 1]]),
        step_mask=torch.tensor([[True, True]]),
    )


def test_balanced_categorical_loss_balances_observed_classes() -> None:
    probabilities = torch.tensor(
        [[[0.8, 0.2], [0.9, 0.1], [0.4, 0.6]]],
        requires_grad=True,
    )
    targets = torch.tensor([[0, 0, 1]])
    mask = torch.ones_like(targets, dtype=torch.bool)
    loss = _balanced_categorical_loss(probabilities, targets, mask)
    assert loss is not None
    expected = 0.5 * (
        -torch.tensor([0.8, 0.9]).log().mean() - torch.tensor(0.6).log()
    )
    assert torch.allclose(loss, expected)
    loss.backward()
    assert bool(torch.isfinite(probabilities.grad).all())


def test_schedule_loss_and_counts_cover_all_heads() -> None:
    schedule = _schedule()
    targets = _targets()
    loss, parts = _schedule_loss(schedule, targets)
    assert bool(torch.isfinite(loss))
    assert set(parts) == {
        "opcode",
        "source",
        "target",
        "relation",
        "type_index",
        "value_code",
    }
    counts = _schedule_counts(schedule, targets)
    assert counts["opcode"] == (2, 2)
    assert counts["joint"] == (2, 2)
