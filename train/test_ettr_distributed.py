from __future__ import annotations

import pytest
import torch

from endogenous_typed_theory_reactor import TheoryReactorError
from ettr_distributed import (
    ETTRDistributedCursor,
    ETTRDistributedGradientAverager,
)


def test_distributed_cursor_drops_only_complete_update_remainder() -> None:
    cursor = ETTRDistributedCursor(epoch=2, position=0)
    assert cursor.validate(
        core_batches=17,
        world_size=3,
        accumulation=2,
    ) == 12
    cursor = cursor.advance(
        core_batches=17,
        world_size=3,
        accumulation=2,
    )
    assert cursor == ETTRDistributedCursor(epoch=2, position=6)
    cursor = cursor.advance(
        core_batches=17,
        world_size=3,
        accumulation=2,
    )
    assert cursor == ETTRDistributedCursor(epoch=3, position=0)


def test_distributed_cursor_rejects_mid_update_resume() -> None:
    with pytest.raises(TheoryReactorError, match="update boundary"):
        ETTRDistributedCursor(epoch=0, position=3).validate(
            core_batches=16,
            world_size=2,
            accumulation=2,
        )


def test_gradient_averager_reduces_fixed_buckets() -> None:
    first = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    second = torch.nn.Parameter(torch.tensor([3.0]))
    first.grad = torch.tensor([2.0, 4.0])
    second.grad = None
    calls = 0

    def reduce_sum(value: torch.Tensor) -> None:
        nonlocal calls
        calls += 1
        if value.dtype == torch.int32:
            value.copy_(torch.tensor([2, 1], dtype=torch.int32))
        else:
            value.add_(torch.tensor([6.0, 8.0, 10.0]))

    average = ETTRDistributedGradientAverager(
        world_size=2,
        all_reduce_sum=reduce_sum,
    )
    average((first, second))
    assert calls == 2
    torch.testing.assert_close(first.grad, torch.tensor([4.0, 6.0]))
    torch.testing.assert_close(second.grad, torch.tensor([5.0]))


def test_gradient_averager_rejects_globally_empty_update() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))

    def reduce_sum(_value: torch.Tensor) -> None:
        return

    average = ETTRDistributedGradientAverager(
        world_size=2,
        all_reduce_sum=reduce_sum,
    )
    with pytest.raises(TheoryReactorError, match="no gradients"):
        average((parameter,))
