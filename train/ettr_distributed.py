"""Distributed gradient and stream-cursor primitives for ETTR training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn as nn

from endogenous_typed_theory_reactor import TheoryReactorError


@dataclass(frozen=True, slots=True)
class ETTRDistributedCursor:
    """A rank-independent cursor at an optimizer-update boundary."""

    epoch: int
    position: int

    def validate(
        self,
        *,
        core_batches: int,
        world_size: int,
        accumulation: int,
    ) -> int:
        values = (
            self.epoch,
            self.position,
            core_batches,
            world_size,
            accumulation,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in values
        ):
            raise TheoryReactorError(
                "ETTR distributed cursor contains a non-integer"
            )
        if (
            self.epoch < 0
            or self.position < 0
            or core_batches < 1
            or world_size < 1
            or accumulation < 1
        ):
            raise TheoryReactorError("ETTR distributed cursor differs")
        update_span = world_size * accumulation
        usable = core_batches - core_batches % update_span
        if self.position > usable or self.position % update_span:
            raise TheoryReactorError(
                "ETTR distributed cursor is not an update boundary"
            )
        return usable

    def advance(
        self,
        *,
        core_batches: int,
        world_size: int,
        accumulation: int,
    ) -> "ETTRDistributedCursor":
        usable = self.validate(
            core_batches=core_batches,
            world_size=world_size,
            accumulation=accumulation,
        )
        next_position = self.position + world_size * accumulation
        if next_position < usable:
            return ETTRDistributedCursor(self.epoch, next_position)
        if next_position == usable:
            return ETTRDistributedCursor(self.epoch + 1, 0)
        raise TheoryReactorError(
            "ETTR distributed cursor advanced beyond the usable epoch"
        )


class ETTRDistributedGradientAverager:
    """Average dense gradients with fixed-order bounded-size collectives."""

    def __init__(
        self,
        *,
        world_size: int,
        all_reduce_sum: Callable[[torch.Tensor], None],
        max_bucket_bytes: int = 64 * 1024 * 1024,
    ):
        if (
            not isinstance(world_size, int)
            or isinstance(world_size, bool)
            or world_size < 1
            or not callable(all_reduce_sum)
            or not isinstance(max_bucket_bytes, int)
            or max_bucket_bytes < 1024
        ):
            raise TheoryReactorError(
                "ETTR distributed gradient contract differs"
            )
        self.world_size = world_size
        self.all_reduce_sum = all_reduce_sum
        self.max_bucket_bytes = max_bucket_bytes

    def __call__(self, parameters: Sequence[nn.Parameter]) -> None:
        values = tuple(parameters)
        if not values:
            raise TheoryReactorError(
                "ETTR distributed gradient set is empty"
            )
        if self.world_size == 1:
            return
        devices = {parameter.device for parameter in values}
        if len(devices) != 1:
            raise TheoryReactorError(
                "ETTR distributed parameters span devices"
            )
        device = next(iter(devices))
        presence = torch.tensor(
            [parameter.grad is not None for parameter in values],
            dtype=torch.int32,
            device=device,
        )
        self.all_reduce_sum(presence)
        if torch.any((presence < 0) | (presence > self.world_size)):
            raise TheoryReactorError(
                "ETTR distributed gradient presence differs"
            )

        active: list[nn.Parameter] = []
        for parameter, count in zip(values, presence.tolist(), strict=True):
            if count == 0:
                continue
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            if parameter.grad.is_sparse:
                raise TheoryReactorError(
                    "ETTR distributed gradients must be dense"
                )
            active.append(parameter)
        if not active:
            raise TheoryReactorError(
                "ETTR distributed update has no gradients"
            )

        bucket: list[nn.Parameter] = []
        bucket_bytes = 0
        bucket_key: tuple[torch.device, torch.dtype] | None = None
        for parameter in active:
            gradient = parameter.grad
            if gradient is None:
                raise TheoryReactorError(
                    "ETTR distributed gradient disappeared"
                )
            key = (gradient.device, gradient.dtype)
            size = gradient.numel() * gradient.element_size()
            if bucket and (
                key != bucket_key
                or bucket_bytes + size > self.max_bucket_bytes
            ):
                self._reduce_bucket(bucket)
                bucket = []
                bucket_bytes = 0
            bucket.append(parameter)
            bucket_bytes += size
            bucket_key = key
        if bucket:
            self._reduce_bucket(bucket)

    def _reduce_bucket(self, parameters: Sequence[nn.Parameter]) -> None:
        gradients = [parameter.grad for parameter in parameters]
        if any(gradient is None for gradient in gradients):
            raise TheoryReactorError(
                "ETTR distributed bucket contains a missing gradient"
            )
        dense = [gradient for gradient in gradients if gradient is not None]
        flat = torch.cat([gradient.reshape(-1) for gradient in dense])
        self.all_reduce_sum(flat)
        flat.div_(self.world_size)
        cursor = 0
        for gradient in dense:
            end = cursor + gradient.numel()
            gradient.copy_(flat[cursor:end].view_as(gradient))
            cursor = end


__all__ = [
    "ETTRDistributedCursor",
    "ETTRDistributedGradientAverager",
]
