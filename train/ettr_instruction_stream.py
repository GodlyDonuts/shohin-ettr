"""Deterministic completion-masked packed batches for native ETTR training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping

import numpy as np
import torch

from endogenous_typed_theory_reactor import TheoryReactorError


@dataclass(frozen=True, slots=True)
class InstructionStreamReceipt:
    epoch: int
    batch_in_epoch: int
    batches_consumed: int
    supervised_positions: int
    forward_positions: int


class WeightedPackedInstructionStream:
    """Replay a frozen packed corpus with deterministic weighted group sampling."""

    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        groups: np.ndarray,
        *,
        batch_size: int,
        sample_weights: Mapping[str, float],
        seed: int,
        receipt: InstructionStreamReceipt | None = None,
    ) -> None:
        if (
            not isinstance(inputs, np.ndarray)
            or not isinstance(targets, np.ndarray)
            or not isinstance(groups, np.ndarray)
            or inputs.ndim != 2
            or targets.shape != inputs.shape
            or groups.shape != (len(inputs),)
            or inputs.dtype.kind not in {"i", "u"}
            or targets.dtype.kind not in {"i", "u"}
            or len(inputs) < batch_size
            or not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
            or not isinstance(seed, int)
            or isinstance(seed, bool)
            or not 0 <= seed < 2**63
        ):
            raise TheoryReactorError("instruction packed stream tensors differ")
        normalized_weights = {
            str(name): float(weight)
            for name, weight in sample_weights.items()
        }
        if (
            not normalized_weights
            or any(
                not np.isfinite(weight) or weight <= 0
                for weight in normalized_weights.values()
            )
        ):
            raise TheoryReactorError("instruction sample weights differ")
        group_names = {str(value) for value in groups.tolist()}
        missing = set(normalized_weights) - group_names
        if missing:
            raise TheoryReactorError(
                "instruction sample weight names are absent from the corpus"
            )
        self.inputs = np.asarray(inputs, dtype=np.int64, order="C")
        self.targets = np.asarray(targets, dtype=np.int64, order="C")
        self.groups = np.asarray(groups, dtype=object)
        self.batch_size = batch_size
        self.sample_weights = normalized_weights
        self.seed = seed
        self._group_indices = {
            name: np.flatnonzero(self.groups == name)
            for name in sorted(normalized_weights)
        }
        self.batches_per_epoch = len(inputs) // batch_size
        self._receipt = (
            InstructionStreamReceipt(0, 0, 0, 0, 0)
            if receipt is None
            else receipt
        )
        self._validate_receipt(self._receipt)
        self._order = self._epoch_order(self._receipt.epoch)
        self._pending: tuple[np.ndarray, int] | None = None

    @property
    def receipt(self) -> InstructionStreamReceipt:
        return self._receipt

    def peek(self) -> tuple[np.ndarray, np.ndarray, int]:
        if self._pending is not None:
            indices, supervised = self._pending
            return self.inputs[indices], self.targets[indices], supervised
        start = self._receipt.batch_in_epoch * self.batch_size
        stop = start + self.batch_size
        indices = self._order[start:stop]
        if len(indices) != self.batch_size:
            raise TheoryReactorError("instruction batch order differs")
        supervised = int(np.count_nonzero(self.targets[indices] != -1))
        if supervised <= 0:
            raise TheoryReactorError(
                "instruction batch has no supervised positions"
            )
        self._pending = (indices, supervised)
        return self.inputs[indices], self.targets[indices], supervised

    def advance(self) -> InstructionStreamReceipt:
        if self._pending is None:
            raise TheoryReactorError(
                "instruction stream cannot advance before a batch is read"
            )
        _, supervised = self._pending
        batch_in_epoch = self._receipt.batch_in_epoch + 1
        epoch = self._receipt.epoch
        if batch_in_epoch == self.batches_per_epoch:
            epoch += 1
            batch_in_epoch = 0
        receipt = InstructionStreamReceipt(
            epoch=epoch,
            batch_in_epoch=batch_in_epoch,
            batches_consumed=self._receipt.batches_consumed + 1,
            supervised_positions=(
                self._receipt.supervised_positions + supervised
            ),
            forward_positions=(
                self._receipt.forward_positions
                + self.batch_size * self.inputs.shape[1]
            ),
        )
        self._validate_receipt(receipt)
        self._receipt = receipt
        self._pending = None
        if epoch != self._epoch_from_order:
            self._order = self._epoch_order(epoch)
        return receipt

    def state_dict(self) -> dict[str, object]:
        if self._pending is not None:
            raise TheoryReactorError(
                "instruction stream state cannot be saved during a batch"
            )
        return {
            "batch_size": self.batch_size,
            "receipt": asdict(self._receipt),
            "sample_weights": dict(sorted(self.sample_weights.items())),
            "schema": "shohin-ettr-weighted-instruction-stream-v1",
            "seed": self.seed,
        }

    def _epoch_order(self, epoch: int) -> np.ndarray:
        digest = hashlib.sha256(
            json.dumps(
                {"epoch": epoch, "seed": self.seed},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).digest()
        rng = np.random.default_rng(
            int.from_bytes(digest[:8], "big", signed=False)
        )
        names = tuple(sorted(self.sample_weights))
        probabilities = np.asarray(
            [self.sample_weights[name] for name in names],
            dtype=np.float64,
        )
        probabilities /= probabilities.sum()
        count = self.batches_per_epoch * self.batch_size
        choices = rng.choice(len(names), size=count, p=probabilities)
        order = np.empty(count, dtype=np.int64)
        for group_index, name in enumerate(names):
            slots = np.flatnonzero(choices == group_index)
            if len(slots):
                order[slots] = rng.choice(
                    self._group_indices[name],
                    size=len(slots),
                    replace=True,
                )
        self._epoch_from_order = epoch
        return order

    def _validate_receipt(self, receipt: InstructionStreamReceipt) -> None:
        if (
            not isinstance(receipt, InstructionStreamReceipt)
            or receipt.epoch < 0
            or not 0 <= receipt.batch_in_epoch < self.batches_per_epoch
            or receipt.batches_consumed < 0
            or receipt.supervised_positions < 0
            or receipt.forward_positions < 0
            or (
                receipt.batches_consumed
                != receipt.epoch * self.batches_per_epoch
                + receipt.batch_in_epoch
            )
            or (
                (receipt.supervised_positions == 0)
                != (receipt.batches_consumed == 0)
            )
            or (
                receipt.forward_positions
                != receipt.batches_consumed
                * self.batch_size
                * self.inputs.shape[1]
            )
        ):
            raise TheoryReactorError("instruction stream receipt differs")


def to_device_batch(
    inputs: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize one immutable CPU batch on the selected accelerator."""

    return (
        torch.from_numpy(np.asarray(inputs, dtype=np.int64)).to(device),
        torch.from_numpy(np.asarray(targets, dtype=np.int64)).to(device),
    )


__all__ = [
    "InstructionStreamReceipt",
    "WeightedPackedInstructionStream",
    "to_device_batch",
]
