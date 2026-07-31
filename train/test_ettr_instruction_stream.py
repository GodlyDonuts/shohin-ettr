from __future__ import annotations

import numpy as np
import pytest
import torch

from endogenous_typed_theory_reactor import TheoryReactorError
from ettr_instruction_stream import (
    WeightedPackedInstructionStream,
    to_device_batch,
)


def _arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inputs = np.arange(96, dtype=np.int64).reshape(24, 4)
    targets = inputs.copy()
    targets[:, :2] = -1
    groups = np.asarray(["math"] * 12 + ["code"] * 12, dtype=object)
    return inputs, targets, groups


def test_weighted_stream_is_deterministic_and_counts_positions() -> None:
    inputs, targets, groups = _arrays()
    first = WeightedPackedInstructionStream(
        inputs,
        targets,
        groups,
        batch_size=4,
        sample_weights={"math": 0.6, "code": 0.4},
        seed=17,
    )
    second = WeightedPackedInstructionStream(
        inputs,
        targets,
        groups,
        batch_size=4,
        sample_weights={"math": 0.6, "code": 0.4},
        seed=17,
    )
    for _ in range(10):
        x1, y1, charged1 = first.peek()
        x2, y2, charged2 = second.peek()
        assert np.array_equal(x1, x2)
        assert np.array_equal(y1, y2)
        assert charged1 == charged2 == 8
        assert first.advance() == second.advance()
    receipt = first.receipt
    assert receipt.batches_consumed == 10
    assert receipt.supervised_positions == 80
    assert receipt.forward_positions == 160


def test_weighted_stream_resume_matches_original() -> None:
    inputs, targets, groups = _arrays()
    stream = WeightedPackedInstructionStream(
        inputs,
        targets,
        groups,
        batch_size=4,
        sample_weights={"math": 1, "code": 1},
        seed=29,
    )
    for _ in range(7):
        stream.peek()
        receipt = stream.advance()
    resumed = WeightedPackedInstructionStream(
        inputs,
        targets,
        groups,
        batch_size=4,
        sample_weights={"math": 1, "code": 1},
        seed=29,
        receipt=receipt,
    )
    original_batch = stream.peek()
    resumed_batch = resumed.peek()
    assert np.array_equal(original_batch[0], resumed_batch[0])
    assert np.array_equal(original_batch[1], resumed_batch[1])
    assert original_batch[2] == resumed_batch[2]


def test_weighted_stream_rejects_unknown_group_and_double_advance() -> None:
    inputs, targets, groups = _arrays()
    with pytest.raises(TheoryReactorError, match="absent"):
        WeightedPackedInstructionStream(
            inputs,
            targets,
            groups,
            batch_size=4,
            sample_weights={"missing": 1},
            seed=1,
        )
    stream = WeightedPackedInstructionStream(
        inputs,
        targets,
        groups,
        batch_size=4,
        sample_weights={"math": 1, "code": 1},
        seed=1,
    )
    with pytest.raises(TheoryReactorError, match="before a batch"):
        stream.advance()


def test_to_device_batch_materializes_long_tensors() -> None:
    inputs, targets, _ = _arrays()
    x, y = to_device_batch(inputs[:2], targets[:2], torch.device("cpu"))
    assert x.dtype == torch.long
    assert y.dtype == torch.long
    assert x.shape == y.shape == (2, 4)
