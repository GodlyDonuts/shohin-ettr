#!/usr/bin/env python3
"""Batch-preserving IDR1 evaluation sharding checks."""

from hf_idr1_evaluate_reviser import IDR1EvaluationError, shard_bounds


def _check(total: int, shards: int, batch_size: int) -> None:
    bounds = [shard_bounds(total, index, shards, batch_size) for index in range(shards)]
    assert bounds[0][0] == 0
    assert bounds[-1][1] == total
    assert all(left[1] == right[0] for left, right in zip(bounds, bounds[1:]))
    assert all(start % batch_size == 0 for start, _ in bounds)
    assert all(end % batch_size == 0 for _, end in bounds[:-1])
    assert sum(end - start for start, end in bounds) == total


def test_shard_bounds_preserve_batches() -> None:
    _check(1_289, 4, 2)
    _check(1_279, 4, 2)
    _check(1_280, 8, 4)


def test_shard_bounds_reject_empty_or_invalid_geometry() -> None:
    for args in ((0, 0, 1, 2), (2, 0, 0, 2), (2, 2, 2, 2), (2, 0, 4, 2)):
        try:
            shard_bounds(*args)
        except IDR1EvaluationError:
            pass
        else:
            raise AssertionError(f"accepted invalid shard geometry: {args}")
