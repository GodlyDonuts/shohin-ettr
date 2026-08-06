#!/usr/bin/env python3
"""Contract tests for DIVERGE-QTG1."""

from __future__ import annotations

from dataclasses import replace

import torch

from diverge_qtg1_data import FIELD_QUERIES
from diverge_qtg1_runtime import (
    FIELD_COUNT,
    MAX_CANDIDATES,
    NONE_VALUE,
    REGISTER_COUNT,
    VALUE_COUNT,
    QTG1Config,
    QTG1ContractError,
    QueryConditionedGatherer,
    QueryGatherLogits,
    source_audit,
)


def _perfect_logits() -> tuple[QueryGatherLogits, torch.Tensor]:
    pointer = torch.full((1, FIELD_COUNT, MAX_CANDIDATES), -20.0)
    value = torch.full(
        (1, FIELD_COUNT, MAX_CANDIDATES, VALUE_COUNT + 1), -20.0
    )
    value[..., NONE_VALUE] = 10.0
    for field in range(FIELD_COUNT):
        pointer[:, field, field] = 20.0
        value[:, field, field, NONE_VALUE] = -20.0
        value[:, field, field, 30 + field] = 20.0
    numeric_margin = value[..., :VALUE_COUNT].logsumexp(-1) - value[..., NONE_VALUE]
    return QueryGatherLogits(pointer, value, pointer + numeric_margin), torch.ones(
        1, FIELD_COUNT, MAX_CANDIDATES, dtype=torch.bool
    )


def test_queries_are_complete_and_disjoint() -> None:
    assert len(FIELD_QUERIES) == FIELD_COUNT
    assert len(set(FIELD_QUERIES)) == FIELD_COUNT
    assert not any("alpha" in query or "antecedent" in query for query in FIELD_QUERIES)


def test_atomic_query_binding() -> None:
    model = QueryConditionedGatherer(
        QTG1Config(input_width=8, width=16, heads=4, layers=1, pointer_width=8)
    )
    logits, mask = _perfect_logits()
    binding = model.decode(logits, mask)
    assert bool(binding.valid.item())
    assert binding.before.tolist() == [[30, 31, 32, 33, 34]]
    assert binding.after.tolist() == [[35, 36, 37, 38, 39]]
    assert len(set(binding.provenance[0].tolist())) == FIELD_COUNT

    shuffled = model.decode(
        replace(
            logits,
            pointer=logits.pointer.roll(1, dims=1),
            value=logits.value.roll(1, dims=1),
            field=logits.field.roll(1, dims=1),
        ),
        mask,
    )
    assert bool(shuffled.valid.item())
    assert not shuffled.selected_values.equal(binding.selected_values)


def test_short_source_and_audit() -> None:
    model = QueryConditionedGatherer(
        QTG1Config(input_width=8, width=16, heads=4, layers=1, pointer_width=8)
    )
    logits, mask = _perfect_logits()
    short = QueryGatherLogits(
        logits.pointer[..., :9], logits.value[..., :9, :], logits.field[..., :9]
    )
    try:
        model.decode(short, mask[..., :9])
    except QTG1ContractError:
        pass
    else:
        raise AssertionError("QTG1 accepted an incomplete source")
    assert source_audit()["pass"]


if __name__ == "__main__":
    test_queries_are_complete_and_disjoint()
    test_atomic_query_binding()
    test_short_source_and_audit()
    print("DIVERGE-QTG1 tests passed")
