"""Tests for frozen ETTR-IL-v3 initializer mechanics."""

from __future__ import annotations

import json

import pytest

from ettr_il_v3_protocol import (
    CHARGED_POSITIONS_PER_ROW,
    CURRICULUM_STAGES,
    FAMILIES,
    MASTER_SEED_SHA256,
    POSITIONS_PER_CORE,
    ProtocolError,
    SPLITS,
    SPLIT_CORES,
    TRAIN_STAGE_CORES,
    candidate_floor,
    corpus_budget,
    cyclic_balanced_allocation,
    orbit_owner,
    protocol_receipt,
    split_family_allocation,
    split_stage_allocation,
    split_stage_family_allocation,
    surplus,
    train_stage_family_allocation,
)


def test_frozen_geometry_and_training_budget() -> None:
    assert CHARGED_POSITIONS_PER_ROW == 528
    assert POSITIONS_PER_CORE == 33_792
    budget = corpus_budget("train")
    assert budget.cores == 40_000
    assert budget.views == 160_000
    assert budget.rows == 2_560_000
    assert budget.charged_positions == 1_351_680_000
    assert sum(TRAIN_STAGE_CORES.values()) == SPLIT_CORES["train"]


def test_every_split_budget_is_exact() -> None:
    for split in SPLITS:
        budget = corpus_budget(split)
        budget.validate()
        assert budget.cores == SPLIT_CORES[split]


def test_surplus_and_candidate_floor_are_frozen() -> None:
    assert surplus(0) == 16
    assert surplus(1) == 16
    assert surplus(64) == 16
    assert surplus(65) == 17
    assert candidate_floor(64) == 240
    with pytest.raises(ProtocolError):
        surplus(-1)
    with pytest.raises(ProtocolError):
        surplus(True)


def test_balanced_allocations_preserve_cardinality() -> None:
    for split in SPLITS:
        allocation = split_family_allocation(split)
        assert tuple(allocation) == FAMILIES
        assert sum(allocation.values()) == SPLIT_CORES[split]
        assert max(allocation.values()) - min(allocation.values()) <= 1
    for stage in CURRICULUM_STAGES:
        allocation = train_stage_family_allocation(stage)
        assert sum(allocation.values()) == TRAIN_STAGE_CORES[stage]
        assert max(allocation.values()) - min(allocation.values()) <= 1
    for split in SPLITS:
        stage_allocation = split_stage_allocation(split)
        matrix = split_stage_family_allocation(split)
        assert tuple(stage_allocation) == CURRICULUM_STAGES
        assert sum(stage_allocation.values()) == SPLIT_CORES[split]
        assert tuple(matrix) == CURRICULUM_STAGES
        for stage in CURRICULUM_STAGES:
            assert tuple(matrix[stage]) == FAMILIES
            assert sum(matrix[stage].values()) == stage_allocation[stage]
            assert max(matrix[stage].values()) - min(
                matrix[stage].values()
            ) <= 1
        family_allocation = split_family_allocation(split)
        assert {
            family: sum(matrix[stage][family] for stage in CURRICULUM_STAGES)
            for family in FAMILIES
        } == family_allocation


def test_train_matrix_has_both_frozen_marginals() -> None:
    matrix = split_stage_family_allocation("train")
    assert {
        stage: sum(matrix[stage].values())
        for stage in CURRICULUM_STAGES
    } == TRAIN_STAGE_CORES
    assert {
        family: sum(matrix[stage][family] for stage in CURRICULUM_STAGES)
        for family in FAMILIES
    } == split_family_allocation("train")


def test_allocation_is_context_bound_and_deterministic() -> None:
    first = cyclic_balanced_allocation(
        10,
        ("a", "b", "c"),
        context={"cell": 1},
    )
    assert first == cyclic_balanced_allocation(
        10,
        ("a", "b", "c"),
        context={"cell": 1},
    )
    assert sum(first.values()) == 10
    with pytest.raises(ProtocolError):
        cyclic_balanced_allocation(1, (), context={})
    with pytest.raises(ProtocolError):
        cyclic_balanced_allocation(1, ("a", "a"), context={})


def test_orbit_owner_ignores_renderer_metadata_not_in_orbit() -> None:
    orbit = {"edges": [[0, 1]], "values": [2, 3]}
    assert orbit_owner(orbit) == orbit_owner(dict(reversed(tuple(orbit.items()))))
    assert orbit_owner(orbit) in {"train", "development", "confirmation"}


def test_protocol_receipt_is_canonical_and_self_bound() -> None:
    first = protocol_receipt()
    second = protocol_receipt()
    assert first == second
    assert first["master_seed_sha256"] == MASTER_SEED_SHA256
    assert len(first["receipt_sha256"]) == 64
    assert (
        first["split_budgets"]["train"]["charged_positions"]
        == 1_351_680_000
    )
    json.dumps(first, allow_nan=False, sort_keys=True)
