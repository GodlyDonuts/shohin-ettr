from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from pipeline.episode_functor_law_collision_board import (
    PathObservationClause,
    audit_version_space,
    delete_law,
    parse_source,
)
from pipeline.episode_functor_law_collision_family import (
    CELL_NAMES,
    FAMILY_RECEIPT_SCHEMA,
    FEATURE_NAMES,
    STATUS,
    GeneratedCollisionFamilyConfig,
    GeneratedCollisionFamilyError,
    ShallowExample,
    audit_generated_collision_family,
    audit_shallow_shortcuts,
)


@pytest.fixture(scope="module")
def family():
    return audit_generated_collision_family()


def test_family_is_deterministic_and_canonical(family) -> None:
    repeated = audit_generated_collision_family()
    assert family.receipt.schema == FAMILY_RECEIPT_SCHEMA
    assert family.receipt.status == STATUS
    assert family.receipt.canonical_bytes() == repeated.receipt.canonical_bytes()
    assert family.receipt.receipt_sha256 == repeated.receipt.receipt_sha256


def test_family_has_many_balanced_variable_units(family) -> None:
    assert len(family.units) == 32
    assert family.receipt.cell_count == 128
    assert family.receipt.target_counts == ((0, 64), (1, 64))
    assert len(family.receipt.geometries) == 8
    assert {geometry[0] for geometry in family.receipt.geometries} == {4, 5, 6, 7}
    assert {geometry[1] for geometry in family.receipt.geometries} == {2, 3}
    assert {geometry[2] for geometry in family.receipt.geometries} == {1, 2}
    assert family.receipt.query_address_count >= 16
    assert len(
        {
            (
                unit.late_query.start_index,
                unit.late_query.action_indices,
                unit.late_query.observer_index,
            )
            for unit in family.units
        }
    ) >= 16


def test_every_unit_is_an_exact_opposite_law_fact_collision(family) -> None:
    for unit in family.units:
        assert unit.gates == dict(unit.receipt.gates)
        assert all(unit.gates.values())
        assert Counter(unit.receipt.late_answer_indices) == Counter({0: 2, 1: 2})
        expected = (0, 1, 1, 0) if unit.parity == 0 else (1, 0, 0, 1)
        assert unit.receipt.late_answer_indices == expected
        assert len(set(unit.receipt.source_sha256s)) == 4
        assert unit.receipt.fact_invariant_sha256s[0] != (
            unit.receipt.fact_invariant_sha256s[1]
        )
        assert (
            unit.fact_invariants[0].complete_action_cycle_types
            != unit.fact_invariants[1].complete_action_cycle_types
        )
        for source, source_audit in zip(
            unit.sources, unit.source_audits, strict=True
        ):
            assert audit_version_space(source).receipt == source_audit.receipt
            assert source_audit.receipt.direct_completion_count == 2
            assert source_audit.receipt.law_completion_count == 1
            assert source_audit.receipt.resolution == "unique-completion"
            deleted = audit_version_space(delete_law(source))
            assert deleted.receipt.direct_completion_count == 2
            assert deleted.receipt.law_completion_count == 2


def test_law_twins_share_facts_and_swap_only_expected_answer(family) -> None:
    for unit in family.units:
        for left_index, right_index in ((0, 1), (2, 3)):
            left = parse_source(unit.sources[left_index])
            right = parse_source(unit.sources[right_index])
            assert left.evidence == right.evidence
            assert len(left.clauses) == len(right.clauses) == 1
            left_clause = left.clauses[0]
            right_clause = right.clauses[0]
            assert isinstance(left_clause, PathObservationClause)
            assert isinstance(right_clause, PathObservationClause)
            assert left_clause.start == right_clause.start
            assert left_clause.actions == right_clause.actions
            assert left_clause.observer == right_clause.observer
            assert left_clause.expected == right_clause.alternate
            assert left_clause.alternate == right_clause.expected


def test_receipts_bind_sources_audits_queries_and_invariants(family) -> None:
    assert family.receipt.unit_receipt_sha256s == tuple(
        unit.receipt.receipt_sha256 for unit in family.units
    )
    assert family.receipt.shortcut_audit_receipt_sha256 == (
        family.shortcut_audit.receipt_sha256
    )
    for unit in family.units:
        assert unit.receipt.source_sha256s == tuple(
            audit.receipt.source_sha256 for audit in unit.source_audits
        )
        assert unit.receipt.source_receipt_sha256s == tuple(
            audit.receipt.receipt_sha256 for audit in unit.source_audits
        )
        assert unit.receipt.non_law_sha256s[0] == unit.receipt.non_law_sha256s[1]
        assert unit.receipt.non_law_sha256s[2] == unit.receipt.non_law_sha256s[3]
        assert unit.receipt.non_law_sha256s[0] != unit.receipt.non_law_sha256s[2]


def test_preregistered_shortcut_attacks_are_held_out_and_chance_compatible(
    family,
) -> None:
    shortcut = family.shortcut_audit
    assert set(shortcut.training_unit_ids).isdisjoint(
        shortcut.evaluation_unit_ids
    )
    assert shortcut.training_examples == shortcut.evaluation_examples == 64
    assert shortcut.target_counts == ((0, 64), (1, 64))
    assert shortcut.feature_names == FEATURE_NAMES
    assert shortcut.chance_compatible
    assert shortcut.decision == "mechanics-family-retained-no-reasoning-claim"
    assert all(classifier.accuracy_gate for classifier in shortcut.classifiers)
    assert all(
        classifier.randomization_gate for classifier in shortcut.classifiers
    )
    assert family.receipt.all_exact_unit_gates_passed
    assert family.receipt.shortcut_chance_gates_passed
    assert family.receipt.decision == (
        "mechanics-family-retained-no-reasoning-claim"
    )
    assert not family.receipt.promotion_eligible
    assert "not" in family.receipt.claim_boundary.lower()
    assert "reasoning claim" in family.receipt.claim_boundary.lower()


def test_shortcut_audit_returns_explicit_no_go_for_a_leaked_feature() -> None:
    examples = []
    for unit in range(16):
        split = "train" if unit < 8 else "evaluation"
        for cell, name in enumerate(CELL_NAMES):
            target = cell % 2
            features = (float(target),) + (0.0,) * (len(FEATURE_NAMES) - 1)
            examples.append(
                ShallowExample(
                    unit_id=f"synthetic-{unit}",
                    split=split,
                    cell_name=name,
                    features=features,
                    target=target,
                )
            )
    receipt = audit_shallow_shortcuts(examples)
    assert not receipt.chance_compatible
    assert receipt.decision == "no-go-shallow-shortcut-leakage"
    assert any(
        classifier.accuracy == 1.0 and not classifier.accuracy_gate
        for classifier in receipt.classifiers
    )


def test_shortcut_audit_rejects_cell_level_split_leakage() -> None:
    examples = [
        ShallowExample(
            unit_id="same-unit",
            split="train" if index < 2 else "evaluation",
            cell_name=CELL_NAMES[index],
            features=(0.0,) * len(FEATURE_NAMES),
            target=index % 2,
        )
        for index in range(4)
    ]
    with pytest.raises(
        GeneratedCollisionFamilyError,
        match="leaked cells",
    ):
        audit_shallow_shortcuts(examples)


@pytest.mark.parametrize(
    "replacement, message",
    (
        ({"unit_count": 31}, "multiple of 32"),
        ({"unit_count": 48}, "multiple of 32"),
        ({"maximum_law_depth": 5}, "law depths"),
        ({"shortcut_accuracy_ceiling": 0.5}, "accuracy ceiling"),
    ),
)
def test_invalid_family_configs_fail_closed(replacement, message) -> None:
    with pytest.raises(GeneratedCollisionFamilyError, match=message):
        GeneratedCollisionFamilyConfig(**replacement)


def test_audit_does_not_promote_even_if_ceiling_is_relaxed(family) -> None:
    relaxed = replace(
        family.config,
        shortcut_accuracy_ceiling=0.99,
        shortcut_alpha=1e-9,
    )
    audit = audit_generated_collision_family(relaxed)
    assert not audit.receipt.promotion_eligible
    assert audit.receipt.status == STATUS
