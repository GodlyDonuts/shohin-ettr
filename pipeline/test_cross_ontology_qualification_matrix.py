from __future__ import annotations

from collections import Counter

from cross_ontology_qualification_matrix import (
    CHALLENGES_PER_VARIANT,
    FOLDS,
    PRIMARY_EXECUTIONS,
    THEORIES_PER_FOLD,
    VARIANTS_PER_THEORY,
    audit_qualification_matrix,
    build_qualification_matrix,
)


def test_primary_matrix_has_frozen_geometry_and_passes_audit() -> None:
    rows = build_qualification_matrix()
    receipt = audit_qualification_matrix(rows)
    assert len(rows) == PRIMARY_EXECUTIONS == 2_688
    assert receipt.fold_count == FOLDS == 3
    assert receipt.heldout_theory_count == (
        FOLDS * THEORIES_PER_FOLD
    )
    assert receipt.source_world_count == (
        FOLDS * THEORIES_PER_FOLD * VARIANTS_PER_THEORY
    )
    assert receipt.canonical_challenge_count == (
        FOLDS * THEORIES_PER_FOLD * CHALLENGES_PER_VARIANT
    )
    assert receipt.primary_execution_count == 2_688
    assert receipt.semantic_separation_execution_count > 0
    assert receipt.abstention_execution_count == (
        FOLDS * THEORIES_PER_FOLD * CHALLENGES_PER_VARIANT
    )
    assert receipt.family_label_leak_count == 0
    assert receipt.unique_row_count == 2_688
    assert receipt.unique_theory_hash_count == 24
    assert len(receipt.payload_sha256) == 64
    assert receipt.all_contracts_pass


def test_every_fold_theory_variant_and_challenge_is_balanced() -> None:
    rows = build_qualification_matrix()
    assert Counter(row.fold for row in rows) == {
        fold: 896 for fold in range(3)
    }
    assert set(
        Counter(
            (row.fold, row.theory_index)
            for row in rows
        ).values()
    ) == {112}
    assert set(
        Counter(
            (row.fold, row.theory_index, row.variant)
            for row in rows
        ).values()
    ) == {16}


def test_source_and_assessor_hash_surfaces_are_separate() -> None:
    rows = build_qualification_matrix()
    assert all(
        len(row.compiler_source_sha256) == 64
        and len(row.challenge_sha256) == 64
        and len(row.expected_sha256) == 64
        and len(row.row_sha256) == 64
        for row in rows
    )
    assert all(
        row.compiler_source_sha256 != row.expected_sha256
        and row.challenge_sha256 != row.expected_sha256
        for row in rows
    )
