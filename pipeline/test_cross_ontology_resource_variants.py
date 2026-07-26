from __future__ import annotations

import inspect

from cross_ontology_resource_board import (
    PLACE_SPECS,
    ProcessStatus,
    THEORIES,
    challenge_cases,
    execute_sequence,
)
from cross_ontology_resource_variants import (
    AnswerDirective,
    CHALLENGES_PER_VARIANT,
    ExecutionSemantics,
    PairExpectation,
    QUALIFICATION_THEORY_INDICES,
    ResourceVariantName,
    STRUCTURAL_EQUIVALENTS,
    VARIANT_ORDER,
    build_resource_qualification_matrix,
    build_resource_variant_cases,
    build_resource_variants,
)


def _case_map(
    theory_index: int,
) -> dict[
    tuple[ResourceVariantName, int],
    object,
]:
    return {
        (case.variant, case.challenge_index): case
        for case in build_resource_variant_cases(theory_index)
    }


def test_matrix_has_exact_preregistered_resource_geometry() -> None:
    matrix = build_resource_qualification_matrix()
    assert len(QUALIFICATION_THEORY_INDICES) == 8
    assert len(set(QUALIFICATION_THEORY_INDICES)) == 8
    assert all(0 <= index < len(THEORIES) for index in QUALIFICATION_THEORY_INDICES)
    assert len(VARIANT_ORDER) == 7
    assert set(VARIANT_ORDER) == {
        ResourceVariantName.BASE,
        ResourceVariantName.ALPHA_REORDER,
        ResourceVariantName.ALIAS_SPLIT,
        ResourceVariantName.RELATION_REIFICATION,
        ResourceVariantName.TYPE_TWIN,
        ResourceVariantName.EXECUTION_SEMANTICS_TWIN,
        ResourceVariantName.AMBIGUITY_DELETED_TWIN,
    }
    assert len(matrix) == 8 * 7 * CHALLENGES_PER_VARIANT == 896
    assert len(
        {(case.theory_index, case.variant, case.challenge_index) for case in matrix}
    ) == len(matrix)
    for theory_index in QUALIFICATION_THEORY_INDICES:
        cases = [case for case in matrix if case.theory_index == theory_index]
        assert {
            variant: sum(case.variant == variant for case in cases)
            for variant in VARIANT_ORDER
        } == {variant: 16 for variant in VARIANT_ORDER}
        challenge_sets = {
            variant: {case.challenge_index for case in cases if case.variant == variant}
            for variant in VARIANT_ORDER
        }
        assert len({frozenset(value) for value in challenge_sets.values()}) == 1


def test_structural_variants_are_genuine_and_exactly_invariant() -> None:
    original_types = tuple(place.resource_kind for place in PLACE_SPECS)
    for theory_index in QUALIFICATION_THEORY_INDICES:
        variants = {
            variant.name: variant for variant in build_resource_variants(theory_index)
        }
        base = variants[ResourceVariantName.BASE]
        alpha = variants[ResourceVariantName.ALPHA_REORDER]
        alias = variants[ResourceVariantName.ALIAS_SPLIT]
        reified = variants[ResourceVariantName.RELATION_REIFICATION]

        assert alpha.presentation.place_keys_by_base != (
            base.presentation.place_keys_by_base
        )
        assert alpha.presentation.operator_keys_by_symbol != (
            base.presentation.operator_keys_by_symbol
        )
        assert (
            alpha.presentation.place_order != base.presentation.place_order
            or alpha.presentation.operator_order != base.presentation.operator_order
        )
        assert len(alias.presentation.alias_pairs) == 1
        assert (
            sum(len(keys) for keys in alias.presentation.place_keys_by_base)
            == len(PLACE_SPECS) + 1
        )
        assert "A " in alias.source
        assert reified.presentation.reified
        assert "\nN " in reified.source
        assert "\nE " in reified.source
        assert "\nI " not in reified.source
        assert "\nT " not in reified.source

        assert len({variant.source for variant in variants.values()}) == 7
        assert all(
            variant.presentation.place_kind_by_base == original_types
            for name, variant in variants.items()
            if name != ResourceVariantName.TYPE_TWIN
        )

        cases = _case_map(theory_index)
        challenge_indices = {
            index for variant, index in cases if variant == ResourceVariantName.BASE
        }
        for challenge_index in challenge_indices:
            reference = cases[
                (ResourceVariantName.BASE, challenge_index)
            ].expected_outcome
            assert reference is not None
            for name in STRUCTURAL_EQUIVALENTS:
                case = cases[(name, challenge_index)]
                assert case.pair_expectation == PairExpectation.EXACT_INVARIANCE
                assert case.directive == AnswerDirective.ANSWER
                assert case.expected_outcome == reference


def test_type_twin_changes_only_typed_state_on_all_cases() -> None:
    for theory_index in QUALIFICATION_THEORY_INDICES:
        variants = {
            variant.name: variant for variant in build_resource_variants(theory_index)
        }
        base_types = variants[ResourceVariantName.BASE].presentation.place_kind_by_base
        twin_types = variants[
            ResourceVariantName.TYPE_TWIN
        ].presentation.place_kind_by_base
        assert sum(left != right for left, right in zip(base_types, twin_types)) == 1

        cases = _case_map(theory_index)
        for challenge_index in {
            index for variant, index in cases if variant == ResourceVariantName.BASE
        }:
            base = cases[(ResourceVariantName.BASE, challenge_index)].expected_outcome
            twin = cases[
                (ResourceVariantName.TYPE_TWIN, challenge_index)
            ].expected_outcome
            assert base is not None and twin is not None
            assert twin.execution_projection == base.execution_projection
            assert twin.resource_kinds != base.resource_kinds
            assert (
                cases[(ResourceVariantName.TYPE_TWIN, challenge_index)].pair_expectation
                == PairExpectation.TYPE_ONLY_SEPARATION
            )


def test_execution_twin_has_exact_balanced_match_and_separation_contract() -> None:
    for theory_index in QUALIFICATION_THEORY_INDICES:
        cases = _case_map(theory_index)
        twin_cases = [
            case
            for (variant, _), case in cases.items()
            if variant == ResourceVariantName.EXECUTION_SEMANTICS_TWIN
        ]
        separated = [
            case
            for case in twin_cases
            if case.pair_expectation == PairExpectation.EXECUTION_SEPARATION
        ]
        invariant = [
            case
            for case in twin_cases
            if case.pair_expectation == PairExpectation.EXACT_INVARIANCE
        ]
        assert len(separated) == 8
        assert len(invariant) == 8
        for case in separated:
            base = cases[
                (ResourceVariantName.BASE, case.challenge_index)
            ].expected_outcome
            assert base is not None and case.expected_outcome is not None
            assert base.status == ProcessStatus.DEADLOCK
            assert case.expected_outcome.status == ProcessStatus.HALT
            assert case.expected_outcome.execution_projection != (
                base.execution_projection
            )
        for case in invariant:
            base = cases[
                (ResourceVariantName.BASE, case.challenge_index)
            ].expected_outcome
            assert case.expected_outcome == base


def test_ambiguity_deletion_exactly_changes_epistemic_decision() -> None:
    for theory_index in QUALIFICATION_THEORY_INDICES:
        variants = {
            variant.name: variant for variant in build_resource_variants(theory_index)
        }
        base = variants[ResourceVariantName.BASE]
        ambiguous = variants[ResourceVariantName.AMBIGUITY_DELETED_TWIN]
        assert base.behavioral_class_count == 1
        assert theory_index in base.consistent_theory_indices
        assert ambiguous.behavioral_class_count >= 2
        assert theory_index in ambiguous.consistent_theory_indices
        assert len(ambiguous.demonstrations) + 1 == len(base.demonstrations)

        cases = _case_map(theory_index)
        ambiguous_cases = [
            case
            for (variant, _), case in cases.items()
            if variant == ResourceVariantName.AMBIGUITY_DELETED_TWIN
        ]
        assert all(
            case.directive == AnswerDirective.ABSTAIN
            and case.expected_outcome is None
            and case.pair_expectation == PairExpectation.AMBIGUITY_ABSTENTION
            for case in ambiguous_cases
        )
        assert any(len(case.possible_outcomes) > 1 for case in ambiguous_cases)
        for case in ambiguous_cases:
            base_case = cases[(ResourceVariantName.BASE, case.challenge_index)]
            assert base_case.directive == AnswerDirective.ANSWER
            assert base_case.expected_outcome in case.possible_outcomes


def test_transformed_executor_agrees_with_board_on_every_atomic_matrix_case() -> None:
    for theory_index in QUALIFICATION_THEORY_INDICES:
        cases = _case_map(theory_index)
        for (variant, challenge_index), case in cases.items():
            if variant in {
                ResourceVariantName.EXECUTION_SEMANTICS_TWIN,
                ResourceVariantName.AMBIGUITY_DELETED_TWIN,
            }:
                continue
            marking, sequence = challenge_cases()[challenge_index]
            board = execute_sequence(THEORIES[theory_index], marking, sequence)
            expected = case.expected_outcome
            assert expected is not None
            assert expected.execution_projection == (
                board.marking.multiplicities,
                board.cursor,
                board.status,
            )


def test_sources_are_deterministic_and_source_deleted_compatible() -> None:
    first = build_resource_qualification_matrix()
    second = build_resource_qualification_matrix()
    assert first == second
    for theory_index in QUALIFICATION_THEORY_INDICES:
        variants = build_resource_variants(theory_index)
        cases = build_resource_variant_cases(theory_index)
        for variant in variants:
            assert variant.source.isascii()
            assert variant.source.endswith("\n")
            related = [case for case in cases if case.variant == variant.name]
            assert all(case.challenge_source.isascii() for case in related)
            assert all(variant.source not in case.challenge_source for case in related)

    module_source = inspect.getsource(__import__("cross_ontology_resource_variants"))
    assert "train." not in module_source
    assert "candidate_runtime" not in module_source
    assert "torch" not in module_source
    assert "execute_sequence(" not in module_source
    assert ExecutionSemantics.ATOMIC_DEADLOCK.value in module_source
