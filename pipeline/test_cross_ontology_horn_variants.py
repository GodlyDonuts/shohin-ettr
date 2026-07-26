from __future__ import annotations

from collections import Counter

from audit_cross_ontology_horn_board import (
    independent_closure,
    independent_consistent_theories,
)
from cross_ontology_horn_board import (
    Demonstration,
    GroundAtom,
    PREDICATES,
    THEORIES,
)
from cross_ontology_horn_variants import (
    HornExecutionSemantics,
    HornExpectationRelation,
    HornSurfaceFact,
    HornVariantKind,
    VARIANT_EXPECTATIONS,
    VARIANT_ORDER,
    audit_horn_variant_set,
    materialize_horn_variant_set,
    normalized_evidence,
)


def _independent_decode(
    facts: tuple[HornSurfaceFact, ...],
    *,
    schema: str,
    predicate_alignment: tuple[tuple[str, int], ...],
    object_alignment: tuple[tuple[str, int], ...],
    role_alignment: tuple[tuple[str, int], ...],
) -> tuple[GroundAtom, ...]:
    predicates = dict(predicate_alignment)
    objects = dict(object_alignment)
    roles = dict(role_alignment)
    if schema == "direct-relations-v1":
        return tuple(
            sorted(
                GroundAtom(
                    predicates[fact.relation],
                    tuple(objects[value] for value in fact.arguments),
                )
                for fact in facts
            )
        )
    assert schema == "reified-incidence-v1"
    nodes = {
        fact.arguments[0]
        for fact in facts
        if fact.relation == "@fact"
    }
    node_predicates = {
        fact.arguments[0]: predicates[fact.arguments[1]]
        for fact in facts
        if fact.relation == "@predicate"
    }
    node_arguments: dict[str, dict[int, int]] = {}
    for fact in facts:
        if fact.relation != "@argument":
            continue
        node, role, object_symbol = fact.arguments
        node_arguments.setdefault(node, {})[roles[role]] = objects[
            object_symbol
        ]
    assert nodes == set(node_predicates) == set(node_arguments)
    result = []
    for node in sorted(nodes):
        predicate = node_predicates[node]
        arity = len(PREDICATES[predicate].argument_types)
        assert set(node_arguments[node]) == set(range(arity))
        result.append(
            GroundAtom(
                predicate,
                tuple(
                    node_arguments[node][index]
                    for index in range(arity)
                ),
            )
        )
    return tuple(sorted(result))


def _independent_evidence(variant) -> tuple[Demonstration, ...]:
    alignment = variant.assessor_only_alignment
    return tuple(
        Demonstration(
            _independent_decode(
                demo.initial,
                schema=variant.presentation.schema,
                predicate_alignment=alignment.predicate_symbols,
                object_alignment=alignment.object_symbols,
                role_alignment=alignment.role_symbols,
            ),
            _independent_decode(
                demo.terminal,
                schema=variant.presentation.schema,
                predicate_alignment=alignment.predicate_symbols,
                object_alignment=alignment.object_symbols,
                role_alignment=alignment.role_symbols,
            ),
        )
        for demo in variant.presentation.demonstrations
    )


def _independent_challenge(variant, challenge) -> tuple[GroundAtom, ...]:
    alignment = variant.assessor_only_alignment
    return _independent_decode(
        challenge.presented_initial,
        schema=variant.presentation.schema,
        predicate_alignment=alignment.predicate_symbols,
        object_alignment=alignment.object_symbols,
        role_alignment=alignment.role_symbols,
    )


def test_variant_registry_freezes_seven_explicit_contracts() -> None:
    assert VARIANT_ORDER == (
        HornVariantKind.BASE,
        HornVariantKind.ALPHA_REORDER,
        HornVariantKind.ALIAS_SPLIT,
        HornVariantKind.RELATION_REIFICATION,
        HornVariantKind.TYPE_TWIN,
        HornVariantKind.EXECUTION_SEMANTICS_TWIN,
        HornVariantKind.AMBIGUITY_DELETED_TWIN,
    )
    for kind in VARIANT_ORDER[:5]:
        expectation = VARIANT_EXPECTATIONS[kind]
        assert (
            expectation.relation_to_base
            == HornExpectationRelation.CANONICALLY_INVARIANT
        )
        assert expectation.evidence_invariant_after_alignment
        assert expectation.challenge_outputs_invariant_after_alignment
        assert not expectation.requires_execution_separation
        assert not expectation.requires_version_space_separation
    execution = VARIANT_EXPECTATIONS[
        HornVariantKind.EXECUTION_SEMANTICS_TWIN
    ]
    assert execution.requires_execution_separation
    assert (
        execution.relation_to_base
        == HornExpectationRelation.EXECUTION_SEPARATE
    )
    ambiguity = VARIANT_EXPECTATIONS[
        HornVariantKind.AMBIGUITY_DELETED_TWIN
    ]
    assert ambiguity.requires_version_space_separation
    assert (
        ambiguity.relation_to_base
        == HornExpectationRelation.IDENTIFIABILITY_SEPARATE
    )


def test_all_theories_pass_exact_variant_contracts() -> None:
    for theory_index in range(len(THEORIES)):
        variants = materialize_horn_variant_set(
            theory_index,
            seed=2026072500 + theory_index,
        )
        report = audit_horn_variant_set(variants)
        assert report.variant_count == 7
        assert report.challenge_count_per_variant == 16
        assert report.invariant_variant_count == 5
        assert report.execution_separation_count == 16
        assert report.ambiguity_separating_challenge_count >= 1


def test_invariant_variants_round_trip_with_independent_decoder() -> None:
    variants = materialize_horn_variant_set(7, seed=2026072517)
    base = variants[0]
    base_evidence = normalized_evidence(base.canonical_evidence)
    for variant in variants[:5]:
        assert normalized_evidence(
            _independent_evidence(variant)
        ) == base_evidence
        assert (
            independent_consistent_theories(
                _independent_evidence(variant)
            )
            == base.consistent_theory_indices
        )
        for challenge, base_challenge in zip(
            variant.challenges,
            base.challenges,
            strict=True,
        ):
            initial = _independent_challenge(variant, challenge)
            assert initial == challenge.canonical_initial
            assert initial == base_challenge.canonical_initial
            assert challenge.expected_terminals == (
                independent_closure(
                    THEORIES[variant.target_theory_index],
                    initial,
                ),
            )


def test_structural_variants_are_not_renderer_labels() -> None:
    base, alpha, alias, reified, type_twin, _, _ = (
        materialize_horn_variant_set(4, seed=2026072524)
    )
    assert alpha.source != base.source
    assert (
        alpha.assessor_only_alignment.predicate_symbols
        != base.assessor_only_alignment.predicate_symbols
    )
    assert (
        alpha.assessor_only_alignment.object_symbols
        != base.assessor_only_alignment.object_symbols
    )
    assert alpha.presentation.declarations == tuple(
        reversed(sorted(alpha.presentation.declarations))
    )

    alias_counts = Counter(
        value
        for _, value in alias.assessor_only_alignment.object_symbols
    )
    assert alias_counts == Counter({index: 2 for index in range(6)})
    alias_edges = {
        fact.arguments
        for fact in alias.presentation.declarations
        if fact.relation == "@alias"
    }
    assert len(alias_edges) == 6
    assert all(
        left != right for left, right in alias_edges
    )

    assert reified.presentation.schema == "reified-incidence-v1"
    payload = tuple(
        fact
        for demo in reified.presentation.demonstrations
        for fact in (*demo.initial, *demo.terminal)
    )
    assert payload
    assert all(fact.relation.startswith("@") for fact in payload)
    assert {fact.relation for fact in payload} == {
        "@fact",
        "@predicate",
        "@argument",
    }

    type_counts = Counter(
        value
        for _, value in type_twin.assessor_only_alignment.type_symbols
    )
    assert type_counts == Counter({0: 2, 1: 2})
    twin_edges = tuple(
        fact
        for fact in type_twin.presentation.declarations
        if fact.relation == "@type_twin"
    )
    assert len(twin_edges) == 2
    assert all(
        len(fact.arguments) == 2
        and fact.arguments[0] != fact.arguments[1]
        for fact in twin_edges
    )


def test_execution_twin_has_independent_exact_separation() -> None:
    variants = materialize_horn_variant_set(12, seed=2026072532)
    base = variants[0]
    execution = variants[5]
    assert (
        execution.execution_semantics
        == HornExecutionSemantics.DERIVED_ONLY_CLOSURE
    )
    assert execution.behavioral_class_count == 1
    for challenge, base_challenge in zip(
        execution.challenges,
        base.challenges,
        strict=True,
    ):
        initial = _independent_challenge(execution, challenge)
        closure = independent_closure(
            THEORIES[execution.target_theory_index],
            initial,
        )
        derived_only = tuple(
            atom for atom in closure if atom not in set(initial)
        )
        assert challenge.expected_terminals == (derived_only,)
        assert base_challenge.expected_terminals == (closure,)
        assert derived_only != closure


def test_ambiguity_deletion_requires_version_space_abstention() -> None:
    variants = materialize_horn_variant_set(18, seed=2026072548)
    base = variants[0]
    ambiguity = variants[6]
    independent_consistent = independent_consistent_theories(
        _independent_evidence(ambiguity)
    )
    assert independent_consistent == ambiguity.consistent_theory_indices
    assert len(ambiguity.canonical_evidence) == (
        len(base.canonical_evidence) - 1
    )
    assert ambiguity.behavioral_class_count >= 2
    assert ambiguity.target_theory_index in independent_consistent
    separating = 0
    for challenge in ambiguity.challenges:
        initial = _independent_challenge(ambiguity, challenge)
        possible = tuple(
            sorted(
                {
                    independent_closure(THEORIES[index], initial)
                    for index in independent_consistent
                }
            )
        )
        assert challenge.requires_abstention
        assert challenge.expected_terminals == possible
        separating += len(possible) > 1
    assert separating >= 1


def test_materialization_is_deterministic_and_candidate_views_are_narrow() -> None:
    first = materialize_horn_variant_set(3, seed=2026072564)
    second = materialize_horn_variant_set(3, seed=2026072564)
    changed_seed = materialize_horn_variant_set(3, seed=2026072565)
    assert first == second
    assert tuple(variant.source for variant in first) != tuple(
        variant.source for variant in changed_seed
    )
    for variant in first:
        assert variant.compiler_source() == variant.source
        assert "target_theory_index" not in variant.source
        assert "consistent_theory_indices" not in variant.source
        assert "sha256" not in variant.source
        for index, challenge in enumerate(variant.challenges):
            assert (
                variant.late_challenge_source(index)
                == challenge.source
            )
            assert "expected" not in challenge.source
            assert "theory" not in challenge.source
