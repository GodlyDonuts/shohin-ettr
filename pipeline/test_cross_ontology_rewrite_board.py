from __future__ import annotations

from audit_cross_ontology_rewrite_board import (
    audit_static_board,
    independent_behavioral_class_count,
    independent_consistent_theories,
    independent_normal_forms,
)
from cross_ontology_rewrite_board import (
    HELDOUT_THEORY_INDICES,
    RULE_LIBRARY,
    TRAIN_THEORY_INDICES,
    THEORIES,
    EvidenceDisposition,
    GroundTerm,
    build_episode,
    challenge_terms,
    execute_normal_forms,
    one_step_reducts,
    reference_theory_state,
)


def test_independent_normal_form_oracle_agrees_exhaustively() -> None:
    comparisons = 0
    for theory_index in range(len(THEORIES)):
        for term in challenge_terms():
            assert independent_normal_forms(
                theory_index,
                term,
            ) == execute_normal_forms(theory_index, term)
            comparisons += 1
    receipt = audit_static_board()
    assert comparisons == receipt.exhaustive_oracle_cases


def test_rule_combinations_are_held_out_but_primitives_are_seen() -> None:
    train_pairs = {
        THEORIES[index].rule_indices
        for index in TRAIN_THEORY_INDICES
    }
    heldout_pairs = {
        THEORIES[index].rule_indices
        for index in HELDOUT_THEORY_INDICES
    }
    assert train_pairs.isdisjoint(heldout_pairs)
    assert {
        rule_index
        for pair in train_pairs
        for rule_index in pair
    } == set(range(len(RULE_LIBRARY)))
    for seed in range(len(HELDOUT_THEORY_INDICES)):
        episode = build_episode(
            seed,
            EvidenceDisposition.SINGLETON,
            renderer=seed % 4,
        )
        assert episode.target_theory_index in HELDOUT_THEORY_INDICES


def test_repeated_variables_and_ordered_child_roles_are_semantic() -> None:
    a = GroundTerm(0, 0)
    b = GroundTerm(0, 1)
    f_a = GroundTerm(0, 4, (a,))
    equal_pair = GroundTerm(0, 5, (a, a))
    unequal_pair = GroundTerm(0, 5, (a, b))
    left_nested = GroundTerm(0, 5, (f_a, b))
    right_nested = GroundTerm(0, 5, (b, f_a))
    repeated_variable_theory = THEORIES[
        next(
            index
            for index, theory in enumerate(THEORIES)
            if theory.rule_indices == (0, 2)
        )
    ]
    ordered_role_theory = THEORIES[
        next(
            index
            for index, theory in enumerate(THEORIES)
            if theory.rule_indices == (3, 5)
        )
    ]
    assert one_step_reducts(repeated_variable_theory, equal_pair)
    assert not one_step_reducts(repeated_variable_theory, unequal_pair)
    assert one_step_reducts(ordered_role_theory, left_nested)
    assert not one_step_reducts(ordered_role_theory, right_nested)


def test_exact_version_space_dispositions_and_independent_audit() -> None:
    disposition_counts = {
        disposition: 0
        for disposition in EvidenceDisposition
    }
    for seed in range(len(HELDOUT_THEORY_INDICES)):
        for disposition in EvidenceDisposition:
            episode = build_episode(
                seed,
                disposition,
                renderer=seed % 4,
            )
            independent = independent_consistent_theories(
                episode.evidence
            )
            assert independent == episode.version_space.theory_indices
            class_count = independent_behavioral_class_count(independent)
            assert class_count == (
                episode.version_space.behavioral_class_count
            )
            target_present = (
                episode.target_theory_index in independent
            )
            if disposition == EvidenceDisposition.SINGLETON:
                assert class_count == 1
                assert target_present
            elif disposition == EvidenceDisposition.AMBIGUOUS:
                assert class_count >= 2
                assert episode.evidence
            elif disposition == EvidenceDisposition.CONTRADICTORY:
                assert class_count == 0
                assert not independent
            else:
                assert class_count == 1
                assert not target_present
            disposition_counts[disposition] += 1
    assert set(disposition_counts.values()) == {
        len(HELDOUT_THEORY_INDICES)
    }


def test_four_opaque_renderers_preserve_episode_semantics() -> None:
    episodes = tuple(
        build_episode(
            2,
            EvidenceDisposition.SINGLETON,
            renderer=renderer,
        )
        for renderer in range(4)
    )
    assert len({episode.source for episode in episodes}) == 4
    assert len({episode.evidence for episode in episodes}) == 1
    assert len(
        {
            episode.version_space.theory_indices
            for episode in episodes
        }
    ) == 1
    assert all("rule" not in episode.source for episode in episodes)


def test_reference_packets_use_only_generic_typed_schema() -> None:
    for theory_index in range(len(THEORIES)):
        state = reference_theory_state(theory_index)
        assert state.halted
        assert state.root is not None
        assert state.capacity == 32
        assert state.type_count == 4
        assert {edge.relation_index for edge in state.edges} <= set(
            range(7)
        )
        # Relation 3 and 4 are distinct ordered child roles.
        assert any(edge.relation_index == 3 for edge in state.edges)
        assert state.from_deployed_wire(state.deployed_wire()) == state


def test_board_cardinalities_are_small_and_exact() -> None:
    receipt = audit_static_board()
    assert receipt.term_count == 64
    assert receipt.theory_count == 15
    assert receipt.training_combination_count == 7
    assert receipt.heldout_combination_count == 8
    assert receipt.exhaustive_oracle_cases == 960
    assert receipt.distinct_behavior_classes >= 10
    assert receipt.renderer_count == 4
    assert receipt.version_space_episode_count == 32
