from __future__ import annotations

from audit_cross_ontology_resource_board import (
    build_exact_audit,
    independent_consistent_theories,
    independent_execute_sequence,
)
from cross_ontology_resource_board import (
    EvidenceDisposition,
    THEORIES,
    behavior_signature,
    build_episode,
    challenge_cases,
    execute_sequence,
    heldout_programs,
    identifying_evidence,
    input_markings,
    reference_theory_state,
    single_step_cases,
)


def test_independent_oracle_agrees_on_every_heldout_execution() -> None:
    for theory in THEORIES:
        for marking, sequence in challenge_cases():
            assert independent_execute_sequence(
                theory,
                marking,
                sequence,
            ) == execute_sequence(theory, marking, sequence)


def test_version_spaces_are_exact_for_every_theory() -> None:
    for seed in range(len(THEORIES)):
        episodes = {
            disposition: build_episode(
                seed,
                disposition,
                renderer=seed % 4,
            )
            for disposition in EvidenceDisposition
        }
        singleton = episodes[EvidenceDisposition.SINGLETON]
        ambiguous = episodes[EvidenceDisposition.AMBIGUOUS]
        contradictory = episodes[EvidenceDisposition.CONTRADICTORY]
        alternate = episodes[EvidenceDisposition.COHERENT_ALTERNATE]
        assert singleton.behavioral_class_count == 1
        assert ambiguous.behavioral_class_count >= 2
        assert contradictory.behavioral_class_count == 0
        assert alternate.behavioral_class_count == 1
        assert singleton.target_theory_index in (singleton.consistent_theory_indices)
        assert ambiguous.target_theory_index in (ambiguous.consistent_theory_indices)
        assert not contradictory.consistent_theory_indices
        assert alternate.target_theory_index not in (
            alternate.consistent_theory_indices
        )
        for episode in episodes.values():
            assert independent_consistent_theories(episode.evidence) == (
                episode.consistent_theory_indices
            )


def test_operator_compositions_are_strictly_held_out() -> None:
    assert all(
        len(demo.sequence) == 1
        for theory_index in range(len(THEORIES))
        for demo in identifying_evidence(theory_index)
    )
    evidence_programs = {sequence for _, sequence in single_step_cases()}
    assert len(heldout_programs()) == 36
    assert all(len(sequence) in {2, 3} for sequence in heldout_programs())
    assert not evidence_programs.intersection(heldout_programs())


def test_four_opaque_renderers_preserve_episode_semantics() -> None:
    episodes = tuple(
        build_episode(
            17,
            EvidenceDisposition.SINGLETON,
            renderer=renderer,
        )
        for renderer in range(4)
    )
    assert len({episode.source for episode in episodes}) == 4
    assert len({episode.evidence for episode in episodes}) == 1
    assert len({episode.consistent_theory_indices for episode in episodes}) == 1
    assert len({episode.behavioral_class_count for episode in episodes}) == 1


def test_reference_packets_are_inert_generic_and_round_trip() -> None:
    for theory_index in range(len(THEORIES)):
        state = reference_theory_state(theory_index)
        assert state.halted
        assert state.root == 0
        assert len(state.cells) <= 32
        assert state.from_deployed_wire(state.deployed_wire()) == state


def test_board_geometry_and_exact_audit_counts() -> None:
    audit = build_exact_audit()
    assert audit == {
        "theory_count": 60,
        "operator_law_count": 5,
        "input_marking_count": 81,
        "single_step_evidence_case_count": 243,
        "heldout_program_count": 36,
        "challenge_case_count": 2916,
        "oracle_execution_comparisons": 174960,
        "nominal_operator_slots": 481140,
        "halt_outcome_count": 13362,
        "deadlock_outcome_count": 161598,
        "transition_table_rows": 1280,
        "exact_oracle_agreement": True,
    }
    assert audit["halt_outcome_count"] > 0
    assert audit["deadlock_outcome_count"] > 0
    assert (
        audit["halt_outcome_count"] + audit["deadlock_outcome_count"]
        == audit["oracle_execution_comparisons"]
    )
    assert len(
        {behavior_signature(theory_index) for theory_index in range(len(THEORIES))}
    ) == len(THEORIES)
    assert len(input_markings()) == 81
