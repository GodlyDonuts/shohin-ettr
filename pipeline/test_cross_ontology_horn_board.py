from __future__ import annotations

from audit_cross_ontology_horn_board import (
    independent_closure,
    independent_consistent_theories,
)
from cross_ontology_horn_board import (
    EvidenceDisposition,
    THEORIES,
    all_ground_atoms,
    behavior_signature,
    build_episode,
    challenge_initials,
    execute_closure,
    reference_theory_state,
)


def test_independent_oracle_agrees_exhaustively() -> None:
    for theory in THEORIES:
        for initial in challenge_initials():
            assert independent_closure(theory, initial) == execute_closure(
                theory,
                initial,
            )


def test_version_space_dispositions_are_exact() -> None:
    for seed in range(len(THEORIES)):
        singleton = build_episode(
            seed,
            EvidenceDisposition.SINGLETON,
            renderer=seed % 4,
        )
        ambiguous = build_episode(
            seed,
            EvidenceDisposition.AMBIGUOUS,
            renderer=seed % 4,
        )
        contradictory = build_episode(
            seed,
            EvidenceDisposition.CONTRADICTORY,
            renderer=seed % 4,
        )
        alternate = build_episode(
            seed,
            EvidenceDisposition.COHERENT_ALTERNATE,
            renderer=seed % 4,
        )
        assert singleton.behavioral_class_count == 1
        assert ambiguous.behavioral_class_count >= 2
        assert contradictory.behavioral_class_count == 0
        assert alternate.behavioral_class_count == 1
        assert singleton.target_theory_index in (
            singleton.consistent_theory_indices
        )
        assert alternate.target_theory_index not in (
            alternate.consistent_theory_indices
        )
        for episode in (
            singleton,
            ambiguous,
            contradictory,
            alternate,
        ):
            assert independent_consistent_theories(episode.evidence) == (
                episode.consistent_theory_indices
            )


def test_renderers_change_source_not_semantics() -> None:
    episodes = tuple(
        build_episode(
            7,
            EvidenceDisposition.SINGLETON,
            renderer=renderer,
        )
        for renderer in range(4)
    )
    assert len({episode.source for episode in episodes}) == 4
    assert len({episode.evidence for episode in episodes}) == 1
    assert len(
        {
            episode.consistent_theory_indices
            for episode in episodes
        }
    ) == 1


def test_reference_packet_is_generic_and_round_trips() -> None:
    for theory_index in range(len(THEORIES)):
        state = reference_theory_state(theory_index)
        assert state.halted
        assert state.root == 5
        assert len(state.cells) <= 32
        assert state.from_deployed_wire(state.deployed_wire()) == state


def test_behavioral_hypotheses_are_nontrivial() -> None:
    assert len(all_ground_atoms()) == 27
    assert len(challenge_initials()) == 378
    assert len({behavior_signature(index) for index in range(len(THEORIES))}) > 10
