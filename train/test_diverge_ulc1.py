#!/usr/bin/env python3
"""Contract tests for the DIVERGE-ULC1 exact CPU mechanism."""

from __future__ import annotations

from dataclasses import replace

from diverge_ulc1 import (
    ACTIVE_RIGHT,
    DelayedObservation,
    DivergeContractError,
    SealedULC1Packet,
    apply_certified_observation,
    execute_ulc1,
    packet_bytes_ulc1,
)
from diverge_ulc1_reference import (
    build_ulc1_episode,
    certify_observation,
    exact_factorized_parity,
    expected_query_decisions,
    materialized_particle_bytes,
)
from diverge_v0 import (
    ABSTAIN,
    ANSWER,
    GuardedPatch,
    build_packet,
    enumerate_assignments,
    factorized_query_execution,
    packet_bytes,
)


def _episode(width: int, serial: int = 0):
    return build_ulc1_episode(
        seed=202608057400,
        serial=serial + width,
        split="calibration",
        ontology="register-workshop",
        renderer="calibration-ledger",
        record_count=width,
    )


def _fully_refine(episode, sealed=None):
    execution = execute_ulc1(sealed or episode.sealed)
    for observation in episode.observations:
        for certificate in certify_observation(execution.sealed, observation):
            execution = apply_certified_observation(execution, certificate)
            assert exact_factorized_parity(execution)
            assert episode.gold_assignment in enumerate_assignments(
                execution.sealed.packet
            )
    return execution


def test_world_counts_parity_recovery_and_query_contract() -> None:
    for width, expected_worlds in enumerate((2, 6, 16, 42), start=1):
        episode = _episode(width)
        initial = execute_ulc1(episode.sealed)
        assert episode.represented_worlds == expected_worlds
        assert episode.initial_top1 != episode.gold_assignment
        assert exact_factorized_parity(initial)
        assert (
            factorized_query_execution(
                initial.sealed.packet,
                initial.receipt,
                episode.invariant_query,
            ).disposition
            == ANSWER
        )
        assert (
            factorized_query_execution(
                initial.sealed.packet,
                initial.receipt,
                episode.underdetermined_query,
            ).disposition
            == ABSTAIN
        )
        refined = _fully_refine(episode)
        assert refined.receipt.represented_worlds == 1
        expected = expected_query_decisions(episode, refined.sealed)
        assert (
            factorized_query_execution(
                refined.sealed.packet,
                refined.receipt,
                episode.sensitive_query,
            )
            == expected["sensitive"]
        )


def test_source_is_sealed_and_packet_swap_fails_closed() -> None:
    left = _episode(3, 10)
    right = _episode(3, 20)
    before = packet_bytes_ulc1(left.sealed)
    poisoned_assessor = replace(left, source_text="POISONED SOURCE BYTES")
    assert packet_bytes_ulc1(poisoned_assessor.sealed) == before
    assert left.source_text.encode("utf-8") not in before
    wrong_observation = right.observations[0]
    try:
        certify_observation(left.sealed, wrong_observation)
    except DivergeContractError as error:
        assert "different source" in str(error)
    else:
        raise AssertionError("cross-packet evidence did not fail closed")


def test_shuffled_record_provenance_cannot_forge_a_conflict() -> None:
    episode = _episode(3, 30)
    first, second = episode.observations[:2]
    shuffled = DelayedObservation(
        first.source_commitment,
        second.record_provenance,
        first.state_slot,
        first.observed_value,
        first.evidence_commitment,
    )
    try:
        certify_observation(episode.sealed, shuffled)
    except DivergeContractError as error:
        assert "conflict core" in str(error)
    else:
        raise AssertionError("shuffled provenance produced a valid conflict")


def test_factorization_has_a_measured_high_ambiguity_storage_advantage() -> None:
    for width in (3, 4):
        episode = _episode(width, 40)
        initial = execute_ulc1(episode.sealed)
        packed = (
            len(packet_bytes_ulc1(episode.sealed)) + initial.receipt.peak_group_bytes
        )
        particles = materialized_particle_bytes(initial)
        assert particles >= 2 * packed
        assert (
            initial.receipt.logical_transaction_applications
            > initial.receipt.unique_transactions
        )


def test_noncommuting_source_order_is_behaviorally_necessary() -> None:
    episode = _episode(2, 50)
    baseline = _fully_refine(episode)
    baseline_answer = factorized_query_execution(
        baseline.sealed.packet,
        baseline.receipt,
        episode.sensitive_query,
    ).answer
    first_record = episode.sealed.records[0]
    variable = episode.sealed.variable_id(first_record.interpretation_provenance)
    right = first_record.domain_interpretations.index(ACTIVE_RIGHT)
    matching = [
        index
        for index, patch in enumerate(episode.sealed.packet.patches)
        if len(patch.guard.literals) == 1
        and patch.guard.literals[0].variable_id == variable
        and patch.guard.literals[0].option == right
    ]
    assert len(matching) == 4
    patches = list(episode.sealed.packet.patches)
    left, right_index = matching[:2]
    patches[left] = GuardedPatch(
        patches[left].index,
        patches[left].guard,
        patches[right_index].transaction,
        patches[left].provenance,
    )
    patches[right_index] = GuardedPatch(
        patches[right_index].index,
        patches[right_index].guard,
        episode.sealed.packet.patches[left].transaction,
        patches[right_index].provenance,
    )
    modified_packet = build_packet(
        source_commitment=episode.sealed.packet.source_commitment,
        shared_state=episode.sealed.packet.shared_state,
        variables=episode.sealed.packet.variables,
        hard_factors=episode.sealed.packet.hard_factors,
        support_factors=episode.sealed.packet.support_factors,
        patches=patches,
        caps=episode.sealed.packet.caps,
    )
    modified = SealedULC1Packet(modified_packet, episode.sealed.records)
    intervention = _fully_refine(episode, modified)
    intervention_answer = factorized_query_execution(
        intervention.sealed.packet,
        intervention.receipt,
        episode.sensitive_query,
    ).answer
    assert intervention_answer != baseline_answer


def test_overflow_discards_partial_support() -> None:
    episode = _episode(4, 60)
    caps = replace(episode.sealed.packet.caps, max_worlds=8)
    overflow = build_packet(
        source_commitment=episode.sealed.packet.source_commitment,
        shared_state=episode.sealed.packet.shared_state,
        variables=episode.sealed.packet.variables,
        hard_factors=episode.sealed.packet.hard_factors,
        support_factors=episode.sealed.packet.support_factors,
        patches=episode.sealed.packet.patches,
        caps=caps,
    )
    assert overflow.overflow and not overflow.variables and not overflow.patches
    sealed = SealedULC1Packet(overflow, episode.sealed.records)
    assert execute_ulc1(sealed).receipt.overflow
    assert b'"overflow":true' in packet_bytes(overflow)


def main() -> None:
    test_world_counts_parity_recovery_and_query_contract()
    test_source_is_sealed_and_packet_swap_fails_closed()
    test_shuffled_record_provenance_cannot_forge_a_conflict()
    test_factorization_has_a_measured_high_ambiguity_storage_advantage()
    test_noncommuting_source_order_is_behaviorally_necessary()
    test_overflow_discards_partial_support()
    print("DIVERGE-ULC1 tests: passed")


if __name__ == "__main__":
    main()
