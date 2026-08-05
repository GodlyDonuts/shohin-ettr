#!/usr/bin/env python3
"""Focused deterministic tests for the complete DIVERGE A--G gate."""

from __future__ import annotations

import diverge_v0_neural_pilot as pilot
from diverge_v0 import (
    ABSTAIN,
    ANSWER,
    enumerate_assignments,
    execute_packet,
    execute_packet_factorized,
    materialized_world_bytes,
    packet_bytes,
    query_execution,
    refine_factorized_receipt,
)
from diverge_v0_matched_gate import (
    CompiledEpisode,
    _bind_delayed_evidence,
    _decisions_for_arm,
    _mean_field_decision,
    _primary_record,
    _refine_packet,
    _replace_primary_semantics,
    _select_full_particles,
    _truth_prediction,
    build_gate_board,
)


def test_board_balances_wrong_primary_program_and_three_query_types() -> None:
    board = build_gate_board(202608056000, 2)
    assert len(board) == 72
    assert {episode.family for episode in board} == {
        "register-workshop",
        "parcel-relation",
        "signal-routing",
    }
    answers = []
    for gate in board:
        truth = _truth_prediction(gate.source)
        packet, canonical, _ = pilot._build_predicted_packet(gate.source, truth)
        assert packet is not None and not packet.overflow
        initial = execute_packet(packet)
        assert query_execution(initial, gate.invariant_query).disposition == ANSWER
        assert query_execution(initial, gate.underdetermined_query).disposition == ABSTAIN
        primary = next(
            record
            for record in gate.source.records
            if record.record_id == gate.source.primary_record_id
        )
        variable = canonical[gate.source.primary_record_id]
        top = min(initial.worlds, key=lambda world: (-world.mass, world.assignment))
        assert top.assignment[variable] != primary.gold_option
        gold_world = next(
            world for world in initial.worlds if world.assignment[variable] == primary.gold_option
        )
        assert gold_world.state is not None
        answers.append(gold_world.state.cells[0].value)
    assert answers.count(10) == answers.count(13) == 36


def test_primary_rewrite_keeps_source_self_consistent() -> None:
    episode = pilot.generate_episode(
        seed=17,
        split="confirmation",
        width=4,
        renderer=3,
        ontology="signal-routing",
    )
    rewritten = _replace_primary_semantics(episode, gold_program=0)
    primary = next(
        record for record in rewritten.records if record.record_id == rewritten.primary_record_id
    )
    gold = primary.options[primary.gold_option]
    other = primary.options[1 - primary.gold_option]
    assert gold.program == 0 and gold.prior_class == 1
    assert other.program == 1 and other.prior_class == 0
    assert rewritten.evidence_alias == gold.alias
    assert rewritten.source_text == "\n".join(record.text for record in rewritten.records)


def test_soft_control_commits_on_uncertain_query() -> None:
    gate = build_gate_board(202608056000, 1)[-1]
    truth = _truth_prediction(gate.source)
    packet, _, _ = pilot._build_predicted_packet(gate.source, truth)
    assert packet is not None
    decision = _mean_field_decision(packet, gate.underdetermined_query, None)
    assert decision.disposition == ANSWER
    assert query_execution(execute_packet(packet), gate.underdetermined_query).disposition == ABSTAIN


def test_delayed_binder_uses_only_sealed_packet_and_new_evidence() -> None:
    gate = build_gate_board(202608056000, 1)[0]
    truth = _truth_prediction(gate.source)
    packet, canonical, _ = pilot._build_predicted_packet(gate.source, truth)
    assert packet is not None
    certificate = _bind_delayed_evidence(packet, gate.source.evidence_text)
    assert certificate is not None
    primary = next(
        record for record in gate.source.records if record.record_id == gate.source.primary_record_id
    )
    assert certificate.variable_id == canonical[gate.source.primary_record_id]
    assert certificate.confirmed_option == primary.gold_option
    assert _bind_delayed_evidence(packet, "evidence without a sealed key") is None


def test_factorized_arm_and_full_particles_use_the_same_measured_budget() -> None:
    gate = build_gate_board(202608056000, 1)[-1]
    prediction = _truth_prediction(gate.source)
    packet, canonical, _ = pilot._build_predicted_packet(gate.source, prediction)
    assert packet is not None and not packet.overflow
    primary = _primary_record(gate.source)
    variable = canonical[gate.source.primary_record_id]
    valid = tuple(
        assignment
        for assignment in enumerate_assignments(packet)
        if assignment[variable] == primary.gold_option
    )
    refined_packet, verification = _refine_packet(
        packet,
        variable=variable,
        confirmed=primary.gold_option,
        valid_assignments=valid,
        evidence_text=gate.source.evidence_text,
    )
    assert verification.accepted
    initial = execute_packet(packet, commute_disjoint=True)
    refined = execute_packet(refined_packet, commute_disjoint=True)
    factorized_initial = execute_packet_factorized(packet)
    factorized_refined = refine_factorized_receipt(refined_packet, factorized_initial)
    expected = {
        "sensitive": query_execution(refined, gate.sensitive_query),
        "invariant": query_execution(initial, gate.invariant_query),
        "underdetermined": query_execution(initial, gate.underdetermined_query),
    }
    compiled = CompiledEpisode(
        gate=gate,
        packet=packet,
        refined_packet=refined_packet,
        prediction=prediction,
        packet_exact=True,
        gold_support_recalled=True,
        primary_variable=variable,
        evidence_variable=variable,
        evidence_option=primary.gold_option,
        primary_gold_option=primary.gold_option,
        initial=initial,
        refined=refined,
        factorized_initial=factorized_initial,
        factorized_refined=factorized_refined,
        expected=expected,
        verifier_calls=1,
        valid_support_preserved=True,
    )
    g_decisions, g_resources = _decisions_for_arm(
        "G_diverge", compiled, particle_seed=17
    )
    assert g_decisions == expected
    assert g_resources["bytes"] == max(
        len(packet_bytes(packet)) + factorized_initial.peak_group_bytes,
        len(packet_bytes(refined_packet)) + factorized_refined.peak_group_bytes,
    )
    selected, b_resources = _select_full_particles(compiled)
    assert b_resources["bytes"] == sum(
        materialized_world_bytes(packet, world) for world in selected
    )
    assert b_resources["bytes"] <= g_resources["bytes"]
    assert b_resources["transactions"] <= g_resources["transactions"]


def main() -> None:
    test_board_balances_wrong_primary_program_and_three_query_types()
    test_primary_rewrite_keeps_source_self_consistent()
    test_soft_control_commits_on_uncertain_query()
    test_delayed_binder_uses_only_sealed_packet_and_new_evidence()
    print("DIVERGE matched gate tests: passed")


if __name__ == "__main__":
    main()
