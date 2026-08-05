#!/usr/bin/env python3
"""Focused deterministic tests for the complete DIVERGE A--G gate."""

from __future__ import annotations

import diverge_v0_neural_pilot as pilot
from diverge_v0 import ABSTAIN, ANSWER, execute_packet, query_execution
from diverge_v0_matched_gate import (
    _mean_field_decision,
    _replace_primary_semantics,
    _truth_prediction,
    build_gate_board,
)


def test_board_balances_wrong_primary_program_and_three_query_types() -> None:
    board = build_gate_board(202608056000, 1)
    assert len(board) == 36
    assert {episode.family for episode in board} == {
        "register-workshop",
        "parcel-relation",
        "signal-routing",
    }
    answers = set()
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
        answers.add(gold_world.state.cells[0].value)
    assert answers == {10, 13}


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


def main() -> None:
    test_board_balances_wrong_primary_program_and_three_query_types()
    test_primary_rewrite_keeps_source_self_consistent()
    test_soft_control_commits_on_uncertain_query()
    print("DIVERGE matched gate tests: passed")


if __name__ == "__main__":
    main()
