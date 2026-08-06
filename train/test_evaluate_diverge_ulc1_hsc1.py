#!/usr/bin/env python3
"""Contract tests for the frozen-HSC1 DIVERGE-ULC1 evaluator."""

from __future__ import annotations

from dataclasses import replace

from diverge_ulc1_mdd import RuntimeChoice, execute_choice_path, execute_mdd, query_mdd
from diverge_v0 import (
    ANSWER,
    DivergeContractError,
    Query,
    TypedCell,
    TypedState,
    TypedTransaction,
    named_commitment,
    read_query,
)
from evaluate_diverge_ulc1_hsc1 import (
    CompiledEpisode,
    _certify_delayed_evidence,
    _delayed_evidence,
    _evaluate_compiled,
    _factorized_packet_bytes,
    _packet_swap_rejected,
    _select_equal_budget_particles,
    _source_poison_invariant,
)


def _choice(
    record: int,
    value: int,
    *,
    mass: int,
    delta: int,
    witness: int,
) -> RuntimeChoice:
    return RuntimeChoice(
        record,
        value,
        mass,
        (
            TypedTransaction("ADD_VALUE", (0, delta)),
            TypedTransaction("SET_VALUE", (5 + record, witness)),
        ),
        witness,
        f"r{record}-v{value}",
        named_commitment("ulc1-hsc1-test-choice", f"{record}:{value}"),
        (("record", record), ("value", value)),
    )


def _compiled(name: str) -> CompiledEpisode:
    initial = TypedState(
        (
            TypedCell(0, 0, 0),
            TypedCell(1, 0, 9),
            TypedCell(2, 0, 2),
            TypedCell(3, 0, 3),
            TypedCell(4, 0, 4),
            TypedCell(5, 1, 0),
            TypedCell(6, 1, 0),
        )
    )
    rows = (
        (
            _choice(0, 0, mass=9, delta=1, witness=10),
            _choice(0, 1, mass=1, delta=4, witness=11),
        ),
        (
            _choice(1, 0, mass=9, delta=2, witness=20),
            _choice(1, 1, mass=1, delta=8, witness=21),
        ),
    )
    execution = execute_mdd(initial, rows)
    expected = execute_choice_path(initial, rows, (1, 1))
    assert expected is not None
    return CompiledEpisode(
        name,
        "test",
        named_commitment("ulc1-hsc1-test-source", name),
        rows,
        (1, 1),
        (11, 21),
        initial,
        expected,
        execution,
        Query("READ_VALUE", (0,)),
        Query("EDGE_COUNT", ()),
        True,
        True,
        12,
    )


def test_delayed_evidence_recovers_the_gold_lineage() -> None:
    compiled = _compiled("left")
    evidence = _delayed_evidence(compiled)
    allowed = _certify_delayed_evidence(compiled, evidence)
    assert allowed == {0: frozenset({1}), 1: frozenset({1})}
    decision = query_mdd(
        compiled.execution,
        compiled.sensitive_query,
        allowed=allowed,
    )
    assert decision.disposition == ANSWER
    assert decision.answer == read_query(
        compiled.expected_state, compiled.sensitive_query
    )


def test_evidence_provenance_and_packet_swaps_fail_closed() -> None:
    left = _compiled("left")
    right = _compiled("right")
    assert _packet_swap_rejected(left, _delayed_evidence(right))
    tampered = list(_delayed_evidence(left))
    tampered[0] = replace(
        tampered[0],
        record_provenance=named_commitment("ulc1-hsc1-tamper", "record"),
    )
    try:
        _certify_delayed_evidence(left, tampered)
    except DivergeContractError:
        pass
    else:
        raise AssertionError("tampered evidence provenance was accepted")


def test_source_seal_is_charged_and_post_seal_poison_is_invariant() -> None:
    compiled = _compiled("left")
    before = _factorized_packet_bytes(compiled)
    altered = replace(
        compiled,
        source_commitment=named_commitment("ulc1-hsc1-test-source", "altered"),
    )
    assert _factorized_packet_bytes(altered) != before
    assert _source_poison_invariant(compiled, _delayed_evidence(compiled))


def test_equal_budget_particles_respect_both_resource_caps() -> None:
    compiled = _compiled("left")
    particles, resources = _select_equal_budget_particles(compiled)
    assert particles
    assert resources["bytes"] <= resources["byte_budget"]
    assert resources["transactions"] <= resources["transaction_budget"]


def test_packet_swap_metric_is_measured_not_hardcoded() -> None:
    compiled = _compiled("left")
    own = _delayed_evidence(compiled)
    own_result = _evaluate_compiled(compiled, donor_evidence=own)
    donor_result = _evaluate_compiled(
        compiled,
        donor_evidence=_delayed_evidence(_compiled("right")),
    )
    assert not own_result["packet_swap_rejected"]
    assert donor_result["packet_swap_rejected"]


def main() -> None:
    test_delayed_evidence_recovers_the_gold_lineage()
    test_evidence_provenance_and_packet_swaps_fail_closed()
    test_source_seal_is_charged_and_post_seal_poison_is_invariant()
    test_equal_budget_particles_respect_both_resource_caps()
    test_packet_swap_metric_is_measured_not_hardcoded()
    print("DIVERGE-ULC1 frozen-HSC1 evaluator tests: passed")


if __name__ == "__main__":
    main()
