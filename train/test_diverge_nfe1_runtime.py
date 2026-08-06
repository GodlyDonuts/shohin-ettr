#!/usr/bin/env python3
"""Exact factorized runtime tests for DIVERGE-NFE1."""

from __future__ import annotations

from diverge_nfe1_runtime import (
    ABSTAIN,
    ANSWER,
    REJECT,
    CompiledMention,
    compile_episode,
    compile_query,
    enumerate_extensional_map,
    execute_factorized,
    factorized_total_bytes,
    issue_evidence,
    mutate_receipt,
    query_receipt,
    receipt_extensional_map,
)


def _mentions(lhs: int, argument: int, rhs: int) -> tuple[CompiledMention, ...]:
    return (
        CompiledMention("LHS", 0, 1, lhs, 1.0),
        CompiledMention("ARGUMENT", 2, 3, argument, 1.0),
        CompiledMention("RHS", 4, 5, rhs, 1.0),
    )


def main() -> None:
    sources = ("8 - 5 = 13", "13 * 2 = 26")
    packet = compile_episode(
        "1" * 64,
        sources,
        (_mentions(8, 5, 13), _mentions(13, 2, 26)),
        ((0.0, 3.0, -1.0), (-1.0, 3.0, 0.0)),
    )
    evidence = issue_evidence(packet)
    query = compile_query(packet, "Return the terminal scalar.")

    no_evidence = execute_factorized(packet)
    assert not no_evidence.rejected
    assert query_receipt(packet, no_evidence, query).disposition == ABSTAIN
    assert receipt_extensional_map(packet, no_evidence) == enumerate_extensional_map(
        packet
    )

    full = execute_factorized(packet, evidence)
    decision = query_receipt(packet, full, query)
    assert decision.disposition == ANSWER and decision.answer == 26
    assert full.represented_worlds == 1
    assert receipt_extensional_map(packet, full) == enumerate_extensional_map(
        packet, evidence
    )
    assert factorized_total_bytes(packet, full, evidence, query) > 0

    reset = execute_factorized(packet, evidence, reset_initial_state=True)
    shifted = execute_factorized(packet, evidence, operand_semantic_shift=True)
    assert query_receipt(packet, reset, query).disposition == REJECT
    assert query_receipt(packet, shifted, query).disposition == REJECT

    bad = list(evidence)
    bad[0] = mutate_receipt(bad[0], "value")
    assert execute_factorized(packet, bad).rejected
    swapped_query = compile_query(
        compile_episode(
            "2" * 64,
            sources,
            (_mentions(8, 5, 13), _mentions(13, 2, 26)),
            ((0.0, 3.0, -1.0), (-1.0, 3.0, 0.0)),
        ),
        "Return the terminal scalar.",
    )
    assert query_receipt(packet, full, swapped_query).disposition == REJECT
    print("diverge NFE1 runtime tests passed")


if __name__ == "__main__":
    main()
