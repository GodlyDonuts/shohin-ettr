#!/usr/bin/env python3
"""Structural checks for the DIVERGE-TOL2 anchor decoder."""

from diverge_tol1_ir import Action, Atom, Instruction
from diverge_tol2_anchor_decoder import (
    decode_direct_action,
    decode_query,
    decode_swap,
    semantic_instruction_equal,
    split_guard,
)


def main() -> None:
    symbols = ("atlas", "blaze", "coral", "drift")
    assert decode_direct_action("to atlas, add blaze", "ADD", symbols) == Action(
        "ADD", "atlas", Atom("REF", "blaze")
    )
    assert decode_direct_action("by 3/2, multiply coral", "MULTIPLY", symbols) == Action(
        "MULTIPLY", "coral", Atom("CONST", "3/2")
    )
    assert decode_direct_action("assign -2 into drift", "SET", None) == Action(
        "SET", "drift", Atom("CONST", "-2")
    )
    assert decode_query("report the value in atlas", symbols) == Instruction(
        "QUERY", query="atlas"
    )
    forward = decode_swap("with blaze, swap atlas", symbols)
    reverse = Instruction("SWAP", swap_left="blaze", swap_right="atlas")
    assert semantic_instruction_equal(forward, reverse)
    regions = split_guard(
        "add blaze to atlas if coral > 0; otherwise decrease drift by 2."
    )
    assert regions.predicate == "coral > 0"
    assert regions.true_action == "add blaze to atlas"
    assert regions.false_action == "decrease drift by 2"
    reversed_regions = split_guard(
        "otherwise decrease drift by 2; if coral > 0, then add blaze to atlas."
    )
    assert reversed_regions == regions
    print("diverge TOL2 anchor decoder tests passed")


if __name__ == "__main__":
    main()
