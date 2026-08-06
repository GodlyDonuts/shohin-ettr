#!/usr/bin/env python3
"""Focused exactness checks for the DIVERGE-TOL1 typed machine."""

from fractions import Fraction

from diverge_tol1_ir import Action, Atom, Instruction, Predicate, execute_program


def main() -> None:
    program = (
        Instruction("SET", action=Action("SET", "alpha", Atom("CONST", "3/2"))),
        Instruction("SET", action=Action("SET", "beta", Atom("CONST", "-2"))),
        Instruction("ADD", action=Action("ADD", "alpha", Atom("REF", "beta"))),
        Instruction(
            "GUARD",
            predicate=Predicate("LT", "alpha", Atom("CONST", "0")),
            true_action=Action("MULTIPLY", "alpha", Atom("CONST", "-2")),
            false_action=Action("ADD", "beta", Atom("CONST", "7")),
        ),
        Instruction("SWAP", swap_left="alpha", swap_right="beta"),
        Instruction("QUERY", query="beta"),
    )
    answer, trajectory = execute_program(program)
    assert answer == Fraction(1, 1)
    assert trajectory[-1] == {"alpha": "-2", "beta": "1"}
    print("diverge TOL1 IR tests passed")


if __name__ == "__main__":
    main()
