from fractions import Fraction

from build_cte1_canonical_transactions import expression_atom, render_transaction_trace
from draft_transaction_compiler import compile_draft_transactions
from natural_microcode_program import parse_program
from typed_microcode_graph import execute_fraction


def test_canonical_trace_round_trips_through_causal_state() -> None:
    program = parse_program(
        "<MICROCODE_V1>\n"
        "R0 P:3 P:2 M\n"
        "R1 L:0 P:2 M\n"
        "R2 L:1 P:52 M\n"
        "C:2\n"
        "</MICROCODE_V1>"
    )
    response, final, load_count = render_transaction_trace(program)
    assert response == (
        "<<(3*2)=6>>\n"
        "<<(6*2)=12>>\n"
        "<<(12*52)=624>>\n"
        "#### 624"
    )
    assert final == Fraction(624)
    assert load_count == 2
    graph, receipt = compile_draft_transactions(
        "James writes 3 pages to 2 friends twice weekly for 52 weeks.", response
    )
    assert execute_fraction(graph) == 624
    assert receipt.state_reads == 2


def test_fraction_atoms_are_parenthesized_before_division() -> None:
    assert expression_atom(Fraction(1, 4)) == "(1/4)"
    program = parse_program(
        "<MICROCODE_V1>\n"
        "R0 P:1/4 P:1/20 D\n"
        "C:0\n"
        "</MICROCODE_V1>"
    )
    response, final, _ = render_transaction_trace(program)
    graph, _ = compile_draft_transactions("Use 1/4 and 1/20.", response)
    assert final == 5
    assert execute_fraction(graph) == 5
