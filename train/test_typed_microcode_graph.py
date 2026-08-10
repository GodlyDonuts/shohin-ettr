from fractions import Fraction

from natural_microcode_program import parse_program
from typed_microcode_graph import (
    LITERAL,
    SOURCE,
    STATE,
    compile_typed_graph,
    execute_fraction,
    number_spans,
    operand_count,
)


def test_source_fraction_normalizes_percent_and_commas() -> None:
    spans = number_spans("A $1,200 value falls by 25%.")
    assert [span.value for span in spans] == [Fraction(1200), Fraction(1, 4)]


def test_compiles_source_state_and_implicit_literal_owners() -> None:
    program = parse_program("""<MICROCODE_V1>
R0 P:3 P:2 M
R1 L:0 P:52 M
C:1
</MICROCODE_V1>""")
    graph = compile_typed_graph("Write 3 pages to 2 friends each week.", program)
    assert execute_fraction(graph) == 312
    assert operand_count(graph, SOURCE) == 2
    assert operand_count(graph, STATE) == 1
    assert operand_count(graph, LITERAL) == 1


def test_lowers_nested_postfix_expression_to_causal_binary_graph() -> None:
    program = parse_program("""<MICROCODE_V1>
R0 P:2 P:3 A P:4 M
C:0
</MICROCODE_V1>""")
    graph = compile_typed_graph("Use 2, 3, and 4.", program)
    assert len(graph.instructions) == 2
    assert graph.instructions[1].left.kind == STATE
    assert graph.instructions[1].left.indices == (0,)
    assert execute_fraction(graph) == 20


def test_materializes_identity_final_as_copy_state() -> None:
    program = parse_program("""<MICROCODE_V1>
R0 P:48
C:0
</MICROCODE_V1>""")
    graph = compile_typed_graph("There are 48 items.", program)
    assert graph.instructions[-1].operation == "COPY"
    assert graph.final.kind == STATE
    assert execute_fraction(graph) == 48
