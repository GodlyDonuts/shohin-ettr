from natural_microcode_program import parse_program
from typed_microcode_graph import Operand, SOURCE, compile_typed_graph
from eval_tmc1_development import operand_equivalent, source_shuffle


def test_equal_source_values_are_equivalent_across_mentions() -> None:
    program = parse_program("""<MICROCODE_V1>
R0 P:3 P:2 A
C:0
</MICROCODE_V1>""")
    graph = compile_typed_graph("3 plus another 3 and 2.", program)
    assert operand_equivalent(
        Operand(SOURCE, (1,)), graph.instructions[0].left, graph, graph
    )


def test_source_shuffle_preserves_register_depth() -> None:
    rows = [
        {"identity_sha256": f"{index:064x}", "register_depth": depth}
        for depth in (1, 2)
        for index in range(depth * 10, depth * 10 + 3)
    ]
    mapping = source_shuffle(rows)
    by_identity = {str(row["identity_sha256"]): row for row in rows}
    assert set(mapping) == set(by_identity)
    for identity, donor in mapping.items():
        assert identity != donor["identity_sha256"]
        assert by_identity[identity]["register_depth"] == donor["register_depth"]
