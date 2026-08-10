import torch

from natural_microcode_program import parse_program
from typed_microcode_compiler import (
    MAX_SOURCE_SPANS,
    MAX_STEPS,
    REFERENCE_CLASSES,
    TypedMicrocodeCompiler,
    graph_labels,
    typed_compiler_loss,
)
from typed_microcode_graph import compile_typed_graph


def _graphs():
    program = parse_program("""<MICROCODE_V1>
R0 P:3 P:2 M
R1 L:0 P:52 M
C:1
</MICROCODE_V1>""")
    return [compile_typed_graph("Use 3 and 2 each week.", program)]


def test_graph_labels_preserve_typed_owners() -> None:
    labels = graph_labels(_graphs(), torch.device("cpu"))
    assert labels["length"].tolist() == [1]
    assert labels["active"].sum() == 2
    assert labels["reference_targets"][0, 0, 0, 0]
    assert labels["reference_targets"][0, 1, 0, MAX_SOURCE_SPANS]
    assert labels["reference_targets"][0, 1, 1, REFERENCE_CLASSES - 1]
    assert labels["literal_mask"][0, 1, 1]


def test_compiler_shapes_and_causal_reference_mask() -> None:
    compiler = TypedMicrocodeCompiler(
        32, width=32, source_layers=1, decoder_layers=1, heads=4
    )
    source = torch.randn(2, 7, 32)
    attention = torch.ones(2, 7, dtype=torch.bool)
    candidates = torch.zeros(2, MAX_SOURCE_SPANS, 7, dtype=torch.bool)
    candidates[:, 0, 1] = True
    output = compiler(source, attention, candidates, torch.tensor([1, 1]))
    assert output.operation_logits.shape == (2, MAX_STEPS, 6)
    assert output.left_reference_logits.shape == (2, MAX_STEPS, REFERENCE_CLASSES)
    assert (output.left_reference_logits[:, 0, MAX_SOURCE_SPANS:-1] < -1e20).all()
    labels = graph_labels(_graphs() * 2, torch.device("cpu"))
    loss, components = typed_compiler_loss(output, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in compiler.parameters())
    assert set(components) == {
        "length",
        "operation",
        "left_reference",
        "right_reference",
        "literal",
    }
