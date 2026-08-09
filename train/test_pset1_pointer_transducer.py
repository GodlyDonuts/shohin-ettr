import pytest
import torch

from pset1_pointer_transducer import (
    EditProgram,
    KEEP,
    PSET1Config,
    PSET1Error,
    PSET1PointerHead,
    REPLACE,
    execute_program,
)


def test_executor_preserves_or_splices_exact_bytes() -> None:
    draft = "answer: 41"
    offsets = [[0, 6], [6, 7], [7, 9], [9, 10]]
    assert execute_program(draft, offsets, EditProgram(KEEP)) == draft
    assert execute_program(draft, offsets, EditProgram(REPLACE, 9, 9, "2")) == "answer: 42"


def test_executor_fails_closed() -> None:
    with pytest.raises(PSET1Error):
        execute_program("x", [[0, 1]], EditProgram(REPLACE, 1, 1, "y"))


def test_pointer_head_shapes() -> None:
    config = PSET1Config(host_hidden_size=16, width=8, attention_heads=2, ff_width=16)
    head = PSET1PointerHead(config)
    source, fused, action, pointers = head.encode(
        torch.randn(2, 3, 16),
        torch.ones(2, 3, dtype=torch.bool),
        torch.randn(2, 5, 16),
        torch.ones(2, 5, dtype=torch.bool),
        torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]),
        torch.tensor([[65, 66, 67, 68, 69], [65, 66, 67, 68, 69]]),
        torch.ones(2, 5, dtype=torch.bool),
    )
    assert source.shape == (2, 3, 8)
    assert fused.shape == (2, 5, 8)
    assert action.shape == (2, 2)
    assert pointers.shape == (2, 2, 5)
