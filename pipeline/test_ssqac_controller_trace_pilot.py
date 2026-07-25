from __future__ import annotations

import torch

from pipeline.ssqac_controller_trace_pilot import (
    _batch_state,
    _instruction_loss,
    compile_reference_program,
    generate_examples,
)
from episode_functor_algebra_machine import (
    OP_HALT,
    execute_program,
    verify_reduction_program,
)
from episode_functor_neural_algebra_controller import (
    ControllerConfig,
    NeuralAlgebraController,
)


def test_reference_trace_is_preparation_only_and_vm_certified() -> None:
    matrix = ((0, 2, 0), (1, 1, 0), (0, 0, 0))
    program = compile_reference_program(matrix)
    assert program[-1].opcode == OP_HALT
    state = execute_program(matrix, program, register_count=4)
    receipt = verify_reduction_program(matrix, state)
    assert receipt.passed
    assert receipt.rank == 2


def test_reference_trace_repairs_reverse_pivot_order() -> None:
    for matrix in (
        ((0, 1, 0), (1, 0, 0)),
        ((0, 0, 1), (0, 1, 0), (1, 0, 0)),
    ):
        program = compile_reference_program(matrix)
        state = execute_program(matrix, program, register_count=4)
        receipt = verify_reduction_program(matrix, state)
        assert receipt.passed
        assert state.rows == tuple(
            tuple(int(row == column) for column in range(len(matrix[0])))
            for row in range(len(matrix))
        )


def test_generated_train_and_evaluation_seeds_are_disjoint() -> None:
    left = generate_examples(
        seed=11,
        count=8,
        maximum_rows=3,
        maximum_columns=4,
        register_count=4,
    )
    right = generate_examples(
        seed=12,
        count=8,
        maximum_rows=3,
        maximum_columns=4,
        register_count=4,
    )
    assert not ({example.matrix for example in left} & {example.matrix for example in right})
    assert all(
        len(example.program) == len(example.snapshots) for example in (*left, *right)
    )


def test_one_teacher_forced_controller_update_is_finite() -> None:
    config = ControllerConfig(
        maximum_rows=3,
        maximum_columns=4,
        register_count=4,
        width=32,
        layers=1,
        heads=4,
        feedforward=64,
        maximum_steps=128,
    )
    examples = generate_examples(
        seed=17,
        count=2,
        maximum_rows=3,
        maximum_columns=4,
        register_count=4,
    )
    controller = NeuralAlgebraController(config)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=1e-3)
    hidden = controller.initial_hidden(len(examples))
    inputs, active = _batch_state(
        examples,
        0,
        config,
        torch.device("cpu"),
    )
    logits, hidden = controller(hidden=hidden, **inputs)
    loss, correct, decisions = _instruction_loss(
        logits,
        examples,
        0,
        active,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert 0 <= correct <= decisions
    assert decisions > 0
    assert torch.isfinite(hidden).all()
