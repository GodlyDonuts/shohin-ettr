from __future__ import annotations

from dataclasses import replace

import pytest

from episode_functor_algebra_machine import (
    OP_AXPY,
    OP_HALT,
    OP_INV,
    OP_LOAD,
    OP_NEG,
    OP_SCALE,
    OP_SWAP,
    AlgebraInstruction,
    AlgebraMachineError,
    execute_program,
    verify_reduction_program,
)


def _valid_program() -> tuple[AlgebraInstruction, ...]:
    # [[2,1,0], [1,1,1]] -> [[1,0,256], [0,1,2]] over F_257.
    return (
        AlgebraInstruction(OP_LOAD, 1, 0, 0),
        AlgebraInstruction(OP_INV, 0, 1),
        AlgebraInstruction(OP_SCALE, 1, 1),
        AlgebraInstruction(OP_LOAD, 0, 0, 0),
        AlgebraInstruction(OP_NEG, 0, 2),
        AlgebraInstruction(OP_AXPY, 0, 1, 2),
        AlgebraInstruction(OP_SWAP, 0, 1),
        AlgebraInstruction(OP_LOAD, 1, 1, 0),
        AlgebraInstruction(OP_INV, 0, 1),
        AlgebraInstruction(OP_SCALE, 1, 1),
        AlgebraInstruction(OP_LOAD, 0, 1, 0),
        AlgebraInstruction(OP_NEG, 0, 2),
        AlgebraInstruction(OP_AXPY, 0, 1, 2),
        AlgebraInstruction(OP_HALT),
    )


def test_explicit_program_reduces_and_certifies_exactly() -> None:
    matrix = ((2, 1, 0), (1, 1, 1))
    state = execute_program(matrix, _valid_program())
    assert state.halted
    assert state.rows == ((1, 0, 256), (0, 1, 2))
    receipt = verify_reduction_program(matrix, state)
    assert receipt.passed
    assert receipt.rank == 2
    assert receipt.executed_instructions == len(_valid_program())
    assert receipt.field_multiply_adds == 12
    assert len(receipt.trace_sha256) == 64


def test_machine_contains_no_automatic_pivot_or_halt() -> None:
    matrix = ((2, 1, 0), (1, 1, 1))
    state = execute_program(matrix, _valid_program()[:-1])
    assert not state.halted
    with pytest.raises(AlgebraMachineError, match="did not emit HALT"):
        verify_reduction_program(matrix, state)

    early = execute_program(matrix, (AlgebraInstruction(OP_HALT),))
    with pytest.raises(AlgebraMachineError, match="pivot"):
        verify_reduction_program(matrix, early)


def test_invalid_controller_actions_fail_closed() -> None:
    matrix = ((0, 1), (1, 0))
    with pytest.raises(AlgebraMachineError, match="invert zero"):
        execute_program(
            matrix,
            (
                AlgebraInstruction(OP_LOAD, 0, 0, 0),
                AlgebraInstruction(OP_INV, 0, 1),
            ),
        )
    with pytest.raises(AlgebraMachineError, match="out of range"):
        execute_program(
            matrix,
            (AlgebraInstruction(OP_SWAP, 0, 2),),
        )
    with pytest.raises(AlgebraMachineError, match="after HALT"):
        execute_program(
            matrix,
            (
                AlgebraInstruction(OP_HALT),
                AlgebraInstruction(OP_SWAP, 0, 1),
            ),
        )


def test_provenance_and_rref_tampering_are_rejected() -> None:
    matrix = ((2, 1, 0), (1, 1, 1))
    state = execute_program(matrix, _valid_program())
    bad_provenance = replace(
        state,
        provenance=((1, 0), state.provenance[1]),
    )
    with pytest.raises(AlgebraMachineError, match="does not reconstruct"):
        verify_reduction_program(matrix, bad_provenance)

    bad_rows = replace(
        state,
        rows=((1, 1, 0), state.rows[1]),
    )
    with pytest.raises(AlgebraMachineError):
        verify_reduction_program(matrix, bad_rows)


def test_noninvertible_scale_and_resource_bounds_fail_closed() -> None:
    with pytest.raises(AlgebraMachineError, match="remain invertible"):
        execute_program(
            ((1,),),
            (AlgebraInstruction(OP_SCALE, 0, 0),),
        )
    with pytest.raises(AlgebraMachineError, match="instruction limit"):
        execute_program(
            ((1,),),
            (AlgebraInstruction(OP_HALT), AlgebraInstruction(OP_HALT)),
            maximum_instructions=1,
        )
