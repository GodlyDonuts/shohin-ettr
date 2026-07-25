from __future__ import annotations

import torch
import pytest

from episode_functor_algebra_machine import OP_AXPY, OPCODES
from episode_functor_neural_algebra_controller import (
    ControllerConfig,
    ControllerLogits,
    NeuralAlgebraController,
    NeuralAlgebraControllerError,
    PREVIOUS_START,
    harden_controller_instruction,
)


def _inputs(config: ControllerConfig, batch: int = 2) -> dict[str, torch.Tensor]:
    rows = torch.zeros(
        batch,
        config.maximum_rows,
        config.maximum_columns,
        dtype=torch.long,
    )
    rows[:, :3, :4] = torch.randint(0, 257, (batch, 3, 4))
    row_mask = torch.zeros(batch, config.maximum_rows, dtype=torch.bool)
    row_mask[:, :3] = True
    column_mask = torch.zeros(batch, config.maximum_columns, dtype=torch.bool)
    column_mask[:, :4] = True
    return {
        "column_mask": column_mask,
        "hidden": torch.zeros(batch, config.width),
        "previous_a": torch.zeros(batch, dtype=torch.long),
        "previous_b": torch.zeros(batch, dtype=torch.long),
        "previous_c": torch.zeros(batch, dtype=torch.long),
        "previous_opcode": torch.full(
            (batch,),
            PREVIOUS_START,
            dtype=torch.long,
        ),
        "registers": torch.zeros(batch, config.register_count, dtype=torch.long),
        "row_mask": row_mask,
        "rows": rows,
        "step": torch.zeros(batch, dtype=torch.long),
    }


def test_controller_is_counted_recurrent_and_geometry_masked() -> None:
    config = ControllerConfig(
        maximum_rows=6,
        maximum_columns=8,
        register_count=4,
        width=48,
        layers=2,
        heads=6,
        feedforward=96,
        maximum_steps=32,
    )
    controller = NeuralAlgebraController(config)
    inputs = _inputs(config)
    logits, hidden = controller(**inputs)
    assert hidden.shape == (2, config.width)
    assert logits.opcode.shape == (2, len(OPCODES))
    assert logits.row_a.shape == (2, config.maximum_rows)
    assert logits.column.shape == (2, config.maximum_columns)
    assert torch.isneginf(logits.row_a[:, 3:]).all()
    assert torch.isneginf(logits.column[:, 4:]).all()
    assert torch.isfinite(logits.row_a[:, :3]).all()
    assert controller.parameter_count == sum(
        parameter.numel() for parameter in controller.parameters()
    )
    assert controller.parameter_count > 0


def test_default_controller_fits_the_twelve_million_parameter_lane() -> None:
    controller = NeuralAlgebraController()
    assert controller.parameter_count < 12_000_000


def test_hardening_emits_only_typed_primitive_instruction() -> None:
    def vector(size: int, winner: int) -> torch.Tensor:
        result = torch.full((1, size), -8.0)
        result[0, winner] = 8.0
        return result

    logits = ControllerLogits(
        opcode=vector(len(OPCODES), OPCODES.index(OP_AXPY)),
        row_a=vector(5, 3),
        row_b=vector(5, 1),
        column=vector(7, 6),
        register_a=vector(4, 2),
        register_b=vector(4, 0),
    )
    decision = harden_controller_instruction(logits, minimum_margin=1.0)
    assert decision.instruction.opcode == OP_AXPY
    assert (
        decision.instruction.a,
        decision.instruction.b,
        decision.instruction.c,
    ) == (3, 1, 2)
    assert decision.minimum_margin == 16.0


def test_hardening_and_forward_fail_closed_on_invalid_inputs() -> None:
    logits = ControllerLogits(
        opcode=torch.zeros(1, len(OPCODES)),
        row_a=torch.zeros(1, 2),
        row_b=torch.zeros(1, 2),
        column=torch.zeros(1, 2),
        register_a=torch.zeros(1, 2),
        register_b=torch.zeros(1, 2),
    )
    with pytest.raises(NeuralAlgebraControllerError, match="below"):
        harden_controller_instruction(logits, minimum_margin=0.1)

    config = ControllerConfig(
        maximum_rows=4,
        maximum_columns=4,
        register_count=2,
        width=24,
        layers=1,
        heads=4,
        feedforward=48,
        maximum_steps=8,
    )
    controller = NeuralAlgebraController(config)
    inputs = _inputs(config, batch=1)
    inputs["row_mask"][:] = False
    with pytest.raises(NeuralAlgebraControllerError, match="at least one"):
        controller(**inputs)
