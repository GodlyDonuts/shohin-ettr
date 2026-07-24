"""Counted recurrent controller for the primitive SSQAC field-row machine.

The controller receives the current matrix, scalar registers, valid geometry,
its previous instruction, and its own recurrent state. It predicts exactly one
primitive VM instruction. No algorithmic cursor, row role, elimination phase,
or reference schedule is supplied at inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import Tensor, nn

from train.episode_functor_algebra_machine import (
    FIELD_MODULUS,
    OP_AXPY,
    OP_HALT,
    OP_INV,
    OP_LOAD,
    OP_NEG,
    OP_SCALE,
    OP_SWAP,
    OPCODES,
    AlgebraInstruction,
)


CONTROLLER_SCHEMA = "ssqac_neural_algebra_controller_v1"
PREVIOUS_START = len(OPCODES)


class NeuralAlgebraControllerError(ValueError):
    """The neural controller interface failed closed."""


def _sinusoidal_positions(count: int, width: int) -> Tensor:
    """Return deterministic coordinates with no untrained position rows."""

    positions = torch.arange(count, dtype=torch.float32)[:, None]
    frequencies = torch.exp(
        torch.arange(0, width, 2, dtype=torch.float32)
        * (-math.log(10_000.0) / width)
    )
    result = torch.zeros(count, width, dtype=torch.float32)
    result[:, 0::2] = torch.sin(positions * frequencies)
    if width > 1:
        result[:, 1::2] = torch.cos(
            positions * frequencies[: result[:, 1::2].shape[1]]
        )
    return result


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    maximum_rows: int = 64
    maximum_columns: int = 128
    register_count: int = 8
    width: int = 192
    layers: int = 4
    heads: int = 6
    feedforward: int = 768
    dropout: float = 0.0
    maximum_steps: int = 4096

    def __post_init__(self) -> None:
        integer_fields = {
            "maximum_rows": self.maximum_rows,
            "maximum_columns": self.maximum_columns,
            "register_count": self.register_count,
            "width": self.width,
            "layers": self.layers,
            "heads": self.heads,
            "feedforward": self.feedforward,
            "maximum_steps": self.maximum_steps,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise NeuralAlgebraControllerError(
                    f"{name} must be a positive integer"
                )
        if self.width % self.heads:
            raise NeuralAlgebraControllerError(
                "controller width must be divisible by attention heads"
            )
        if not isinstance(self.dropout, float) or not 0.0 <= self.dropout < 1.0:
            raise NeuralAlgebraControllerError(
                "dropout must be a float in [0, 1)"
            )


@dataclass(frozen=True, slots=True)
class ControllerLogits:
    opcode: Tensor
    row_a: Tensor
    row_b: Tensor
    column: Tensor
    register_a: Tensor
    register_b: Tensor

    def as_mapping(self) -> Mapping[str, Tensor]:
        return {
            "column": self.column,
            "opcode": self.opcode,
            "register_a": self.register_a,
            "register_b": self.register_b,
            "row_a": self.row_a,
            "row_b": self.row_b,
        }


@dataclass(frozen=True, slots=True)
class HardenedControllerDecision:
    instruction: AlgebraInstruction
    minimum_margin: float


class NeuralAlgebraController(nn.Module):
    """Variable-geometry recurrent instruction policy."""

    def __init__(self, config: ControllerConfig = ControllerConfig()) -> None:
        super().__init__()
        self.config = config
        width = config.width
        self.coefficient_embedding = nn.Embedding(FIELD_MODULUS, width)
        self.register_embedding = nn.Embedding(config.register_count, width)
        self.opcode_embedding = nn.Embedding(len(OPCODES) + 1, width)
        self.maximum_operand = max(
            config.maximum_rows,
            config.maximum_columns,
            config.register_count,
        )
        self.register_buffer(
            "row_positions",
            _sinusoidal_positions(config.maximum_rows, width),
            persistent=True,
        )
        self.register_buffer(
            "column_positions",
            _sinusoidal_positions(config.maximum_columns, width),
            persistent=True,
        )
        self.register_buffer(
            "operand_positions",
            _sinusoidal_positions(self.maximum_operand + 1, width),
            persistent=True,
        )
        self.register_buffer(
            "step_positions",
            _sinusoidal_positions(config.maximum_steps, width),
            persistent=True,
        )
        self.previous_a_projection = nn.Linear(width, width, bias=False)
        self.previous_b_projection = nn.Linear(width, width, bias=False)
        self.previous_c_projection = nn.Linear(width, width, bias=False)
        self.step_projection = nn.Linear(width, width, bias=False)
        self.token_type_embedding = nn.Embedding(3, width)
        self.control_token = nn.Parameter(torch.empty(width))
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config.heads,
            dim_feedforward=config.feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.layers,
            enable_nested_tensor=False,
        )
        self.recurrent = nn.GRUCell(width, width)
        self.output_norm = nn.LayerNorm(width)
        self.opcode_head = nn.Linear(width, len(OPCODES))
        self.row_a_query = nn.Linear(width, width, bias=False)
        self.row_b_query = nn.Linear(width, width, bias=False)
        self.column_query = nn.Linear(width, width, bias=False)
        self.register_a_head = nn.Linear(width, config.register_count)
        self.register_b_head = nn.Linear(width, config.register_count)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.control_token, mean=0.0, std=0.02)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def initial_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise NeuralAlgebraControllerError("batch size must be an integer")
        if batch_size < 1:
            raise NeuralAlgebraControllerError("batch size must be positive")
        reference = self.control_token
        return torch.zeros(
            batch_size,
            self.config.width,
            device=device if device is not None else reference.device,
            dtype=dtype if dtype is not None else reference.dtype,
        )

    def forward(
        self,
        *,
        rows: Tensor,
        registers: Tensor,
        row_mask: Tensor,
        column_mask: Tensor,
        previous_opcode: Tensor,
        previous_a: Tensor,
        previous_b: Tensor,
        previous_c: Tensor,
        step: Tensor,
        hidden: Tensor,
    ) -> tuple[ControllerLogits, Tensor]:
        config = self.config
        if rows.ndim != 3:
            raise NeuralAlgebraControllerError(
                "rows must have shape [batch, rows, columns]"
            )
        batch, row_count, column_count = rows.shape
        if row_count != config.maximum_rows or column_count != config.maximum_columns:
            raise NeuralAlgebraControllerError(
                "rows must use the configured padded geometry"
            )
        expected_shapes = {
            "column_mask": (batch, column_count),
            "hidden": (batch, config.width),
            "previous_a": (batch,),
            "previous_b": (batch,),
            "previous_c": (batch,),
            "previous_opcode": (batch,),
            "registers": (batch, config.register_count),
            "row_mask": (batch, row_count),
            "step": (batch,),
        }
        supplied = {
            "column_mask": column_mask,
            "hidden": hidden,
            "previous_a": previous_a,
            "previous_b": previous_b,
            "previous_c": previous_c,
            "previous_opcode": previous_opcode,
            "registers": registers,
            "row_mask": row_mask,
            "step": step,
        }
        for name, expected in expected_shapes.items():
            if tuple(supplied[name].shape) != expected:
                raise NeuralAlgebraControllerError(
                    f"{name} has shape {tuple(supplied[name].shape)}, "
                    f"expected {expected}"
                )
        if row_mask.dtype != torch.bool or column_mask.dtype != torch.bool:
            raise NeuralAlgebraControllerError(
                "geometry masks must be Boolean"
            )
        if not torch.all(row_mask.any(dim=1)) or not torch.all(
            column_mask.any(dim=1)
        ):
            raise NeuralAlgebraControllerError(
                "every example must expose at least one row and column"
            )
        if torch.any(rows < 0) or torch.any(rows >= FIELD_MODULUS):
            raise NeuralAlgebraControllerError(
                "matrix coefficients leave F_257"
            )
        if torch.any(registers < 0) or torch.any(registers >= FIELD_MODULUS):
            raise NeuralAlgebraControllerError("register values leave F_257")
        if torch.any(previous_opcode < 0) or torch.any(
            previous_opcode > PREVIOUS_START
        ):
            raise NeuralAlgebraControllerError("previous opcode is out of range")
        if torch.any(step < 0) or torch.any(step >= config.maximum_steps):
            raise NeuralAlgebraControllerError("controller step is out of range")

        device = rows.device
        row_positions = torch.arange(row_count, device=device)
        column_positions = torch.arange(column_count, device=device)
        matrix_tokens = (
            self.coefficient_embedding(rows.long())
            + self.row_positions[row_positions][None, :, None, :]
            + self.column_positions[column_positions][None, None, :, :]
            + self.token_type_embedding.weight[0][None, None, None, :]
        ).reshape(batch, row_count * column_count, config.width)
        matrix_mask = (
            row_mask[:, :, None] & column_mask[:, None, :]
        ).reshape(batch, row_count * column_count)

        register_positions = torch.arange(config.register_count, device=device)
        register_tokens = (
            self.coefficient_embedding(registers.long())
            + self.register_embedding(register_positions)[None, :, :]
            + self.token_type_embedding.weight[1][None, None, :]
        )
        control = (
            self.control_token[None, :].expand(batch, -1)
            + self.opcode_embedding(previous_opcode.long())
            + self.previous_a_projection(
                self.operand_positions[
                    previous_a.clamp(0, self.maximum_operand).long()
                ]
            )
            + self.previous_b_projection(
                self.operand_positions[
                    previous_b.clamp(0, self.maximum_operand).long()
                ]
            )
            + self.previous_c_projection(
                self.operand_positions[
                    previous_c.clamp(0, self.maximum_operand).long()
                ]
            )
            + self.step_projection(self.step_positions[step.long()])
            + self.token_type_embedding.weight[2][None, :]
        )
        tokens = torch.cat(
            (control[:, None, :], matrix_tokens, register_tokens),
            dim=1,
        )
        padding_mask = torch.cat(
            (
                torch.zeros(batch, 1, dtype=torch.bool, device=device),
                ~matrix_mask,
                torch.zeros(
                    batch,
                    config.register_count,
                    dtype=torch.bool,
                    device=device,
                ),
            ),
            dim=1,
        )
        encoded = self.encoder(
            tokens,
            src_key_padding_mask=padding_mask,
        )
        next_hidden = self.recurrent(encoded[:, 0, :], hidden)
        output = self.output_norm(next_hidden)
        encoded_matrix = encoded[
            :,
            1 : 1 + row_count * column_count,
            :,
        ].reshape(batch, row_count, column_count, config.width)
        visible = (row_mask[:, :, None] & column_mask[:, None, :]).to(
            encoded_matrix.dtype
        )
        row_representations = (
            (encoded_matrix * visible[:, :, :, None]).sum(dim=2)
            / visible.sum(dim=2).clamp_min(1.0)[:, :, None]
        )
        column_representations = (
            (encoded_matrix * visible[:, :, :, None]).sum(dim=1)
            / visible.sum(dim=1).clamp_min(1.0)[:, :, None]
        )
        scale = math.sqrt(config.width)
        row_a_logits = torch.einsum(
            "brd,bd->br",
            row_representations,
            self.row_a_query(output),
        ) / scale
        row_b_logits = torch.einsum(
            "brd,bd->br",
            row_representations,
            self.row_b_query(output),
        ) / scale
        column_logits = torch.einsum(
            "bcd,bd->bc",
            column_representations,
            self.column_query(output),
        ) / scale
        row_invalid = ~row_mask
        column_invalid = ~column_mask
        logits = ControllerLogits(
            opcode=self.opcode_head(output),
            row_a=row_a_logits.masked_fill(row_invalid, -torch.inf),
            row_b=row_b_logits.masked_fill(row_invalid, -torch.inf),
            column=column_logits.masked_fill(
                column_invalid,
                -torch.inf,
            ),
            register_a=self.register_a_head(output),
            register_b=self.register_b_head(output),
        )
        return logits, next_hidden


def _winner_and_margin(logits: Tensor) -> tuple[int, float]:
    if logits.ndim != 1 or logits.numel() < 2:
        raise NeuralAlgebraControllerError(
            "hardening requires a one-dimensional categorical vector"
        )
    if not torch.isfinite(logits).any():
        raise NeuralAlgebraControllerError("all categorical logits are invalid")
    values, indices = torch.topk(logits, k=2)
    if not torch.isfinite(values[0]):
        raise NeuralAlgebraControllerError("categorical winner is invalid")
    margin = (
        math.inf
        if not torch.isfinite(values[1])
        else float((values[0] - values[1]).item())
    )
    return int(indices[0].item()), margin


def harden_controller_instruction(
    logits: ControllerLogits,
    *,
    minimum_margin: float,
) -> HardenedControllerDecision:
    """Harden one batch-size-one prediction into a typed VM instruction."""

    if not isinstance(minimum_margin, float) or minimum_margin < 0.0:
        raise NeuralAlgebraControllerError(
            "minimum margin must be a nonnegative float"
        )
    vectors = {}
    margins = {}
    for name, tensor in logits.as_mapping().items():
        if tensor.ndim != 2 or tensor.shape[0] != 1:
            raise NeuralAlgebraControllerError(
                "hardening requires batch size one"
            )
        vectors[name], margins[name] = _winner_and_margin(tensor[0])
    opcode = OPCODES[vectors["opcode"]]
    if opcode == OP_LOAD:
        required = ("opcode", "row_a", "column", "register_a")
        instruction = AlgebraInstruction(
            opcode,
            vectors["row_a"],
            vectors["column"],
            vectors["register_a"],
        )
    elif opcode in (OP_INV, OP_NEG):
        required = ("opcode", "register_a", "register_b")
        instruction = AlgebraInstruction(
            opcode,
            vectors["register_a"],
            vectors["register_b"],
        )
    elif opcode == OP_SCALE:
        required = ("opcode", "row_a", "register_a")
        instruction = AlgebraInstruction(
            opcode,
            vectors["row_a"],
            vectors["register_a"],
        )
    elif opcode == OP_AXPY:
        required = ("opcode", "row_a", "row_b", "register_a")
        instruction = AlgebraInstruction(
            opcode,
            vectors["row_a"],
            vectors["row_b"],
            vectors["register_a"],
        )
    elif opcode == OP_SWAP:
        required = ("opcode", "row_a", "row_b")
        instruction = AlgebraInstruction(
            opcode,
            vectors["row_a"],
            vectors["row_b"],
        )
    elif opcode == OP_HALT:
        required = ("opcode",)
        instruction = AlgebraInstruction(opcode)
    else:
        raise NeuralAlgebraControllerError("unreachable hardened opcode")
    minimum = min(margins[name] for name in required)
    if minimum < minimum_margin:
        raise NeuralAlgebraControllerError(
            "categorical hardening margin is below the frozen threshold"
        )
    return HardenedControllerDecision(
        instruction=instruction,
        minimum_margin=minimum,
    )
