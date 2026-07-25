#!/usr/bin/env python3
"""Vectorized reactive SSQAC controller pilot.

The preparation oracle is used once to produce a flattened state dataset.
Optimization never unrolls trajectories: all matrix states, masks, previous
primitive instructions, and typed labels are tensorized before the first
optimizer update.  The candidate is a step-free feed-forward policy with
shared content-pointer heads.  It has no recurrent state, no row or column
coordinate features, and no learned absolute row or column tables.  Matrix
mixing is built only from shared cell transforms, masked row/column/global
reductions, and typed previous-pointer role markers.  Consequently the policy
is mathematically equivariant to arbitrary valid-row and valid-column
permutations (up to ordinary floating-point reduction-order noise).

Final evaluation crosses a hard boundary to matrix-only inputs and invokes
only the learned policy, the primitive field-row VM, and the independent RREF
verifier.  This remains a mechanics falsifier, not a reasoning claim, unless
a sufficiently large strictly held-out geometry split materially passes the
frozen exact-certification gate.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from episode_functor_algebra_machine import (
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
    AlgebraMachineError,
    execute_program,
    verify_reduction_program,
)
from pipeline.ssqac_controller_trace_pilot import (
    TraceExample,
    generate_examples,
)


PILOT_SCHEMA = "ssqac_vectorized_reactive_pilot_v2"
DATASET_SCHEMA = "ssqac_vectorized_reactive_dataset_v1"
MODEL_SCHEMA = "ssqac_permutation_equivariant_reactive_policy_v2"
STATUS_NOT_REASONING = "vectorized_reactive_mechanics_falsifier_only_not_reasoning"
STATUS_MATERIAL_PASS = (
    "strict_heldout_certification_materially_passed_replication_required"
)
CANDIDATE_RUNTIME = "step_free_feed_forward_content_pointer_policy_plus_primitive_vm"
ORACLE_BOUNDARY = "preparation_only_flattened_labels_never_autonomous_evaluation"
PREVIOUS_START = len(OPCODES)
IGNORE_INDEX = -100

_OPCODE_TO_INDEX = {opcode: index for index, opcode in enumerate(OPCODES)}
_ROW_A_OPCODES = frozenset((OP_LOAD, OP_SCALE, OP_AXPY, OP_SWAP))
_ROW_B_OPCODES = frozenset((OP_AXPY, OP_SWAP))
_REGISTER_A_OPCODES = frozenset((OP_LOAD, OP_INV, OP_NEG, OP_SCALE, OP_AXPY))
_REGISTER_B_OPCODES = frozenset((OP_INV, OP_NEG))

_INPUT_TENSOR_NAMES = (
    "rows",
    "registers",
    "row_mask",
    "column_mask",
    "previous_opcode",
    "previous_a",
    "previous_b",
    "previous_c",
)
_TARGET_TENSOR_NAMES = (
    "target_opcode",
    "target_row_a",
    "target_row_b",
    "target_column",
    "target_register_a",
    "target_register_b",
)


class VectorizedReactivePilotError(ValueError):
    """The isolated pilot contract failed closed."""


@dataclass(frozen=True, slots=True)
class ReactivePolicyConfig:
    """Geometry bounds and shape-independent policy capacity."""

    maximum_rows: int = 6
    maximum_columns: int = 8
    register_count: int = 4
    width: int = 256
    blocks: int = 6
    feedforward: int = 768
    dropout: float = 0.0

    def __post_init__(self) -> None:
        integer_fields = {
            "maximum_rows": self.maximum_rows,
            "maximum_columns": self.maximum_columns,
            "register_count": self.register_count,
            "width": self.width,
            "blocks": self.blocks,
            "feedforward": self.feedforward,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise VectorizedReactivePilotError(f"{name} must be a positive integer")
        if self.maximum_columns < self.maximum_rows:
            raise VectorizedReactivePilotError(
                "maximum columns must admit every row geometry"
            )
        if not isinstance(self.dropout, float) or not 0.0 <= self.dropout < 1.0:
            raise VectorizedReactivePilotError("dropout must be a float in [0, 1)")

    @property
    def canonical_sha256(self) -> str:
        return sha256(
            json.dumps(
                asdict(self),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class FlattenedExpertState:
    """One preparation-labeled state with no trajectory index or step."""

    rows: tuple[tuple[int, ...], ...]
    registers: tuple[int, ...]
    previous_instruction: AlgebraInstruction | None
    target_instruction: AlgebraInstruction

    def canonical_data(self) -> list[object]:
        previous = (
            ["START", 0, 0, 0]
            if self.previous_instruction is None
            else self.previous_instruction.canonical_data()
        )
        return [
            [list(row) for row in self.rows],
            list(self.registers),
            previous,
            self.target_instruction.canonical_data(),
        ]

    @property
    def sha256(self) -> str:
        return sha256(
            json.dumps(
                self.canonical_data(),
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class ControllerObservation:
    """The complete candidate-visible state for one primitive decision."""

    rows: tuple[tuple[int, ...], ...]
    registers: tuple[int, ...]
    previous_instruction: AlgebraInstruction | None


@dataclass(frozen=True, slots=True)
class MatrixOnlyCase:
    """Final candidate input after every expert artifact is deleted."""

    matrix: tuple[tuple[int, ...], ...]

    @property
    def matrix_sha256(self) -> str:
        return sha256(
            json.dumps(
                self.matrix,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class TensorizedStateDataset:
    """A fully padded, mask-bearing state dataset."""

    tensors: Mapping[str, Tensor]
    examples: int
    manifest_sha256: str
    source_state_manifest_sha256: str

    def __post_init__(self) -> None:
        expected = frozenset((*_INPUT_TENSOR_NAMES, *_TARGET_TENSOR_NAMES))
        if frozenset(self.tensors) != expected:
            raise VectorizedReactivePilotError(
                "tensorized dataset fields differ from the frozen schema"
            )
        if self.examples < 1:
            raise VectorizedReactivePilotError(
                "tensorized dataset must contain examples"
            )
        for name, tensor in self.tensors.items():
            if tensor.shape[0] != self.examples:
                raise VectorizedReactivePilotError(
                    f"dataset tensor {name} has a different leading dimension"
                )

    def to(self, device: torch.device) -> TensorizedStateDataset:
        return TensorizedStateDataset(
            tensors={
                name: tensor.to(device, non_blocking=True)
                for name, tensor in self.tensors.items()
            },
            examples=self.examples,
            manifest_sha256=self.manifest_sha256,
            source_state_manifest_sha256=self.source_state_manifest_sha256,
        )


@dataclass(frozen=True, slots=True)
class PolicyLogits:
    opcode: Tensor
    row_a: Tensor
    row_b: Tensor
    column: Tensor
    register_a: Tensor
    register_b: Tensor

    def as_mapping(self) -> Mapping[str, Tensor]:
        return {
            "opcode": self.opcode,
            "row_a": self.row_a,
            "row_b": self.row_b,
            "column": self.column,
            "register_a": self.register_a,
            "register_b": self.register_b,
        }


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    optimizer_updates: int
    mean_training_loss: float
    final_training_loss: float


@dataclass(frozen=True, slots=True)
class NamedAccuracy:
    name: str
    correct: int
    total: int
    accuracy: float


@dataclass(frozen=True, slots=True)
class TeacherForcedMetrics:
    full_instruction_correct: int
    full_instruction_total: int
    full_instruction_accuracy: float
    components: tuple[NamedAccuracy, ...]


@dataclass(frozen=True, slots=True)
class RolloutMetrics:
    certified: int
    invalid: int
    overlong: int
    oracle_calls: int
    model_batches: int
    model_decisions: int

    @property
    def total(self) -> int:
        return self.certified + self.invalid + self.overlong

    @property
    def certification_rate(self) -> float:
        return self.certified / self.total if self.total else 0.0


@dataclass(frozen=True, slots=True)
class VectorizedReactiveReport:
    schema: str
    status: str
    model_schema: str
    candidate_runtime: str
    preparation_oracle_boundary: str
    autonomous_input_fields: tuple[str, ...]
    seed: int
    device: str
    amp_bfloat16: bool
    torch_compile: bool
    flattened_state_dataset: bool
    dataset_resident_on_device: bool
    step_signal_exposed: bool
    recurrent_state: bool
    learned_absolute_row_table: bool
    learned_absolute_column_table: bool
    row_coordinate_features: bool
    column_coordinate_features: bool
    exact_row_permutation_equivariance: bool
    exact_column_permutation_equivariance: bool
    preparation_oracle_order_sensitive: bool
    shared_content_pointer_heads: bool
    train_matrices: int
    audit_matrices: int
    evaluation_matrices: int
    flattened_train_states: int
    flattened_audit_states: int
    fit_maximum_rows: int
    fit_maximum_columns: int
    evaluation_minimum_rows: int
    evaluation_minimum_columns: int
    evaluation_maximum_rows: int
    evaluation_maximum_columns: int
    maximum_rollout_steps: int
    controller_parameters: int
    optimizer_updates: int
    mean_training_loss: float
    final_training_loss: float
    teacher_forced_full_instruction_correct: int
    teacher_forced_full_instruction_total: int
    teacher_forced_full_instruction_accuracy: float
    teacher_forced_components: tuple[NamedAccuracy, ...]
    closed_loop_certified: int
    closed_loop_total: int
    closed_loop_certification_rate: float
    invalid_programs: int
    overlong_programs: int
    final_rollout_oracle_calls: int
    final_rollout_model_batches: int
    final_rollout_model_decisions: int
    material_minimum_evaluation_cases: int
    material_minimum_certification_rate: float
    material_certification_gate_passed: bool
    strict_geometry_disjoint: bool
    train_matrix_manifest_sha256: str
    audit_matrix_manifest_sha256: str
    evaluation_matrix_manifest_sha256: str
    train_state_manifest_sha256: str
    audit_state_manifest_sha256: str
    train_tensor_manifest_sha256: str
    audit_tensor_manifest_sha256: str
    model_config_sha256: str
    model_state_sha256: str

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                asdict(self),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
            + b"\n"
        )


def _previous_data(
    instruction: AlgebraInstruction | None,
) -> tuple[int, int, int, int]:
    if instruction is None:
        return PREVIOUS_START, 0, 0, 0
    return (
        _OPCODE_TO_INDEX[instruction.opcode],
        instruction.a,
        instruction.b,
        instruction.c,
    )


def flatten_expert_states(
    examples: Iterable[TraceExample],
) -> tuple[FlattenedExpertState, ...]:
    """Flatten complete traces before optimization begins."""

    result: list[FlattenedExpertState] = []
    for example in examples:
        if len(example.program) != len(example.snapshots):
            raise VectorizedReactivePilotError(
                "trace labels and snapshots have different lengths"
            )
        previous: AlgebraInstruction | None = None
        for snapshot, target in zip(
            example.snapshots,
            example.program,
            strict=True,
        ):
            result.append(
                FlattenedExpertState(
                    rows=snapshot.rows,
                    registers=snapshot.registers,
                    previous_instruction=previous,
                    target_instruction=target,
                )
            )
            previous = target
    if not result:
        raise VectorizedReactivePilotError("flattened expert state dataset is empty")
    return tuple(result)


def _state_manifest(states: Iterable[FlattenedExpertState]) -> str:
    frozen = tuple(states)
    return sha256(
        (
            DATASET_SCHEMA + "\n" + "\n".join(state.sha256 for state in frozen) + "\n"
        ).encode("ascii")
    ).hexdigest()


def _matrix_manifest(
    cases: Iterable[TraceExample | MatrixOnlyCase],
) -> str:
    frozen = tuple(cases)
    return sha256(
        (
            PILOT_SCHEMA
            + "\n"
            + "\n".join(case.matrix_sha256 for case in frozen)
            + "\n"
        ).encode("ascii")
    ).hexdigest()


def _tensor_manifest(tensors: Mapping[str, Tensor]) -> str:
    digest = sha256()
    digest.update(DATASET_SCHEMA.encode("ascii"))
    digest.update(b"\0")
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(
                list(tensor.shape),
                separators=(",", ":"),
            ).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def model_state_sha256(model: nn.Module) -> str:
    digest = sha256()
    digest.update(MODEL_SCHEMA.encode("ascii"))
    digest.update(b"\0")
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_observation(
    observation: ControllerObservation | FlattenedExpertState,
    config: ReactivePolicyConfig,
) -> tuple[int, int]:
    row_count = len(observation.rows)
    if not 1 <= row_count <= config.maximum_rows:
        raise VectorizedReactivePilotError("state row count leaves controller geometry")
    column_count = len(observation.rows[0])
    if not 1 <= column_count <= config.maximum_columns:
        raise VectorizedReactivePilotError(
            "state column count leaves controller geometry"
        )
    if any(len(row) != column_count for row in observation.rows):
        raise VectorizedReactivePilotError("state matrix is not rectangular")
    if len(observation.registers) != config.register_count:
        raise VectorizedReactivePilotError(
            "state register count differs from controller configuration"
        )
    if any(
        value < 0 or value >= FIELD_MODULUS for row in observation.rows for value in row
    ):
        raise VectorizedReactivePilotError("state matrix coefficient leaves F_257")
    if any(value < 0 or value >= FIELD_MODULUS for value in observation.registers):
        raise VectorizedReactivePilotError("register value leaves F_257")
    return row_count, column_count


def tensorize_observations(
    observations: Sequence[ControllerObservation | FlattenedExpertState],
    config: ReactivePolicyConfig,
) -> dict[str, Tensor]:
    """Pad a complete batch once and expose exact row/column masks."""

    if not observations:
        raise VectorizedReactivePilotError("observation batch is empty")
    batch = len(observations)
    rows = torch.zeros(
        batch,
        config.maximum_rows,
        config.maximum_columns,
        dtype=torch.long,
    )
    registers = torch.zeros(
        batch,
        config.register_count,
        dtype=torch.long,
    )
    row_mask = torch.zeros(
        batch,
        config.maximum_rows,
        dtype=torch.bool,
    )
    column_mask = torch.zeros(
        batch,
        config.maximum_columns,
        dtype=torch.bool,
    )
    previous = torch.zeros(batch, 4, dtype=torch.long)
    for index, observation in enumerate(observations):
        row_count, column_count = _validate_observation(observation, config)
        rows[index, :row_count, :column_count] = torch.tensor(
            observation.rows,
            dtype=torch.long,
        )
        registers[index] = torch.tensor(
            observation.registers,
            dtype=torch.long,
        )
        row_mask[index, :row_count] = True
        column_mask[index, :column_count] = True
        previous[index] = torch.tensor(
            _previous_data(observation.previous_instruction),
            dtype=torch.long,
        )
    return {
        "rows": rows,
        "registers": registers,
        "row_mask": row_mask,
        "column_mask": column_mask,
        "previous_opcode": previous[:, 0],
        "previous_a": previous[:, 1],
        "previous_b": previous[:, 2],
        "previous_c": previous[:, 3],
    }


def _target_operands(
    target: AlgebraInstruction,
) -> tuple[int, int, int, int, int, int]:
    opcode = _OPCODE_TO_INDEX[target.opcode]
    row_a = target.a if target.opcode in _ROW_A_OPCODES else IGNORE_INDEX
    row_b = target.b if target.opcode in _ROW_B_OPCODES else IGNORE_INDEX
    column = target.b if target.opcode == OP_LOAD else IGNORE_INDEX
    if target.opcode == OP_LOAD:
        register_a = target.c
    elif target.opcode in (OP_INV, OP_NEG):
        register_a = target.a
    elif target.opcode == OP_SCALE:
        register_a = target.b
    elif target.opcode == OP_AXPY:
        register_a = target.c
    else:
        register_a = IGNORE_INDEX
    register_b = target.b if target.opcode in _REGISTER_B_OPCODES else IGNORE_INDEX
    return opcode, row_a, row_b, column, register_a, register_b


def tensorize_states(
    states: Sequence[FlattenedExpertState],
    config: ReactivePolicyConfig,
) -> TensorizedStateDataset:
    """Create the complete resident training tensor set."""

    inputs = tensorize_observations(states, config)
    targets = torch.tensor(
        [_target_operands(state.target_instruction) for state in states],
        dtype=torch.long,
    )
    tensors = {
        **inputs,
        "target_opcode": targets[:, 0],
        "target_row_a": targets[:, 1],
        "target_row_b": targets[:, 2],
        "target_column": targets[:, 3],
        "target_register_a": targets[:, 4],
        "target_register_b": targets[:, 5],
    }
    return TensorizedStateDataset(
        tensors=tensors,
        examples=len(states),
        manifest_sha256=_tensor_manifest(tensors),
        source_state_manifest_sha256=_state_manifest(states),
    )


def _masked_mean(
    values: Tensor,
    mask: Tensor,
    *,
    dimension: int,
) -> Tensor:
    weights = mask.to(values.dtype)
    numerator = (values * weights.unsqueeze(-1)).sum(dim=dimension)
    denominator = weights.sum(dim=dimension).clamp_min(1.0).unsqueeze(-1)
    return numerator / denominator


def _typed_previous_markers(
    *,
    previous_opcode: Tensor,
    previous_a: Tensor,
    previous_b: Tensor,
    previous_c: Tensor,
    row_count: int,
    column_count: int,
    register_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    device = previous_opcode.device
    batch = previous_opcode.shape[0]
    row_indices = torch.arange(device=device, end=row_count)[None, :]
    column_indices = torch.arange(device=device, end=column_count)[None, :]
    register_indices = torch.arange(device=device, end=register_count)[None, :]

    def is_opcode(*opcodes: str) -> Tensor:
        result = torch.zeros(batch, dtype=torch.bool, device=device)
        for opcode in opcodes:
            result |= previous_opcode == _OPCODE_TO_INDEX[opcode]
        return result

    row_a_active = is_opcode(OP_LOAD, OP_SCALE, OP_AXPY, OP_SWAP)
    row_b_active = is_opcode(OP_AXPY, OP_SWAP)
    row_markers = torch.stack(
        (
            (row_indices == previous_a[:, None]) & row_a_active[:, None],
            (row_indices == previous_b[:, None]) & row_b_active[:, None],
        ),
        dim=-1,
    ).to(torch.float32)

    column_active = is_opcode(OP_LOAD)
    column_markers = (
        (column_indices == previous_b[:, None]) & column_active[:, None]
    ).to(torch.float32)[..., None]

    primary = torch.zeros_like(previous_a)
    primary = torch.where(
        is_opcode(OP_LOAD, OP_AXPY),
        previous_c,
        primary,
    )
    primary = torch.where(
        is_opcode(OP_INV, OP_NEG),
        previous_a,
        primary,
    )
    primary = torch.where(is_opcode(OP_SCALE), previous_b, primary)
    primary_active = is_opcode(OP_LOAD, OP_INV, OP_NEG, OP_SCALE, OP_AXPY)
    secondary_active = is_opcode(OP_INV, OP_NEG)
    register_markers = torch.stack(
        (
            (register_indices == primary[:, None]) & primary_active[:, None],
            (register_indices == previous_b[:, None]) & secondary_active[:, None],
        ),
        dim=-1,
    ).to(torch.float32)
    return row_markers, column_markers, register_markers


class EquivariantMixBlock(nn.Module):
    """Shared cell update from cell, row, column, and global summaries."""

    def __init__(
        self,
        width: int,
        feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.cell_norm = nn.LayerNorm(width)
        self.cell_projection = nn.Linear(width, width, bias=False)
        self.row_projection = nn.Linear(width, width, bias=False)
        self.column_projection = nn.Linear(width, width, bias=False)
        self.global_projection = nn.Linear(width, width, bias=False)
        self.update = nn.Sequential(
            nn.GELU(),
            nn.Linear(width, feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward, width),
        )

    def forward(self, cells: Tensor, visible: Tensor) -> Tensor:
        normalized = self.cell_norm(cells)
        row_summary = _masked_mean(
            normalized,
            visible,
            dimension=2,
        )
        column_summary = _masked_mean(
            normalized,
            visible,
            dimension=1,
        )
        global_summary = _masked_mean(
            row_summary,
            visible.any(dim=2),
            dimension=1,
        )
        message = (
            self.cell_projection(normalized)
            + self.row_projection(row_summary)[:, :, None, :]
            + self.column_projection(column_summary)[:, None, :, :]
            + self.global_projection(global_summary)[:, None, None, :]
        )
        updated = cells + self.update(message)
        return updated * visible.unsqueeze(-1).to(updated.dtype)


class GeometryEquivariantReactivePolicy(nn.Module):
    """Feed-forward shared-pointer policy over a padded matrix batch."""

    def __init__(
        self,
        config: ReactivePolicyConfig = ReactivePolicyConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        width = config.width
        self.coefficient_embedding = nn.Embedding(FIELD_MODULUS, width)
        self.previous_opcode_embedding = nn.Embedding(
            len(OPCODES) + 1,
            width,
        )
        self.register_identity_embedding = nn.Embedding(
            config.register_count,
            width,
        )
        self.row_previous_role_projection = nn.Linear(2, width, bias=False)
        self.column_previous_role_projection = nn.Linear(
            1,
            width,
            bias=False,
        )
        self.register_previous_role_projection = nn.Linear(
            2,
            width,
            bias=False,
        )
        self.cell_input_norm = nn.LayerNorm(width)
        self.blocks = nn.ModuleList(
            EquivariantMixBlock(
                width,
                config.feedforward,
                config.dropout,
            )
            for _ in range(config.blocks)
        )
        self.geometry_projection = nn.Linear(4, width, bias=False)
        self.control_token = nn.Parameter(torch.empty(width))
        self.control_update = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, config.feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward, width),
            nn.LayerNorm(width),
        )
        self.opcode_head = nn.Linear(width, len(OPCODES))
        self.row_key = nn.Linear(width, width, bias=False)
        self.column_key = nn.Linear(width, width, bias=False)
        self.register_key = nn.Linear(width, width, bias=False)
        self.row_a_query = nn.Linear(width, width, bias=False)
        self.row_b_query = nn.Linear(width, width, bias=False)
        self.column_query = nn.Linear(width, width, bias=False)
        self.register_a_query = nn.Linear(width, width, bias=False)
        self.register_b_query = nn.Linear(width, width, bias=False)
        self.output_norm = nn.LayerNorm(width)
        nn.init.normal_(self.control_token, mean=0.0, std=0.02)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _validate_inputs(
        *,
        rows: Tensor,
        registers: Tensor,
        row_mask: Tensor,
        column_mask: Tensor,
        previous_opcode: Tensor,
        previous_a: Tensor,
        previous_b: Tensor,
        previous_c: Tensor,
        config: ReactivePolicyConfig,
    ) -> tuple[int, int, int]:
        if rows.ndim != 3:
            raise VectorizedReactivePilotError(
                "rows must have shape [batch, rows, columns]"
            )
        batch, row_count, column_count = rows.shape
        expected = {
            "registers": (batch, config.register_count),
            "row_mask": (batch, row_count),
            "column_mask": (batch, column_count),
            "previous_opcode": (batch,),
            "previous_a": (batch,),
            "previous_b": (batch,),
            "previous_c": (batch,),
        }
        supplied = {
            "registers": registers,
            "row_mask": row_mask,
            "column_mask": column_mask,
            "previous_opcode": previous_opcode,
            "previous_a": previous_a,
            "previous_b": previous_b,
            "previous_c": previous_c,
        }
        for name, shape in expected.items():
            if tuple(supplied[name].shape) != shape:
                raise VectorizedReactivePilotError(
                    f"{name} has shape {tuple(supplied[name].shape)}, expected {shape}"
                )
        if row_count > config.maximum_rows:
            raise VectorizedReactivePilotError("row tensor leaves configured geometry")
        if column_count > config.maximum_columns:
            raise VectorizedReactivePilotError(
                "column tensor leaves configured geometry"
            )
        if row_mask.dtype != torch.bool or column_mask.dtype != torch.bool:
            raise VectorizedReactivePilotError("geometry masks must be Boolean")
        if not torch.all(row_mask.any(dim=1)):
            raise VectorizedReactivePilotError("every state must expose a row")
        if not torch.all(column_mask.any(dim=1)):
            raise VectorizedReactivePilotError("every state must expose a column")
        if torch.any(rows < 0) or torch.any(rows >= FIELD_MODULUS):
            raise VectorizedReactivePilotError("matrix coefficient leaves F_257")
        if torch.any(registers < 0) or torch.any(registers >= FIELD_MODULUS):
            raise VectorizedReactivePilotError("register coefficient leaves F_257")
        if torch.any(previous_opcode < 0) or torch.any(
            previous_opcode > PREVIOUS_START
        ):
            raise VectorizedReactivePilotError("previous opcode leaves the typed range")
        return batch, row_count, column_count

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
    ) -> PolicyLogits:
        config = self.config
        batch, row_count, column_count = self._validate_inputs(
            rows=rows,
            registers=registers,
            row_mask=row_mask,
            column_mask=column_mask,
            previous_opcode=previous_opcode,
            previous_a=previous_a,
            previous_b=previous_b,
            previous_c=previous_c,
            config=config,
        )
        visible = row_mask[:, :, None] & column_mask[:, None, :]
        register_mask = torch.ones(
            batch,
            config.register_count,
            dtype=torch.bool,
            device=rows.device,
        )
        row_markers, column_markers, register_markers = _typed_previous_markers(
            previous_opcode=previous_opcode,
            previous_a=previous_a,
            previous_b=previous_b,
            previous_c=previous_c,
            row_count=row_count,
            column_count=column_count,
            register_count=config.register_count,
        )
        dtype = self.control_token.dtype
        row_markers = row_markers.to(dtype)
        column_markers = column_markers.to(dtype)
        register_markers = register_markers.to(dtype)

        cells = (
            self.coefficient_embedding(rows.long())
            + self.row_previous_role_projection(row_markers)[:, :, None, :]
            + self.column_previous_role_projection(column_markers)[:, None, :, :]
        )
        cells = self.cell_input_norm(cells)
        cells = cells * visible.unsqueeze(-1).to(cells.dtype)
        for block in self.blocks:
            cells = block(cells, visible)

        row_tokens = _masked_mean(cells, visible, dimension=2)
        row_tokens = row_tokens + self.row_previous_role_projection(row_markers)
        column_tokens = _masked_mean(cells, visible, dimension=1)
        column_tokens = column_tokens + self.column_previous_role_projection(
            column_markers
        )
        register_indices = torch.arange(
            config.register_count,
            device=rows.device,
        )
        register_tokens = (
            self.coefficient_embedding(registers.long())
            + self.register_identity_embedding(register_indices)[None, :, :]
            + self.register_previous_role_projection(register_markers)
        )
        matrix_global = _masked_mean(
            row_tokens,
            row_mask,
            dimension=1,
        )
        row_global = _masked_mean(row_tokens, row_mask, dimension=1)
        column_global = _masked_mean(
            column_tokens,
            column_mask,
            dimension=1,
        )
        register_global = register_tokens.mean(dim=1)
        row_lengths = row_mask.sum(dim=1).to(torch.float32)
        column_lengths = column_mask.sum(dim=1).to(torch.float32)
        geometry = torch.stack(
            (
                torch.log1p(row_lengths),
                torch.log1p(column_lengths),
                row_lengths / column_lengths.clamp_min(1.0),
                column_lengths / row_lengths.clamp_min(1.0),
            ),
            dim=-1,
        ).to(dtype)
        control = (
            self.control_token[None, :]
            + self.previous_opcode_embedding(previous_opcode.long())
            + matrix_global
            + row_global
            + column_global
            + register_global
            + self.geometry_projection(geometry)
        )
        control = self.output_norm(control + self.control_update(control))
        scale = math.sqrt(config.width)

        def pointer(
            tokens: Tensor,
            key: nn.Linear,
            query: nn.Linear,
            mask: Tensor,
        ) -> Tensor:
            logits = (
                torch.einsum(
                    "bnd,bd->bn",
                    key(tokens),
                    query(control),
                )
                / scale
            )
            return logits.masked_fill(~mask, -torch.inf)

        return PolicyLogits(
            opcode=self.opcode_head(control),
            row_a=pointer(
                row_tokens,
                self.row_key,
                self.row_a_query,
                row_mask,
            ),
            row_b=pointer(
                row_tokens,
                self.row_key,
                self.row_b_query,
                row_mask,
            ),
            column=pointer(
                column_tokens,
                self.column_key,
                self.column_query,
                column_mask,
            ),
            register_a=pointer(
                register_tokens,
                self.register_key,
                self.register_a_query,
                register_mask,
            ),
            register_b=pointer(
                register_tokens,
                self.register_key,
                self.register_b_query,
                register_mask,
            ),
        )


def _select_inputs(
    dataset: TensorizedStateDataset,
    indices: Tensor,
) -> dict[str, Tensor]:
    return {
        name: dataset.tensors[name].index_select(0, indices)
        for name in _INPUT_TENSOR_NAMES
    }


def _select_targets(
    dataset: TensorizedStateDataset,
    indices: Tensor,
) -> dict[str, Tensor]:
    return {
        name: dataset.tensors[name].index_select(0, indices)
        for name in _TARGET_TENSOR_NAMES
    }


def _masked_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor | None:
    active = targets != IGNORE_INDEX
    if not torch.any(active):
        return None
    return F.cross_entropy(logits[active], targets[active])


def vectorized_instruction_loss(
    logits: PolicyLogits,
    targets: Mapping[str, Tensor],
) -> Tensor:
    losses = [
        F.cross_entropy(logits.opcode, targets["target_opcode"]),
    ]
    heads = {
        "target_row_a": logits.row_a,
        "target_row_b": logits.row_b,
        "target_column": logits.column,
        "target_register_a": logits.register_a,
        "target_register_b": logits.register_b,
    }
    for name, head in heads.items():
        loss = _masked_cross_entropy(head, targets[name])
        if loss is not None:
            losses.append(loss)
    return torch.stack(losses).mean()


def _autocast_context(
    device: torch.device,
    amp_bfloat16: bool,
) -> object:
    if device.type == "cuda" and amp_bfloat16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def train_vectorized_policy(
    model: GeometryEquivariantReactivePolicy,
    dataset: TensorizedStateDataset,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    amp_bfloat16: bool,
    compile_model: bool,
    shuffle_seed: int,
) -> TrainingMetrics:
    """Fit entirely from resident flattened tensors."""

    if epochs < 1 or batch_size < 1:
        raise VectorizedReactivePilotError("epochs and batch size must be positive")
    if learning_rate <= 0.0 or weight_decay < 0.0:
        raise VectorizedReactivePilotError(
            "optimizer hyperparameters leave the valid range"
        )
    resident = dataset.to(device)
    model.train()
    candidate: nn.Module = model
    if compile_model:
        compiler = getattr(torch, "compile", None)
        if compiler is None:
            raise VectorizedReactivePilotError("torch.compile is unavailable")
        candidate = compiler(model, mode="reduce-overhead")
    optimizer_kwargs: dict[str, object] = {
        "lr": learning_rate,
        "weight_decay": weight_decay,
    }
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(shuffle_seed)
    losses: list[float] = []
    updates = 0
    for _ in range(epochs):
        order = torch.randperm(
            dataset.examples,
            generator=generator,
            device="cpu",
        ).to(device)
        for offset in range(0, dataset.examples, batch_size):
            indices = order[offset : offset + batch_size]
            inputs = _select_inputs(resident, indices)
            targets = _select_targets(resident, indices)
            with _autocast_context(device, amp_bfloat16):
                logits = candidate(**inputs)
                loss = vectorized_instruction_loss(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("vectorized training loss became nonfinite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            updates += 1
    return TrainingMetrics(
        optimizer_updates=updates,
        mean_training_loss=sum(losses) / len(losses),
        final_training_loss=losses[-1],
    )


def _component_counts(
    logits: PolicyLogits,
    targets: Mapping[str, Tensor],
) -> tuple[dict[str, tuple[int, int]], Tensor]:
    predictions = {
        "opcode": logits.opcode.argmax(dim=-1),
        "row_a": logits.row_a.argmax(dim=-1),
        "row_b": logits.row_b.argmax(dim=-1),
        "column": logits.column.argmax(dim=-1),
        "register_a": logits.register_a.argmax(dim=-1),
        "register_b": logits.register_b.argmax(dim=-1),
    }
    target_names = {
        "opcode": "target_opcode",
        "row_a": "target_row_a",
        "row_b": "target_row_b",
        "column": "target_column",
        "register_a": "target_register_a",
        "register_b": "target_register_b",
    }
    full = torch.ones_like(targets["target_opcode"], dtype=torch.bool)
    counts: dict[str, tuple[int, int]] = {}
    for name, target_name in target_names.items():
        target = targets[target_name]
        active = (
            torch.ones_like(target, dtype=torch.bool)
            if name == "opcode"
            else target != IGNORE_INDEX
        )
        correct = predictions[name] == target
        full &= ~active | correct
        counts[name] = (
            int((correct & active).sum().item()),
            int(active.sum().item()),
        )
    return counts, full


@torch.no_grad()
def teacher_forced_metrics(
    model: GeometryEquivariantReactivePolicy,
    dataset: TensorizedStateDataset,
    *,
    batch_size: int,
    device: torch.device,
    amp_bfloat16: bool,
) -> TeacherForcedMetrics:
    resident = dataset.to(device)
    model.eval()
    totals = {
        name: [0, 0]
        for name in (
            "opcode",
            "row_a",
            "row_b",
            "column",
            "register_a",
            "register_b",
        )
    }
    full_correct = 0
    for offset in range(0, dataset.examples, batch_size):
        indices = torch.arange(
            offset,
            min(offset + batch_size, dataset.examples),
            device=device,
        )
        with _autocast_context(device, amp_bfloat16):
            logits = model(**_select_inputs(resident, indices))
        counts, full = _component_counts(
            logits,
            _select_targets(resident, indices),
        )
        full_correct += int(full.sum().item())
        for name, (correct, total) in counts.items():
            totals[name][0] += correct
            totals[name][1] += total
    components = tuple(
        NamedAccuracy(
            name=name,
            correct=correct,
            total=total,
            accuracy=correct / total if total else 0.0,
        )
        for name, (correct, total) in sorted(totals.items())
    )
    return TeacherForcedMetrics(
        full_instruction_correct=full_correct,
        full_instruction_total=dataset.examples,
        full_instruction_accuracy=full_correct / dataset.examples,
        components=components,
    )


def _harden_batch(logits: PolicyLogits) -> tuple[AlgebraInstruction, ...]:
    winners = {
        name: tensor.argmax(dim=-1).detach().cpu().tolist()
        for name, tensor in logits.as_mapping().items()
    }
    result: list[AlgebraInstruction] = []
    for index, opcode_index in enumerate(winners["opcode"]):
        opcode = OPCODES[opcode_index]
        if opcode == OP_LOAD:
            instruction = AlgebraInstruction(
                opcode,
                winners["row_a"][index],
                winners["column"][index],
                winners["register_a"][index],
            )
        elif opcode in (OP_INV, OP_NEG):
            instruction = AlgebraInstruction(
                opcode,
                winners["register_a"][index],
                winners["register_b"][index],
            )
        elif opcode == OP_SCALE:
            instruction = AlgebraInstruction(
                opcode,
                winners["row_a"][index],
                winners["register_a"][index],
            )
        elif opcode == OP_AXPY:
            instruction = AlgebraInstruction(
                opcode,
                winners["row_a"][index],
                winners["row_b"][index],
                winners["register_a"][index],
            )
        elif opcode == OP_SWAP:
            instruction = AlgebraInstruction(
                opcode,
                winners["row_a"][index],
                winners["row_b"][index],
            )
        elif opcode == OP_HALT:
            instruction = AlgebraInstruction(opcode)
        else:
            raise RuntimeError("unreachable opcode hardening branch")
        result.append(instruction)
    return tuple(result)


@torch.no_grad()
def predict_observations(
    model: GeometryEquivariantReactivePolicy,
    observations: Sequence[ControllerObservation],
    *,
    device: torch.device,
    amp_bfloat16: bool,
) -> tuple[AlgebraInstruction, ...]:
    inputs = {
        name: tensor.to(device, non_blocking=True)
        for name, tensor in tensorize_observations(
            observations,
            model.config,
        ).items()
    }
    model.eval()
    with _autocast_context(device, amp_bfloat16):
        logits = model(**inputs)
    return _harden_batch(logits)


@torch.no_grad()
def autonomous_matrix_only_evaluate(
    model: GeometryEquivariantReactivePolicy,
    cases: Sequence[MatrixOnlyCase],
    *,
    maximum_rollout_steps: int,
    device: torch.device,
    amp_bfloat16: bool,
) -> RolloutMetrics:
    """Run batched policy calls with no preparation-oracle reference."""

    if maximum_rollout_steps < 1:
        raise VectorizedReactivePilotError("maximum rollout steps must be positive")
    programs: list[list[AlgebraInstruction]] = [[] for _ in cases]
    previous: list[AlgebraInstruction | None] = [None for _ in cases]
    active = list(range(len(cases)))
    certified = 0
    invalid = 0
    model_batches = 0
    model_decisions = 0
    for _ in range(maximum_rollout_steps):
        if not active:
            break
        observations: list[ControllerObservation] = []
        surviving: list[int] = []
        for case_index in active:
            try:
                snapshot = execute_program(
                    cases[case_index].matrix,
                    programs[case_index],
                    register_count=model.config.register_count,
                )
            except AlgebraMachineError:
                invalid += 1
                continue
            observations.append(
                ControllerObservation(
                    rows=snapshot.rows,
                    registers=snapshot.registers,
                    previous_instruction=previous[case_index],
                )
            )
            surviving.append(case_index)
        if not surviving:
            active = []
            break
        instructions = predict_observations(
            model,
            observations,
            device=device,
            amp_bfloat16=amp_bfloat16,
        )
        model_batches += 1
        model_decisions += len(instructions)
        next_active: list[int] = []
        for case_index, instruction in zip(
            surviving,
            instructions,
            strict=True,
        ):
            programs[case_index].append(instruction)
            previous[case_index] = instruction
            try:
                state = execute_program(
                    cases[case_index].matrix,
                    programs[case_index],
                    register_count=model.config.register_count,
                )
            except AlgebraMachineError:
                invalid += 1
                continue
            if instruction.opcode == OP_HALT:
                try:
                    verify_reduction_program(cases[case_index].matrix, state)
                except AlgebraMachineError:
                    invalid += 1
                else:
                    certified += 1
            else:
                next_active.append(case_index)
        active = next_active
    overlong = len(active)
    return RolloutMetrics(
        certified=certified,
        invalid=invalid,
        overlong=overlong,
        oracle_calls=0,
        model_batches=model_batches,
        model_decisions=model_decisions,
    )


def _generate_disjoint_examples(
    *,
    seed: int,
    count: int,
    maximum_rows: int,
    maximum_columns: int,
    register_count: int,
    minimum_rows: int,
    minimum_columns: int,
    excluded: set[tuple[tuple[int, ...], ...]],
) -> tuple[TraceExample, ...]:
    result: list[TraceExample] = []
    for attempt in range(100):
        if len(result) == count:
            break
        generated = generate_examples(
            seed=seed + attempt * 104_729,
            count=max(8, count - len(result)),
            maximum_rows=maximum_rows,
            maximum_columns=maximum_columns,
            register_count=register_count,
            minimum_rows=minimum_rows,
            minimum_columns=minimum_columns,
        )
        for example in generated:
            if example.matrix in excluded:
                continue
            excluded.add(example.matrix)
            result.append(example)
            if len(result) == count:
                break
    if len(result) != count:
        raise RuntimeError("could not generate an exact disjoint split")
    return tuple(result)


def delete_expert_artifacts(
    examples: Iterable[TraceExample],
) -> tuple[MatrixOnlyCase, ...]:
    return tuple(MatrixOnlyCase(matrix=example.matrix) for example in examples)


def _material_gate(
    rollout: RolloutMetrics,
    *,
    minimum_cases: int,
    minimum_rate: float,
) -> bool:
    return (
        rollout.oracle_calls == 0
        and rollout.total >= minimum_cases
        and rollout.certification_rate >= minimum_rate
    )


def _validate_args(args: argparse.Namespace) -> None:
    if not 2 <= args.fit_maximum_rows < args.evaluation_minimum_rows:
        raise VectorizedReactivePilotError(
            "evaluation rows must be strictly larger than fit rows"
        )
    if not 2 <= args.fit_maximum_columns < args.evaluation_minimum_columns:
        raise VectorizedReactivePilotError(
            "evaluation columns must be strictly larger than fit columns"
        )
    if args.fit_maximum_columns < args.fit_maximum_rows:
        raise VectorizedReactivePilotError(
            "fit columns must admit every fit row geometry"
        )
    if not args.evaluation_minimum_rows <= args.maximum_rows:
        raise VectorizedReactivePilotError("evaluation rows leave controller geometry")
    if not args.evaluation_minimum_columns <= args.maximum_columns:
        raise VectorizedReactivePilotError(
            "evaluation columns leave controller geometry"
        )
    if args.maximum_columns < args.maximum_rows:
        raise VectorizedReactivePilotError(
            "controller columns must admit every row geometry"
        )
    count_names = (
        "train_examples",
        "audit_examples",
        "evaluation_examples",
        "epochs",
        "batch_size",
        "maximum_rollout_steps",
        "material_minimum_evaluation_cases",
    )
    for name in count_names:
        if getattr(args, name) < 1:
            raise VectorizedReactivePilotError(f"{name} must be positive")
    if not 0.0 < args.material_minimum_certification_rate <= 1.0:
        raise VectorizedReactivePilotError(
            "material certification rate must be in (0, 1]"
        )


def run_pilot(args: argparse.Namespace) -> VectorizedReactiveReport:
    _validate_args(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    config = ReactivePolicyConfig(
        maximum_rows=args.maximum_rows,
        maximum_columns=args.maximum_columns,
        register_count=args.registers,
        width=args.width,
        blocks=args.blocks,
        feedforward=args.feedforward,
        dropout=args.dropout,
    )
    excluded: set[tuple[tuple[int, ...], ...]] = set()
    train_examples = _generate_disjoint_examples(
        seed=args.seed,
        count=args.train_examples,
        maximum_rows=args.fit_maximum_rows,
        maximum_columns=args.fit_maximum_columns,
        register_count=config.register_count,
        minimum_rows=2,
        minimum_columns=2,
        excluded=excluded,
    )
    audit_examples = _generate_disjoint_examples(
        seed=args.seed + 1,
        count=args.audit_examples,
        maximum_rows=args.fit_maximum_rows,
        maximum_columns=args.fit_maximum_columns,
        register_count=config.register_count,
        minimum_rows=2,
        minimum_columns=2,
        excluded=excluded,
    )
    evaluation_examples = _generate_disjoint_examples(
        seed=args.seed + 10_000,
        count=args.evaluation_examples,
        maximum_rows=config.maximum_rows,
        maximum_columns=config.maximum_columns,
        register_count=config.register_count,
        minimum_rows=args.evaluation_minimum_rows,
        minimum_columns=args.evaluation_minimum_columns,
        excluded=excluded,
    )
    train_states = flatten_expert_states(train_examples)
    audit_states = flatten_expert_states(audit_examples)
    train_dataset = tensorize_states(train_states, config)
    audit_dataset = tensorize_states(audit_states, config)
    evaluation_cases = delete_expert_artifacts(evaluation_examples)
    strict_geometry_disjoint = (
        args.fit_maximum_rows < args.evaluation_minimum_rows
        and args.fit_maximum_columns < args.evaluation_minimum_columns
        and len(excluded)
        == (len(train_examples) + len(audit_examples) + len(evaluation_examples))
    )
    if not strict_geometry_disjoint:
        raise RuntimeError("strict geometry split invariant failed")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise VectorizedReactivePilotError("CUDA device is unavailable")
    model = GeometryEquivariantReactivePolicy(config).to(device)
    training = train_vectorized_policy(
        model,
        train_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        amp_bfloat16=args.amp_bfloat16,
        compile_model=args.compile,
        shuffle_seed=args.seed + 20_000,
    )
    teacher = teacher_forced_metrics(
        model,
        audit_dataset,
        batch_size=args.batch_size,
        device=device,
        amp_bfloat16=args.amp_bfloat16,
    )
    rollout = autonomous_matrix_only_evaluate(
        model,
        evaluation_cases,
        maximum_rollout_steps=args.maximum_rollout_steps,
        device=device,
        amp_bfloat16=args.amp_bfloat16,
    )
    if rollout.oracle_calls != 0:
        raise RuntimeError("autonomous rollout crossed the oracle boundary")
    material_pass = _material_gate(
        rollout,
        minimum_cases=args.material_minimum_evaluation_cases,
        minimum_rate=args.material_minimum_certification_rate,
    )
    status = STATUS_MATERIAL_PASS if material_pass else STATUS_NOT_REASONING
    model_hash = model_state_sha256(model)
    if args.model_output is not None:
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": MODEL_SCHEMA,
                "config": asdict(config),
                "model_state_sha256": model_hash,
                "state_dict": {
                    name: tensor.detach().cpu()
                    for name, tensor in model.state_dict().items()
                },
            },
            args.model_output,
        )
    return VectorizedReactiveReport(
        schema=PILOT_SCHEMA,
        status=status,
        model_schema=MODEL_SCHEMA,
        candidate_runtime=CANDIDATE_RUNTIME,
        preparation_oracle_boundary=ORACLE_BOUNDARY,
        autonomous_input_fields=("matrix",),
        seed=args.seed,
        device=str(device),
        amp_bfloat16=bool(args.amp_bfloat16 and device.type == "cuda"),
        torch_compile=bool(args.compile),
        flattened_state_dataset=True,
        dataset_resident_on_device=True,
        step_signal_exposed=False,
        recurrent_state=False,
        learned_absolute_row_table=False,
        learned_absolute_column_table=False,
        row_coordinate_features=False,
        column_coordinate_features=False,
        exact_row_permutation_equivariance=True,
        exact_column_permutation_equivariance=True,
        preparation_oracle_order_sensitive=True,
        shared_content_pointer_heads=True,
        train_matrices=len(train_examples),
        audit_matrices=len(audit_examples),
        evaluation_matrices=len(evaluation_cases),
        flattened_train_states=len(train_states),
        flattened_audit_states=len(audit_states),
        fit_maximum_rows=args.fit_maximum_rows,
        fit_maximum_columns=args.fit_maximum_columns,
        evaluation_minimum_rows=args.evaluation_minimum_rows,
        evaluation_minimum_columns=args.evaluation_minimum_columns,
        evaluation_maximum_rows=config.maximum_rows,
        evaluation_maximum_columns=config.maximum_columns,
        maximum_rollout_steps=args.maximum_rollout_steps,
        controller_parameters=model.parameter_count,
        optimizer_updates=training.optimizer_updates,
        mean_training_loss=training.mean_training_loss,
        final_training_loss=training.final_training_loss,
        teacher_forced_full_instruction_correct=(teacher.full_instruction_correct),
        teacher_forced_full_instruction_total=teacher.full_instruction_total,
        teacher_forced_full_instruction_accuracy=(teacher.full_instruction_accuracy),
        teacher_forced_components=teacher.components,
        closed_loop_certified=rollout.certified,
        closed_loop_total=rollout.total,
        closed_loop_certification_rate=rollout.certification_rate,
        invalid_programs=rollout.invalid,
        overlong_programs=rollout.overlong,
        final_rollout_oracle_calls=rollout.oracle_calls,
        final_rollout_model_batches=rollout.model_batches,
        final_rollout_model_decisions=rollout.model_decisions,
        material_minimum_evaluation_cases=(args.material_minimum_evaluation_cases),
        material_minimum_certification_rate=(args.material_minimum_certification_rate),
        material_certification_gate_passed=material_pass,
        strict_geometry_disjoint=strict_geometry_disjoint,
        train_matrix_manifest_sha256=_matrix_manifest(train_examples),
        audit_matrix_manifest_sha256=_matrix_manifest(audit_examples),
        evaluation_matrix_manifest_sha256=_matrix_manifest(evaluation_cases),
        train_state_manifest_sha256=_state_manifest(train_states),
        audit_state_manifest_sha256=_state_manifest(audit_states),
        train_tensor_manifest_sha256=train_dataset.manifest_sha256,
        audit_tensor_manifest_sha256=audit_dataset.manifest_sha256,
        model_config_sha256=config.canonical_sha256,
        model_state_sha256=model_hash,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--train-examples", type=int, default=4096)
    parser.add_argument("--audit-examples", type=int, default=512)
    parser.add_argument("--evaluation-examples", type=int, default=256)
    parser.add_argument("--fit-maximum-rows", type=int, default=4)
    parser.add_argument("--fit-maximum-columns", type=int, default=6)
    parser.add_argument("--evaluation-minimum-rows", type=int, default=5)
    parser.add_argument("--evaluation-minimum-columns", type=int, default=7)
    parser.add_argument("--maximum-rows", type=int, default=6)
    parser.add_argument("--maximum-columns", type=int, default=8)
    parser.add_argument("--registers", type=int, default=4)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--feedforward", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--maximum-rollout-steps", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--amp-bfloat16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--material-minimum-evaluation-cases",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--material-minimum-certification-rate",
        type=float,
        default=0.8,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_pilot(args)
    payload = report.canonical_bytes()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
    print(payload.decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
