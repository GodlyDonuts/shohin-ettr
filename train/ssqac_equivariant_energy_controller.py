#!/usr/bin/env python3
"""Permutation-equivariant SSQAC controller driven by algebraic defect energy.

This module is an isolated mechanics experiment.  The candidate receives only
the current F_257 matrix.  It has no source, query, workspace, absolute
position, recurrent state, previous action, or step input.

The controller chooses among hard, invertible local row transitions:

* ``NORMALIZE(r, c)`` multiplies row ``r`` by the inverse of its nonzero
  coefficient in column ``c``.
* ``ELIMINATE(dst, src, c)`` clears column ``c`` in ``dst`` using a source
  whose coefficient in ``c`` is exactly one.
* ``HALT`` is exposed only when the explicit defect energy is zero.

Every nonterminal action compiles to the existing primitive field-row VM.  A
geometry-free bipartite message-passing network supplies a bounded residual
correction to the exact one-step energy reduction.  The residual magnitude is
strictly below one half, so it cannot reverse an integer energy advantage; it
can only select among equal-energy local moves.  This makes the architecture
novel but directly falsifiable: if the chosen energy has bad plateaus, the
learned residual must resolve them without hidden state or an external oracle.

No result emitted by this file is a reasoning claim.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from episode_functor_algebra_machine import (
    FIELD_MODULUS,
    OP_AXPY,
    OP_HALT,
    OP_INV,
    OP_LOAD,
    OP_NEG,
    OP_SCALE,
    AlgebraInstruction,
    AlgebraMachineError,
    execute_program,
    verify_reduction_program,
)


ARCHITECTURE_SCHEMA = "ssqac_equivariant_energy_controller_v1"
EXPERIMENT_SCHEMA = "ssqac_equivariant_energy_experiment_v2"
STATUS = "isolated_energy_descent_mechanics_not_reasoning"

ACTION_NORMALIZE = "NORMALIZE"
ACTION_ELIMINATE = "ELIMINATE"
ACTION_HALT = "HALT"
ACTION_TYPES = (ACTION_NORMALIZE, ACTION_ELIMINATE, ACTION_HALT)

MAX_EXACT_ENERGY_COLUMNS = 12


class EnergyControllerError(ValueError):
    """The energy controller contract failed closed."""


def _plain_positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EnergyControllerError(f"{label} must be a positive integer")
    return value


def canonical_matrix(
    rows: Iterable[Iterable[int]],
) -> tuple[tuple[int, ...], ...]:
    """Freeze a bounded, nonempty rectangular matrix over F_257."""

    matrix = tuple(
        tuple(int(value) % FIELD_MODULUS for value in row)
        for row in rows
    )
    if not matrix or not matrix[0]:
        raise EnergyControllerError("matrix must be nonempty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise EnergyControllerError("matrix rows have inconsistent widths")
    if width > MAX_EXACT_ENERGY_COLUMNS:
        raise EnergyControllerError(
            "exact defect energy supports at most "
            f"{MAX_EXACT_ENERGY_COLUMNS} columns"
        )
    return matrix


def matrix_sha256(rows: Iterable[Iterable[int]]) -> str:
    matrix = canonical_matrix(rows)
    return sha256(
        json.dumps(matrix, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def field_rank(rows: Iterable[Iterable[int]]) -> int:
    """Compute exact rank over F_257 without choosing a deployed schedule."""

    matrix = [list(row) for row in canonical_matrix(rows)]
    row_count = len(matrix)
    column_count = len(matrix[0])
    pivot_row = 0
    for column in range(column_count):
        source = next(
            (
                row
                for row in range(pivot_row, row_count)
                if matrix[row][column]
            ),
            None,
        )
        if source is None:
            continue
        matrix[pivot_row], matrix[source] = (
            matrix[source],
            matrix[pivot_row],
        )
        inverse = pow(matrix[pivot_row][column], -1, FIELD_MODULUS)
        matrix[pivot_row] = [
            inverse * value % FIELD_MODULUS
            for value in matrix[pivot_row]
        ]
        for row in range(row_count):
            if row == pivot_row:
                continue
            factor = matrix[row][column]
            if factor:
                matrix[row] = [
                    (left - factor * right) % FIELD_MODULUS
                    for left, right in zip(
                        matrix[row],
                        matrix[pivot_row],
                        strict=True,
                    )
                ]
        pivot_row += 1
        if pivot_row == row_count:
            break
    return pivot_row


@lru_cache(maxsize=262_144)
def _defect_energy_cached(
    matrix: tuple[tuple[int, ...], ...],
) -> int:
    """Minimum matching defect for an unordered reduced row basis.

    For rank ``k``, select ``k`` distinct rows and columns.  A selected
    row/column pair pays one high-weight defect if its pivot is not one and
    one high-weight defect for every other nonzero in that pivot column.
    Every nonzero in an unselected row pays one low-weight defect.

    The minimum is an exact cardinality-constrained bipartite matching,
    computed by row dynamic programming.  The high weight exceeds the total
    number of cells, so local pivot repair dominates incidental support
    changes.  Energy zero is equivalent to an unordered reduced basis:
    exactly ``rank`` nonzero rows, each owning at least one isolated unit
    pivot column.
    """

    rank = field_rank(matrix)
    if rank == 0:
        return 0
    row_count = len(matrix)
    column_count = len(matrix[0])
    row_nonzero = tuple(
        sum(value != 0 for value in row) for row in matrix
    )
    column_nonzero = tuple(
        sum(matrix[row][column] != 0 for row in range(row_count))
        for column in range(column_count)
    )
    total_nonzero = sum(row_nonzero)
    structural_weight = row_count * column_count + 1

    # State: (selected rows, selected-column mask) -> adjustment to the
    # baseline cost in which every row is unselected.
    dynamic: dict[tuple[int, int], int] = {(0, 0): 0}
    for row_index, row in enumerate(matrix):
        updated = dict(dynamic)
        for (selected, column_mask), cost in dynamic.items():
            if selected >= rank:
                continue
            for column, value in enumerate(row):
                if value == 0 or column_mask & (1 << column):
                    continue
                structural_defects = (
                    int(value != 1) + column_nonzero[column] - 1
                )
                pair_adjustment = (
                    structural_weight * structural_defects
                    - row_nonzero[row_index]
                )
                key = (selected + 1, column_mask | (1 << column))
                candidate = cost + pair_adjustment
                if key not in updated or candidate < updated[key]:
                    updated[key] = candidate
        dynamic = updated
    feasible = [
        cost
        for (selected, _), cost in dynamic.items()
        if selected == rank
    ]
    if not feasible:
        raise EnergyControllerError(
            "rank has no support matching; matrix invariant is inconsistent"
        )
    energy = total_nonzero + min(feasible)
    if energy < 0:
        raise EnergyControllerError("defect energy became negative")
    return energy


def defect_energy(rows: Iterable[Iterable[int]]) -> int:
    """Return the explicit permutation-invariant algebraic defect energy."""

    return _defect_energy_cached(canonical_matrix(rows))


@dataclass(frozen=True, slots=True, order=True)
class LocalEnergyAction:
    """One hard local transition with geometry-relative operands."""

    kind: str
    row_a: int = 0
    row_b: int = 0
    column: int = 0

    def __post_init__(self) -> None:
        if self.kind not in ACTION_TYPES:
            raise EnergyControllerError(f"unknown action type {self.kind!r}")
        for label, value in (
            ("row_a", self.row_a),
            ("row_b", self.row_b),
            ("column", self.column),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise EnergyControllerError(f"{label} must be an integer")

    def canonical_data(self) -> list[object]:
        return [self.kind, self.row_a, self.row_b, self.column]


@dataclass(frozen=True, slots=True)
class EnergyTransition:
    action: LocalEnergyAction
    rows: tuple[tuple[int, ...], ...]
    energy_before: int
    energy_after: int

    @property
    def explicit_reduction(self) -> int:
        return self.energy_before - self.energy_after


def enumerate_legal_actions(
    rows: Iterable[Iterable[int]],
) -> tuple[LocalEnergyAction, ...]:
    """Enumerate all hard local actions, independent of absolute geometry."""

    matrix = canonical_matrix(rows)
    if defect_energy(matrix) == 0:
        return (LocalEnergyAction(ACTION_HALT),)
    row_count = len(matrix)
    column_count = len(matrix[0])
    actions: list[LocalEnergyAction] = []
    for row in range(row_count):
        for column in range(column_count):
            value = matrix[row][column]
            if value not in (0, 1):
                actions.append(
                    LocalEnergyAction(
                        ACTION_NORMALIZE,
                        row_a=row,
                        column=column,
                    )
                )
    for source in range(row_count):
        for column in range(column_count):
            if matrix[source][column] != 1:
                continue
            for destination in range(row_count):
                if (
                    destination != source
                    and matrix[destination][column] != 0
                ):
                    actions.append(
                        LocalEnergyAction(
                            ACTION_ELIMINATE,
                            row_a=destination,
                            row_b=source,
                            column=column,
                        )
                    )
    if not actions:
        raise EnergyControllerError(
            "positive-energy state has no legal local transition"
        )
    return tuple(actions)


def apply_local_action(
    rows: Iterable[Iterable[int]],
    action: LocalEnergyAction,
) -> tuple[tuple[int, ...], ...]:
    """Apply one exact hard transition over F_257."""

    matrix = canonical_matrix(rows)
    row_count = len(matrix)
    column_count = len(matrix[0])
    if not isinstance(action, LocalEnergyAction):
        raise EnergyControllerError("action has the wrong type")
    if action.kind == ACTION_HALT:
        if defect_energy(matrix) != 0:
            raise EnergyControllerError("HALT is legal only at zero energy")
        return matrix
    if not 0 <= action.row_a < row_count:
        raise EnergyControllerError("row_a is out of range")
    if not 0 <= action.column < column_count:
        raise EnergyControllerError("column is out of range")
    mutable = [list(row) for row in matrix]
    if action.kind == ACTION_NORMALIZE:
        value = mutable[action.row_a][action.column]
        if value in (0, 1):
            raise EnergyControllerError(
                "NORMALIZE requires a nonunit nonzero coefficient"
            )
        factor = pow(value, -1, FIELD_MODULUS)
        mutable[action.row_a] = [
            factor * coefficient % FIELD_MODULUS
            for coefficient in mutable[action.row_a]
        ]
    elif action.kind == ACTION_ELIMINATE:
        if not 0 <= action.row_b < row_count:
            raise EnergyControllerError("row_b is out of range")
        if action.row_a == action.row_b:
            raise EnergyControllerError(
                "ELIMINATE source and destination must differ"
            )
        if mutable[action.row_b][action.column] != 1:
            raise EnergyControllerError(
                "ELIMINATE source coefficient must equal one"
            )
        target = mutable[action.row_a][action.column]
        if target == 0:
            raise EnergyControllerError(
                "ELIMINATE destination coefficient must be nonzero"
            )
        factor = (-target) % FIELD_MODULUS
        mutable[action.row_a] = [
            (left + factor * right) % FIELD_MODULUS
            for left, right in zip(
                mutable[action.row_a],
                mutable[action.row_b],
                strict=True,
            )
        ]
    else:
        raise EnergyControllerError("unreachable transition dispatch")
    return tuple(tuple(row) for row in mutable)


def compile_action_to_vm_primitives(
    rows: Iterable[Iterable[int]],
    action: LocalEnergyAction,
) -> tuple[AlgebraInstruction, ...]:
    """Compile one local transition to the existing primitive field-row VM."""

    matrix = canonical_matrix(rows)
    # Validate first so the emitted sequence cannot encode an illegal macro.
    apply_local_action(matrix, action)
    if action.kind == ACTION_NORMALIZE:
        return (
            AlgebraInstruction(OP_LOAD, action.row_a, action.column, 0),
            AlgebraInstruction(OP_INV, 0, 1),
            AlgebraInstruction(OP_SCALE, action.row_a, 1),
        )
    if action.kind == ACTION_ELIMINATE:
        return (
            AlgebraInstruction(OP_LOAD, action.row_a, action.column, 0),
            AlgebraInstruction(OP_NEG, 0, 2),
            AlgebraInstruction(OP_AXPY, action.row_a, action.row_b, 2),
        )
    return (AlgebraInstruction(OP_HALT),)


def compile_action_trace_to_vm_primitives(
    input_rows: Iterable[Iterable[int]],
    actions: Sequence[LocalEnergyAction],
) -> tuple[AlgebraInstruction, ...]:
    """Replay a complete macro trace into one primitive VM program.

    The compiler validates every macro against the matrix produced by its
    predecessors.  A complete trace must contain exactly one ``HALT`` and it
    must be the final macro, so strict certification cannot be performed on a
    partial endpoint.
    """

    matrix = canonical_matrix(input_rows)
    frozen_actions = tuple(actions)
    if not frozen_actions or frozen_actions[-1].kind != ACTION_HALT:
        raise EnergyControllerError(
            "complete action trace must terminate with HALT"
        )
    if any(action.kind == ACTION_HALT for action in frozen_actions[:-1]):
        raise EnergyControllerError(
            "action trace contains an instruction after HALT"
        )
    program: list[AlgebraInstruction] = []
    for action in frozen_actions:
        program.extend(compile_action_to_vm_primitives(matrix, action))
        matrix = apply_local_action(matrix, action)
    if sum(instruction.opcode == OP_HALT for instruction in program) != 1:
        raise EnergyControllerError(
            "compiled primitive trace must contain exactly one HALT"
        )
    return tuple(program)


def verify_strict_canonical_action_trace(
    input_rows: Iterable[Iterable[int]],
    actions: Sequence[LocalEnergyAction],
) -> bool:
    """Verify a macro trace with the repository's canonical RREF contract."""

    source = canonical_matrix(input_rows)
    try:
        program = compile_action_trace_to_vm_primitives(source, actions)
        final_state = execute_program(source, program)
        receipt = verify_reduction_program(source, final_state)
    except (AlgebraMachineError, EnergyControllerError):
        return False
    return receipt.passed


def evaluate_transitions(
    rows: Iterable[Iterable[int]],
) -> tuple[EnergyTransition, ...]:
    matrix = canonical_matrix(rows)
    before = defect_energy(matrix)
    return tuple(
        EnergyTransition(
            action=action,
            rows=successor,
            energy_before=before,
            energy_after=defect_energy(successor),
        )
        for action in enumerate_legal_actions(matrix)
        for successor in (apply_local_action(matrix, action),)
    )


def _identity(size: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(1 if row == column else 0 for column in range(size))
        for row in range(size)
    )


def _apply_action_to_provenance(
    provenance: tuple[tuple[int, ...], ...],
    rows_before: tuple[tuple[int, ...], ...],
    action: LocalEnergyAction,
) -> tuple[tuple[int, ...], ...]:
    if action.kind == ACTION_HALT:
        return provenance
    mutable = [list(row) for row in provenance]
    if action.kind == ACTION_NORMALIZE:
        factor = pow(
            rows_before[action.row_a][action.column],
            -1,
            FIELD_MODULUS,
        )
        mutable[action.row_a] = [
            factor * value % FIELD_MODULUS
            for value in mutable[action.row_a]
        ]
    else:
        factor = (-rows_before[action.row_a][action.column]) % FIELD_MODULUS
        mutable[action.row_a] = [
            (left + factor * right) % FIELD_MODULUS
            for left, right in zip(
                mutable[action.row_a],
                mutable[action.row_b],
                strict=True,
            )
        ]
    return tuple(tuple(row) for row in mutable)


def verify_unordered_reduced_basis(
    input_rows: Iterable[Iterable[int]],
    output_rows: Iterable[Iterable[int]],
    provenance: Iterable[Iterable[int]],
) -> bool:
    """Independently certify the invariant terminal condition."""

    source = canonical_matrix(input_rows)
    output = canonical_matrix(output_rows)
    witness = canonical_matrix(provenance)
    if len(output) != len(source) or len(output[0]) != len(source[0]):
        return False
    if len(witness) != len(source) or len(witness[0]) != len(source):
        return False
    reconstructed = tuple(
        tuple(
            sum(
                witness[row][origin] * source[origin][column]
                for origin in range(len(source))
            )
            % FIELD_MODULUS
            for column in range(len(source[0]))
        )
        for row in range(len(source))
    )
    if reconstructed != output:
        return False
    rank = field_rank(source)
    if field_rank(output) != rank:
        return False
    nonzero_rows = [
        row for row, values in enumerate(output) if any(values)
    ]
    if len(nonzero_rows) != rank:
        return False
    for row in nonzero_rows:
        owns_isolated_unit = any(
            output[row][column] == 1
            and all(
                other == row or output[other][column] == 0
                for other in range(len(output))
            )
            for column in range(len(output[0]))
        )
        if not owns_isolated_unit:
            return False
    return defect_energy(output) == 0


@dataclass(frozen=True, slots=True)
class EnergyControllerConfig:
    width: int = 64
    message_layers: int = 3
    residual_hidden: int = 128
    field_harmonics: int = 4
    residual_bound: float = 0.49

    def __post_init__(self) -> None:
        for label, value in (
            ("width", self.width),
            ("message_layers", self.message_layers),
            ("residual_hidden", self.residual_hidden),
            ("field_harmonics", self.field_harmonics),
        ):
            _plain_positive_int(value, label=label)
        if (
            not isinstance(self.residual_bound, float)
            or not 0.0 < self.residual_bound < 0.5
        ):
            raise EnergyControllerError(
                "residual_bound must be a float strictly between zero and 0.5"
            )


class _EquivariantMessageLayer(nn.Module):
    """One row/column permutation-equivariant bipartite update."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.row_mlp = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.column_mlp = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.cell_mlp = nn.Sequential(
            nn.Linear(4 * width, 2 * width),
            nn.SiLU(),
            nn.Linear(2 * width, width),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, cells: Tensor) -> Tensor:
        if cells.ndim != 3:
            raise EnergyControllerError(
                "cell state must have shape [rows, columns, width]"
            )
        row_count, column_count, width = cells.shape
        global_state = cells.mean(dim=(0, 1))
        row_state = self.row_mlp(
            torch.cat(
                (
                    cells.mean(dim=1),
                    global_state.expand(row_count, width),
                ),
                dim=-1,
            )
        )
        column_state = self.column_mlp(
            torch.cat(
                (
                    cells.mean(dim=0),
                    global_state.expand(column_count, width),
                ),
                dim=-1,
            )
        )
        update = self.cell_mlp(
            torch.cat(
                (
                    cells,
                    row_state[:, None, :].expand(
                        row_count,
                        column_count,
                        width,
                    ),
                    column_state[None, :, :].expand(
                        row_count,
                        column_count,
                        width,
                    ),
                    global_state[None, None, :].expand(
                        row_count,
                        column_count,
                        width,
                    ),
                ),
                dim=-1,
            )
        )
        return self.norm(cells + update)


@dataclass(frozen=True, slots=True)
class ScoredEnergyActions:
    actions: tuple[LocalEnergyAction, ...]
    energy_before: int
    energy_after: tuple[int, ...]
    explicit_reduction: Tensor
    learned_residual: Tensor
    total_score: Tensor


class EquivariantEnergyController(nn.Module):
    """Memoryless geometry-equivariant residual scorer."""

    def __init__(
        self,
        config: EnergyControllerConfig = EnergyControllerConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        feature_width = 5 + 4 * config.field_harmonics
        width = config.width
        self.field_encoder = nn.Sequential(
            nn.Linear(feature_width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.message_layers = nn.ModuleList(
            _EquivariantMessageLayer(width)
            for _ in range(config.message_layers)
        )
        self.action_type = nn.Embedding(len(ACTION_TYPES), width)
        self.energy_projection = nn.Sequential(
            nn.Linear(6, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(8 * width, config.residual_hidden),
            nn.SiLU(),
            nn.Linear(config.residual_hidden, config.residual_hidden),
            nn.SiLU(),
            nn.Linear(config.residual_hidden, 1),
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def parameter_count_breakdown(self) -> Mapping[str, int]:
        breakdown: dict[str, int] = {}
        for name, parameter in self.named_parameters():
            owner = name.split(".", 1)[0]
            breakdown[owner] = breakdown.get(owner, 0) + parameter.numel()
        breakdown["total"] = sum(
            value for key, value in breakdown.items() if key != "total"
        )
        return dict(sorted(breakdown.items()))

    def _field_features(self, rows: Tensor) -> Tensor:
        values = rows.to(dtype=torch.float32)
        inverse_long = torch.zeros_like(rows)
        nonzero = rows != 0
        if nonzero.any():
            inverse_long[nonzero] = torch.tensor(
                [
                    pow(int(value), -1, FIELD_MODULUS)
                    for value in rows[nonzero].detach().cpu().tolist()
                ],
                dtype=rows.dtype,
                device=rows.device,
            )
        inverse = inverse_long.to(dtype=torch.float32)
        scale = float(FIELD_MODULUS - 1)
        features = [
            (rows == 0).to(dtype=torch.float32),
            (rows == 1).to(dtype=torch.float32),
            nonzero.to(dtype=torch.float32),
            torch.minimum(values, FIELD_MODULUS - values) / (scale / 2.0),
            torch.minimum(inverse, FIELD_MODULUS - inverse) / (scale / 2.0),
        ]
        angle = 2.0 * math.pi / FIELD_MODULUS
        for harmonic in range(1, self.config.field_harmonics + 1):
            features.extend(
                (
                    torch.sin(angle * harmonic * values),
                    torch.cos(angle * harmonic * values),
                    torch.sin(angle * harmonic * inverse),
                    torch.cos(angle * harmonic * inverse),
                )
            )
        return torch.stack(features, dim=-1)

    def encode_matrix(self, rows: Tensor) -> Tensor:
        if rows.ndim != 2:
            raise EnergyControllerError(
                "rows must have shape [rows, columns]"
            )
        if rows.dtype not in (torch.int32, torch.int64):
            raise EnergyControllerError("rows must use an integer dtype")
        if rows.shape[0] < 1 or rows.shape[1] < 1:
            raise EnergyControllerError("rows must be nonempty")
        if torch.any(rows < 0) or torch.any(rows >= FIELD_MODULUS):
            raise EnergyControllerError("matrix coefficients leave F_257")
        cells = self.field_encoder(self._field_features(rows))
        for layer in self.message_layers:
            cells = layer(cells)
        return cells

    def forward(
        self,
        rows: Tensor,
        actions: Sequence[LocalEnergyAction],
        *,
        energy_before: int,
        energy_after: Sequence[int],
    ) -> Tensor:
        """Return bounded learned residuals for supplied legal actions."""

        frozen_actions = tuple(actions)
        frozen_after = tuple(energy_after)
        if not frozen_actions or len(frozen_actions) != len(frozen_after):
            raise EnergyControllerError(
                "actions and successor energies must be nonempty and aligned"
            )
        cells = self.encode_matrix(rows)
        row_count, column_count, width = cells.shape
        row_state = cells.mean(dim=1)
        column_state = cells.mean(dim=0)
        global_state = cells.mean(dim=(0, 1))
        zeros = torch.zeros(width, dtype=cells.dtype, device=cells.device)
        encodings = []
        energy_scale = float(
            max(1, row_count * column_count * (row_count * column_count + 1))
        )
        for action, after in zip(
            frozen_actions,
            frozen_after,
            strict=True,
        ):
            if action.kind == ACTION_HALT:
                row_a = row_b = column = cell_a = cell_b = zeros
            else:
                if not (
                    0 <= action.row_a < row_count
                    and 0 <= action.column < column_count
                ):
                    raise EnergyControllerError(
                        "action operands leave the matrix geometry"
                    )
                row_a = row_state[action.row_a]
                column = column_state[action.column]
                cell_a = cells[action.row_a, action.column]
                if action.kind == ACTION_ELIMINATE:
                    if not 0 <= action.row_b < row_count:
                        raise EnergyControllerError("row_b is out of range")
                    row_b = row_state[action.row_b]
                    cell_b = cells[action.row_b, action.column]
                else:
                    row_b = cell_b = zeros
            action_type = self.action_type(
                torch.tensor(
                    ACTION_TYPES.index(action.kind),
                    dtype=torch.long,
                    device=cells.device,
                )
            )
            scalars = torch.tensor(
                [
                    energy_before / energy_scale,
                    after / energy_scale,
                    (energy_before - after) / energy_scale,
                    row_count / 16.0,
                    column_count / 16.0,
                    field_rank(rows.detach().cpu().tolist())
                    / max(1.0, float(min(row_count, column_count))),
                ],
                dtype=cells.dtype,
                device=cells.device,
            )
            energy_state = self.energy_projection(scalars)
            encodings.append(
                torch.cat(
                    (
                        global_state,
                        row_a,
                        row_b,
                        column,
                        cell_a,
                        cell_b,
                        action_type,
                        energy_state,
                    )
                )
            )
        raw = self.residual_head(torch.stack(encodings)).squeeze(-1)
        return self.config.residual_bound * torch.tanh(raw)

    def score_actions(
        self,
        rows: Iterable[Iterable[int]],
        *,
        learned_residual: bool = True,
    ) -> ScoredEnergyActions:
        matrix = canonical_matrix(rows)
        transitions = evaluate_transitions(matrix)
        actions = tuple(transition.action for transition in transitions)
        before = transitions[0].energy_before
        after = tuple(transition.energy_after for transition in transitions)
        reference = next(self.parameters())
        row_tensor = torch.tensor(
            matrix,
            dtype=torch.long,
            device=reference.device,
        )
        explicit = torch.tensor(
            [before - value for value in after],
            dtype=reference.dtype,
            device=reference.device,
        )
        if learned_residual:
            residual = self(
                row_tensor,
                actions,
                energy_before=before,
                energy_after=after,
            )
        else:
            residual = torch.zeros_like(explicit)
        return ScoredEnergyActions(
            actions=actions,
            energy_before=before,
            energy_after=after,
            explicit_reduction=explicit,
            learned_residual=residual,
            total_score=explicit + residual,
        )


@dataclass(frozen=True, slots=True)
class LabeledEnergyState:
    rows: tuple[tuple[int, ...], ...]
    target_indices: tuple[int, ...]

    @property
    def sha256(self) -> str:
        return sha256(
            json.dumps(
                [self.rows, self.target_indices],
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()


@dataclass(slots=True)
class OracleCounter:
    calls: int = 0


def expert_action_indices(
    rows: Iterable[Iterable[int]],
    *,
    counter: OracleCounter,
) -> tuple[int, ...]:
    """Preparation-only two-ply tie resolver.

    The expert cannot override the best exact one-step energy reduction.  It
    only asks which tied move exposes the lowest best successor energy.
    """

    counter.calls += 1
    matrix = canonical_matrix(rows)
    transitions = evaluate_transitions(matrix)
    reductions = tuple(
        transition.explicit_reduction for transition in transitions
    )
    maximum_reduction = max(reductions)
    tied = [
        index
        for index, reduction in enumerate(reductions)
        if reduction == maximum_reduction
    ]
    if len(tied) == 1 or defect_energy(matrix) == 0:
        return tuple(tied)
    lookahead: dict[int, int] = {}
    for index in tied:
        successor = transitions[index].rows
        if defect_energy(successor) == 0:
            lookahead[index] = 0
            continue
        next_transitions = evaluate_transitions(successor)
        lookahead[index] = min(
            transition.energy_after for transition in next_transitions
        )
    best = min(lookahead.values())
    return tuple(index for index in tied if lookahead[index] == best)


def build_expert_states(
    matrices: Iterable[Iterable[Iterable[int]]],
    *,
    maximum_steps: int,
    counter: OracleCounter,
) -> tuple[LabeledEnergyState, ...]:
    """Collect bounded expert trajectories for preparation only."""

    limit = _plain_positive_int(maximum_steps, label="maximum_steps")
    states: dict[str, LabeledEnergyState] = {}
    for raw in matrices:
        matrix = canonical_matrix(raw)
        visited: set[tuple[tuple[int, ...], ...]] = set()
        for _ in range(limit):
            targets = expert_action_indices(matrix, counter=counter)
            labeled = LabeledEnergyState(matrix, targets)
            prior = states.get(matrix_sha256(matrix))
            if prior is not None and prior != labeled:
                raise EnergyControllerError("expert labels conflict")
            states[matrix_sha256(matrix)] = labeled
            actions = enumerate_legal_actions(matrix)
            chosen = next(
                (
                    index
                    for index in targets
                    if apply_local_action(matrix, actions[index])
                    not in visited
                ),
                targets[0],
            )
            action = actions[chosen]
            if action.kind == ACTION_HALT:
                break
            visited.add(matrix)
            successor = apply_local_action(matrix, action)
            if successor in visited:
                break
            matrix = successor
    return tuple(states[key] for key in sorted(states))


def make_random_label_control(
    states: Sequence[LabeledEnergyState],
    *,
    seed: int,
) -> tuple[LabeledEnergyState, ...]:
    """Replace expert labels with seeded random nonexpert legal labels.

    Whenever more than one legal action exists, the control is forced away
    from the expert target set.  This deliberately asks the bounded residual
    to fit labels that may contradict the explicit energy ordering; failure
    to fit them is an expected causal control rather than a training defect.
    """

    rng = random.Random(seed)
    result = []
    for state in states:
        transitions = evaluate_transitions(state.rows)
        candidates = [
            index
            for index in range(len(transitions))
            if index not in state.target_indices
        ]
        if not candidates:
            candidates = list(range(len(transitions)))
        result.append(
            replace(state, target_indices=(rng.choice(candidates),))
        )
    return tuple(result)


def labeled_state_manifest(
    states: Iterable[LabeledEnergyState],
) -> str:
    return sha256(
        (
            "\n".join(
                state.sha256 for state in sorted(states, key=lambda item: item.sha256)
            )
            + "\n"
        ).encode("ascii")
    ).hexdigest()


def _listwise_loss(
    controller: EquivariantEnergyController,
    state: LabeledEnergyState,
) -> Tensor:
    scored = controller.score_actions(state.rows)
    target = torch.tensor(
        state.target_indices,
        dtype=torch.long,
        device=scored.total_score.device,
    )
    return (
        torch.logsumexp(scored.total_score, dim=0)
        - torch.logsumexp(scored.total_score[target], dim=0)
    )


def train_energy_controller(
    controller: EquivariantEnergyController,
    states: Sequence[LabeledEnergyState],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    shuffle_seed: int,
    maximum_updates: int,
) -> int:
    """Run a deterministic, explicitly bounded listwise fit."""

    epoch_count = _plain_positive_int(epochs, label="epochs")
    batch = _plain_positive_int(batch_size, label="batch_size")
    update_limit = _plain_positive_int(
        maximum_updates,
        label="maximum_updates",
    )
    if not states:
        raise EnergyControllerError("training states must be nonempty")
    if not isinstance(learning_rate, float) or learning_rate <= 0.0:
        raise EnergyControllerError("learning_rate must be positive")
    optimizer = torch.optim.AdamW(
        controller.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )
    rng = random.Random(shuffle_seed)
    updates = 0
    controller.train()
    for _ in range(epoch_count):
        order = list(range(len(states)))
        rng.shuffle(order)
        for offset in range(0, len(order), batch):
            optimizer.zero_grad(set_to_none=True)
            losses = [
                _listwise_loss(controller, states[index])
                for index in order[offset : offset + batch]
            ]
            torch.stack(losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
            optimizer.step()
            updates += 1
            if updates >= update_limit:
                return updates
    return updates


@torch.no_grad()
def label_accuracy(
    controller: EquivariantEnergyController,
    states: Sequence[LabeledEnergyState],
) -> float:
    controller.eval()
    if not states:
        return 0.0
    correct = 0
    for state in states:
        predicted = int(
            controller.score_actions(state.rows).total_score.argmax().item()
        )
        correct += predicted in state.target_indices
    return correct / len(states)


def model_state_sha256(model: EquivariantEnergyController) -> str:
    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RolloutResult:
    unordered_certified: bool
    strict_canonical_certified: bool
    invalid: bool
    overlong: bool
    actions: tuple[LocalEnergyAction, ...]
    initial_energy: int
    final_energy: int
    oracle_calls: int

    @property
    def certified(self) -> bool:
        """Backward-compatible alias for the unordered endpoint metric."""

        return self.unordered_certified


@torch.no_grad()
def final_oracle_free_rollout(
    controller: EquivariantEnergyController,
    rows: Iterable[Iterable[int]],
    *,
    maximum_steps: int,
    learned_residual: bool = True,
) -> RolloutResult:
    """Run the deployed matrix-only dynamics with no oracle call path."""

    limit = _plain_positive_int(maximum_steps, label="maximum_steps")
    source = canonical_matrix(rows)
    matrix = source
    provenance = _identity(len(source))
    emitted: list[LocalEnergyAction] = []
    visited: set[tuple[tuple[int, ...], ...]] = set()
    initial_energy = defect_energy(source)
    controller.eval()
    for _ in range(limit):
        scored = controller.score_actions(
            matrix,
            learned_residual=learned_residual,
        )
        choice = int(scored.total_score.argmax().item())
        action = scored.actions[choice]
        if action.kind == ACTION_HALT:
            actions = tuple((*emitted, action))
            unordered_certified = verify_unordered_reduced_basis(
                source,
                matrix,
                provenance,
            )
            strict_canonical_certified = (
                verify_strict_canonical_action_trace(source, actions)
            )
            return RolloutResult(
                unordered_certified=unordered_certified,
                strict_canonical_certified=strict_canonical_certified,
                invalid=not unordered_certified,
                overlong=False,
                actions=actions,
                initial_energy=initial_energy,
                final_energy=defect_energy(matrix),
                oracle_calls=0,
            )
        if int(scored.explicit_reduction[choice].item()) < 0:
            return RolloutResult(
                unordered_certified=False,
                strict_canonical_certified=False,
                invalid=True,
                overlong=False,
                actions=tuple(emitted),
                initial_energy=initial_energy,
                final_energy=defect_energy(matrix),
                oracle_calls=0,
            )
        if matrix in visited:
            return RolloutResult(
                unordered_certified=False,
                strict_canonical_certified=False,
                invalid=True,
                overlong=False,
                actions=tuple(emitted),
                initial_energy=initial_energy,
                final_energy=defect_energy(matrix),
                oracle_calls=0,
            )
        visited.add(matrix)
        provenance = _apply_action_to_provenance(
            provenance,
            matrix,
            action,
        )
        matrix = apply_local_action(matrix, action)
        emitted.append(action)
    return RolloutResult(
        unordered_certified=False,
        strict_canonical_certified=False,
        invalid=False,
        overlong=True,
        actions=tuple(emitted),
        initial_energy=initial_energy,
        final_energy=defect_energy(matrix),
        oracle_calls=0,
    )


@dataclass(frozen=True, slots=True)
class RolloutSummary:
    unordered_certified: int
    strict_canonical_certified: int
    invalid: int
    overlong: int
    oracle_calls: int

    @property
    def certified(self) -> int:
        """Backward-compatible alias for the unordered endpoint metric."""

        return self.unordered_certified

    @property
    def total(self) -> int:
        return self.unordered_certified + self.invalid + self.overlong


def evaluate_matrices_oracle_free(
    controller: EquivariantEnergyController,
    matrices: Sequence[tuple[tuple[int, ...], ...]],
    *,
    maximum_steps: int,
    learned_residual: bool = True,
) -> RolloutSummary:
    unordered = strict = invalid = overlong = oracle_calls = 0
    for matrix in matrices:
        result = final_oracle_free_rollout(
            controller,
            matrix,
            maximum_steps=maximum_steps,
            learned_residual=learned_residual,
        )
        unordered += result.unordered_certified
        strict += result.strict_canonical_certified
        invalid += result.invalid
        overlong += result.overlong
        oracle_calls += result.oracle_calls
    return RolloutSummary(
        unordered_certified=unordered,
        strict_canonical_certified=strict,
        invalid=invalid,
        overlong=overlong,
        oracle_calls=oracle_calls,
    )


def generate_matrices(
    *,
    seed: int,
    count: int,
    minimum_rows: int,
    maximum_rows: int,
    minimum_columns: int,
    maximum_columns: int,
    excluded: set[tuple[tuple[int, ...], ...]] | None = None,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Generate deterministic bounded sparse matrices with no hidden traces."""

    target = _plain_positive_int(count, label="count")
    min_rows = _plain_positive_int(minimum_rows, label="minimum_rows")
    max_rows = _plain_positive_int(maximum_rows, label="maximum_rows")
    min_columns = _plain_positive_int(
        minimum_columns,
        label="minimum_columns",
    )
    max_columns = _plain_positive_int(
        maximum_columns,
        label="maximum_columns",
    )
    if min_rows > max_rows or min_columns > max_columns:
        raise EnergyControllerError("matrix generation bounds are inverted")
    if max_columns > MAX_EXACT_ENERGY_COLUMNS:
        raise EnergyControllerError("generation exceeds exact energy bound")
    rng = random.Random(seed)
    forbidden = set() if excluded is None else set(excluded)
    result: list[tuple[tuple[int, ...], ...]] = []
    seen = set(forbidden)
    attempts = 0
    while len(result) < target and attempts < target * 10_000:
        attempts += 1
        row_count = rng.randint(min_rows, max_rows)
        column_count = rng.randint(
            max(row_count, min_columns),
            max_columns,
        )
        matrix = tuple(
            tuple(
                0 if rng.random() < 0.55 else rng.randrange(1, FIELD_MODULUS)
                for _ in range(column_count)
            )
            for _ in range(row_count)
        )
        if matrix in seen or not any(value for row in matrix for value in row):
            continue
        seen.add(matrix)
        result.append(matrix)
    if len(result) != target:
        raise EnergyControllerError("matrix generator exhausted its bound")
    return tuple(result)


def matrix_manifest(
    matrices: Iterable[Iterable[Iterable[int]]],
) -> str:
    return sha256(
        (
            "\n".join(matrix_sha256(matrix) for matrix in matrices) + "\n"
        ).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class EnergyExperimentConfig:
    seed: int = 20260724
    train_matrices: int = 96
    evaluation_matrices: int = 32
    train_maximum_rows: int = 3
    train_maximum_columns: int = 4
    evaluation_minimum_rows: int = 4
    evaluation_minimum_columns: int = 5
    evaluation_maximum_rows: int = 4
    evaluation_maximum_columns: int = 6
    maximum_expert_steps: int = 96
    maximum_rollout_steps: int = 192
    epochs: int = 12
    batch_size: int = 8
    learning_rate: float = 1e-3
    maximum_updates: int = 2_048
    controller: EnergyControllerConfig = EnergyControllerConfig()

    def __post_init__(self) -> None:
        for label, value in (
            ("train_matrices", self.train_matrices),
            ("evaluation_matrices", self.evaluation_matrices),
            ("train_maximum_rows", self.train_maximum_rows),
            ("train_maximum_columns", self.train_maximum_columns),
            ("evaluation_minimum_rows", self.evaluation_minimum_rows),
            ("evaluation_minimum_columns", self.evaluation_minimum_columns),
            ("evaluation_maximum_rows", self.evaluation_maximum_rows),
            ("evaluation_maximum_columns", self.evaluation_maximum_columns),
            ("maximum_expert_steps", self.maximum_expert_steps),
            ("maximum_rollout_steps", self.maximum_rollout_steps),
            ("epochs", self.epochs),
            ("batch_size", self.batch_size),
            ("maximum_updates", self.maximum_updates),
        ):
            _plain_positive_int(value, label=label)
        if self.evaluation_minimum_rows <= self.train_maximum_rows:
            raise EnergyControllerError(
                "evaluation rows must be strictly larger than training rows"
            )
        if self.evaluation_minimum_columns <= self.train_maximum_columns:
            raise EnergyControllerError(
                "evaluation columns must be strictly larger than training columns"
            )
        if self.evaluation_minimum_rows > self.evaluation_maximum_rows:
            raise EnergyControllerError("evaluation row bounds are inverted")
        if self.evaluation_minimum_columns > self.evaluation_maximum_columns:
            raise EnergyControllerError("evaluation column bounds are inverted")
        if self.evaluation_maximum_columns > MAX_EXACT_ENERGY_COLUMNS:
            raise EnergyControllerError("evaluation exceeds exact energy bound")
        if not isinstance(self.learning_rate, float) or self.learning_rate <= 0:
            raise EnergyControllerError("learning_rate must be positive")


@dataclass(frozen=True, slots=True)
class EnergyExperimentReport:
    schema: str
    status: str
    seed: int
    controller_parameters: int
    parameter_count_breakdown: Mapping[str, int]
    train_matrices: int
    train_states: int
    evaluation_matrices: int
    train_maximum_rows: int
    train_maximum_columns: int
    evaluation_minimum_rows: int
    evaluation_minimum_columns: int
    evaluation_maximum_rows: int
    evaluation_maximum_columns: int
    expert_oracle_calls_preparation_only: int
    final_rollout_oracle_calls: int
    expert_optimizer_updates: int
    random_label_optimizer_updates: int
    expert_train_label_accuracy: float
    random_control_label_accuracy: float
    random_control_true_expert_accuracy: float
    energy_only_unordered_certified: int
    expert_model_unordered_certified: int
    random_label_model_unordered_certified: int
    energy_only_strict_canonical_certified: int
    expert_model_strict_canonical_certified: int
    random_label_model_strict_canonical_certified: int
    energy_only_invalid: int
    expert_model_invalid: int
    random_label_model_invalid: int
    energy_only_overlong: int
    expert_model_overlong: int
    random_label_model_overlong: int
    train_matrix_manifest_sha256: str
    evaluation_matrix_manifest_sha256: str
    expert_state_manifest_sha256: str
    random_state_manifest_sha256: str
    expert_model_sha256: str
    random_label_model_sha256: str

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                asdict(self),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
            + b"\n"
        )


def _fresh_identical_controllers(
    config: EnergyControllerConfig,
    *,
    seed: int,
) -> tuple[EquivariantEnergyController, EquivariantEnergyController]:
    torch.manual_seed(seed)
    first = EquivariantEnergyController(config)
    second = EquivariantEnergyController(config)
    second.load_state_dict(first.state_dict())
    return first, second


def run_bounded_experiment(
    config: EnergyExperimentConfig,
) -> EnergyExperimentReport:
    """Run the deterministic expert/random-control falsification package."""

    if not isinstance(config, EnergyExperimentConfig):
        raise EnergyControllerError("experiment config has the wrong type")
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    train = generate_matrices(
        seed=config.seed,
        count=config.train_matrices,
        minimum_rows=2,
        maximum_rows=config.train_maximum_rows,
        minimum_columns=2,
        maximum_columns=config.train_maximum_columns,
    )
    evaluation = generate_matrices(
        seed=config.seed + 1,
        count=config.evaluation_matrices,
        minimum_rows=config.evaluation_minimum_rows,
        maximum_rows=config.evaluation_maximum_rows,
        minimum_columns=config.evaluation_minimum_columns,
        maximum_columns=config.evaluation_maximum_columns,
        excluded=set(train),
    )
    oracle_counter = OracleCounter()
    expert_states = build_expert_states(
        train,
        maximum_steps=config.maximum_expert_steps,
        counter=oracle_counter,
    )
    random_states = make_random_label_control(
        expert_states,
        seed=config.seed + 2,
    )
    expert_model, random_model = _fresh_identical_controllers(
        config.controller,
        seed=config.seed + 3,
    )
    expert_updates = train_energy_controller(
        expert_model,
        expert_states,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        shuffle_seed=config.seed + 4,
        maximum_updates=config.maximum_updates,
    )
    random_updates = train_energy_controller(
        random_model,
        random_states,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        shuffle_seed=config.seed + 4,
        maximum_updates=config.maximum_updates,
    )
    energy_only = evaluate_matrices_oracle_free(
        expert_model,
        evaluation,
        maximum_steps=config.maximum_rollout_steps,
        learned_residual=False,
    )
    expert_result = evaluate_matrices_oracle_free(
        expert_model,
        evaluation,
        maximum_steps=config.maximum_rollout_steps,
    )
    random_result = evaluate_matrices_oracle_free(
        random_model,
        evaluation,
        maximum_steps=config.maximum_rollout_steps,
    )
    final_oracle_calls = (
        energy_only.oracle_calls
        + expert_result.oracle_calls
        + random_result.oracle_calls
    )
    if final_oracle_calls != 0:
        raise EnergyControllerError("oracle was called during final rollout")
    return EnergyExperimentReport(
        schema=EXPERIMENT_SCHEMA,
        status=STATUS,
        seed=config.seed,
        controller_parameters=expert_model.parameter_count,
        parameter_count_breakdown=expert_model.parameter_count_breakdown(),
        train_matrices=len(train),
        train_states=len(expert_states),
        evaluation_matrices=len(evaluation),
        train_maximum_rows=config.train_maximum_rows,
        train_maximum_columns=config.train_maximum_columns,
        evaluation_minimum_rows=config.evaluation_minimum_rows,
        evaluation_minimum_columns=config.evaluation_minimum_columns,
        evaluation_maximum_rows=config.evaluation_maximum_rows,
        evaluation_maximum_columns=config.evaluation_maximum_columns,
        expert_oracle_calls_preparation_only=oracle_counter.calls,
        final_rollout_oracle_calls=final_oracle_calls,
        expert_optimizer_updates=expert_updates,
        random_label_optimizer_updates=random_updates,
        expert_train_label_accuracy=label_accuracy(
            expert_model,
            expert_states,
        ),
        random_control_label_accuracy=label_accuracy(
            random_model,
            random_states,
        ),
        random_control_true_expert_accuracy=label_accuracy(
            random_model,
            expert_states,
        ),
        energy_only_unordered_certified=energy_only.unordered_certified,
        expert_model_unordered_certified=(
            expert_result.unordered_certified
        ),
        random_label_model_unordered_certified=(
            random_result.unordered_certified
        ),
        energy_only_strict_canonical_certified=(
            energy_only.strict_canonical_certified
        ),
        expert_model_strict_canonical_certified=(
            expert_result.strict_canonical_certified
        ),
        random_label_model_strict_canonical_certified=(
            random_result.strict_canonical_certified
        ),
        energy_only_invalid=energy_only.invalid,
        expert_model_invalid=expert_result.invalid,
        random_label_model_invalid=random_result.invalid,
        energy_only_overlong=energy_only.overlong,
        expert_model_overlong=expert_result.overlong,
        random_label_model_overlong=random_result.overlong,
        train_matrix_manifest_sha256=matrix_manifest(train),
        evaluation_matrix_manifest_sha256=matrix_manifest(evaluation),
        expert_state_manifest_sha256=labeled_state_manifest(expert_states),
        random_state_manifest_sha256=labeled_state_manifest(random_states),
        expert_model_sha256=model_state_sha256(expert_model),
        random_label_model_sha256=model_state_sha256(random_model),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--train-matrices", type=int, default=96)
    parser.add_argument("--evaluation-matrices", type=int, default=32)
    parser.add_argument("--train-maximum-rows", type=int, default=3)
    parser.add_argument("--train-maximum-columns", type=int, default=4)
    parser.add_argument("--evaluation-minimum-rows", type=int, default=4)
    parser.add_argument("--evaluation-minimum-columns", type=int, default=5)
    parser.add_argument("--evaluation-maximum-rows", type=int, default=4)
    parser.add_argument("--evaluation-maximum-columns", type=int, default=6)
    parser.add_argument("--maximum-expert-steps", type=int, default=96)
    parser.add_argument("--maximum-rollout-steps", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--maximum-updates", type=int, default=2_048)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--message-layers", type=int, default=3)
    parser.add_argument("--residual-hidden", type=int, default=128)
    parser.add_argument("--field-harmonics", type=int, default=4)
    parser.add_argument("--residual-bound", type=float, default=0.49)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_bounded_experiment(
        EnergyExperimentConfig(
            seed=args.seed,
            train_matrices=args.train_matrices,
            evaluation_matrices=args.evaluation_matrices,
            train_maximum_rows=args.train_maximum_rows,
            train_maximum_columns=args.train_maximum_columns,
            evaluation_minimum_rows=args.evaluation_minimum_rows,
            evaluation_minimum_columns=args.evaluation_minimum_columns,
            evaluation_maximum_rows=args.evaluation_maximum_rows,
            evaluation_maximum_columns=args.evaluation_maximum_columns,
            maximum_expert_steps=args.maximum_expert_steps,
            maximum_rollout_steps=args.maximum_rollout_steps,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            maximum_updates=args.maximum_updates,
            controller=EnergyControllerConfig(
                width=args.width,
                message_layers=args.message_layers,
                residual_hidden=args.residual_hidden,
                field_harmonics=args.field_harmonics,
                residual_bound=args.residual_bound,
            ),
        )
    )
    payload = report.canonical_bytes()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
    print(payload.decode("ascii"), end="")


if __name__ == "__main__":
    main()
