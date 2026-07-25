#!/usr/bin/env python3
"""Canonical defect-energy controller for the SSQAC algebra machine.

This is an isolated mechanics falsifier.  The deployed candidate receives only
the current matrix over F_257.  It receives no source, query, workspace,
target, oracle, recurrent state, previous action, or step signal.

The explicit integer energy is the number of rank slots not yet covered by a
settled canonical pivot prefix.  A prefix row is settled only when its leading
column is the earliest support still available in the remaining rows, its
pivot is one, and its pivot column is reduced everywhere else.  Therefore the
energy is zero exactly at canonical RREF, while an ordinary Gauss-Jordan
frontier action never has to damage already settled work.

An ordered rank-matching diagnostic additionally searches for a rank-sized
assignment of current rows to canonical row slots and a strictly increasing
assignment of semantic pivot columns.  It reports:

* assigning a row to the wrong canonical row slot;
* a nonunit pivot;
* nonzero coefficients before a proposed pivot;
* nonzero coefficients elsewhere in a proposed pivot column; and
* support left in rows outside the rank-sized assignment.

The diagnostic is visible to the bounded residual and can therefore rank
equal-integer-energy actions, although it cannot change the integer term.
The frontier channel additionally exposes exact host-computed source, pivot,
and action-admissibility facts.  Matched inference ablations and a
zero-parameter fixed Gauss-Jordan schedule are reported explicitly so this
host algorithmic assistance cannot be mistaken for a learned primitive.
Because every deployed transition compiles to an invertible row operation in
the existing primitive VM, provenance reconstruction and row-span
preservation remain inductive invariants.  Final acceptance still calls the
unchanged ``verify_reduction_program`` assessor; energy never substitutes for
it.

The learned network emits a residual strictly inside (-0.5, 0.5).  It can
resolve actions with equal integer energy reduction, but cannot reverse a
one-unit explicit advantage.  Columns have fixed semantic order.  Row order is
part of the state and can be changed only by an explicit ``SWAP`` transition.
All coordinate features are deterministic, so unseen rows or columns do not
create untrained embedding-table entries.

No result from this module is, by itself, evidence of native or general
reasoning.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from hashlib import sha256
from itertools import combinations, permutations
import json
import math
from pathlib import Path
import random
from typing import Iterable, Iterator, Mapping, Sequence

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
    OP_SWAP,
    AlgebraInstruction,
    AlgebraMachineError,
    execute_program,
    verify_reduction_program,
)


ARCHITECTURE_SCHEMA = "ssqac_canonical_energy_controller_v2"
SEED_REPORT_SCHEMA = "ssqac_canonical_energy_seed_report_v3"
MULTISEED_REPORT_SCHEMA = "ssqac_canonical_energy_multiseed_report_v3"
STATUS = "isolated_canonical_energy_mechanics_not_reasoning"
HYBRID_INTERPRETATION = (
    "hybrid_host_algorithm_mechanics_not_learned_or_native_reasoning"
)
LEARNED_INTERPRETATION = "learned_matrix_policy_mechanics_not_native_reasoning"
ABLATION_COLLAPSE_RULE = (
    "strict_certifications_retained_below_50_percent_of_full_expert"
)
LEARNED_CLAIM_DOWNGRADE_RULE = (
    "fixed_schedule_reaches_100_percent_or_any_ablation_collapses"
)

ACTION_NORMALIZE = "NORMALIZE"
ACTION_ELIMINATE = "ELIMINATE"
ACTION_SWAP = "SWAP"
ACTION_HALT = "HALT"
ACTION_TYPES = (
    ACTION_NORMALIZE,
    ACTION_ELIMINATE,
    ACTION_SWAP,
    ACTION_HALT,
)

FRONTIER_FULL = "full"
FRONTIER_ZERO_ALL = "zero_all"
FRONTIER_MASK_ACTION_CORRECTNESS = "mask_action_correctness"
FRONTIER_ABLATIONS = (
    FRONTIER_FULL,
    FRONTIER_ZERO_ALL,
    FRONTIER_MASK_ACTION_CORRECTNESS,
)
FRONTIER_FEATURE_COUNT = 18
FRONTIER_ACTION_CORRECTNESS_START = 14

MAX_EXACT_ROWS = 8
MAX_EXACT_COLUMNS = 12


class CanonicalEnergyError(ValueError):
    """A canonical-energy contract failed closed."""


@dataclass(frozen=True, slots=True)
class MechanicsResourceCounts:
    """Auditable structural work performed inside one measured arm."""

    field_rank_calls: int = 0
    field_rank_coefficient_tests: int = 0
    field_rank_pivots: int = 0
    field_rank_inversions: int = 0
    field_rank_scale_cells: int = 0
    field_rank_axpy_cells: int = 0
    settled_prefix_cache_misses: int = 0
    matching_cache_misses: int = 0
    matching_row_permutations: int = 0
    matching_candidate_assignments: int = 0
    matching_slot_evaluations: int = 0
    action_successor_batches: int = 0
    action_successor_evaluations: int = 0
    reference_schedule_calls: int = 0
    reference_schedule_actions: int = 0
    strict_verifier_calls: int = 0

    def __add__(self, other: object) -> MechanicsResourceCounts:
        if not isinstance(other, MechanicsResourceCounts):
            return NotImplemented
        return MechanicsResourceCounts(
            **{
                name: getattr(self, name) + getattr(other, name)
                for name in self.__dataclass_fields__
            }
        )


@dataclass(slots=True)
class _MutableMechanicsResources:
    field_rank_calls: int = 0
    field_rank_coefficient_tests: int = 0
    field_rank_pivots: int = 0
    field_rank_inversions: int = 0
    field_rank_scale_cells: int = 0
    field_rank_axpy_cells: int = 0
    settled_prefix_cache_misses: int = 0
    matching_cache_misses: int = 0
    matching_row_permutations: int = 0
    matching_candidate_assignments: int = 0
    matching_slot_evaluations: int = 0
    action_successor_batches: int = 0
    action_successor_evaluations: int = 0
    reference_schedule_calls: int = 0
    reference_schedule_actions: int = 0
    strict_verifier_calls: int = 0

    def freeze(self) -> MechanicsResourceCounts:
        return MechanicsResourceCounts(
            **{
                name: getattr(self, name)
                for name in MechanicsResourceCounts.__dataclass_fields__
            }
        )


_RESOURCE_COUNTER: ContextVar[_MutableMechanicsResources | None] = ContextVar(
    "ssqac_canonical_resource_counter",
    default=None,
)


def _resources() -> _MutableMechanicsResources | None:
    return _RESOURCE_COUNTER.get()


@contextmanager
def mechanics_resource_accounting() -> Iterator[_MutableMechanicsResources]:
    counter = _MutableMechanicsResources()
    token = _RESOURCE_COUNTER.set(counter)
    try:
        yield counter
    finally:
        _RESOURCE_COUNTER.reset(token)


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CanonicalEnergyError(f"{label} must be a positive integer")
    return value


def _validate_frontier_ablation(value: str) -> str:
    if value not in FRONTIER_ABLATIONS:
        raise CanonicalEnergyError(
            f"frontier ablation must be one of {FRONTIER_ABLATIONS!r}"
        )
    return value


def _ablation_collapsed(*, ablated: int, full: int) -> bool:
    if ablated < 0 or full < 0:
        raise CanonicalEnergyError("certification counts must be nonnegative")
    return 2 * ablated < full


def canonical_matrix(
    rows: Iterable[Iterable[int]],
) -> tuple[tuple[int, ...], ...]:
    """Freeze a nonempty rectangular matrix over F_257."""

    matrix = tuple(tuple(int(value) % FIELD_MODULUS for value in row) for row in rows)
    if not matrix or not matrix[0]:
        raise CanonicalEnergyError("matrix must be nonempty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise CanonicalEnergyError("matrix rows have inconsistent widths")
    if len(matrix) > MAX_EXACT_ROWS:
        raise CanonicalEnergyError(
            f"exact energy supports at most {MAX_EXACT_ROWS} rows"
        )
    if width > MAX_EXACT_COLUMNS:
        raise CanonicalEnergyError(
            f"exact energy supports at most {MAX_EXACT_COLUMNS} columns"
        )
    return matrix


def matrix_sha256(rows: Iterable[Iterable[int]]) -> str:
    matrix = canonical_matrix(rows)
    return sha256(json.dumps(matrix, separators=(",", ":")).encode("ascii")).hexdigest()


def field_rank(rows: Iterable[Iterable[int]]) -> int:
    """Return exact rank over F_257 without emitting a deployed schedule."""

    matrix = [list(row) for row in canonical_matrix(rows)]
    row_count = len(matrix)
    column_count = len(matrix[0])
    resources = _resources()
    if resources is not None:
        resources.field_rank_calls += 1
    pivot_row = 0
    for column in range(column_count):
        source = None
        for row in range(pivot_row, row_count):
            if resources is not None:
                resources.field_rank_coefficient_tests += 1
            if matrix[row][column]:
                source = row
                break
        if source is None:
            continue
        matrix[pivot_row], matrix[source] = (
            matrix[source],
            matrix[pivot_row],
        )
        inverse = pow(matrix[pivot_row][column], -1, FIELD_MODULUS)
        if resources is not None:
            resources.field_rank_pivots += 1
            resources.field_rank_inversions += 1
            resources.field_rank_scale_cells += column_count
        matrix[pivot_row] = [
            inverse * value % FIELD_MODULUS for value in matrix[pivot_row]
        ]
        for row in range(row_count):
            if row == pivot_row:
                continue
            factor = matrix[row][column]
            if factor:
                if resources is not None:
                    resources.field_rank_axpy_cells += column_count
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


@dataclass(frozen=True, slots=True)
class CanonicalEnergyWitness:
    """Canonical frontier energy plus its ordered-matching diagnostic."""

    energy: int
    rank: int
    settled_prefix: int
    structural_weight: int
    structural_defects: int
    tail_nonzero: int
    assigned_rows: tuple[int, ...]
    pivot_columns: tuple[int, ...]


@lru_cache(maxsize=262_144)
def _canonical_settled_prefix_cached(
    matrix: tuple[tuple[int, ...], ...],
) -> int:
    """Count the consecutive canonical pivots fixed at the row frontier."""

    resources = _resources()
    if resources is not None:
        resources.settled_prefix_cache_misses += 1
    rank = field_rank(matrix)
    previous_pivot = -1
    settled = 0
    for row_index in range(rank):
        row = matrix[row_index]
        nonzero = tuple(column for column, value in enumerate(row) if value)
        if not nonzero:
            break
        pivot = nonzero[0]
        if pivot <= previous_pivot or row[pivot] != 1:
            break
        if any(
            other != row_index and matrix[other][pivot] != 0
            for other in range(len(matrix))
        ):
            break
        if any(
            matrix[later][column] != 0
            for later in range(row_index + 1, len(matrix))
            for column in range(pivot)
        ):
            break
        settled += 1
        previous_pivot = pivot
    return settled


def canonical_settled_prefix(rows: Iterable[Iterable[int]]) -> int:
    return _canonical_settled_prefix_cached(canonical_matrix(rows))


@lru_cache(maxsize=262_144)
def _canonical_energy_witness_cached(
    matrix: tuple[tuple[int, ...], ...],
) -> CanonicalEnergyWitness:
    """Compute frontier energy and the ordered rank-matching diagnostic.

    A rank-k endpoint is canonical exactly when rows 0..k-1 can be assigned
    to increasing pivot columns with no structural defect and every remaining
    row is zero.  Requiring nonzero assigned entries keeps the witness tied to
    actual row support rather than to fabricated pivot locations.
    """

    resources = _resources()
    if resources is not None:
        resources.matching_cache_misses += 1
    row_count = len(matrix)
    column_count = len(matrix[0])
    rank = field_rank(matrix)
    settled_prefix = _canonical_settled_prefix_cached(matrix)
    energy = rank - settled_prefix
    structural_weight = row_count * column_count + 1
    if rank == 0:
        nonzero = sum(value != 0 for row in matrix for value in row)
        return CanonicalEnergyWitness(
            energy=energy,
            rank=0,
            settled_prefix=settled_prefix,
            structural_weight=structural_weight,
            structural_defects=0,
            tail_nonzero=nonzero,
            assigned_rows=(),
            pivot_columns=(),
        )

    column_nonzero = tuple(
        sum(matrix[row][column] != 0 for row in range(row_count))
        for column in range(column_count)
    )
    row_nonzero = tuple(sum(value != 0 for value in row) for row in matrix)
    best: tuple[int, int, int, tuple[int, ...], tuple[int, ...]] | None = None
    for assigned_rows in permutations(range(row_count), rank):
        if resources is not None:
            resources.matching_row_permutations += 1
        selected = frozenset(assigned_rows)
        tail_nonzero = sum(
            row_nonzero[row] for row in range(row_count) if row not in selected
        )
        for pivot_columns in combinations(range(column_count), rank):
            if resources is not None:
                resources.matching_candidate_assignments += 1
            structural = 0
            feasible = True
            for slot, (row, column) in enumerate(
                zip(assigned_rows, pivot_columns, strict=True)
            ):
                if resources is not None:
                    resources.matching_slot_evaluations += 1
                value = matrix[row][column]
                if value == 0:
                    feasible = False
                    break
                structural += int(row != slot)
                structural += int(value != 1)
                structural += column_nonzero[column] - 1
                structural += sum(matrix[row][prior] != 0 for prior in range(column))
            if not feasible:
                continue
            matching_defect = structural_weight * structural + tail_nonzero
            candidate = (
                matching_defect,
                structural,
                tail_nonzero,
                tuple(assigned_rows),
                tuple(pivot_columns),
            )
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise CanonicalEnergyError(
            "rank has no nonzero ordered matching; rank invariant failed"
        )
    _, structural, tail, assigned_rows, pivot_columns = best
    return CanonicalEnergyWitness(
        energy=energy,
        rank=rank,
        settled_prefix=settled_prefix,
        structural_weight=structural_weight,
        structural_defects=structural,
        tail_nonzero=tail,
        assigned_rows=assigned_rows,
        pivot_columns=pivot_columns,
    )


def canonical_energy_witness(
    rows: Iterable[Iterable[int]],
) -> CanonicalEnergyWitness:
    return _canonical_energy_witness_cached(canonical_matrix(rows))


def canonical_defect_energy(rows: Iterable[Iterable[int]]) -> int:
    """Return rank minus the settled canonical pivot prefix."""

    matrix = canonical_matrix(rows)
    return field_rank(matrix) - _canonical_settled_prefix_cached(matrix)


def clear_mechanics_caches() -> None:
    """Clear structural caches before a matched resource measurement."""

    _canonical_settled_prefix_cached.cache_clear()
    _canonical_energy_witness_cached.cache_clear()


def is_canonical_rref_structure(
    rows: Iterable[Iterable[int]],
) -> bool:
    """Check the endpoint structure, including trailing zero-row order."""

    matrix = canonical_matrix(rows)
    pivots: list[int] = []
    saw_zero = False
    for row in matrix:
        nonzero = tuple(column for column, value in enumerate(row) if value)
        if not nonzero:
            saw_zero = True
            continue
        if saw_zero:
            return False
        pivot = nonzero[0]
        if pivots and pivot <= pivots[-1]:
            return False
        if row[pivot] != 1:
            return False
        pivots.append(pivot)
    for row_index, pivot in enumerate(pivots):
        if any(
            other != row_index and matrix[other][pivot] != 0
            for other in range(len(matrix))
        ):
            return False
    return True


@dataclass(frozen=True, slots=True, order=True)
class CanonicalAction:
    """One legal matrix-only macro transition."""

    kind: str
    row_a: int = 0
    row_b: int = 0
    column: int = 0

    def __post_init__(self) -> None:
        if self.kind not in ACTION_TYPES:
            raise CanonicalEnergyError(f"unknown action type {self.kind!r}")
        for label, value in (
            ("row_a", self.row_a),
            ("row_b", self.row_b),
            ("column", self.column),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise CanonicalEnergyError(f"{label} must be an integer")

    def canonical_data(self) -> tuple[object, ...]:
        return (self.kind, self.row_a, self.row_b, self.column)


@dataclass(frozen=True, slots=True)
class CanonicalTransition:
    action: CanonicalAction
    rows: tuple[tuple[int, ...], ...]
    energy_before: int
    energy_after: int

    @property
    def explicit_reduction(self) -> int:
        return self.energy_before - self.energy_after


def enumerate_legal_actions(
    rows: Iterable[Iterable[int]],
) -> tuple[CanonicalAction, ...]:
    """Enumerate invertible local row operations from matrix state alone."""

    matrix = canonical_matrix(rows)
    if canonical_defect_energy(matrix) == 0:
        return (CanonicalAction(ACTION_HALT),)
    row_count = len(matrix)
    column_count = len(matrix[0])
    actions: list[CanonicalAction] = []
    for row in range(row_count):
        for column in range(column_count):
            if matrix[row][column] not in (0, 1):
                actions.append(
                    CanonicalAction(
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
                if destination != source and matrix[destination][column] != 0:
                    actions.append(
                        CanonicalAction(
                            ACTION_ELIMINATE,
                            row_a=destination,
                            row_b=source,
                            column=column,
                        )
                    )
    for left in range(row_count):
        for right in range(left + 1, row_count):
            if matrix[left] != matrix[right]:
                actions.append(
                    CanonicalAction(
                        ACTION_SWAP,
                        row_a=left,
                        row_b=right,
                    )
                )
    if not actions:
        raise CanonicalEnergyError(
            "positive-energy state has no legal invertible transition"
        )
    return tuple(actions)


def apply_action(
    rows: Iterable[Iterable[int]],
    action: CanonicalAction,
) -> tuple[tuple[int, ...], ...]:
    """Apply one exact macro transition over F_257."""

    matrix = canonical_matrix(rows)
    if not isinstance(action, CanonicalAction):
        raise CanonicalEnergyError("action has the wrong type")
    row_count = len(matrix)
    column_count = len(matrix[0])
    if action.kind == ACTION_HALT:
        if canonical_defect_energy(matrix) != 0:
            raise CanonicalEnergyError("HALT is legal only at zero energy")
        return matrix
    if not 0 <= action.row_a < row_count:
        raise CanonicalEnergyError("row_a is out of range")
    mutable = [list(row) for row in matrix]
    if action.kind == ACTION_SWAP:
        if not 0 <= action.row_b < row_count:
            raise CanonicalEnergyError("row_b is out of range")
        if action.row_a >= action.row_b:
            raise CanonicalEnergyError("SWAP requires canonical row_a < row_b")
        if mutable[action.row_a] == mutable[action.row_b]:
            raise CanonicalEnergyError("SWAP requires distinct row values")
        mutable[action.row_a], mutable[action.row_b] = (
            mutable[action.row_b],
            mutable[action.row_a],
        )
        return tuple(tuple(row) for row in mutable)
    if not 0 <= action.column < column_count:
        raise CanonicalEnergyError("column is out of range")
    if action.kind == ACTION_NORMALIZE:
        value = mutable[action.row_a][action.column]
        if value in (0, 1):
            raise CanonicalEnergyError(
                "NORMALIZE requires a nonunit nonzero coefficient"
            )
        factor = pow(value, -1, FIELD_MODULUS)
        mutable[action.row_a] = [
            factor * coefficient % FIELD_MODULUS
            for coefficient in mutable[action.row_a]
        ]
    elif action.kind == ACTION_ELIMINATE:
        if not 0 <= action.row_b < row_count:
            raise CanonicalEnergyError("row_b is out of range")
        if action.row_a == action.row_b:
            raise CanonicalEnergyError("ELIMINATE source and destination must differ")
        if mutable[action.row_b][action.column] != 1:
            raise CanonicalEnergyError("ELIMINATE source coefficient must equal one")
        target = mutable[action.row_a][action.column]
        if target == 0:
            raise CanonicalEnergyError(
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
        raise CanonicalEnergyError("unreachable action dispatch")
    return tuple(tuple(row) for row in mutable)


def compile_action_to_vm(
    rows: Iterable[Iterable[int]],
    action: CanonicalAction,
) -> tuple[AlgebraInstruction, ...]:
    """Compile one validated macro to the unchanged primitive row VM."""

    matrix = canonical_matrix(rows)
    apply_action(matrix, action)
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
    if action.kind == ACTION_SWAP:
        return (AlgebraInstruction(OP_SWAP, action.row_a, action.row_b),)
    return (AlgebraInstruction(OP_HALT),)


def compile_action_trace_to_vm(
    input_rows: Iterable[Iterable[int]],
    actions: Sequence[CanonicalAction],
) -> tuple[AlgebraInstruction, ...]:
    """Replay and compile a complete trace ending in exactly one HALT."""

    matrix = canonical_matrix(input_rows)
    frozen = tuple(actions)
    if not frozen or frozen[-1].kind != ACTION_HALT:
        raise CanonicalEnergyError("complete trace must terminate with HALT")
    if any(action.kind == ACTION_HALT for action in frozen[:-1]):
        raise CanonicalEnergyError("trace contains an action after HALT")
    program: list[AlgebraInstruction] = []
    for action in frozen:
        program.extend(compile_action_to_vm(matrix, action))
        matrix = apply_action(matrix, action)
    if sum(item.opcode == OP_HALT for item in program) != 1:
        raise CanonicalEnergyError("compiled trace must contain one HALT")
    return tuple(program)


def strictly_verify_action_trace(
    input_rows: Iterable[Iterable[int]],
    actions: Sequence[CanonicalAction],
) -> bool:
    """Run the unchanged verifier plus the explicit zero-row-order gate."""

    resources = _resources()
    if resources is not None:
        resources.strict_verifier_calls += 1
    source = canonical_matrix(input_rows)
    try:
        program = compile_action_trace_to_vm(source, actions)
        state = execute_program(source, program)
        receipt = verify_reduction_program(source, state)
    except (AlgebraMachineError, CanonicalEnergyError):
        return False
    return (
        receipt.passed
        and canonical_defect_energy(state.rows) == 0
        and is_canonical_rref_structure(state.rows)
    )


def evaluate_transitions(
    rows: Iterable[Iterable[int]],
) -> tuple[CanonicalTransition, ...]:
    matrix = canonical_matrix(rows)
    before = canonical_defect_energy(matrix)
    actions = enumerate_legal_actions(matrix)
    resources = _resources()
    if resources is not None:
        resources.action_successor_batches += 1
        resources.action_successor_evaluations += len(actions)
    return tuple(
        CanonicalTransition(
            action=action,
            rows=successor,
            energy_before=before,
            energy_after=canonical_defect_energy(successor),
        )
        for action in actions
        for successor in (apply_action(matrix, action),)
    )


def canonical_reference_schedule(
    rows: Iterable[Iterable[int]],
) -> tuple[CanonicalAction, ...]:
    """Return the deterministic host Gauss-Jordan schedule."""

    resources = _resources()
    if resources is not None:
        resources.reference_schedule_calls += 1
    matrix = canonical_matrix(rows)
    row_count = len(matrix)
    column_count = len(matrix[0])
    pivot_row = 0
    minimum_column = 0
    actions: list[CanonicalAction] = []
    while pivot_row < row_count and minimum_column < column_count:
        pivot_column = next(
            (
                column
                for column in range(minimum_column, column_count)
                if any(matrix[row][column] for row in range(pivot_row, row_count))
            ),
            None,
        )
        if pivot_column is None:
            break
        source = next(
            row for row in range(pivot_row, row_count) if matrix[row][pivot_column]
        )
        if source != pivot_row:
            swap = CanonicalAction(
                ACTION_SWAP,
                row_a=pivot_row,
                row_b=source,
            )
            actions.append(swap)
            matrix = apply_action(matrix, swap)
        if matrix[pivot_row][pivot_column] != 1:
            normalize = CanonicalAction(
                ACTION_NORMALIZE,
                row_a=pivot_row,
                column=pivot_column,
            )
            actions.append(normalize)
            matrix = apply_action(matrix, normalize)
        for destination in range(row_count):
            if destination != pivot_row and matrix[destination][pivot_column] != 0:
                eliminate = CanonicalAction(
                    ACTION_ELIMINATE,
                    row_a=destination,
                    row_b=pivot_row,
                    column=pivot_column,
                )
                actions.append(eliminate)
                matrix = apply_action(matrix, eliminate)
        pivot_row += 1
        minimum_column = pivot_column + 1
    if canonical_defect_energy(matrix) != 0:
        raise CanonicalEnergyError("reference schedule failed canonical energy closure")
    if resources is not None:
        resources.reference_schedule_actions += len(actions)
    return tuple(actions)


@dataclass(frozen=True, slots=True)
class CanonicalControllerConfig:
    width: int = 64
    message_layers: int = 3
    residual_hidden: int = 128
    field_harmonics: int = 4
    coordinate_harmonics: int = 4
    residual_bound: float = 0.49

    def __post_init__(self) -> None:
        for label, value in (
            ("width", self.width),
            ("message_layers", self.message_layers),
            ("residual_hidden", self.residual_hidden),
            ("field_harmonics", self.field_harmonics),
            ("coordinate_harmonics", self.coordinate_harmonics),
        ):
            _positive_int(value, label=label)
        if (
            not isinstance(self.residual_bound, float)
            or not 0.0 < self.residual_bound < 0.5
        ):
            raise CanonicalEnergyError(
                "residual_bound must be strictly between zero and 0.5"
            )


class _CanonicalMessageLayer(nn.Module):
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
            raise CanonicalEnergyError(
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
class ScoredCanonicalActions:
    actions: tuple[CanonicalAction, ...]
    energy_before: int
    energy_after: tuple[int, ...]
    explicit_reduction: Tensor
    learned_residual: Tensor
    total_score: Tensor


class CanonicalEnergyController(nn.Module):
    """Memoryless matrix-only scorer with deterministic coordinates."""

    def __init__(
        self,
        config: CanonicalControllerConfig = CanonicalControllerConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        coordinate_width = 4 + 2 * config.coordinate_harmonics
        field_width = 5 + 4 * config.field_harmonics
        width = config.width
        self.cell_encoder = nn.Sequential(
            nn.Linear(field_width + 2 * coordinate_width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.message_layers = nn.ModuleList(
            _CanonicalMessageLayer(width) for _ in range(config.message_layers)
        )
        self.action_type = nn.Embedding(len(ACTION_TYPES), width)
        self.frontier_projection = nn.Sequential(
            nn.Linear(18, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.energy_projection = nn.Sequential(
            nn.Linear(8, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(9 * width, config.residual_hidden),
            nn.SiLU(),
            nn.Linear(config.residual_hidden, config.residual_hidden),
            nn.SiLU(),
            nn.Linear(config.residual_hidden, 1),
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def parameter_count_breakdown(self) -> Mapping[str, int]:
        result: dict[str, int] = {}
        for name, parameter in self.named_parameters():
            owner = name.split(".", 1)[0]
            result[owner] = result.get(owner, 0) + parameter.numel()
        result["total"] = sum(value for key, value in result.items() if key != "total")
        return dict(sorted(result.items()))

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

    def _coordinate_features(
        self,
        count: int,
        *,
        maximum: int,
        device: torch.device,
    ) -> Tensor:
        indices = torch.arange(count, dtype=torch.float32, device=device)
        normalized = indices / float(maximum - 1)
        relative = indices / float(max(1, count - 1))
        features = [
            normalized,
            relative,
            (indices == 0).to(torch.float32),
            (indices == count - 1).to(torch.float32),
        ]
        for harmonic in range(
            1,
            self.config.coordinate_harmonics + 1,
        ):
            angle = math.pi * harmonic * normalized
            features.extend((torch.sin(angle), torch.cos(angle)))
        return torch.stack(features, dim=-1)

    def encode_matrix(self, rows: Tensor) -> Tensor:
        if rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] < 1:
            raise CanonicalEnergyError(
                "rows must have shape [positive rows, positive columns]"
            )
        if rows.dtype not in (torch.int32, torch.int64):
            raise CanonicalEnergyError("rows must use integer dtype")
        if torch.any(rows < 0) or torch.any(rows >= FIELD_MODULUS):
            raise CanonicalEnergyError("matrix coefficients leave F_257")
        row_count, column_count = rows.shape
        if row_count > MAX_EXACT_ROWS or column_count > MAX_EXACT_COLUMNS:
            raise CanonicalEnergyError("matrix exceeds exact architecture bound")
        row_coordinates = self._coordinate_features(
            row_count,
            maximum=MAX_EXACT_ROWS,
            device=rows.device,
        )
        column_coordinates = self._coordinate_features(
            column_count,
            maximum=MAX_EXACT_COLUMNS,
            device=rows.device,
        )
        coordinates = torch.cat(
            (
                row_coordinates[:, None, :].expand(
                    row_count,
                    column_count,
                    -1,
                ),
                column_coordinates[None, :, :].expand(
                    row_count,
                    column_count,
                    -1,
                ),
            ),
            dim=-1,
        )
        cells = self.cell_encoder(
            torch.cat((self._field_features(rows), coordinates), dim=-1)
        )
        for layer in self.message_layers:
            cells = layer(cells)
        return cells

    def _frontier_features(
        self,
        matrix: tuple[tuple[int, ...], ...],
        action: CanonicalAction,
        witness: CanonicalEnergyWitness,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """Describe an action relative to the settled canonical frontier."""

        row_count = len(matrix)
        column_count = len(matrix[0])
        frontier = witness.settled_prefix
        earliest = 0
        source = 0
        pivot_is_unit = False
        if frontier < witness.rank:
            earliest_value = next(
                (
                    column
                    for column in range(column_count)
                    if any(matrix[row][column] for row in range(frontier, row_count))
                ),
                None,
            )
            if earliest_value is None:
                raise CanonicalEnergyError("unsettled rank has no remaining support")
            earliest = earliest_value
            source = next(
                row for row in range(frontier, row_count) if matrix[row][earliest]
            )
            pivot_is_unit = source == frontier and matrix[frontier][earliest] == 1
        active = action.kind != ACTION_HALT
        row_a = action.row_a if active else 0
        row_b = action.row_b if action.kind in (ACTION_ELIMINATE, ACTION_SWAP) else 0
        column = (
            action.column if action.kind in (ACTION_NORMALIZE, ACTION_ELIMINATE) else 0
        )
        swap_places_source = (
            action.kind == ACTION_SWAP
            and source != frontier
            and row_a == frontier
            and row_b == source
        )
        normalize_frontier = (
            action.kind == ACTION_NORMALIZE
            and source == frontier
            and matrix[frontier][earliest] not in (0, 1)
            and row_a == frontier
            and column == earliest
        )
        eliminate_frontier = (
            action.kind == ACTION_ELIMINATE
            and pivot_is_unit
            and row_b == frontier
            and column == earliest
        )
        frontier_admissible = (
            swap_places_source
            or normalize_frontier
            or eliminate_frontier
            or (action.kind == ACTION_HALT and witness.energy == 0)
        )
        values = (
            row_a / float(MAX_EXACT_ROWS),
            row_b / float(MAX_EXACT_ROWS),
            column / float(MAX_EXACT_COLUMNS),
            frontier / float(MAX_EXACT_ROWS),
            earliest / float(MAX_EXACT_COLUMNS),
            source / float(MAX_EXACT_ROWS),
            float(active and row_a == frontier),
            float(active and row_b == frontier),
            float(active and row_a == source),
            float(active and row_b == source),
            float(active and column == earliest),
            float(
                active
                and (
                    row_a < frontier
                    or (
                        action.kind in (ACTION_ELIMINATE, ACTION_SWAP)
                        and row_b < frontier
                    )
                )
            ),
            float(source != frontier),
            float(pivot_is_unit),
            float(swap_places_source),
            float(normalize_frontier),
            float(eliminate_frontier),
            float(frontier_admissible),
        )
        return torch.tensor(values, dtype=dtype, device=device)

    def _ablate_frontier_features(
        self,
        features: Tensor,
        *,
        frontier_ablation: str,
    ) -> Tensor:
        mode = _validate_frontier_ablation(frontier_ablation)
        if features.shape != (FRONTIER_FEATURE_COUNT,):
            raise CanonicalEnergyError("frontier feature width differs")
        if mode == FRONTIER_ZERO_ALL:
            return torch.zeros_like(features)
        if mode == FRONTIER_MASK_ACTION_CORRECTNESS:
            masked = features.clone()
            masked[FRONTIER_ACTION_CORRECTNESS_START:] = 0
            return masked
        return features

    def forward(
        self,
        rows: Tensor,
        actions: Sequence[CanonicalAction],
        *,
        energy_before: int,
        energy_after: Sequence[int],
        frontier_ablation: str = FRONTIER_FULL,
    ) -> Tensor:
        """Return bounded residuals for supplied legal transitions."""

        frozen_actions = tuple(actions)
        frozen_after = tuple(energy_after)
        if not frozen_actions or len(frozen_actions) != len(frozen_after):
            raise CanonicalEnergyError("actions and successor energies must align")
        cells = self.encode_matrix(rows)
        row_count, column_count, width = cells.shape
        row_state = cells.mean(dim=1)
        column_state = cells.mean(dim=0)
        global_state = cells.mean(dim=(0, 1))
        zeros = torch.zeros(width, dtype=cells.dtype, device=cells.device)
        witness = canonical_energy_witness(rows.detach().cpu().tolist())
        matrix = canonical_matrix(rows.detach().cpu().tolist())
        scale = float(
            max(
                1,
                witness.structural_weight * (row_count * column_count + row_count),
            )
        )
        encodings: list[Tensor] = []
        for action, after in zip(
            frozen_actions,
            frozen_after,
            strict=True,
        ):
            row_a = row_b = column = cell_a = cell_b = zeros
            if action.kind != ACTION_HALT:
                row_a = row_state[action.row_a]
                if action.kind in (ACTION_ELIMINATE, ACTION_SWAP):
                    row_b = row_state[action.row_b]
                if action.kind in (ACTION_NORMALIZE, ACTION_ELIMINATE):
                    column = column_state[action.column]
                    cell_a = cells[action.row_a, action.column]
                    if action.kind == ACTION_ELIMINATE:
                        cell_b = cells[action.row_b, action.column]
            action_type = self.action_type(
                torch.tensor(
                    ACTION_TYPES.index(action.kind),
                    dtype=torch.long,
                    device=cells.device,
                )
            )
            scalars = torch.tensor(
                [
                    energy_before / scale,
                    after / scale,
                    (energy_before - after) / scale,
                    witness.structural_defects
                    / max(1.0, float(row_count * column_count)),
                    witness.tail_nonzero / max(1.0, float(row_count * column_count)),
                    witness.rank / max(1.0, float(min(row_count, column_count))),
                    row_count / float(MAX_EXACT_ROWS),
                    column_count / float(MAX_EXACT_COLUMNS),
                ],
                dtype=cells.dtype,
                device=cells.device,
            )
            energy_state = self.energy_projection(scalars)
            frontier_state = self.frontier_projection(
                self._ablate_frontier_features(
                    self._frontier_features(
                        matrix,
                        action,
                        witness,
                        dtype=cells.dtype,
                        device=cells.device,
                    ),
                    frontier_ablation=frontier_ablation,
                )
            )
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
                        frontier_state,
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
        frontier_ablation: str = FRONTIER_FULL,
    ) -> ScoredCanonicalActions:
        mode = _validate_frontier_ablation(frontier_ablation)
        matrix = canonical_matrix(rows)
        transitions = evaluate_transitions(matrix)
        actions = tuple(item.action for item in transitions)
        before = transitions[0].energy_before
        after = tuple(item.energy_after for item in transitions)
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
        residual = (
            self(
                row_tensor,
                actions,
                energy_before=before,
                energy_after=after,
                frontier_ablation=mode,
            )
            if learned_residual
            else torch.zeros_like(explicit)
        )
        return ScoredCanonicalActions(
            actions=actions,
            energy_before=before,
            energy_after=after,
            explicit_reduction=explicit,
            learned_residual=residual,
            total_score=explicit + residual,
        )


@dataclass(frozen=True, slots=True)
class LabeledCanonicalState:
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
    """Preparation-only tie resolver subordinate to integer energy."""

    counter.calls += 1
    matrix = canonical_matrix(rows)
    transitions = evaluate_transitions(matrix)
    reductions = tuple(item.explicit_reduction for item in transitions)
    maximum = max(reductions)
    tied = [index for index, reduction in enumerate(reductions) if reduction == maximum]
    if len(tied) == 1 or canonical_defect_energy(matrix) == 0:
        return (tied[0],)
    reference = canonical_reference_schedule(matrix)
    if reference:
        for index in tied:
            if transitions[index].action == reference[0]:
                return (index,)
    chosen = min(
        tied,
        key=lambda index: (
            len(canonical_reference_schedule(transitions[index].rows)),
            transitions[index].action.canonical_data(),
        ),
    )
    return (chosen,)


def build_expert_states(
    matrices: Iterable[Iterable[Iterable[int]]],
    *,
    maximum_steps: int,
    counter: OracleCounter,
) -> tuple[LabeledCanonicalState, ...]:
    """Collect bounded expert trajectories for preparation only."""

    limit = _positive_int(maximum_steps, label="maximum_steps")
    result: dict[str, LabeledCanonicalState] = {}
    for raw in matrices:
        matrix = canonical_matrix(raw)
        visited: set[tuple[tuple[int, ...], ...]] = set()
        for _ in range(limit):
            targets = expert_action_indices(matrix, counter=counter)
            state = LabeledCanonicalState(matrix, targets)
            digest = matrix_sha256(matrix)
            prior = result.get(digest)
            if prior is not None and prior != state:
                raise CanonicalEnergyError("expert labels conflict")
            result[digest] = state
            actions = enumerate_legal_actions(matrix)
            action = actions[targets[0]]
            if action.kind == ACTION_HALT:
                break
            successor = apply_action(matrix, action)
            if successor in visited:
                # The canonical reference is an admissible escape if the
                # minimum-energy tie resolver entered a reversible plateau.
                reference = canonical_reference_schedule(matrix)
                if not reference:
                    break
                successor = apply_action(matrix, reference[0])
                if successor in visited:
                    break
            visited.add(matrix)
            matrix = successor
    return tuple(result[key] for key in sorted(result))


def make_random_label_control(
    states: Sequence[LabeledCanonicalState],
    *,
    seed: int,
) -> tuple[LabeledCanonicalState, ...]:
    """Randomize only within the same integer-energy action plateau."""

    rng = random.Random(seed)
    randomized: list[LabeledCanonicalState] = []
    for state in states:
        transitions = evaluate_transitions(state.rows)
        target_reduction = transitions[state.target_indices[0]].explicit_reduction
        alternatives = [
            index
            for index, transition in enumerate(transitions)
            if transition.explicit_reduction == target_reduction
            and index not in state.target_indices
        ]
        target = rng.choice(alternatives) if alternatives else state.target_indices[0]
        randomized.append(replace(state, target_indices=(target,)))
    return tuple(randomized)


def state_manifest(states: Iterable[LabeledCanonicalState]) -> str:
    return sha256(
        (
            "\n".join(
                state.sha256 for state in sorted(states, key=lambda item: item.sha256)
            )
            + "\n"
        ).encode("ascii")
    ).hexdigest()


def _listwise_loss(
    controller: CanonicalEnergyController,
    state: LabeledCanonicalState,
) -> Tensor:
    scored = controller.score_actions(state.rows)
    target = torch.tensor(
        state.target_indices,
        dtype=torch.long,
        device=scored.total_score.device,
    )
    return torch.logsumexp(scored.total_score, dim=0) - torch.logsumexp(
        scored.total_score[target], dim=0
    )


def train_controller(
    controller: CanonicalEnergyController,
    states: Sequence[LabeledCanonicalState],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    shuffle_seed: int,
    maximum_updates: int,
) -> int:
    """Run a deterministic bounded residual fit."""

    epoch_count = _positive_int(epochs, label="epochs")
    batch = _positive_int(batch_size, label="batch_size")
    update_limit = _positive_int(
        maximum_updates,
        label="maximum_updates",
    )
    if not states:
        raise CanonicalEnergyError("training states must be nonempty")
    if not isinstance(learning_rate, float) or learning_rate <= 0.0:
        raise CanonicalEnergyError("learning_rate must be positive")
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
    controller: CanonicalEnergyController,
    states: Sequence[LabeledCanonicalState],
) -> float:
    controller.eval()
    if not states:
        return 0.0
    correct = sum(
        int(
            int(controller.score_actions(state.rows).total_score.argmax().item())
            in state.target_indices
        )
        for state in states
    )
    return correct / len(states)


def model_sha256(model: CanonicalEnergyController) -> str:
    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CanonicalRolloutResult:
    strict_canonical_certified: bool
    invalid: bool
    overlong: bool
    actions: tuple[CanonicalAction, ...]
    initial_energy: int
    final_energy: int
    oracle_calls: int


@torch.no_grad()
def final_oracle_free_rollout(
    controller: CanonicalEnergyController,
    rows: Iterable[Iterable[int]],
    *,
    maximum_steps: int,
    learned_residual: bool = True,
    frontier_ablation: str = FRONTIER_FULL,
) -> CanonicalRolloutResult:
    """Run matrix-only dynamics; the final path contains no oracle call."""

    limit = _positive_int(maximum_steps, label="maximum_steps")
    mode = _validate_frontier_ablation(frontier_ablation)
    source = canonical_matrix(rows)
    matrix = source
    initial_energy = canonical_defect_energy(source)
    emitted: list[CanonicalAction] = []
    visited: set[tuple[tuple[int, ...], ...]] = set()
    controller.eval()
    for _ in range(limit):
        scored = controller.score_actions(
            matrix,
            learned_residual=learned_residual,
            frontier_ablation=mode,
        )
        choice = int(scored.total_score.argmax().item())
        action = scored.actions[choice]
        if action.kind == ACTION_HALT:
            actions = tuple((*emitted, action))
            certified = strictly_verify_action_trace(source, actions)
            return CanonicalRolloutResult(
                strict_canonical_certified=certified,
                invalid=not certified,
                overlong=False,
                actions=actions,
                initial_energy=initial_energy,
                final_energy=canonical_defect_energy(matrix),
                oracle_calls=0,
            )
        if matrix in visited:
            return CanonicalRolloutResult(
                strict_canonical_certified=False,
                invalid=True,
                overlong=False,
                actions=tuple(emitted),
                initial_energy=initial_energy,
                final_energy=canonical_defect_energy(matrix),
                oracle_calls=0,
            )
        visited.add(matrix)
        matrix = apply_action(matrix, action)
        emitted.append(action)
    return CanonicalRolloutResult(
        strict_canonical_certified=False,
        invalid=False,
        overlong=True,
        actions=tuple(emitted),
        initial_energy=initial_energy,
        final_energy=canonical_defect_energy(matrix),
        oracle_calls=0,
    )


@dataclass(frozen=True, slots=True)
class CanonicalRolloutSummary:
    strict_canonical_certified: int
    invalid: int
    overlong: int
    oracle_calls: int

    @property
    def total(self) -> int:
        return self.strict_canonical_certified + self.invalid + self.overlong


def evaluate_matrices_oracle_free(
    controller: CanonicalEnergyController,
    matrices: Sequence[tuple[tuple[int, ...], ...]],
    *,
    maximum_steps: int,
    learned_residual: bool = True,
    frontier_ablation: str = FRONTIER_FULL,
) -> CanonicalRolloutSummary:
    mode = _validate_frontier_ablation(frontier_ablation)
    certified = invalid = overlong = oracle_calls = 0
    for matrix in matrices:
        result = final_oracle_free_rollout(
            controller,
            matrix,
            maximum_steps=maximum_steps,
            learned_residual=learned_residual,
            frontier_ablation=mode,
        )
        certified += result.strict_canonical_certified
        invalid += result.invalid
        overlong += result.overlong
        oracle_calls += result.oracle_calls
    return CanonicalRolloutSummary(
        strict_canonical_certified=certified,
        invalid=invalid,
        overlong=overlong,
        oracle_calls=oracle_calls,
    )


def fixed_schedule_strict_rollout(
    rows: Iterable[Iterable[int]],
    *,
    maximum_steps: int,
) -> CanonicalRolloutResult:
    """Run the deterministic host Gauss-Jordan baseline and strict assessor."""

    limit = _positive_int(maximum_steps, label="maximum_steps")
    source = canonical_matrix(rows)
    initial_energy = canonical_defect_energy(source)
    schedule = canonical_reference_schedule(source)
    actions = tuple((*schedule, CanonicalAction(ACTION_HALT)))
    if len(actions) > limit:
        return CanonicalRolloutResult(
            strict_canonical_certified=False,
            invalid=False,
            overlong=True,
            actions=tuple(schedule[:limit]),
            initial_energy=initial_energy,
            final_energy=initial_energy,
            oracle_calls=0,
        )
    certified = strictly_verify_action_trace(source, actions)
    return CanonicalRolloutResult(
        strict_canonical_certified=certified,
        invalid=not certified,
        overlong=False,
        actions=actions,
        initial_energy=initial_energy,
        final_energy=0 if certified else initial_energy,
        oracle_calls=0,
    )


def evaluate_fixed_schedule_strict(
    matrices: Sequence[tuple[tuple[int, ...], ...]],
    *,
    maximum_steps: int,
) -> CanonicalRolloutSummary:
    """Evaluate the fixed host algorithm with the same strict verifier."""

    certified = invalid = overlong = 0
    for matrix in matrices:
        result = fixed_schedule_strict_rollout(
            matrix,
            maximum_steps=maximum_steps,
        )
        certified += result.strict_canonical_certified
        invalid += result.invalid
        overlong += result.overlong
    return CanonicalRolloutSummary(
        strict_canonical_certified=certified,
        invalid=invalid,
        overlong=overlong,
        oracle_calls=0,
    )


def _measure_matrix_policy_arm(
    controller: CanonicalEnergyController,
    matrices: Sequence[tuple[tuple[int, ...], ...]],
    *,
    maximum_steps: int,
    learned_residual: bool,
    frontier_ablation: str = FRONTIER_FULL,
) -> tuple[CanonicalRolloutSummary, MechanicsResourceCounts]:
    clear_mechanics_caches()
    with mechanics_resource_accounting() as resources:
        summary = evaluate_matrices_oracle_free(
            controller,
            matrices,
            maximum_steps=maximum_steps,
            learned_residual=learned_residual,
            frontier_ablation=frontier_ablation,
        )
    return summary, resources.freeze()


def _measure_fixed_schedule_arm(
    matrices: Sequence[tuple[tuple[int, ...], ...]],
    *,
    maximum_steps: int,
) -> tuple[CanonicalRolloutSummary, MechanicsResourceCounts]:
    clear_mechanics_caches()
    with mechanics_resource_accounting() as resources:
        summary = evaluate_fixed_schedule_strict(
            matrices,
            maximum_steps=maximum_steps,
        )
    return summary, resources.freeze()


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
    """Generate deterministic sparse matrices with no hidden traces."""

    target = _positive_int(count, label="count")
    min_rows = _positive_int(minimum_rows, label="minimum_rows")
    max_rows = _positive_int(maximum_rows, label="maximum_rows")
    min_columns = _positive_int(
        minimum_columns,
        label="minimum_columns",
    )
    max_columns = _positive_int(
        maximum_columns,
        label="maximum_columns",
    )
    if min_rows > max_rows or min_columns > max_columns:
        raise CanonicalEnergyError("generation bounds are inverted")
    if max_rows > MAX_EXACT_ROWS or max_columns > MAX_EXACT_COLUMNS:
        raise CanonicalEnergyError("generation exceeds exact energy bounds")
    rng = random.Random(seed)
    seen = set() if excluded is None else set(excluded)
    result: list[tuple[tuple[int, ...], ...]] = []
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
        raise CanonicalEnergyError("matrix generator exhausted its bound")
    return tuple(result)


def matrix_manifest(
    matrices: Iterable[Iterable[Iterable[int]]],
) -> str:
    return sha256(
        ("\n".join(matrix_sha256(matrix) for matrix in matrices) + "\n").encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class CanonicalExperimentConfig:
    seed: int = 20260724
    train_matrices: int = 64
    evaluation_matrices: int = 32
    train_maximum_rows: int = 3
    train_maximum_columns: int = 4
    evaluation_minimum_rows: int = 4
    evaluation_maximum_rows: int = 4
    evaluation_minimum_columns: int = 5
    evaluation_maximum_columns: int = 6
    maximum_expert_steps: int = 128
    maximum_rollout_steps: int = 256
    epochs: int = 12
    batch_size: int = 8
    learning_rate: float = 1e-3
    maximum_updates: int = 2_048
    controller: CanonicalControllerConfig = CanonicalControllerConfig()

    def __post_init__(self) -> None:
        for label, value in (
            ("train_matrices", self.train_matrices),
            ("evaluation_matrices", self.evaluation_matrices),
            ("train_maximum_rows", self.train_maximum_rows),
            ("train_maximum_columns", self.train_maximum_columns),
            ("maximum_expert_steps", self.maximum_expert_steps),
            ("maximum_rollout_steps", self.maximum_rollout_steps),
            ("epochs", self.epochs),
            ("batch_size", self.batch_size),
            ("maximum_updates", self.maximum_updates),
        ):
            _positive_int(value, label=label)
        if self.train_maximum_rows > 3 or self.train_maximum_columns > 4:
            raise CanonicalEnergyError("training geometry must be at most 3x4")
        if (
            self.evaluation_minimum_rows != 4
            or self.evaluation_maximum_rows != 4
            or self.evaluation_minimum_columns != 5
            or self.evaluation_maximum_columns != 6
        ):
            raise CanonicalEnergyError("evaluation geometry must be exactly 4x5-6")
        if not isinstance(self.learning_rate, float) or self.learning_rate <= 0:
            raise CanonicalEnergyError("learning_rate must be positive")


@dataclass(frozen=True, slots=True)
class CanonicalSeedReport:
    schema: str
    status: str
    ablation_collapse_rule: str
    learned_claim_downgrade_rule: str
    learned_claim_downgraded: bool
    fixed_schedule_reaches_ceiling: bool
    zero_frontier_ablation_collapsed: bool
    action_correctness_ablation_collapsed: bool
    seed: int
    controller_parameters: int
    parameter_count_breakdown: Mapping[str, int]
    train_matrices: int
    train_states: int
    evaluation_matrices: int
    train_maximum_rows: int
    train_maximum_columns: int
    evaluation_rows: int
    evaluation_minimum_columns: int
    evaluation_maximum_columns: int
    preparation_oracle_calls: int
    final_rollout_oracle_calls: int
    expert_optimizer_updates: int
    random_optimizer_updates: int
    expert_train_label_accuracy: float
    random_train_label_accuracy: float
    random_true_expert_accuracy: float
    energy_only_strict_canonical_certified: int
    expert_strict_canonical_certified: int
    expert_zero_frontier_strict_canonical_certified: int
    expert_masked_action_bits_strict_canonical_certified: int
    random_strict_canonical_certified: int
    fixed_schedule_strict_canonical_certified: int
    energy_only_invalid: int
    expert_invalid: int
    expert_zero_frontier_invalid: int
    expert_masked_action_bits_invalid: int
    random_invalid: int
    fixed_schedule_invalid: int
    energy_only_overlong: int
    expert_overlong: int
    expert_zero_frontier_overlong: int
    expert_masked_action_bits_overlong: int
    random_overlong: int
    fixed_schedule_overlong: int
    preparation_training_resources: MechanicsResourceCounts
    energy_only_resources: MechanicsResourceCounts
    expert_full_resources: MechanicsResourceCounts
    expert_zero_frontier_resources: MechanicsResourceCounts
    expert_masked_action_bits_resources: MechanicsResourceCounts
    random_resources: MechanicsResourceCounts
    fixed_schedule_resources: MechanicsResourceCounts
    train_manifest_sha256: str
    evaluation_manifest_sha256: str
    expert_state_manifest_sha256: str
    random_state_manifest_sha256: str
    expert_model_sha256: str
    random_model_sha256: str


def _identical_controllers(
    config: CanonicalControllerConfig,
    *,
    seed: int,
) -> tuple[CanonicalEnergyController, CanonicalEnergyController]:
    torch.manual_seed(seed)
    expert = CanonicalEnergyController(config)
    random_model = CanonicalEnergyController(config)
    random_model.load_state_dict(expert.state_dict())
    return expert, random_model


def run_seed_experiment(
    config: CanonicalExperimentConfig,
) -> CanonicalSeedReport:
    """Run one bounded expert/energy/random-control experiment."""

    if not isinstance(config, CanonicalExperimentConfig):
        raise CanonicalEnergyError("experiment config has the wrong type")
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
        minimum_rows=4,
        maximum_rows=4,
        minimum_columns=5,
        maximum_columns=6,
        excluded=set(train),
    )
    clear_mechanics_caches()
    with mechanics_resource_accounting() as preparation_counter:
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
        expert_model, random_model = _identical_controllers(
            config.controller,
            seed=config.seed + 3,
        )
        expert_updates = train_controller(
            expert_model,
            expert_states,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            shuffle_seed=config.seed + 4,
            maximum_updates=config.maximum_updates,
        )
        random_updates = train_controller(
            random_model,
            random_states,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            shuffle_seed=config.seed + 4,
            maximum_updates=config.maximum_updates,
        )
        expert_train_accuracy = label_accuracy(
            expert_model,
            expert_states,
        )
        random_train_accuracy = label_accuracy(
            random_model,
            random_states,
        )
        random_true_expert_accuracy = label_accuracy(
            random_model,
            expert_states,
        )
    preparation_resources = preparation_counter.freeze()

    energy_only, energy_only_resources = _measure_matrix_policy_arm(
        expert_model,
        evaluation,
        maximum_steps=config.maximum_rollout_steps,
        learned_residual=False,
    )
    expert_result, expert_resources = _measure_matrix_policy_arm(
        expert_model,
        evaluation,
        maximum_steps=config.maximum_rollout_steps,
        learned_residual=True,
        frontier_ablation=FRONTIER_FULL,
    )
    zero_frontier, zero_frontier_resources = _measure_matrix_policy_arm(
        expert_model,
        evaluation,
        maximum_steps=config.maximum_rollout_steps,
        learned_residual=True,
        frontier_ablation=FRONTIER_ZERO_ALL,
    )
    masked_action_bits, masked_action_bits_resources = _measure_matrix_policy_arm(
        expert_model,
        evaluation,
        maximum_steps=config.maximum_rollout_steps,
        learned_residual=True,
        frontier_ablation=FRONTIER_MASK_ACTION_CORRECTNESS,
    )
    random_result, random_resources = _measure_matrix_policy_arm(
        random_model,
        evaluation,
        maximum_steps=config.maximum_rollout_steps,
        learned_residual=True,
        frontier_ablation=FRONTIER_FULL,
    )
    fixed_schedule, fixed_schedule_resources = _measure_fixed_schedule_arm(
        evaluation,
        maximum_steps=config.maximum_rollout_steps,
    )
    final_calls = (
        energy_only.oracle_calls
        + expert_result.oracle_calls
        + zero_frontier.oracle_calls
        + masked_action_bits.oracle_calls
        + random_result.oracle_calls
        + fixed_schedule.oracle_calls
    )
    if final_calls != 0:
        raise CanonicalEnergyError("oracle leaked into final rollout")
    fixed_schedule_reaches_ceiling = fixed_schedule.strict_canonical_certified == len(
        evaluation
    )
    zero_frontier_collapsed = _ablation_collapsed(
        ablated=zero_frontier.strict_canonical_certified,
        full=expert_result.strict_canonical_certified,
    )
    action_bits_collapsed = _ablation_collapsed(
        ablated=masked_action_bits.strict_canonical_certified,
        full=expert_result.strict_canonical_certified,
    )
    interpretation = (
        HYBRID_INTERPRETATION
        if (
            fixed_schedule_reaches_ceiling
            or zero_frontier_collapsed
            or action_bits_collapsed
        )
        else LEARNED_INTERPRETATION
    )
    return CanonicalSeedReport(
        schema=SEED_REPORT_SCHEMA,
        status=interpretation,
        ablation_collapse_rule=ABLATION_COLLAPSE_RULE,
        learned_claim_downgrade_rule=LEARNED_CLAIM_DOWNGRADE_RULE,
        learned_claim_downgraded=interpretation == HYBRID_INTERPRETATION,
        fixed_schedule_reaches_ceiling=fixed_schedule_reaches_ceiling,
        zero_frontier_ablation_collapsed=zero_frontier_collapsed,
        action_correctness_ablation_collapsed=action_bits_collapsed,
        seed=config.seed,
        controller_parameters=expert_model.parameter_count,
        parameter_count_breakdown=expert_model.parameter_count_breakdown(),
        train_matrices=len(train),
        train_states=len(expert_states),
        evaluation_matrices=len(evaluation),
        train_maximum_rows=config.train_maximum_rows,
        train_maximum_columns=config.train_maximum_columns,
        evaluation_rows=4,
        evaluation_minimum_columns=5,
        evaluation_maximum_columns=6,
        preparation_oracle_calls=oracle_counter.calls,
        final_rollout_oracle_calls=final_calls,
        expert_optimizer_updates=expert_updates,
        random_optimizer_updates=random_updates,
        expert_train_label_accuracy=expert_train_accuracy,
        random_train_label_accuracy=random_train_accuracy,
        random_true_expert_accuracy=random_true_expert_accuracy,
        energy_only_strict_canonical_certified=(energy_only.strict_canonical_certified),
        expert_strict_canonical_certified=(expert_result.strict_canonical_certified),
        expert_zero_frontier_strict_canonical_certified=(
            zero_frontier.strict_canonical_certified
        ),
        expert_masked_action_bits_strict_canonical_certified=(
            masked_action_bits.strict_canonical_certified
        ),
        random_strict_canonical_certified=(random_result.strict_canonical_certified),
        fixed_schedule_strict_canonical_certified=(
            fixed_schedule.strict_canonical_certified
        ),
        energy_only_invalid=energy_only.invalid,
        expert_invalid=expert_result.invalid,
        expert_zero_frontier_invalid=zero_frontier.invalid,
        expert_masked_action_bits_invalid=masked_action_bits.invalid,
        random_invalid=random_result.invalid,
        fixed_schedule_invalid=fixed_schedule.invalid,
        energy_only_overlong=energy_only.overlong,
        expert_overlong=expert_result.overlong,
        expert_zero_frontier_overlong=zero_frontier.overlong,
        expert_masked_action_bits_overlong=masked_action_bits.overlong,
        random_overlong=random_result.overlong,
        fixed_schedule_overlong=fixed_schedule.overlong,
        preparation_training_resources=preparation_resources,
        energy_only_resources=energy_only_resources,
        expert_full_resources=expert_resources,
        expert_zero_frontier_resources=zero_frontier_resources,
        expert_masked_action_bits_resources=masked_action_bits_resources,
        random_resources=random_resources,
        fixed_schedule_resources=fixed_schedule_resources,
        train_manifest_sha256=matrix_manifest(train),
        evaluation_manifest_sha256=matrix_manifest(evaluation),
        expert_state_manifest_sha256=state_manifest(expert_states),
        random_state_manifest_sha256=state_manifest(random_states),
        expert_model_sha256=model_sha256(expert_model),
        random_model_sha256=model_sha256(random_model),
    )


@dataclass(frozen=True, slots=True)
class CanonicalMultiseedReport:
    schema: str
    status: str
    ablation_collapse_rule: str
    learned_claim_downgrade_rule: str
    learned_claim_downgraded: bool
    fixed_schedule_reaches_ceiling: bool
    zero_frontier_ablation_collapsed: bool
    action_correctness_ablation_collapsed: bool
    fixed_schedule_ceiling_seed_count: int
    zero_frontier_ablation_collapsed_seed_count: int
    action_correctness_ablation_collapsed_seed_count: int
    seeds: tuple[int, ...]
    seed_reports: tuple[CanonicalSeedReport, ...]
    evaluation_cases_per_arm: int
    final_rollout_oracle_calls: int
    energy_only_strict_canonical_certified: int
    expert_strict_canonical_certified: int
    expert_zero_frontier_strict_canonical_certified: int
    expert_masked_action_bits_strict_canonical_certified: int
    random_strict_canonical_certified: int
    fixed_schedule_strict_canonical_certified: int
    preparation_training_resources: MechanicsResourceCounts
    energy_only_resources: MechanicsResourceCounts
    expert_full_resources: MechanicsResourceCounts
    expert_zero_frontier_resources: MechanicsResourceCounts
    expert_masked_action_bits_resources: MechanicsResourceCounts
    random_resources: MechanicsResourceCounts
    fixed_schedule_resources: MechanicsResourceCounts

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                asdict(self),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
            + b"\n"
        )


def run_multiseed_experiment(
    config: CanonicalExperimentConfig = CanonicalExperimentConfig(),
    *,
    seeds: Sequence[int] = (20260724, 20260725, 20260726),
) -> CanonicalMultiseedReport:
    """Report all six audited arms over at least three independent seeds."""

    frozen_seeds = tuple(int(seed) for seed in seeds)
    if len(frozen_seeds) < 3 or len(set(frozen_seeds)) != len(frozen_seeds):
        raise CanonicalEnergyError(
            "multiseed report requires at least three distinct seeds"
        )
    reports = tuple(
        run_seed_experiment(replace(config, seed=seed)) for seed in frozen_seeds
    )
    final_calls = sum(report.final_rollout_oracle_calls for report in reports)
    if final_calls != 0:
        raise CanonicalEnergyError("oracle leaked into multiseed rollout")
    evaluation_cases = sum(report.evaluation_matrices for report in reports)
    energy_certified = sum(
        report.energy_only_strict_canonical_certified for report in reports
    )
    expert_certified = sum(
        report.expert_strict_canonical_certified for report in reports
    )
    zero_frontier_certified = sum(
        report.expert_zero_frontier_strict_canonical_certified for report in reports
    )
    masked_action_bits_certified = sum(
        report.expert_masked_action_bits_strict_canonical_certified
        for report in reports
    )
    random_certified = sum(
        report.random_strict_canonical_certified for report in reports
    )
    fixed_certified = sum(
        report.fixed_schedule_strict_canonical_certified for report in reports
    )
    fixed_schedule_reaches_ceiling = fixed_certified == evaluation_cases
    zero_frontier_collapsed = _ablation_collapsed(
        ablated=zero_frontier_certified,
        full=expert_certified,
    )
    action_bits_collapsed = _ablation_collapsed(
        ablated=masked_action_bits_certified,
        full=expert_certified,
    )
    learned_claim_downgraded = (
        fixed_schedule_reaches_ceiling
        or zero_frontier_collapsed
        or action_bits_collapsed
    )

    def summed_resources(attribute: str) -> MechanicsResourceCounts:
        total = MechanicsResourceCounts()
        for report in reports:
            value = getattr(report, attribute)
            if not isinstance(value, MechanicsResourceCounts):
                raise CanonicalEnergyError("resource report has the wrong type")
            total += value
        return total

    return CanonicalMultiseedReport(
        schema=MULTISEED_REPORT_SCHEMA,
        status=(
            HYBRID_INTERPRETATION
            if learned_claim_downgraded
            else LEARNED_INTERPRETATION
        ),
        ablation_collapse_rule=ABLATION_COLLAPSE_RULE,
        learned_claim_downgrade_rule=LEARNED_CLAIM_DOWNGRADE_RULE,
        learned_claim_downgraded=learned_claim_downgraded,
        fixed_schedule_reaches_ceiling=fixed_schedule_reaches_ceiling,
        zero_frontier_ablation_collapsed=zero_frontier_collapsed,
        action_correctness_ablation_collapsed=action_bits_collapsed,
        fixed_schedule_ceiling_seed_count=sum(
            report.fixed_schedule_reaches_ceiling for report in reports
        ),
        zero_frontier_ablation_collapsed_seed_count=sum(
            report.zero_frontier_ablation_collapsed for report in reports
        ),
        action_correctness_ablation_collapsed_seed_count=sum(
            report.action_correctness_ablation_collapsed for report in reports
        ),
        seeds=frozen_seeds,
        seed_reports=reports,
        evaluation_cases_per_arm=evaluation_cases,
        final_rollout_oracle_calls=final_calls,
        energy_only_strict_canonical_certified=energy_certified,
        expert_strict_canonical_certified=expert_certified,
        expert_zero_frontier_strict_canonical_certified=zero_frontier_certified,
        expert_masked_action_bits_strict_canonical_certified=(
            masked_action_bits_certified
        ),
        random_strict_canonical_certified=random_certified,
        fixed_schedule_strict_canonical_certified=fixed_certified,
        preparation_training_resources=summed_resources(
            "preparation_training_resources"
        ),
        energy_only_resources=summed_resources("energy_only_resources"),
        expert_full_resources=summed_resources("expert_full_resources"),
        expert_zero_frontier_resources=summed_resources(
            "expert_zero_frontier_resources"
        ),
        expert_masked_action_bits_resources=summed_resources(
            "expert_masked_action_bits_resources"
        ),
        random_resources=summed_resources("random_resources"),
        fixed_schedule_resources=summed_resources("fixed_schedule_resources"),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=(20260724, 20260725, 20260726),
    )
    parser.add_argument("--train-matrices", type=int, default=64)
    parser.add_argument("--evaluation-matrices", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--maximum-updates", type=int, default=2_048)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--message-layers", type=int, default=3)
    parser.add_argument("--residual-hidden", type=int, default=128)
    parser.add_argument("--field-harmonics", type=int, default=4)
    parser.add_argument("--coordinate-harmonics", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = CanonicalExperimentConfig(
        train_matrices=args.train_matrices,
        evaluation_matrices=args.evaluation_matrices,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        maximum_updates=args.maximum_updates,
        controller=CanonicalControllerConfig(
            width=args.width,
            message_layers=args.message_layers,
            residual_hidden=args.residual_hidden,
            field_harmonics=args.field_harmonics,
            coordinate_harmonics=args.coordinate_harmonics,
        ),
    )
    report = run_multiseed_experiment(config, seeds=args.seeds)
    payload = report.canonical_bytes()
    if args.output is None:
        print(payload.decode("ascii"), end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)


if __name__ == "__main__":
    main()
