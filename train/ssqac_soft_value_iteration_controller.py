#!/usr/bin/env python3
"""Fixed-compute soft value iteration controller for SSQAC row reduction.

This module is an isolated, falsifiable mechanics experiment.  The deployed
candidate receives only:

* the current matrix over F_257; and
* deterministic features for the currently legal local macro actions.

Each macro action is a node in a variable-size interaction graph.  A single
shared neural backup cell repeatedly exchanges soft value messages between
locally related actions.  After a fixed number of internal iterations, the
controller emits exactly one hard macro action.  There is no beam, frontier,
host search, verifier score, preparation oracle, source, query, or workspace
inside the candidate runtime.

The four macros are NORMALIZE, ELIMINATE, SWAP, and HALT.  HALT is always
available, including in nonterminal states, so action availability cannot leak
the canonical-RREF endpoint.  A separate assessor verifies a completed trace
only after candidate rollout has ended.

Preparation traces may use a deterministic canonical-RREF scheduler.  Final
evaluation uses strictly larger, disjoint matrix geometries and reports random
label and zero-internal-iteration controls.  Even a positive result is a
mechanics result, not evidence of general or native reasoning.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
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
    AlgebraInstruction,
    AlgebraMachineError,
    execute_program,
    verify_reduction_program,
)


ARCHITECTURE_SCHEMA = "ssqac_soft_value_iteration_controller_v1"
EXPERIMENT_SCHEMA = "ssqac_soft_value_iteration_experiment_v1"
HOSTILE_AUDIT_SCHEMA = "ssqac_soft_value_iteration_hostile_audit_v1"
STATUS = "internal_planning_mechanics_falsifier_only_not_reasoning"
HOSTILE_AUDIT_STATUS = "hostile_feature_audit_only_not_reasoning"
HOSTILE_AUDIT_OUTCOME = "causal_feature_audit_completed_no_reasoning_claim"
OUTCOME_INCONCLUSIVE = "falsified_or_inconclusive"
OUTCOME_MATERIAL = "material_mechanics_gate_passed_replication_required"

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

MAX_ROWS = 32
MAX_COLUMNS = 32
DEFAULT_REGISTER_COUNT = 4
PROTECTED_FLAGSHIP_PARAMETERS = 125_081_664
TOTAL_PARAMETER_BUDGET = 200_000_000
ACTION_SCALAR_FEATURES = 20
MINIMAL_ACTION_SCALAR_FEATURES = 7
PAIR_RELATION_FEATURES = 12
FIELD_BASE_FEATURES = 5


class SoftValueIterationError(ValueError):
    """The soft value-iteration controller contract failed closed."""


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SoftValueIterationError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SoftValueIterationError(f"{label} must be a nonnegative integer")
    return value


def canonical_matrix(
    rows: Iterable[Iterable[int]],
) -> tuple[tuple[int, ...], ...]:
    """Freeze a bounded, nonempty rectangular matrix over F_257."""

    matrix = tuple(tuple(int(value) % FIELD_MODULUS for value in row) for row in rows)
    if not matrix or not matrix[0]:
        raise SoftValueIterationError("matrix must be nonempty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise SoftValueIterationError("matrix rows have inconsistent widths")
    if len(matrix) > MAX_ROWS or width > MAX_COLUMNS:
        raise SoftValueIterationError(
            f"matrix exceeds the {MAX_ROWS}x{MAX_COLUMNS} mechanics bound"
        )
    return matrix


def matrix_sha256(rows: Iterable[Iterable[int]]) -> str:
    matrix = canonical_matrix(rows)
    return sha256(json.dumps(matrix, separators=(",", ":")).encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True, order=True)
class MacroAction:
    """One hard local row-reduction macro."""

    kind: str
    row_a: int = 0
    row_b: int = 0
    column: int = 0

    def __post_init__(self) -> None:
        if self.kind not in ACTION_TYPES:
            raise SoftValueIterationError(f"unknown macro action {self.kind!r}")
        for label, value in (
            ("row_a", self.row_a),
            ("row_b", self.row_b),
            ("column", self.column),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise SoftValueIterationError(f"{label} must be an integer")

    def canonical_data(self) -> list[object]:
        return [self.kind, self.row_a, self.row_b, self.column]


def _macro_action_rows(action: MacroAction) -> frozenset[int]:
    if action.kind == ACTION_HALT:
        return frozenset()
    if action.kind in (ACTION_ELIMINATE, ACTION_SWAP):
        return frozenset((action.row_a, action.row_b))
    return frozenset((action.row_a,))


def _pair_relation_values(
    left_index: int,
    left: MacroAction,
    right_index: int,
    right: MacroAction,
) -> tuple[float, ...]:
    left_rows = _macro_action_rows(left)
    right_rows = _macro_action_rows(right)
    same_column = (
        left.kind not in (ACTION_HALT, ACTION_SWAP)
        and right.kind not in (ACTION_HALT, ACTION_SWAP)
        and left.column == right.column
    )
    shares_row = bool(left_rows & right_rows)
    causal_normalize = (
        left.kind == ACTION_NORMALIZE
        and right.kind == ACTION_ELIMINATE
        and left.row_a == right.row_b
        and left.column == right.column
    )
    causal_eliminate = (
        left.kind == ACTION_ELIMINATE
        and right.kind == ACTION_ELIMINATE
        and left.row_b == right.row_b
        and left.column == right.column
    )
    return (
        float(left_index == right_index),
        float(left.row_a == right.row_a),
        float(left.row_b == right.row_b),
        float(same_column),
        float(left.row_a == right.row_b),
        float(left.row_b == right.row_a),
        float(shares_row),
        float(left.kind == ACTION_HALT),
        float(right.kind == ACTION_HALT),
        float(causal_normalize),
        float(causal_eliminate),
        float(left.kind == right.kind),
    )


def _message_neighbor(
    left_index: int,
    left: MacroAction,
    right_index: int,
    right: MacroAction,
) -> bool:
    if left_index == right_index:
        return True
    left_rows = _macro_action_rows(left)
    right_rows = _macro_action_rows(right)
    same_column = (
        left.kind not in (ACTION_HALT, ACTION_SWAP)
        and right.kind not in (ACTION_HALT, ACTION_SWAP)
        and left.column == right.column
    )
    return bool(
        left_rows & right_rows
        or same_column
        or left.kind == ACTION_HALT
        or right.kind == ACTION_HALT
    )


def enumerate_legal_macro_actions(
    rows: Iterable[Iterable[int]],
) -> tuple[MacroAction, ...]:
    """Enumerate local macros without testing whether the state is solved."""

    matrix = canonical_matrix(rows)
    row_count = len(matrix)
    column_count = len(matrix[0])
    actions: list[MacroAction] = []
    for row in range(row_count):
        for column in range(column_count):
            if matrix[row][column] not in (0, 1):
                actions.append(
                    MacroAction(
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
                        MacroAction(
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
                    MacroAction(
                        ACTION_SWAP,
                        row_a=left,
                        row_b=right,
                    )
                )
    # HALT is deliberately unconditional.  The candidate must decide when the
    # endpoint is valid; legality cannot reveal the assessor's answer.
    actions.append(MacroAction(ACTION_HALT))
    return tuple(actions)


def representative_recode_matrix(
    rows: Iterable[Iterable[int]],
    *,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    """Change integer representatives without changing the F_257 matrix."""

    matrix = canonical_matrix(rows)
    rng = random.Random(seed)
    recoded = tuple(
        tuple(
            value + FIELD_MODULUS * rng.choice((-7, -3, -1, 1, 4, 9)) for value in row
        )
        for row in matrix
    )
    if canonical_matrix(recoded) != matrix:
        raise SoftValueIterationError("field representative recoding drifted")
    return recoded


def _validated_permutation(
    order: Sequence[int],
    *,
    count: int,
    label: str,
) -> tuple[int, ...]:
    frozen = tuple(order)
    if len(frozen) != count or set(frozen) != set(range(count)):
        raise SoftValueIterationError(f"{label} is not a permutation")
    return frozen


def permute_matrix(
    rows: Iterable[Iterable[int]],
    *,
    row_order: Sequence[int],
    column_order: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    """Apply an explicit row/column recoding to one matrix."""

    matrix = canonical_matrix(rows)
    row_permutation = _validated_permutation(
        row_order,
        count=len(matrix),
        label="row_order",
    )
    column_permutation = _validated_permutation(
        column_order,
        count=len(matrix[0]),
        label="column_order",
    )
    return tuple(
        tuple(matrix[old_row][old_column] for old_column in column_permutation)
        for old_row in row_permutation
    )


def remap_action_under_permutation(
    action: MacroAction,
    *,
    row_order: Sequence[int],
    column_order: Sequence[int],
) -> MacroAction:
    """Map an action from old coordinates into permuted coordinates."""

    row_permutation = _validated_permutation(
        row_order,
        count=len(row_order),
        label="row_order",
    )
    column_permutation = _validated_permutation(
        column_order,
        count=len(column_order),
        label="column_order",
    )
    old_to_new_row = {old: new for new, old in enumerate(row_permutation)}
    old_to_new_column = {old: new for new, old in enumerate(column_permutation)}
    if action.kind == ACTION_HALT:
        return action
    row_a = old_to_new_row[action.row_a]
    if action.kind == ACTION_NORMALIZE:
        return MacroAction(
            ACTION_NORMALIZE,
            row_a=row_a,
            column=old_to_new_column[action.column],
        )
    row_b = old_to_new_row[action.row_b]
    if action.kind == ACTION_ELIMINATE:
        return MacroAction(
            ACTION_ELIMINATE,
            row_a=row_a,
            row_b=row_b,
            column=old_to_new_column[action.column],
        )
    left, right = sorted((row_a, row_b))
    return MacroAction(ACTION_SWAP, row_a=left, row_b=right)


def apply_macro_action(
    rows: Iterable[Iterable[int]],
    action: MacroAction,
) -> tuple[tuple[int, ...], ...]:
    """Apply one legal macro exactly over F_257."""

    matrix = canonical_matrix(rows)
    if not isinstance(action, MacroAction):
        raise SoftValueIterationError("action has the wrong type")
    legal = enumerate_legal_macro_actions(matrix)
    if action not in legal:
        raise SoftValueIterationError("action is not legal in the supplied matrix")
    if action.kind == ACTION_HALT:
        return matrix
    mutable = [list(row) for row in matrix]
    if action.kind == ACTION_NORMALIZE:
        factor = pow(
            mutable[action.row_a][action.column],
            -1,
            FIELD_MODULUS,
        )
        mutable[action.row_a] = [
            factor * value % FIELD_MODULUS for value in mutable[action.row_a]
        ]
    elif action.kind == ACTION_ELIMINATE:
        factor = (-mutable[action.row_a][action.column]) % FIELD_MODULUS
        mutable[action.row_a] = [
            (left + factor * right) % FIELD_MODULUS
            for left, right in zip(
                mutable[action.row_a],
                mutable[action.row_b],
                strict=True,
            )
        ]
    elif action.kind == ACTION_SWAP:
        mutable[action.row_a], mutable[action.row_b] = (
            mutable[action.row_b],
            mutable[action.row_a],
        )
    else:
        raise SoftValueIterationError("unreachable macro dispatch")
    return tuple(tuple(row) for row in mutable)


def compile_macro_to_primitives(
    rows: Iterable[Iterable[int]],
    action: MacroAction,
) -> tuple[AlgebraInstruction, ...]:
    """Compile one validated macro into the existing primitive VM."""

    matrix = canonical_matrix(rows)
    apply_macro_action(matrix, action)
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


def compile_macro_trace_to_primitives(
    input_rows: Iterable[Iterable[int]],
    actions: Sequence[MacroAction],
) -> tuple[AlgebraInstruction, ...]:
    """Compile a complete macro trace, requiring one final HALT."""

    matrix = canonical_matrix(input_rows)
    frozen = tuple(actions)
    if not frozen or frozen[-1].kind != ACTION_HALT:
        raise SoftValueIterationError("complete macro trace must terminate with HALT")
    if any(action.kind == ACTION_HALT for action in frozen[:-1]):
        raise SoftValueIterationError("macro trace contains an action after HALT")
    program: list[AlgebraInstruction] = []
    for action in frozen:
        program.extend(compile_macro_to_primitives(matrix, action))
        matrix = apply_macro_action(matrix, action)
    return tuple(program)


@dataclass(slots=True)
class PreparationOracleCounter:
    calls: int = 0


def _leading_column(row: Sequence[int]) -> int | None:
    return next(
        (column for column, value in enumerate(row) if value),
        None,
    )


def next_preparation_macro(
    rows: Iterable[Iterable[int]],
    *,
    counter: PreparationOracleCounter,
) -> MacroAction:
    """Return one canonical-RREF macro for preparation traces only."""

    counter.calls += 1
    matrix = canonical_matrix(rows)
    row_count = len(matrix)
    column_count = len(matrix[0])

    settled = 0
    previous_pivot = -1
    while settled < row_count:
        row = matrix[settled]
        pivot = _leading_column(row)
        if pivot is None:
            if any(
                _leading_column(later) is not None for later in matrix[settled + 1 :]
            ):
                break
            return MacroAction(ACTION_HALT)
        later_pivots = tuple(
            _leading_column(later)
            for later in matrix[settled + 1 :]
            if _leading_column(later) is not None
        )
        if (
            pivot <= previous_pivot
            or row[pivot] != 1
            or any(
                other != settled and matrix[other][pivot] != 0
                for other in range(row_count)
            )
            or any(later <= pivot for later in later_pivots)
        ):
            break
        previous_pivot = pivot
        settled += 1
    if settled == row_count:
        return MacroAction(ACTION_HALT)

    pivot_column = next(
        (
            column
            for column in range(previous_pivot + 1, column_count)
            if any(matrix[row][column] for row in range(settled, row_count))
        ),
        None,
    )
    if pivot_column is None:
        # The remaining rows are zero.  This path is equivalent to the
        # terminal branch above but is retained as a fail-closed guard.
        return MacroAction(ACTION_HALT)
    source = next(row for row in range(settled, row_count) if matrix[row][pivot_column])
    if source != settled:
        return MacroAction(
            ACTION_SWAP,
            row_a=settled,
            row_b=source,
        )
    if matrix[settled][pivot_column] != 1:
        return MacroAction(
            ACTION_NORMALIZE,
            row_a=settled,
            column=pivot_column,
        )
    destination = next(
        (
            row
            for row in range(row_count)
            if row != settled and matrix[row][pivot_column]
        ),
        None,
    )
    if destination is None:
        raise SoftValueIterationError(
            "preparation scheduler found no repair for an unsettled row"
        )
    return MacroAction(
        ACTION_ELIMINATE,
        row_a=destination,
        row_b=settled,
        column=pivot_column,
    )


@dataclass(frozen=True, slots=True)
class LabeledPlanningState:
    rows: tuple[tuple[int, ...], ...]
    target_action: MacroAction

    @property
    def sha256(self) -> str:
        return sha256(
            json.dumps(
                [self.rows, self.target_action.canonical_data()],
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()


def build_preparation_states(
    matrices: Iterable[Iterable[Iterable[int]]],
    *,
    maximum_steps: int,
    counter: PreparationOracleCounter,
) -> tuple[LabeledPlanningState, ...]:
    """Collect deduplicated preparation states and exact macro labels."""

    limit = _positive_int(maximum_steps, label="maximum_steps")
    states: dict[str, LabeledPlanningState] = {}
    for raw in matrices:
        source = canonical_matrix(raw)
        matrix = source
        actions: list[MacroAction] = []
        for _ in range(limit):
            target = next_preparation_macro(matrix, counter=counter)
            if target not in enumerate_legal_macro_actions(matrix):
                raise SoftValueIterationError(
                    "preparation target is not a legal local macro"
                )
            state = LabeledPlanningState(matrix, target)
            key = matrix_sha256(matrix)
            prior = states.get(key)
            if prior is not None and prior != state:
                raise SoftValueIterationError(
                    "preparation labels conflict for one matrix state"
                )
            states[key] = state
            actions.append(target)
            if target.kind == ACTION_HALT:
                break
            matrix = apply_macro_action(matrix, target)
        else:
            raise SoftValueIterationError(
                "preparation trace exceeded its fixed step bound"
            )
        program = compile_macro_trace_to_primitives(source, actions)
        final_state = execute_program(
            source,
            program,
            register_count=DEFAULT_REGISTER_COUNT,
        )
        verify_reduction_program(source, final_state)
    return tuple(states[key] for key in sorted(states))


def make_random_label_control(
    states: Sequence[LabeledPlanningState],
    *,
    seed: int,
) -> tuple[LabeledPlanningState, ...]:
    """Replace each preparation label with a seeded nonexpert legal label."""

    rng = random.Random(seed)
    result = []
    for state in states:
        legal = enumerate_legal_macro_actions(state.rows)
        alternatives = [action for action in legal if action != state.target_action]
        chosen = rng.choice(alternatives) if alternatives else state.target_action
        result.append(replace(state, target_action=chosen))
    return tuple(result)


def planning_state_manifest(
    states: Iterable[LabeledPlanningState],
) -> str:
    return sha256(
        (
            "\n".join(
                state.sha256 for state in sorted(states, key=lambda item: item.sha256)
            )
            + "\n"
        ).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class SoftValueIterationConfig:
    """Geometry-independent capacity and fixed internal compute."""

    width: int = 192
    message_layers: int = 3
    action_hidden: int = 384
    transition_hidden: int = 384
    field_harmonics: int = 4
    coordinate_harmonics: int = 4
    backup_iterations: int = 8
    temperature: float = 0.35
    discount: float = 0.95
    raw_matrix_features: bool = True
    structural_action_scalars: bool = True
    pair_relation_features: bool = True
    message_passing: bool = True

    def __post_init__(self) -> None:
        for label, value in (
            ("width", self.width),
            ("message_layers", self.message_layers),
            ("action_hidden", self.action_hidden),
            ("transition_hidden", self.transition_hidden),
            ("field_harmonics", self.field_harmonics),
            ("coordinate_harmonics", self.coordinate_harmonics),
        ):
            _positive_int(value, label=label)
        _nonnegative_int(
            self.backup_iterations,
            label="backup_iterations",
        )
        if not isinstance(self.temperature, float) or self.temperature <= 0.0:
            raise SoftValueIterationError("temperature must be a positive float")
        if not isinstance(self.discount, float) or not 0.0 <= self.discount <= 1.0:
            raise SoftValueIterationError("discount must be a float in [0, 1]")
        for label, value in (
            ("raw_matrix_features", self.raw_matrix_features),
            ("structural_action_scalars", self.structural_action_scalars),
            ("pair_relation_features", self.pair_relation_features),
            ("message_passing", self.message_passing),
        ):
            if not isinstance(value, bool):
                raise SoftValueIterationError(f"{label} must be boolean")

    @property
    def active_raw_matrix_feature_channels(self) -> int:
        return (
            FIELD_BASE_FEATURES + 4 * self.field_harmonics
            if self.raw_matrix_features
            else 0
        )

    @property
    def active_action_scalar_features(self) -> int:
        return (
            ACTION_SCALAR_FEATURES
            if self.structural_action_scalars
            else MINIMAL_ACTION_SCALAR_FEATURES
        )

    @property
    def active_pair_relation_features(self) -> int:
        return PAIR_RELATION_FEATURES if self.pair_relation_features else 0

    @property
    def canonical_sha256(self) -> str:
        return sha256(
            json.dumps(
                asdict(self),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()


class _CellMessageLayer(nn.Module):
    """Shared row/column/cell update with no geometry-sized parameters."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.row_update = nn.Sequential(
            nn.Linear(3 * width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.column_update = nn.Sequential(
            nn.Linear(3 * width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.cell_update = nn.Sequential(
            nn.Linear(4 * width, 2 * width),
            nn.SiLU(),
            nn.Linear(2 * width, width),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, cells: Tensor) -> Tensor:
        if cells.ndim != 3:
            raise SoftValueIterationError(
                "cell state must have shape [rows, columns, width]"
            )
        row_count, column_count, width = cells.shape
        global_mean = cells.mean(dim=(0, 1))
        row_state = self.row_update(
            torch.cat(
                (
                    cells.mean(dim=1),
                    cells.amax(dim=1),
                    global_mean.expand(row_count, width),
                ),
                dim=-1,
            )
        )
        column_state = self.column_update(
            torch.cat(
                (
                    cells.mean(dim=0),
                    cells.amax(dim=0),
                    global_mean.expand(column_count, width),
                ),
                dim=-1,
            )
        )
        update = self.cell_update(
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
                    global_mean[None, None, :].expand(
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
class ResourceCounts:
    """Exact structural compute exposure for candidate forward calls."""

    model_forward_calls: int = 0
    matrix_cells_encoded: int = 0
    raw_matrix_feature_values: int = 0
    coordinate_feature_values: int = 0
    matrix_message_cell_updates: int = 0
    action_nodes_scored: int = 0
    minimal_action_scalar_values: int = 0
    structural_action_scalar_values: int = 0
    transition_pairs_evaluated: int = 0
    active_message_edges: int = 0
    pair_relation_feature_values: int = 0
    internal_backup_iterations: int = 0
    action_value_updates: int = 0

    def __add__(self, other: ResourceCounts) -> ResourceCounts:
        if not isinstance(other, ResourceCounts):
            return NotImplemented
        return ResourceCounts(
            **{
                field: getattr(self, field) + getattr(other, field)
                for field in self.__dataclass_fields__
            }
        )


@dataclass(frozen=True, slots=True)
class ActionValues:
    actions: tuple[MacroAction, ...]
    logits: Tensor
    local_reward: Tensor
    internal_backup_iterations: int
    action_value_backups: int
    resources: ResourceCounts


class SoftValueIterationController(nn.Module):
    """Fixed-depth differentiable planner over legal local action nodes."""

    def __init__(
        self,
        config: SoftValueIterationConfig = SoftValueIterationConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        field_features = 5 + 4 * config.field_harmonics
        coordinate_features = 2 + 2 * config.coordinate_harmonics
        cell_features = field_features + 2 * coordinate_features
        width = config.width
        self.cell_encoder = nn.Sequential(
            nn.Linear(cell_features, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.cell_layers = nn.ModuleList(
            _CellMessageLayer(width) for _ in range(config.message_layers)
        )
        self.action_type = nn.Embedding(len(ACTION_TYPES), width)
        self.scalar_projection = nn.Sequential(
            nn.Linear(ACTION_SCALAR_FEATURES, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(8 * width, config.action_hidden),
            nn.SiLU(),
            nn.Linear(config.action_hidden, width),
            nn.LayerNorm(width),
        )
        self.local_reward = nn.Sequential(
            nn.Linear(width, config.action_hidden),
            nn.SiLU(),
            nn.Linear(config.action_hidden, 1),
        )
        self.transition_energy = nn.Sequential(
            nn.Linear(
                4 * width + PAIR_RELATION_FEATURES,
                config.transition_hidden,
            ),
            nn.SiLU(),
            nn.Linear(config.transition_hidden, 1),
        )
        self.message_projection = nn.Linear(width, width)
        self.backup_projection = nn.Sequential(
            nn.Linear(2 * width + 2, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        # One cell is reused at every internal planning iteration.
        self.shared_backup_cell = nn.GRUCell(width, width)
        self.continuation_value = nn.Sequential(
            nn.Linear(width, config.action_hidden),
            nn.SiLU(),
            nn.Linear(config.action_hidden, 1),
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def complete_system_parameter_count(self) -> int:
        return PROTECTED_FLAGSHIP_PARAMETERS + self.parameter_count

    def parameter_count_breakdown(self) -> Mapping[str, int]:
        result: dict[str, int] = {}
        for name, parameter in self.named_parameters():
            owner = name.split(".", 1)[0]
            result[owner] = result.get(owner, 0) + parameter.numel()
        result["total"] = sum(result.values())
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
        device: torch.device,
    ) -> Tensor:
        coordinate = torch.arange(
            count,
            dtype=torch.float32,
            device=device,
        ) / max(1, count - 1)
        features = [coordinate, 1.0 - coordinate]
        for harmonic in range(
            1,
            self.config.coordinate_harmonics + 1,
        ):
            features.extend(
                (
                    torch.sin(math.pi * harmonic * coordinate),
                    torch.cos(math.pi * harmonic * coordinate),
                )
            )
        return torch.stack(features, dim=-1)

    def encode_matrix(self, rows: Tensor) -> Tensor:
        if rows.ndim != 2:
            raise SoftValueIterationError("rows must have shape [rows, columns]")
        if rows.dtype not in (torch.int32, torch.int64):
            raise SoftValueIterationError("rows must use an integer dtype")
        if rows.shape[0] < 1 or rows.shape[1] < 1:
            raise SoftValueIterationError("rows must be nonempty")
        if torch.any(rows < 0) or torch.any(rows >= FIELD_MODULUS):
            raise SoftValueIterationError("matrix coefficients leave F_257")
        row_coordinates = self._coordinate_features(
            rows.shape[0],
            device=rows.device,
        )
        column_coordinates = self._coordinate_features(
            rows.shape[1],
            device=rows.device,
        )
        coordinate_grid = torch.cat(
            (
                row_coordinates[:, None, :].expand(
                    rows.shape[0],
                    rows.shape[1],
                    row_coordinates.shape[-1],
                ),
                column_coordinates[None, :, :].expand(
                    rows.shape[0],
                    rows.shape[1],
                    column_coordinates.shape[-1],
                ),
            ),
            dim=-1,
        )
        field_features = self._field_features(rows)
        if not self.config.raw_matrix_features:
            field_features = torch.zeros_like(field_features)
        cells = self.cell_encoder(torch.cat((field_features, coordinate_grid), dim=-1))
        for layer in self.cell_layers:
            cells = layer(cells)
        return cells

    @staticmethod
    def _fraction(index: int, count: int) -> float:
        return index / max(1, count - 1)

    def _action_scalars(
        self,
        matrix: tuple[tuple[int, ...], ...],
        action: MacroAction,
    ) -> tuple[float, ...]:
        row_count = len(matrix)
        column_count = len(matrix[0])
        type_features = [float(action.kind == kind) for kind in ACTION_TYPES]
        if action.kind == ACTION_HALT:
            row_a = row_b = column = 0
            coefficient_a = coefficient_b = 0
            row_a_support = row_b_support = 0
            column_support = 0
            leading_a = leading_b = False
            prefix_a = prefix_b = False
        else:
            row_a = action.row_a
            row_b = (
                action.row_b
                if action.kind in (ACTION_ELIMINATE, ACTION_SWAP)
                else action.row_a
            )
            column = action.column if action.kind != ACTION_SWAP else 0
            coefficient_a = matrix[row_a][column] if action.kind != ACTION_SWAP else 0
            coefficient_b = (
                matrix[row_b][column] if action.kind == ACTION_ELIMINATE else 0
            )
            row_a_support = sum(value != 0 for value in matrix[row_a])
            row_b_support = sum(value != 0 for value in matrix[row_b])
            column_support = sum(matrix[row][column] != 0 for row in range(row_count))
            leading_a = _leading_column(matrix[row_a]) == column
            leading_b = _leading_column(matrix[row_b]) == column
            prefix_a = all(matrix[row_a][prior] == 0 for prior in range(column))
            prefix_b = all(matrix[row_b][prior] == 0 for prior in range(column))
        total_nonzero = sum(value != 0 for row in matrix for value in row)
        values = (
            *type_features,
            self._fraction(row_a, row_count),
            self._fraction(row_b, row_count),
            self._fraction(column, column_count),
            coefficient_a / float(FIELD_MODULUS - 1),
            coefficient_b / float(FIELD_MODULUS - 1),
            row_a_support / float(column_count),
            row_b_support / float(column_count),
            column_support / float(row_count),
            float(leading_a),
            float(leading_b),
            float(prefix_a),
            float(prefix_b),
            total_nonzero / float(row_count * column_count),
            row_count / float(MAX_ROWS),
            column_count / float(MAX_COLUMNS),
            float(action.kind == ACTION_HALT),
        )
        if len(values) != ACTION_SCALAR_FEATURES:
            raise SoftValueIterationError("action scalar feature schema drifted")
        if not self.config.structural_action_scalars:
            values = (
                *values[:MINIMAL_ACTION_SCALAR_FEATURES],
                *(
                    0.0
                    for _ in range(
                        ACTION_SCALAR_FEATURES - MINIMAL_ACTION_SCALAR_FEATURES
                    )
                ),
            )
        return values

    def _encode_actions(
        self,
        matrix: tuple[tuple[int, ...], ...],
        cells: Tensor,
        actions: tuple[MacroAction, ...],
    ) -> Tensor:
        row_count, column_count, width = cells.shape
        row_state = cells.mean(dim=1)
        column_state = cells.mean(dim=0)
        global_state = cells.mean(dim=(0, 1))
        zeros = torch.zeros(width, dtype=cells.dtype, device=cells.device)
        structural_states = []
        action_type_indices = []
        scalar_values = []
        for action in actions:
            if action.kind == ACTION_HALT:
                row_a = row_b = column = cell_a = cell_b = zeros
            elif action.kind == ACTION_SWAP:
                row_a = row_state[action.row_a]
                row_b = row_state[action.row_b]
                column = zeros
                cell_a = cells[action.row_a].mean(dim=0)
                cell_b = cells[action.row_b].mean(dim=0)
            else:
                row_a = row_state[action.row_a]
                column = column_state[action.column]
                cell_a = cells[action.row_a, action.column]
                if action.kind == ACTION_ELIMINATE:
                    row_b = row_state[action.row_b]
                    cell_b = cells[action.row_b, action.column]
                else:
                    row_b = cell_b = zeros
            structural_states.append(
                (
                    global_state,
                    row_a,
                    row_b,
                    column,
                    cell_a,
                    cell_b,
                )
            )
            action_type_indices.append(ACTION_TYPES.index(action.kind))
            scalar_values.append(self._action_scalars(matrix, action))
        type_states = self.action_type(
            torch.tensor(
                action_type_indices,
                dtype=torch.long,
                device=cells.device,
            )
        )
        scalar_states = self.scalar_projection(
            torch.tensor(
                scalar_values,
                dtype=cells.dtype,
                device=cells.device,
            )
        )
        encodings = []
        for index, structural in enumerate(structural_states):
            encodings.append(
                torch.cat(
                    (
                        *structural,
                        type_states[index],
                        scalar_states[index],
                    )
                )
            )
        return self.action_encoder(torch.stack(encodings))

    def _pair_relations(
        self,
        actions: tuple[MacroAction, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        relation_rows = []
        neighbor_rows = []
        for left_index, left in enumerate(actions):
            relation_row = []
            neighbor_row = []
            for right_index, right in enumerate(actions):
                values = _pair_relation_values(
                    left_index,
                    left,
                    right_index,
                    right,
                )
                if not self.config.pair_relation_features:
                    values = (0.0,) * PAIR_RELATION_FEATURES
                relation_row.append(values)
                neighbor_row.append(
                    left_index == right_index
                    if not self.config.message_passing
                    else _message_neighbor(
                        left_index,
                        left,
                        right_index,
                        right,
                    )
                )
            relation_rows.append(relation_row)
            neighbor_rows.append(neighbor_row)
        relations = torch.tensor(
            relation_rows,
            dtype=dtype,
            device=device,
        )
        neighbor = torch.tensor(
            neighbor_rows,
            dtype=torch.bool,
            device=device,
        )
        if not torch.all(neighbor.any(dim=1)):
            raise SoftValueIterationError("an action node has no value-backup neighbor")
        return relations, neighbor

    def _plan(
        self,
        rows: Tensor,
        actions: tuple[MacroAction, ...],
    ) -> tuple[Tensor, Tensor, int]:
        matrix = canonical_matrix(rows.detach().cpu().tolist())
        if not actions:
            raise SoftValueIterationError("actions must be nonempty")
        legal = enumerate_legal_macro_actions(matrix)
        if len(actions) != len(legal) or set(actions) != set(legal):
            raise SoftValueIterationError(
                "candidate actions differ from the legal local-action list"
            )
        if len(set(actions)) != len(actions):
            raise SoftValueIterationError(
                "candidate action renderer contains duplicates"
            )
        cells = self.encode_matrix(rows)
        hidden = self._encode_actions(matrix, cells, actions)
        reward = self.local_reward(hidden).squeeze(-1)
        value = reward
        relations, neighbor = self._pair_relations(
            actions,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        active_edges_per_iteration = int(neighbor.sum().item())
        for _ in range(self.config.backup_iterations):
            count, width = hidden.shape
            left = hidden[:, None, :].expand(count, count, width)
            right = hidden[None, :, :].expand(count, count, width)
            pair = torch.cat(
                (
                    left,
                    right,
                    left * right,
                    torch.abs(left - right),
                    relations,
                ),
                dim=-1,
            )
            transition = self.transition_energy(pair).squeeze(-1)
            utility = (transition + value[None, :]) / self.config.temperature
            utility = utility.masked_fill(~neighbor, float("-inf"))
            neighbor_count = neighbor.sum(dim=1).to(hidden.dtype)
            soft_backup = self.config.temperature * (
                torch.logsumexp(utility, dim=1) - torch.log(neighbor_count)
            )
            weights = torch.softmax(utility, dim=1)
            message = weights @ self.message_projection(hidden)
            update = self.backup_projection(
                torch.cat(
                    (
                        hidden,
                        message,
                        soft_backup[:, None],
                        reward[:, None],
                    ),
                    dim=-1,
                )
            )
            hidden = self.shared_backup_cell(update, hidden)
            value = reward + self.config.discount * (
                self.continuation_value(hidden).squeeze(-1)
            )
        return value, reward, active_edges_per_iteration

    def forward(
        self,
        rows: Tensor,
        actions: Sequence[MacroAction],
    ) -> Tensor:
        """Run exactly the configured number of shared value backups."""

        value, _, _ = self._plan(rows, tuple(actions))
        return value

    def score_actions(
        self,
        rows: Iterable[Iterable[int]],
        actions: Sequence[MacroAction] | None = None,
    ) -> ActionValues:
        matrix = canonical_matrix(rows)
        rendered_actions = (
            enumerate_legal_macro_actions(matrix) if actions is None else tuple(actions)
        )
        reference = next(self.parameters())
        row_tensor = torch.tensor(
            matrix,
            dtype=torch.long,
            device=reference.device,
        )
        logits, local_reward, active_edges = self._plan(
            row_tensor,
            rendered_actions,
        )
        row_count = len(matrix)
        column_count = len(matrix[0])
        action_count = len(rendered_actions)
        iterations = self.config.backup_iterations
        coordinate_channels = 2 + 2 * self.config.coordinate_harmonics
        return ActionValues(
            actions=rendered_actions,
            logits=logits,
            local_reward=local_reward,
            internal_backup_iterations=iterations,
            action_value_backups=(iterations * action_count),
            resources=ResourceCounts(
                model_forward_calls=1,
                matrix_cells_encoded=row_count * column_count,
                raw_matrix_feature_values=(
                    row_count
                    * column_count
                    * self.config.active_raw_matrix_feature_channels
                ),
                coordinate_feature_values=(
                    row_count * column_count * 2 * coordinate_channels
                ),
                matrix_message_cell_updates=(
                    row_count * column_count * self.config.message_layers
                ),
                action_nodes_scored=action_count,
                minimal_action_scalar_values=(
                    action_count * MINIMAL_ACTION_SCALAR_FEATURES
                ),
                structural_action_scalar_values=(
                    action_count
                    * (
                        self.config.active_action_scalar_features
                        - MINIMAL_ACTION_SCALAR_FEATURES
                    )
                ),
                transition_pairs_evaluated=(iterations * action_count * action_count),
                active_message_edges=iterations * active_edges,
                pair_relation_feature_values=(
                    iterations
                    * action_count
                    * action_count
                    * self.config.active_pair_relation_features
                ),
                internal_backup_iterations=iterations,
                action_value_updates=iterations * action_count,
            ),
        )


def model_state_sha256(model: SoftValueIterationController) -> str:
    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _target_index(
    actions: Sequence[MacroAction],
    target: MacroAction,
) -> int:
    try:
        return tuple(actions).index(target)
    except ValueError as error:
        raise SoftValueIterationError(
            "training target is absent from legal actions"
        ) from error


def _state_loss(
    controller: SoftValueIterationController,
    state: LabeledPlanningState,
) -> tuple[Tensor, ActionValues]:
    scored = controller.score_actions(state.rows)
    target = torch.tensor(
        [_target_index(scored.actions, state.target_action)],
        dtype=torch.long,
        device=scored.logits.device,
    )
    return F.cross_entropy(scored.logits[None, :], target), scored


@dataclass(frozen=True, slots=True)
class TrainingResourceReceipt:
    """Exact bounded exposure consumed by one controller fit."""

    optimizer_updates: int
    labeled_state_presentations: int
    parameters: int
    resources: ResourceCounts


def train_controller_with_receipt(
    controller: SoftValueIterationController,
    states: Sequence[LabeledPlanningState],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    maximum_updates: int,
    shuffle_seed: int,
) -> TrainingResourceReceipt:
    """Run a deterministic fit and retain exact structural resource counts."""

    epoch_count = _positive_int(epochs, label="epochs")
    batch = _positive_int(batch_size, label="batch_size")
    update_limit = _positive_int(
        maximum_updates,
        label="maximum_updates",
    )
    if not states:
        raise SoftValueIterationError("training states must be nonempty")
    if not isinstance(learning_rate, float) or learning_rate <= 0.0:
        raise SoftValueIterationError("learning_rate must be a positive float")
    optimizer = torch.optim.AdamW(
        controller.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )
    rng = random.Random(shuffle_seed)
    updates = 0
    presentations = 0
    resources = ResourceCounts()
    controller.train()
    for _ in range(epoch_count):
        order = list(range(len(states)))
        rng.shuffle(order)
        for offset in range(0, len(order), batch):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for index in order[offset : offset + batch]:
                loss, scored = _state_loss(controller, states[index])
                losses.append(loss)
                presentations += 1
                resources += scored.resources
            torch.stack(losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
            optimizer.step()
            updates += 1
            if updates >= update_limit:
                return TrainingResourceReceipt(
                    optimizer_updates=updates,
                    labeled_state_presentations=presentations,
                    parameters=controller.parameter_count,
                    resources=resources,
                )
    return TrainingResourceReceipt(
        optimizer_updates=updates,
        labeled_state_presentations=presentations,
        parameters=controller.parameter_count,
        resources=resources,
    )


def train_controller(
    controller: SoftValueIterationController,
    states: Sequence[LabeledPlanningState],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    maximum_updates: int,
    shuffle_seed: int,
) -> int:
    """Compatibility wrapper returning the exact optimizer-update count."""

    return train_controller_with_receipt(
        controller,
        states,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        maximum_updates=maximum_updates,
        shuffle_seed=shuffle_seed,
    ).optimizer_updates


@torch.no_grad()
def label_accuracy(
    controller: SoftValueIterationController,
    states: Sequence[LabeledPlanningState],
) -> float:
    controller.eval()
    if not states:
        return 0.0
    correct = 0
    for state in states:
        scored = controller.score_actions(state.rows)
        predicted = scored.actions[int(scored.logits.argmax().item())]
        correct += predicted == state.target_action
    return correct / len(states)


@dataclass(frozen=True, slots=True)
class CandidateRuntimeAudit:
    model_decisions: int
    fixed_backup_iterations_per_decision: int
    internal_backup_iterations: int
    action_value_backups: int
    oracle_calls: int
    search_calls: int
    verifier_calls: int
    resources: ResourceCounts = ResourceCounts()


@dataclass(frozen=True, slots=True)
class CandidateRollout:
    halted: bool
    invalid: bool
    overlong: bool
    actions: tuple[MacroAction, ...]
    output_rows: tuple[tuple[int, ...], ...]
    audit: CandidateRuntimeAudit


@torch.no_grad()
def candidate_matrix_only_rollout(
    controller: SoftValueIterationController,
    rows: Iterable[Iterable[int]],
    *,
    maximum_steps: int,
    action_renderer_seed: int | None = None,
) -> CandidateRollout:
    """Run the matrix-only candidate without oracle, search, or verifier."""

    limit = _positive_int(maximum_steps, label="maximum_steps")
    matrix = canonical_matrix(rows)
    emitted: list[MacroAction] = []
    decisions = 0
    backups = 0
    action_backups = 0
    resources = ResourceCounts()
    controller.eval()
    for step in range(limit):
        rendered_actions = enumerate_legal_macro_actions(matrix)
        if action_renderer_seed is not None:
            renderer_digest = sha256(
                (f"{action_renderer_seed}:{step}:{matrix_sha256(matrix)}").encode(
                    "ascii"
                )
            ).digest()
            renderer_rng = random.Random(int.from_bytes(renderer_digest[:8], "big"))
            shuffled = list(rendered_actions)
            renderer_rng.shuffle(shuffled)
            rendered_actions = tuple(shuffled)
        scored = controller.score_actions(matrix, rendered_actions)
        decisions += 1
        backups += scored.internal_backup_iterations
        action_backups += scored.action_value_backups
        resources += scored.resources
        action = scored.actions[int(scored.logits.argmax().item())]
        emitted.append(action)
        if action.kind == ACTION_HALT:
            return CandidateRollout(
                halted=True,
                invalid=False,
                overlong=False,
                actions=tuple(emitted),
                output_rows=matrix,
                audit=CandidateRuntimeAudit(
                    model_decisions=decisions,
                    fixed_backup_iterations_per_decision=(
                        controller.config.backup_iterations
                    ),
                    internal_backup_iterations=backups,
                    action_value_backups=action_backups,
                    oracle_calls=0,
                    search_calls=0,
                    verifier_calls=0,
                    resources=resources,
                ),
            )
        try:
            matrix = apply_macro_action(matrix, action)
        except SoftValueIterationError:
            return CandidateRollout(
                halted=False,
                invalid=True,
                overlong=False,
                actions=tuple(emitted),
                output_rows=matrix,
                audit=CandidateRuntimeAudit(
                    model_decisions=decisions,
                    fixed_backup_iterations_per_decision=(
                        controller.config.backup_iterations
                    ),
                    internal_backup_iterations=backups,
                    action_value_backups=action_backups,
                    oracle_calls=0,
                    search_calls=0,
                    verifier_calls=0,
                    resources=resources,
                ),
            )
    return CandidateRollout(
        halted=False,
        invalid=False,
        overlong=True,
        actions=tuple(emitted),
        output_rows=matrix,
        audit=CandidateRuntimeAudit(
            model_decisions=decisions,
            fixed_backup_iterations_per_decision=(controller.config.backup_iterations),
            internal_backup_iterations=backups,
            action_value_backups=action_backups,
            oracle_calls=0,
            search_calls=0,
            verifier_calls=0,
            resources=resources,
        ),
    )


@dataclass(frozen=True, slots=True)
class AssessedRollout:
    strict_canonical_certified: bool
    invalid: bool
    overlong: bool
    assessor_calls: int


def assess_candidate_rollout(
    input_rows: Iterable[Iterable[int]],
    rollout: CandidateRollout,
) -> AssessedRollout:
    """Verify a stopped candidate trace outside the candidate runtime."""

    source = canonical_matrix(input_rows)
    if rollout.overlong:
        return AssessedRollout(
            strict_canonical_certified=False,
            invalid=False,
            overlong=True,
            assessor_calls=0,
        )
    if rollout.invalid or not rollout.halted:
        return AssessedRollout(
            strict_canonical_certified=False,
            invalid=True,
            overlong=False,
            assessor_calls=0,
        )
    try:
        program = compile_macro_trace_to_primitives(
            source,
            rollout.actions,
        )
        final_state = execute_program(
            source,
            program,
            register_count=DEFAULT_REGISTER_COUNT,
        )
        receipt = verify_reduction_program(source, final_state)
    except (AlgebraMachineError, SoftValueIterationError):
        return AssessedRollout(
            strict_canonical_certified=False,
            invalid=True,
            overlong=False,
            assessor_calls=1,
        )
    return AssessedRollout(
        strict_canonical_certified=receipt.passed,
        invalid=not receipt.passed,
        overlong=False,
        assessor_calls=1,
    )


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    strict_canonical_certified: int
    invalid: int
    overlong: int
    model_decisions: int
    internal_backup_iterations: int
    action_value_backups: int
    candidate_oracle_calls: int
    candidate_search_calls: int
    candidate_verifier_calls: int
    posthoc_assessor_calls: int
    resources: ResourceCounts

    @property
    def total(self) -> int:
        return self.strict_canonical_certified + self.invalid + self.overlong

    @property
    def certification_rate(self) -> float:
        return self.strict_canonical_certified / self.total if self.total else 0.0


def evaluate_matrices(
    controller: SoftValueIterationController,
    matrices: Sequence[tuple[tuple[int, ...], ...]],
    *,
    maximum_steps: int,
    action_renderer_seed: int | None = None,
) -> EvaluationSummary:
    certified = invalid = overlong = decisions = backups = 0
    action_backups = oracle_calls = search_calls = verifier_calls = 0
    assessor_calls = 0
    resources = ResourceCounts()
    for matrix in matrices:
        rollout = candidate_matrix_only_rollout(
            controller,
            matrix,
            maximum_steps=maximum_steps,
            action_renderer_seed=action_renderer_seed,
        )
        assessment = assess_candidate_rollout(matrix, rollout)
        certified += assessment.strict_canonical_certified
        invalid += assessment.invalid
        overlong += assessment.overlong
        decisions += rollout.audit.model_decisions
        backups += rollout.audit.internal_backup_iterations
        action_backups += rollout.audit.action_value_backups
        oracle_calls += rollout.audit.oracle_calls
        search_calls += rollout.audit.search_calls
        verifier_calls += rollout.audit.verifier_calls
        assessor_calls += assessment.assessor_calls
        resources += rollout.audit.resources
    return EvaluationSummary(
        strict_canonical_certified=certified,
        invalid=invalid,
        overlong=overlong,
        model_decisions=decisions,
        internal_backup_iterations=backups,
        action_value_backups=action_backups,
        candidate_oracle_calls=oracle_calls,
        candidate_search_calls=search_calls,
        candidate_verifier_calls=verifier_calls,
        posthoc_assessor_calls=assessor_calls,
        resources=resources,
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
    """Generate bounded sparse matrices with exact disjointness."""

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
        raise SoftValueIterationError("matrix generation bounds are inverted")
    if max_rows > MAX_ROWS or max_columns > MAX_COLUMNS:
        raise SoftValueIterationError("matrix generation exceeds mechanics bounds")
    rng = random.Random(seed)
    seen = set() if excluded is None else set(excluded)
    result = []
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
        raise SoftValueIterationError("matrix generator exhausted its bounded attempts")
    return tuple(result)


def matrix_manifest(
    matrices: Iterable[Iterable[Iterable[int]]],
) -> str:
    return sha256(
        ("\n".join(matrix_sha256(matrix) for matrix in matrices) + "\n").encode("ascii")
    ).hexdigest()


def make_permuted_corpus(
    matrices: Sequence[tuple[tuple[int, ...], ...]],
    *,
    seed: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Apply deterministic, nonidentity row and column permutations."""

    result = []
    for index, matrix in enumerate(matrices):
        frozen = canonical_matrix(matrix)
        digest = sha256(f"{seed}:{index}:{matrix_sha256(frozen)}".encode("ascii"))
        rng = random.Random(int.from_bytes(digest.digest()[:8], "big"))
        row_order = list(range(len(frozen)))
        column_order = list(range(len(frozen[0])))
        rng.shuffle(row_order)
        rng.shuffle(column_order)
        if len(row_order) > 1 and row_order == list(range(len(row_order))):
            row_order[0], row_order[1] = row_order[1], row_order[0]
        if len(column_order) > 1 and column_order == list(range(len(column_order))):
            column_order[0], column_order[1] = (
                column_order[1],
                column_order[0],
            )
        result.append(
            permute_matrix(
                frozen,
                row_order=row_order,
                column_order=column_order,
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class FeatureLeakageAudit:
    """Conservative information lower bounds exposed before neural encoding."""

    states: int
    matrix_cells: int
    nonzero_cells: int
    legal_actions: int
    legal_nonzero_cells_revealed: int
    legal_unit_cells_revealed: int
    legal_nonunit_cells_revealed: int
    legal_nonzero_recall: float
    full_scalar_exact_coefficient_cells: int
    full_scalar_nonzero_coefficient_recall: float
    row_inequality_pairs_exposed: int
    minimal_action_scalar_values: int
    structural_action_scalar_values: int
    pair_relation_bits: int
    positive_pair_relation_bits: int
    message_graph_edges: int
    expert_target_is_first_legal: int
    expert_target_is_last_legal: int
    expert_target_first_legal_rate: float
    expert_target_last_legal_rate: float


def audit_feature_leakage(
    states: Sequence[LabeledPlanningState],
) -> FeatureLeakageAudit:
    """Measure information directly recoverable from action-side features."""

    matrix_cells = nonzero_cells = legal_actions = 0
    legal_nonzero = legal_unit = legal_nonunit = 0
    exact_coefficients = row_inequalities = 0
    minimal_scalars = structural_scalars = 0
    pair_bits = positive_pair_bits = message_edges = 0
    first_targets = last_targets = 0
    for state in states:
        matrix = canonical_matrix(state.rows)
        actions = enumerate_legal_macro_actions(matrix)
        matrix_cells += len(matrix) * len(matrix[0])
        nonzero_cells += sum(value != 0 for row in matrix for value in row)
        legal_actions += len(actions)
        normalize_cells = {
            (action.row_a, action.column)
            for action in actions
            if action.kind == ACTION_NORMALIZE
        }
        elimination_cells = {
            coordinate
            for action in actions
            if action.kind == ACTION_ELIMINATE
            for coordinate in (
                (action.row_a, action.column),
                (action.row_b, action.column),
            )
        }
        revealed_nonzero = normalize_cells | elimination_cells
        revealed_unit = revealed_nonzero - normalize_cells
        if any(matrix[row][column] == 0 for row, column in revealed_nonzero):
            raise SoftValueIterationError("legal actions falsely revealed a zero cell")
        if any(matrix[row][column] in (0, 1) for row, column in normalize_cells):
            raise SoftValueIterationError("NORMALIZE leakage class drifted")
        if any(matrix[row][column] != 1 for row, column in revealed_unit):
            raise SoftValueIterationError("legal unit-cell inference drifted")
        legal_nonzero += len(revealed_nonzero)
        legal_unit += len(revealed_unit)
        legal_nonunit += len(normalize_cells)
        exact_coefficients += len(revealed_nonzero)
        row_inequalities += sum(action.kind == ACTION_SWAP for action in actions)
        minimal_scalars += len(actions) * MINIMAL_ACTION_SCALAR_FEATURES
        structural_scalars += len(actions) * (
            ACTION_SCALAR_FEATURES - MINIMAL_ACTION_SCALAR_FEATURES
        )
        for left_index, left in enumerate(actions):
            for right_index, right in enumerate(actions):
                relation = _pair_relation_values(
                    left_index,
                    left,
                    right_index,
                    right,
                )
                pair_bits += len(relation)
                positive_pair_bits += int(sum(relation))
                message_edges += int(
                    _message_neighbor(
                        left_index,
                        left,
                        right_index,
                        right,
                    )
                )
        first_targets += state.target_action == actions[0]
        last_targets += state.target_action == actions[-1]
    state_count = len(states)
    return FeatureLeakageAudit(
        states=state_count,
        matrix_cells=matrix_cells,
        nonzero_cells=nonzero_cells,
        legal_actions=legal_actions,
        legal_nonzero_cells_revealed=legal_nonzero,
        legal_unit_cells_revealed=legal_unit,
        legal_nonunit_cells_revealed=legal_nonunit,
        legal_nonzero_recall=(legal_nonzero / nonzero_cells if nonzero_cells else 0.0),
        full_scalar_exact_coefficient_cells=exact_coefficients,
        full_scalar_nonzero_coefficient_recall=(
            exact_coefficients / nonzero_cells if nonzero_cells else 0.0
        ),
        row_inequality_pairs_exposed=row_inequalities,
        minimal_action_scalar_values=minimal_scalars,
        structural_action_scalar_values=structural_scalars,
        pair_relation_bits=pair_bits,
        positive_pair_relation_bits=positive_pair_bits,
        message_graph_edges=message_edges,
        expert_target_is_first_legal=first_targets,
        expert_target_is_last_legal=last_targets,
        expert_target_first_legal_rate=(
            first_targets / state_count if state_count else 0.0
        ),
        expert_target_last_legal_rate=(
            last_targets / state_count if state_count else 0.0
        ),
    )


@dataclass(frozen=True, slots=True)
class SoftValueExperimentConfig:
    seed: int = 20260724
    train_matrices: int = 96
    evaluation_matrices: int = 64
    train_maximum_rows: int = 3
    train_maximum_columns: int = 4
    evaluation_minimum_rows: int = 4
    evaluation_minimum_columns: int = 5
    evaluation_maximum_rows: int = 4
    evaluation_maximum_columns: int = 6
    maximum_preparation_steps: int = 96
    maximum_rollout_steps: int = 192
    epochs: int = 12
    batch_size: int = 8
    learning_rate: float = 1e-3
    maximum_updates: int = 2_048
    material_minimum_cases: int = 64
    material_minimum_rate: float = 0.80
    material_minimum_control_gap: float = 0.10
    device: str = "cpu"
    controller: SoftValueIterationConfig = SoftValueIterationConfig()

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
            ("maximum_preparation_steps", self.maximum_preparation_steps),
            ("maximum_rollout_steps", self.maximum_rollout_steps),
            ("epochs", self.epochs),
            ("batch_size", self.batch_size),
            ("maximum_updates", self.maximum_updates),
            ("material_minimum_cases", self.material_minimum_cases),
        ):
            _positive_int(value, label=label)
        if self.evaluation_minimum_rows <= self.train_maximum_rows:
            raise SoftValueIterationError(
                "evaluation rows must be strictly larger than training rows"
            )
        if self.evaluation_minimum_columns <= self.train_maximum_columns:
            raise SoftValueIterationError(
                "evaluation columns must be strictly larger than training columns"
            )
        if self.evaluation_minimum_rows > self.evaluation_maximum_rows:
            raise SoftValueIterationError("evaluation row bounds are inverted")
        if self.evaluation_minimum_columns > self.evaluation_maximum_columns:
            raise SoftValueIterationError("evaluation column bounds are inverted")
        if not isinstance(self.learning_rate, float) or (self.learning_rate <= 0.0):
            raise SoftValueIterationError("learning_rate must be a positive float")
        for label, value in (
            ("material_minimum_rate", self.material_minimum_rate),
            (
                "material_minimum_control_gap",
                self.material_minimum_control_gap,
            ),
        ):
            if not isinstance(value, float) or not 0.0 <= value <= 1.0:
                raise SoftValueIterationError(f"{label} must be a float in [0, 1]")
        if self.device not in ("cpu", "cuda"):
            raise SoftValueIterationError("device must be either 'cpu' or 'cuda'")


@dataclass(frozen=True, slots=True)
class SoftValueExperimentReport:
    schema: str
    status: str
    outcome: str
    seed: int
    device: str
    candidate_input_fields: tuple[str, ...]
    candidate_has_host_search: bool
    candidate_has_verifier: bool
    candidate_has_oracle: bool
    fixed_shared_weight_recurrence: bool
    treatment_backup_iterations_per_decision: int
    random_control_backup_iterations_per_decision: int
    zero_control_backup_iterations_per_decision: int
    controller_parameters: int
    complete_system_parameters: int
    total_parameter_budget: int
    parameter_budget_passed: bool
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
    strict_geometry_disjoint: bool
    preparation_oracle_calls: int
    treatment_optimizer_updates: int
    random_control_optimizer_updates: int
    zero_control_optimizer_updates: int
    treatment_train_label_accuracy: float
    random_control_label_accuracy: float
    random_control_true_label_accuracy: float
    zero_control_train_label_accuracy: float
    treatment_strict_canonical_certified: int
    random_control_strict_canonical_certified: int
    zero_control_strict_canonical_certified: int
    treatment_invalid: int
    random_control_invalid: int
    zero_control_invalid: int
    treatment_overlong: int
    random_control_overlong: int
    zero_control_overlong: int
    treatment_model_decisions: int
    treatment_internal_backup_iterations: int
    treatment_action_value_backups: int
    final_candidate_oracle_calls: int
    final_candidate_search_calls: int
    final_candidate_verifier_calls: int
    posthoc_assessor_calls: int
    no_oracle_no_search_no_verifier_gate_passed: bool
    material_minimum_cases: int
    material_minimum_rate: float
    material_minimum_control_gap: float
    material_gate_passed: bool
    train_matrix_manifest_sha256: str
    evaluation_matrix_manifest_sha256: str
    preparation_state_manifest_sha256: str
    random_state_manifest_sha256: str
    treatment_model_sha256: str
    random_control_model_sha256: str
    zero_control_model_sha256: str

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                asdict(self),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
            + b"\n"
        )


@dataclass(frozen=True, slots=True)
class HostileArmReport:
    name: str
    controller_config: Mapping[str, object]
    controller_config_sha256: str
    controller_parameters: int
    complete_system_parameters: int
    initial_model_sha256: str
    trained_model_sha256: str
    training_label_manifest_sha256: str
    training_resources: TrainingResourceReceipt
    training_label_accuracy: float
    true_label_accuracy: float
    evaluation: EvaluationSummary
    no_oracle_no_search_no_verifier_gate_passed: bool


@dataclass(frozen=True, slots=True)
class RecodingAudit:
    cases: int
    action_renderer_seed: int
    action_order_trace_matches: int
    action_order_assessment_matches: int
    representative_trace_matches: int
    representative_assessment_matches: int
    legal_action_permutation_matches: int
    raw_representative_manifest_sha256: str
    permuted_matrix_manifest_sha256: str
    permuted_evaluation: EvaluationSummary
    comparison_resources: ResourceCounts


@dataclass(frozen=True, slots=True)
class HostileAuditReport:
    schema: str
    status: str
    outcome: str
    seed: int
    device: str
    candidate_input_fields: tuple[str, ...]
    candidate_has_host_search: bool
    candidate_has_verifier: bool
    candidate_has_oracle: bool
    train_matrices: int
    train_states: int
    evaluation_matrices: int
    train_matrix_manifest_sha256: str
    evaluation_matrix_manifest_sha256: str
    preparation_state_manifest_sha256: str
    random_state_manifest_sha256: str
    preparation_oracle_calls: int
    feature_leakage: FeatureLeakageAudit
    initial_model_sha256: str
    controller_parameters: int
    complete_system_parameters: int
    total_parameter_budget: int
    parameter_budget_passed: bool
    all_arm_parameter_counts_equal: bool
    all_arm_initializations_equal: bool
    all_arm_optimizer_updates_equal: bool
    all_arm_state_presentations_equal: bool
    arms: tuple[HostileArmReport, ...]
    recoding: RecodingAudit
    strongest_causal_control_name: str
    strongest_causal_control_rate: float
    treatment_rate: float
    treatment_minus_strongest_control: float
    final_candidate_oracle_calls: int
    final_candidate_search_calls: int
    final_candidate_verifier_calls: int
    no_oracle_no_search_no_verifier_gate_passed: bool

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                asdict(self),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
            + b"\n"
        )


def _fresh_control_models(
    config: SoftValueIterationConfig,
    *,
    seed: int,
) -> tuple[
    SoftValueIterationController,
    SoftValueIterationController,
    SoftValueIterationController,
]:
    torch.manual_seed(seed)
    treatment = SoftValueIterationController(config)
    random_control = SoftValueIterationController(config)
    random_control.load_state_dict(treatment.state_dict())
    zero_control = SoftValueIterationController(replace(config, backup_iterations=0))
    zero_control.load_state_dict(treatment.state_dict())
    return treatment, random_control, zero_control


def run_bounded_experiment(
    config: SoftValueExperimentConfig,
) -> SoftValueExperimentReport:
    """Run treatment, random-label, and zero-iteration falsifiers."""

    if not isinstance(config, SoftValueExperimentConfig):
        raise SoftValueIterationError("experiment config has the wrong type")
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SoftValueIterationError("CUDA was requested but is not available")
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
    counter = PreparationOracleCounter()
    states = build_preparation_states(
        train,
        maximum_steps=config.maximum_preparation_steps,
        counter=counter,
    )
    random_states = make_random_label_control(
        states,
        seed=config.seed + 2,
    )
    treatment, random_control, zero_control = _fresh_control_models(
        config.controller,
        seed=config.seed + 3,
    )
    treatment.to(device)
    random_control.to(device)
    zero_control.to(device)
    treatment_updates = train_controller(
        treatment,
        states,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        maximum_updates=config.maximum_updates,
        shuffle_seed=config.seed + 4,
    )
    random_updates = train_controller(
        random_control,
        random_states,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        maximum_updates=config.maximum_updates,
        shuffle_seed=config.seed + 4,
    )
    zero_updates = train_controller(
        zero_control,
        states,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        maximum_updates=config.maximum_updates,
        shuffle_seed=config.seed + 4,
    )
    treatment_result = evaluate_matrices(
        treatment,
        evaluation,
        maximum_steps=config.maximum_rollout_steps,
    )
    random_result = evaluate_matrices(
        random_control,
        evaluation,
        maximum_steps=config.maximum_rollout_steps,
    )
    zero_result = evaluate_matrices(
        zero_control,
        evaluation,
        maximum_steps=config.maximum_rollout_steps,
    )
    final_oracle = (
        treatment_result.candidate_oracle_calls
        + random_result.candidate_oracle_calls
        + zero_result.candidate_oracle_calls
    )
    final_search = (
        treatment_result.candidate_search_calls
        + random_result.candidate_search_calls
        + zero_result.candidate_search_calls
    )
    final_verifier = (
        treatment_result.candidate_verifier_calls
        + random_result.candidate_verifier_calls
        + zero_result.candidate_verifier_calls
    )
    boundary_gate = final_oracle == 0 and final_search == 0 and final_verifier == 0
    if not boundary_gate:
        raise SoftValueIterationError(
            "candidate runtime crossed a forbidden evaluation boundary"
        )
    treatment_rate = treatment_result.certification_rate
    strongest_control = max(
        random_result.certification_rate,
        zero_result.certification_rate,
    )
    material_gate = (
        len(evaluation) >= config.material_minimum_cases
        and treatment_rate >= config.material_minimum_rate
        and treatment_rate - strongest_control >= config.material_minimum_control_gap
    )
    parameter_gate = treatment.complete_system_parameter_count < TOTAL_PARAMETER_BUDGET
    if not parameter_gate:
        raise SoftValueIterationError(
            "controller exceeds the complete-system parameter budget"
        )
    return SoftValueExperimentReport(
        schema=EXPERIMENT_SCHEMA,
        status=STATUS,
        outcome=OUTCOME_MATERIAL if material_gate else OUTCOME_INCONCLUSIVE,
        seed=config.seed,
        device=str(device),
        candidate_input_fields=(
            "matrix",
            "legal_local_action_features",
        ),
        candidate_has_host_search=False,
        candidate_has_verifier=False,
        candidate_has_oracle=False,
        fixed_shared_weight_recurrence=True,
        treatment_backup_iterations_per_decision=(treatment.config.backup_iterations),
        random_control_backup_iterations_per_decision=(
            random_control.config.backup_iterations
        ),
        zero_control_backup_iterations_per_decision=(
            zero_control.config.backup_iterations
        ),
        controller_parameters=treatment.parameter_count,
        complete_system_parameters=(treatment.complete_system_parameter_count),
        total_parameter_budget=TOTAL_PARAMETER_BUDGET,
        parameter_budget_passed=parameter_gate,
        parameter_count_breakdown=(treatment.parameter_count_breakdown()),
        train_matrices=len(train),
        train_states=len(states),
        evaluation_matrices=len(evaluation),
        train_maximum_rows=config.train_maximum_rows,
        train_maximum_columns=config.train_maximum_columns,
        evaluation_minimum_rows=config.evaluation_minimum_rows,
        evaluation_minimum_columns=config.evaluation_minimum_columns,
        evaluation_maximum_rows=config.evaluation_maximum_rows,
        evaluation_maximum_columns=config.evaluation_maximum_columns,
        strict_geometry_disjoint=not bool(set(train) & set(evaluation)),
        preparation_oracle_calls=counter.calls,
        treatment_optimizer_updates=treatment_updates,
        random_control_optimizer_updates=random_updates,
        zero_control_optimizer_updates=zero_updates,
        treatment_train_label_accuracy=label_accuracy(treatment, states),
        random_control_label_accuracy=label_accuracy(
            random_control,
            random_states,
        ),
        random_control_true_label_accuracy=label_accuracy(
            random_control,
            states,
        ),
        zero_control_train_label_accuracy=label_accuracy(
            zero_control,
            states,
        ),
        treatment_strict_canonical_certified=(
            treatment_result.strict_canonical_certified
        ),
        random_control_strict_canonical_certified=(
            random_result.strict_canonical_certified
        ),
        zero_control_strict_canonical_certified=(
            zero_result.strict_canonical_certified
        ),
        treatment_invalid=treatment_result.invalid,
        random_control_invalid=random_result.invalid,
        zero_control_invalid=zero_result.invalid,
        treatment_overlong=treatment_result.overlong,
        random_control_overlong=random_result.overlong,
        zero_control_overlong=zero_result.overlong,
        treatment_model_decisions=treatment_result.model_decisions,
        treatment_internal_backup_iterations=(
            treatment_result.internal_backup_iterations
        ),
        treatment_action_value_backups=(treatment_result.action_value_backups),
        final_candidate_oracle_calls=final_oracle,
        final_candidate_search_calls=final_search,
        final_candidate_verifier_calls=final_verifier,
        posthoc_assessor_calls=(
            treatment_result.posthoc_assessor_calls
            + random_result.posthoc_assessor_calls
            + zero_result.posthoc_assessor_calls
        ),
        no_oracle_no_search_no_verifier_gate_passed=boundary_gate,
        material_minimum_cases=config.material_minimum_cases,
        material_minimum_rate=config.material_minimum_rate,
        material_minimum_control_gap=(config.material_minimum_control_gap),
        material_gate_passed=material_gate,
        train_matrix_manifest_sha256=matrix_manifest(train),
        evaluation_matrix_manifest_sha256=matrix_manifest(evaluation),
        preparation_state_manifest_sha256=planning_state_manifest(states),
        random_state_manifest_sha256=planning_state_manifest(random_states),
        treatment_model_sha256=model_state_sha256(treatment),
        random_control_model_sha256=model_state_sha256(random_control),
        zero_control_model_sha256=model_state_sha256(zero_control),
    )


def _raw_matrix_manifest(
    matrices: Sequence[tuple[tuple[int, ...], ...]],
) -> str:
    return sha256(
        (
            "\n".join(json.dumps(matrix, separators=(",", ":")) for matrix in matrices)
            + "\n"
        ).encode("ascii")
    ).hexdigest()


def _same_rollout(
    left: CandidateRollout,
    right: CandidateRollout,
) -> bool:
    return (
        left.halted,
        left.invalid,
        left.overlong,
        left.actions,
        left.output_rows,
    ) == (
        right.halted,
        right.invalid,
        right.overlong,
        right.actions,
        right.output_rows,
    )


def _same_assessment(
    left: AssessedRollout,
    right: AssessedRollout,
) -> bool:
    return (
        left.strict_canonical_certified,
        left.invalid,
        left.overlong,
    ) == (
        right.strict_canonical_certified,
        right.invalid,
        right.overlong,
    )


def run_recoding_audit(
    controller: SoftValueIterationController,
    matrices: Sequence[tuple[tuple[int, ...], ...]],
    *,
    maximum_steps: int,
    seed: int,
) -> RecodingAudit:
    """Run action-order, field-renderer, and matrix-permutation recodings."""

    action_trace_matches = action_assessment_matches = 0
    representative_trace_matches = representative_assessment_matches = 0
    legal_permutation_matches = 0
    resources = ResourceCounts()
    representatives = []
    for index, matrix in enumerate(matrices):
        canonical = candidate_matrix_only_rollout(
            controller,
            matrix,
            maximum_steps=maximum_steps,
        )
        reordered = candidate_matrix_only_rollout(
            controller,
            matrix,
            maximum_steps=maximum_steps,
            action_renderer_seed=seed,
        )
        representative = representative_recode_matrix(
            matrix,
            seed=seed + index + 1,
        )
        representatives.append(representative)
        rerendered = candidate_matrix_only_rollout(
            controller,
            representative,
            maximum_steps=maximum_steps,
        )
        canonical_assessment = assess_candidate_rollout(matrix, canonical)
        reordered_assessment = assess_candidate_rollout(matrix, reordered)
        representative_assessment = assess_candidate_rollout(
            representative,
            rerendered,
        )
        action_trace_matches += _same_rollout(canonical, reordered)
        action_assessment_matches += _same_assessment(
            canonical_assessment,
            reordered_assessment,
        )
        representative_trace_matches += _same_rollout(
            canonical,
            rerendered,
        )
        representative_assessment_matches += _same_assessment(
            canonical_assessment,
            representative_assessment,
        )
        resources += canonical.audit.resources
        resources += reordered.audit.resources
        resources += rerendered.audit.resources

        row_order = tuple(reversed(range(len(matrix))))
        column_order = tuple(reversed(range(len(matrix[0]))))
        permuted = permute_matrix(
            matrix,
            row_order=row_order,
            column_order=column_order,
        )
        expected_actions = {
            remap_action_under_permutation(
                action,
                row_order=row_order,
                column_order=column_order,
            )
            for action in enumerate_legal_macro_actions(matrix)
        }
        legal_permutation_matches += expected_actions == set(
            enumerate_legal_macro_actions(permuted)
        )
    permuted_matrices = make_permuted_corpus(matrices, seed=seed + 1)
    permuted_evaluation = evaluate_matrices(
        controller,
        permuted_matrices,
        maximum_steps=maximum_steps,
    )
    return RecodingAudit(
        cases=len(matrices),
        action_renderer_seed=seed,
        action_order_trace_matches=action_trace_matches,
        action_order_assessment_matches=action_assessment_matches,
        representative_trace_matches=representative_trace_matches,
        representative_assessment_matches=(representative_assessment_matches),
        legal_action_permutation_matches=legal_permutation_matches,
        raw_representative_manifest_sha256=_raw_matrix_manifest(tuple(representatives)),
        permuted_matrix_manifest_sha256=matrix_manifest(permuted_matrices),
        permuted_evaluation=permuted_evaluation,
        comparison_resources=resources,
    )


def run_hostile_audit_experiment(
    config: SoftValueExperimentConfig,
) -> HostileAuditReport:
    """Fit matched hostile controls and retain exact causal receipts."""

    if not isinstance(config, SoftValueExperimentConfig):
        raise SoftValueIterationError("experiment config has the wrong type")
    if config.controller.backup_iterations < 2:
        raise SoftValueIterationError(
            "hostile audit requires at least two treatment backup iterations"
        )
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SoftValueIterationError("CUDA was requested but is not available")
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
    counter = PreparationOracleCounter()
    states = build_preparation_states(
        train,
        maximum_steps=config.maximum_preparation_steps,
        counter=counter,
    )
    random_states = make_random_label_control(
        states,
        seed=config.seed + 2,
    )
    base_config = config.controller
    arm_specs = (
        ("treatment", base_config, states),
        (
            "structural_action_scalars_removed",
            replace(base_config, structural_action_scalars=False),
            states,
        ),
        (
            "raw_matrix_removed",
            replace(base_config, raw_matrix_features=False),
            states,
        ),
        (
            "pair_relation_features_removed",
            replace(base_config, pair_relation_features=False),
            states,
        ),
        (
            "legal_operands_types_only",
            replace(
                base_config,
                raw_matrix_features=False,
                structural_action_scalars=False,
                pair_relation_features=False,
                message_passing=False,
            ),
            states,
        ),
        (
            "one_backup_iteration",
            replace(base_config, backup_iterations=1),
            states,
        ),
        (
            "message_passing_disabled",
            replace(base_config, message_passing=False),
            states,
        ),
        (
            "zero_backup_iterations",
            replace(base_config, backup_iterations=0),
            states,
        ),
        ("random_labels", base_config, random_states),
    )
    torch.manual_seed(config.seed + 3)
    initial_model = SoftValueIterationController(base_config)
    initial_state = {
        name: tensor.detach().clone()
        for name, tensor in initial_model.state_dict().items()
    }
    initial_hash = model_state_sha256(initial_model)
    arm_reports = []
    initial_hashes = []
    treatment_model: SoftValueIterationController | None = None
    for name, arm_config, labels in arm_specs:
        controller = SoftValueIterationController(arm_config)
        controller.load_state_dict(initial_state)
        arm_initial_hash = model_state_sha256(controller)
        initial_hashes.append(arm_initial_hash)
        if arm_initial_hash != initial_hash:
            raise SoftValueIterationError("hostile arm initialization drifted")
        controller.to(device)
        receipt = train_controller_with_receipt(
            controller,
            labels,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            maximum_updates=config.maximum_updates,
            shuffle_seed=config.seed + 4,
        )
        evaluation_result = evaluate_matrices(
            controller,
            evaluation,
            maximum_steps=config.maximum_rollout_steps,
        )
        boundary_gate = (
            evaluation_result.candidate_oracle_calls == 0
            and evaluation_result.candidate_search_calls == 0
            and evaluation_result.candidate_verifier_calls == 0
        )
        if not boundary_gate:
            raise SoftValueIterationError(
                f"hostile arm {name} crossed a forbidden boundary"
            )
        arm_reports.append(
            HostileArmReport(
                name=name,
                controller_config=asdict(arm_config),
                controller_config_sha256=arm_config.canonical_sha256,
                controller_parameters=controller.parameter_count,
                complete_system_parameters=(controller.complete_system_parameter_count),
                initial_model_sha256=arm_initial_hash,
                trained_model_sha256=model_state_sha256(controller),
                training_label_manifest_sha256=(planning_state_manifest(labels)),
                training_resources=receipt,
                training_label_accuracy=label_accuracy(controller, labels),
                true_label_accuracy=label_accuracy(controller, states),
                evaluation=evaluation_result,
                no_oracle_no_search_no_verifier_gate_passed=boundary_gate,
            )
        )
        if name == "treatment":
            treatment_model = controller
        else:
            del controller
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if treatment_model is None:
        raise SoftValueIterationError("hostile treatment model was not retained")
    recoding = run_recoding_audit(
        treatment_model,
        evaluation,
        maximum_steps=config.maximum_rollout_steps,
        seed=config.seed + 5,
    )
    causal_control_names = {
        "structural_action_scalars_removed",
        "raw_matrix_removed",
        "pair_relation_features_removed",
        "legal_operands_types_only",
        "one_backup_iteration",
        "message_passing_disabled",
        "zero_backup_iterations",
    }
    treatment_report = next(arm for arm in arm_reports if arm.name == "treatment")
    strongest_control = max(
        (arm for arm in arm_reports if arm.name in causal_control_names),
        key=lambda arm: arm.evaluation.certification_rate,
    )
    parameter_counts = {arm.controller_parameters for arm in arm_reports}
    optimizer_updates = {
        arm.training_resources.optimizer_updates for arm in arm_reports
    }
    state_presentations = {
        arm.training_resources.labeled_state_presentations for arm in arm_reports
    }
    final_oracle = (
        sum(arm.evaluation.candidate_oracle_calls for arm in arm_reports)
        + recoding.permuted_evaluation.candidate_oracle_calls
    )
    final_search = (
        sum(arm.evaluation.candidate_search_calls for arm in arm_reports)
        + recoding.permuted_evaluation.candidate_search_calls
    )
    final_verifier = (
        sum(arm.evaluation.candidate_verifier_calls for arm in arm_reports)
        + recoding.permuted_evaluation.candidate_verifier_calls
    )
    final_boundary_gate = (
        final_oracle == 0 and final_search == 0 and final_verifier == 0
    )
    if not final_boundary_gate:
        raise SoftValueIterationError(
            "hostile audit crossed a forbidden candidate boundary"
        )
    parameter_gate = (
        initial_model.complete_system_parameter_count < TOTAL_PARAMETER_BUDGET
    )
    if not parameter_gate:
        raise SoftValueIterationError("hostile controller exceeds the parameter budget")
    return HostileAuditReport(
        schema=HOSTILE_AUDIT_SCHEMA,
        status=HOSTILE_AUDIT_STATUS,
        outcome=HOSTILE_AUDIT_OUTCOME,
        seed=config.seed,
        device=str(device),
        candidate_input_fields=(
            "matrix",
            "legal_local_action_features",
        ),
        candidate_has_host_search=False,
        candidate_has_verifier=False,
        candidate_has_oracle=False,
        train_matrices=len(train),
        train_states=len(states),
        evaluation_matrices=len(evaluation),
        train_matrix_manifest_sha256=matrix_manifest(train),
        evaluation_matrix_manifest_sha256=matrix_manifest(evaluation),
        preparation_state_manifest_sha256=planning_state_manifest(states),
        random_state_manifest_sha256=planning_state_manifest(random_states),
        preparation_oracle_calls=counter.calls,
        feature_leakage=audit_feature_leakage(states),
        initial_model_sha256=initial_hash,
        controller_parameters=initial_model.parameter_count,
        complete_system_parameters=(initial_model.complete_system_parameter_count),
        total_parameter_budget=TOTAL_PARAMETER_BUDGET,
        parameter_budget_passed=parameter_gate,
        all_arm_parameter_counts_equal=len(parameter_counts) == 1,
        all_arm_initializations_equal=(
            len(set(initial_hashes)) == 1 and initial_hashes[0] == initial_hash
        ),
        all_arm_optimizer_updates_equal=len(optimizer_updates) == 1,
        all_arm_state_presentations_equal=len(state_presentations) == 1,
        arms=tuple(arm_reports),
        recoding=recoding,
        strongest_causal_control_name=strongest_control.name,
        strongest_causal_control_rate=(strongest_control.evaluation.certification_rate),
        treatment_rate=treatment_report.evaluation.certification_rate,
        treatment_minus_strongest_control=(
            treatment_report.evaluation.certification_rate
            - strongest_control.evaluation.certification_rate
        ),
        final_candidate_oracle_calls=final_oracle,
        final_candidate_search_calls=final_search,
        final_candidate_verifier_calls=final_verifier,
        no_oracle_no_search_no_verifier_gate_passed=final_boundary_gate,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--train-matrices", type=int, default=96)
    parser.add_argument("--evaluation-matrices", type=int, default=64)
    parser.add_argument("--train-maximum-rows", type=int, default=3)
    parser.add_argument("--train-maximum-columns", type=int, default=4)
    parser.add_argument("--evaluation-minimum-rows", type=int, default=4)
    parser.add_argument("--evaluation-minimum-columns", type=int, default=5)
    parser.add_argument("--evaluation-maximum-rows", type=int, default=4)
    parser.add_argument("--evaluation-maximum-columns", type=int, default=6)
    parser.add_argument("--maximum-preparation-steps", type=int, default=96)
    parser.add_argument("--maximum-rollout-steps", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--maximum-updates", type=int, default=2_048)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--message-layers", type=int, default=3)
    parser.add_argument("--action-hidden", type=int, default=384)
    parser.add_argument("--transition-hidden", type=int, default=384)
    parser.add_argument("--field-harmonics", type=int, default=4)
    parser.add_argument("--coordinate-harmonics", type=int, default=4)
    parser.add_argument("--backup-iterations", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--hostile-audit", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = SoftValueExperimentConfig(
        seed=args.seed,
        train_matrices=args.train_matrices,
        evaluation_matrices=args.evaluation_matrices,
        train_maximum_rows=args.train_maximum_rows,
        train_maximum_columns=args.train_maximum_columns,
        evaluation_minimum_rows=args.evaluation_minimum_rows,
        evaluation_minimum_columns=args.evaluation_minimum_columns,
        evaluation_maximum_rows=args.evaluation_maximum_rows,
        evaluation_maximum_columns=args.evaluation_maximum_columns,
        maximum_preparation_steps=args.maximum_preparation_steps,
        maximum_rollout_steps=args.maximum_rollout_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        maximum_updates=args.maximum_updates,
        device=args.device,
        controller=SoftValueIterationConfig(
            width=args.width,
            message_layers=args.message_layers,
            action_hidden=args.action_hidden,
            transition_hidden=args.transition_hidden,
            field_harmonics=args.field_harmonics,
            coordinate_harmonics=args.coordinate_harmonics,
            backup_iterations=args.backup_iterations,
            temperature=args.temperature,
            discount=args.discount,
        ),
    )
    report = (
        run_hostile_audit_experiment(config)
        if args.hostile_audit
        else run_bounded_experiment(config)
    )
    payload = report.canonical_bytes()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
    print(payload.decode("ascii"), end="")


if __name__ == "__main__":
    main()
