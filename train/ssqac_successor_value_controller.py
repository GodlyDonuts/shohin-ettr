#!/usr/bin/env python3
"""Successor-aware recurrent neural planning falsifier for SSQAC.

The candidate receives only the current raw F_257 matrix, geometry-relative
action descriptors, deterministic coordinate labels, and the raw matrix
produced by applying each legal one-step action.  It never receives rank,
energy, frontier, a reference schedule, search state, verifier output, or an
oracle result.  A shared geometry-general encoder scores the counterfactuals,
then a fixed shared-weight recurrent set planner compares them before emitting
one hard action.

The experiment includes four matched controls:

* successor matrices replaced by zero after still evaluating every successor;
* deterministic derangement of action-to-successor bindings;
* recurrent planning disabled while retaining every recurrent parameter; and
* seeded nonexpert labels with the full treatment architecture.

Preparation labels may use the existing canonical trace oracle.  The oracle is
locked before any optimization or evaluation.  The original strict canonical
verifier is called only by the posthoc assessor after a candidate halts.
This is an isolated mechanics falsifier and cannot authorize pretraining.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from hashlib import sha256
import inspect
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


ARCHITECTURE_SCHEMA = "ssqac_successor_value_controller_v1"
EXPERIMENT_SCHEMA = "ssqac_successor_value_experiment_v1"
RESOURCE_SCHEMA = "ssqac_successor_value_resource_counts_v1"
STATUS = "isolated_successor_planning_falsifier_not_reasoning"
CLAIM_NO_GO = "successor_planning_falsified_or_below_material_gate"
CLAIM_SUGGESTIVE = "suggestive_successor_signal_below_material_gate"
CLAIM_MATERIAL = (
    "material_successor_planning_mechanics_pass_replication_required_not_reasoning"
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
ACTION_TO_INDEX = {name: index for index, name in enumerate(ACTION_TYPES)}

MODE_RAW = "raw_successors"
MODE_ZERO = "zeroed_successors"
MODE_SHUFFLED = "shuffled_action_successor_bindings"
INPUT_MODES = (MODE_RAW, MODE_ZERO, MODE_SHUFFLED)

ARM_TREATMENT = "successor_recurrent"
ARM_PROGRESSIVE = "progressive_randomized_depth"
ARM_ZERO = "successor_zeroed"
ARM_SHUFFLED = "successor_binding_shuffled"
ARM_NO_RECURRENCE = "recurrence_disabled"
ARM_RANDOM_LABELS = "random_labels"
ARMS = (
    ARM_TREATMENT,
    ARM_PROGRESSIVE,
    ARM_ZERO,
    ARM_SHUFFLED,
    ARM_NO_RECURRENCE,
    ARM_RANDOM_LABELS,
)

DEPTH_FIXED = "fixed_depth_only"
DEPTH_PROGRESSIVE = "progressive_paired_random_depth"
DEPTH_DISABLED = "recurrence_disabled"
DEPTH_REGIMES = (DEPTH_FIXED, DEPTH_PROGRESSIVE, DEPTH_DISABLED)

MAX_ROWS = 32
MAX_COLUMNS = 32
DEFAULT_REGISTER_COUNT = 4
PROTECTED_FLAGSHIP_PARAMETERS = 125_081_664
TOTAL_PARAMETER_BUDGET = 200_000_000

ROLE_FEATURES = 8
KIND_FEATURES = len(ACTION_TYPES)


class SuccessorValueError(ValueError):
    """The successor-aware falsifier contract failed closed."""


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SuccessorValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SuccessorValueError(f"{label} must be a nonnegative integer")
    return value


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise SuccessorValueError("value is not canonical ASCII JSON") from error


def _digest(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def canonical_matrix(
    rows: Iterable[Iterable[int]],
) -> tuple[tuple[int, ...], ...]:
    """Freeze a bounded rectangular matrix over F_257."""

    matrix = tuple(tuple(int(value) % FIELD_MODULUS for value in row) for row in rows)
    if not matrix or not matrix[0]:
        raise SuccessorValueError("matrix must be nonempty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise SuccessorValueError("matrix rows have inconsistent widths")
    if len(matrix) > MAX_ROWS or width > MAX_COLUMNS:
        raise SuccessorValueError(
            f"matrix exceeds the {MAX_ROWS}x{MAX_COLUMNS} mechanics bound"
        )
    return matrix


def matrix_sha256(rows: Iterable[Iterable[int]]) -> str:
    return _digest([list(row) for row in canonical_matrix(rows)])


@dataclass(frozen=True, slots=True, order=True)
class SuccessorAction:
    """One legal local macro action."""

    kind: str
    row_a: int = 0
    row_b: int = 0
    column: int = 0

    def __post_init__(self) -> None:
        if self.kind not in ACTION_TYPES:
            raise SuccessorValueError(f"unknown action kind {self.kind!r}")
        for label, value in (
            ("row_a", self.row_a),
            ("row_b", self.row_b),
            ("column", self.column),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise SuccessorValueError(f"{label} must be an integer")

    def canonical_data(self) -> list[object]:
        return [self.kind, self.row_a, self.row_b, self.column]

    @property
    def sha256(self) -> str:
        return _digest(self.canonical_data())


def enumerate_legal_actions(
    rows: Iterable[Iterable[int]],
) -> tuple[SuccessorAction, ...]:
    """Enumerate local actions without consulting endpoint correctness."""

    matrix = canonical_matrix(rows)
    row_count = len(matrix)
    column_count = len(matrix[0])
    actions: list[SuccessorAction] = []
    for row in range(row_count):
        for column in range(column_count):
            if matrix[row][column] not in (0, 1):
                actions.append(
                    SuccessorAction(
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
                        SuccessorAction(
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
                    SuccessorAction(
                        ACTION_SWAP,
                        row_a=left,
                        row_b=right,
                    )
                )
    # HALT is unconditional so the legal-action renderer does not disclose
    # whether the posthoc strict endpoint accepts this state.
    actions.append(SuccessorAction(ACTION_HALT))
    return tuple(actions)


def apply_action(
    rows: Iterable[Iterable[int]],
    action: SuccessorAction,
) -> tuple[tuple[int, ...], ...]:
    """Apply one legal macro exactly, without scoring the result."""

    matrix = canonical_matrix(rows)
    if not isinstance(action, SuccessorAction):
        raise SuccessorValueError("action has the wrong type")
    if action not in enumerate_legal_actions(matrix):
        raise SuccessorValueError("action is not legal in the supplied matrix")
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
        raise SuccessorValueError("unreachable action dispatch")
    return tuple(tuple(row) for row in mutable)


def compile_action_to_primitives(
    rows: Iterable[Iterable[int]],
    action: SuccessorAction,
) -> tuple[AlgebraInstruction, ...]:
    """Compile one validated macro to the original primitive VM."""

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


def compile_trace_to_primitives(
    input_rows: Iterable[Iterable[int]],
    actions: Sequence[SuccessorAction],
) -> tuple[AlgebraInstruction, ...]:
    """Compile a complete hard-action trace, requiring one final HALT."""

    matrix = canonical_matrix(input_rows)
    frozen = tuple(actions)
    if not frozen or frozen[-1].kind != ACTION_HALT:
        raise SuccessorValueError("trace must terminate with HALT")
    if any(action.kind == ACTION_HALT for action in frozen[:-1]):
        raise SuccessorValueError("trace contains an action after HALT")
    program: list[AlgebraInstruction] = []
    for action in frozen:
        program.extend(compile_action_to_primitives(matrix, action))
        matrix = apply_action(matrix, action)
    return tuple(program)


@dataclass(frozen=True, slots=True)
class LabeledSuccessorState:
    rows: tuple[tuple[int, ...], ...]
    target_action: SuccessorAction

    @property
    def sha256(self) -> str:
        return _digest(
            [
                [list(row) for row in self.rows],
                self.target_action.canonical_data(),
            ]
        )


@dataclass(frozen=True, slots=True)
class PreparationResult:
    states: tuple[LabeledSuccessorState, ...]
    oracle_calls: int
    oracle_source_sha256: str


_PREPARATION_LOCKED = False


def lock_preparation_oracle() -> None:
    """Permanently close the preparation boundary in this process."""

    global _PREPARATION_LOCKED
    _PREPARATION_LOCKED = True


def build_preparation_states(
    matrices: Iterable[Iterable[Iterable[int]]],
    *,
    maximum_steps: int,
) -> PreparationResult:
    """Use the existing canonical trace oracle only before the lock."""

    if _PREPARATION_LOCKED:
        raise SuccessorValueError("preparation oracle is locked")
    limit = _positive_int(maximum_steps, label="maximum_steps")
    # The import is deliberately local.  No candidate or evaluation function
    # retains the oracle module or counter.
    import ssqac_soft_value_iteration_controller as preparation_oracle

    source_path = Path(inspect.getsourcefile(preparation_oracle) or "")
    if not source_path.is_file():
        raise SuccessorValueError("preparation oracle source is unavailable")
    source_sha = sha256(source_path.read_bytes()).hexdigest()
    counter = preparation_oracle.PreparationOracleCounter()
    states: dict[str, LabeledSuccessorState] = {}
    for raw in matrices:
        matrix = canonical_matrix(raw)
        for _ in range(limit):
            oracle_action = preparation_oracle.next_preparation_macro(
                matrix,
                counter=counter,
            )
            target = SuccessorAction(
                oracle_action.kind,
                row_a=oracle_action.row_a,
                row_b=oracle_action.row_b,
                column=oracle_action.column,
            )
            if target not in enumerate_legal_actions(matrix):
                raise SuccessorValueError("oracle emitted a nonlocal action")
            state = LabeledSuccessorState(matrix, target)
            prior = states.get(matrix_sha256(matrix))
            if prior is not None and prior != state:
                raise SuccessorValueError("preparation labels conflict")
            states[matrix_sha256(matrix)] = state
            if target.kind == ACTION_HALT:
                break
            matrix = apply_action(matrix, target)
        else:
            raise SuccessorValueError("preparation trace exceeded its step bound")
    return PreparationResult(
        states=tuple(states[key] for key in sorted(states)),
        oracle_calls=counter.calls,
        oracle_source_sha256=source_sha,
    )


def make_random_label_control(
    states: Sequence[LabeledSuccessorState],
    *,
    seed: int,
) -> tuple[LabeledSuccessorState, ...]:
    """Replace each expert label by a deterministic nonexpert legal label."""

    rng = random.Random(seed)
    result = []
    for state in states:
        alternatives = [
            action
            for action in enumerate_legal_actions(state.rows)
            if action != state.target_action
        ]
        target = rng.choice(alternatives) if alternatives else state.target_action
        result.append(
            LabeledSuccessorState(
                rows=state.rows,
                target_action=target,
            )
        )
    return tuple(result)


def state_manifest(states: Iterable[LabeledSuccessorState]) -> str:
    return sha256(
        (
            "\n".join(
                state.sha256 for state in sorted(states, key=lambda item: item.sha256)
            )
            + "\n"
        ).encode("ascii")
    ).hexdigest()


@dataclass(slots=True)
class MutableResourceCounts:
    successor_evaluations: int = 0
    successor_matrix_cells: int = 0
    model_forward_calls: int = 0
    action_candidates_scored: int = 0
    planner_iterations: int = 0
    recurrent_action_updates: int = 0
    oracle_calls: int = 0
    search_calls: int = 0
    verifier_calls: int = 0

    def freeze(self) -> ResourceCounts:
        return ResourceCounts(
            schema=RESOURCE_SCHEMA,
            successor_evaluations=self.successor_evaluations,
            successor_matrix_cells=self.successor_matrix_cells,
            model_forward_calls=self.model_forward_calls,
            action_candidates_scored=self.action_candidates_scored,
            planner_iterations=self.planner_iterations,
            recurrent_action_updates=self.recurrent_action_updates,
            oracle_calls=self.oracle_calls,
            search_calls=self.search_calls,
            verifier_calls=self.verifier_calls,
        )


@dataclass(frozen=True, slots=True)
class ResourceCounts:
    schema: str
    successor_evaluations: int
    successor_matrix_cells: int
    model_forward_calls: int
    action_candidates_scored: int
    planner_iterations: int
    recurrent_action_updates: int
    oracle_calls: int
    search_calls: int
    verifier_calls: int


@dataclass(frozen=True, slots=True)
class RenderedCounterfactuals:
    matrix: tuple[tuple[int, ...], ...]
    actions: tuple[SuccessorAction, ...]
    true_successors: tuple[tuple[tuple[int, ...], ...], ...]
    visible_successors: tuple[tuple[tuple[int, ...], ...], ...]
    binding_manifest_sha256: str


def _binding_derangement(
    matrix: tuple[tuple[int, ...], ...],
    actions: Sequence[SuccessorAction],
    *,
    seed: int,
) -> Mapping[SuccessorAction, SuccessorAction]:
    """Return an order-independent deterministic cyclic derangement."""

    ordered = tuple(sorted(actions, key=lambda action: action.canonical_data()))
    if len(ordered) == 1:
        return {ordered[0]: ordered[0]}
    offset_digest = sha256(
        _canonical_bytes(
            {
                "matrix": [list(row) for row in matrix],
                "seed": seed,
                "actions": [action.canonical_data() for action in ordered],
            }
        )
    ).digest()
    shift = 1 + int.from_bytes(offset_digest[:8], "big") % (len(ordered) - 1)
    return {
        action: ordered[(index + shift) % len(ordered)]
        for index, action in enumerate(ordered)
    }


def render_counterfactuals(
    rows: Iterable[Iterable[int]],
    *,
    mode: str,
    binding_seed: int,
    resources: MutableResourceCounts,
    actions: Sequence[SuccessorAction] | None = None,
) -> RenderedCounterfactuals:
    """Evaluate every raw successor, then apply the requested input ablation."""

    if mode not in INPUT_MODES:
        raise SuccessorValueError(f"unknown input mode {mode!r}")
    matrix = canonical_matrix(rows)
    legal = enumerate_legal_actions(matrix)
    rendered_actions = legal if actions is None else tuple(actions)
    if len(rendered_actions) != len(legal) or set(rendered_actions) != set(legal):
        raise SuccessorValueError("candidate actions differ from the legal action set")
    if len(set(rendered_actions)) != len(rendered_actions):
        raise SuccessorValueError("candidate action sequence contains duplicates")
    successor_by_action = {
        action: apply_action(matrix, action) for action in rendered_actions
    }
    resources.successor_evaluations += len(rendered_actions)
    resources.successor_matrix_cells += (
        len(rendered_actions) * len(matrix) * len(matrix[0])
    )
    true_successors = tuple(
        successor_by_action[action] for action in rendered_actions
    )
    if mode == MODE_RAW:
        visible = true_successors
        binding = {action: action for action in rendered_actions}
    elif mode == MODE_ZERO:
        zero = tuple(tuple(0 for _ in matrix[0]) for _ in matrix)
        visible = tuple(zero for _ in rendered_actions)
        binding = {action: action for action in rendered_actions}
    else:
        binding = _binding_derangement(
            matrix,
            rendered_actions,
            seed=binding_seed,
        )
        visible = tuple(
            successor_by_action[binding[action]] for action in rendered_actions
        )
    binding_manifest = _digest(
        [
            [action.canonical_data(), binding[action].canonical_data()]
            for action in sorted(rendered_actions, key=lambda item: item.canonical_data())
        ]
    )
    return RenderedCounterfactuals(
        matrix=matrix,
        actions=rendered_actions,
        true_successors=true_successors,
        visible_successors=visible,
        binding_manifest_sha256=binding_manifest,
    )


@dataclass(frozen=True, slots=True)
class SuccessorValueConfig:
    field_width: int = 64
    width: int = 256
    cell_hidden: int = 384
    matrix_layers: int = 4
    planner_hidden: int = 512
    planner_iterations: int = 8
    coordinate_harmonics: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for label, value in (
            ("field_width", self.field_width),
            ("width", self.width),
            ("cell_hidden", self.cell_hidden),
            ("matrix_layers", self.matrix_layers),
            ("planner_hidden", self.planner_hidden),
            ("coordinate_harmonics", self.coordinate_harmonics),
        ):
            _positive_int(value, label=label)
        _nonnegative_int(self.planner_iterations, label="planner_iterations")
        if not isinstance(self.dropout, float) or not 0.0 <= self.dropout < 1.0:
            raise SuccessorValueError("dropout must be a float in [0, 1)")


class _EquivariantGridLayer(nn.Module):
    def __init__(self, width: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(width * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, cells: Tensor) -> Tensor:
        row_mean = cells.mean(dim=2, keepdim=True).expand_as(cells)
        column_mean = cells.mean(dim=1, keepdim=True).expand_as(cells)
        global_mean = cells.mean(dim=(1, 2), keepdim=True).expand_as(cells)
        update = self.update(
            torch.cat((cells, row_mean, column_mean, global_mean), dim=-1)
        )
        return self.norm(cells + update)


@dataclass(frozen=True, slots=True)
class SuccessorScores:
    actions: tuple[SuccessorAction, ...]
    logits: Tensor
    binding_manifest_sha256: str
    planner_iterations: int


class SuccessorValueController(nn.Module):
    """Geometry-general counterfactual encoder plus recurrent set planner."""

    def __init__(
        self,
        config: SuccessorValueConfig = SuccessorValueConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        coordinate_width = 4 * config.coordinate_harmonics + 2
        cell_input = (
            2 * config.field_width
            + coordinate_width
            + ROLE_FEATURES
            + KIND_FEATURES
        )
        self.field_embedding = nn.Embedding(FIELD_MODULUS, config.field_width)
        self.cell_projection = nn.Sequential(
            nn.Linear(cell_input, config.cell_hidden),
            nn.GELU(),
            nn.Linear(config.cell_hidden, config.width),
            nn.LayerNorm(config.width),
        )
        self.grid_layers = nn.ModuleList(
            _EquivariantGridLayer(
                config.width,
                config.cell_hidden,
                config.dropout,
            )
            for _ in range(config.matrix_layers)
        )
        pooled_width = config.width * 7 + KIND_FEATURES
        self.action_projection = nn.Sequential(
            nn.Linear(pooled_width, config.planner_hidden),
            nn.GELU(),
            nn.LayerNorm(config.planner_hidden),
        )
        self.planner_input = nn.Sequential(
            nn.Linear(config.planner_hidden * 4, config.planner_hidden),
            nn.GELU(),
        )
        self.raw_recall_projection = nn.Sequential(
            nn.Linear(config.planner_hidden, config.planner_hidden),
            nn.Sigmoid(),
        )
        self.shared_planner_cell = nn.GRUCell(
            config.planner_hidden,
            config.planner_hidden,
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(config.planner_hidden),
            nn.Linear(config.planner_hidden, config.planner_hidden),
            nn.GELU(),
            nn.Linear(config.planner_hidden, 1),
        )
        if self.complete_system_parameter_count >= TOTAL_PARAMETER_BUDGET:
            raise SuccessorValueError("complete system exceeds the parameter budget")

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def complete_system_parameter_count(self) -> int:
        return PROTECTED_FLAGSHIP_PARAMETERS + self.parameter_count

    def parameter_count_breakdown(self) -> Mapping[str, int]:
        groups = {
            "field_embedding": sum(
                parameter.numel() for parameter in self.field_embedding.parameters()
            ),
            "cell_projection": sum(
                parameter.numel() for parameter in self.cell_projection.parameters()
            ),
            "grid_layers": sum(
                parameter.numel() for parameter in self.grid_layers.parameters()
            ),
            "action_projection": sum(
                parameter.numel() for parameter in self.action_projection.parameters()
            ),
            "planner_input": sum(
                parameter.numel() for parameter in self.planner_input.parameters()
            ),
            "raw_recall_projection": sum(
                parameter.numel()
                for parameter in self.raw_recall_projection.parameters()
            ),
            "shared_planner_cell": sum(
                parameter.numel()
                for parameter in self.shared_planner_cell.parameters()
            ),
            "value_head": sum(
                parameter.numel() for parameter in self.value_head.parameters()
            ),
        }
        groups["total"] = sum(groups.values())
        return groups

    def _coordinate_features(
        self,
        row_codes: Tensor,
        column_codes: Tensor,
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        row_denominator = torch.clamp(row_codes.max() + 1.0, min=1.0)
        column_denominator = torch.clamp(column_codes.max() + 1.0, min=1.0)
        row = (row_codes + 0.5) / row_denominator
        column = (column_codes + 0.5) / column_denominator
        row_grid = row[:, None].expand(row.numel(), column.numel())
        column_grid = column[None, :].expand(row.numel(), column.numel())
        features = [row_grid, column_grid]
        for harmonic in range(1, self.config.coordinate_harmonics + 1):
            angle = math.pi * harmonic
            features.extend(
                (
                    torch.sin(angle * row_grid),
                    torch.cos(angle * row_grid),
                    torch.sin(angle * column_grid),
                    torch.cos(angle * column_grid),
                )
            )
        return torch.stack(features, dim=-1).to(dtype=dtype)

    @staticmethod
    def _role_features(
        action_kind: Tensor,
        row_a: Tensor,
        row_b: Tensor,
        column: Tensor,
        *,
        row_count: int,
        column_count: int,
        dtype: torch.dtype,
    ) -> Tensor:
        device = action_kind.device
        row_index = torch.arange(row_count, device=device)
        column_index = torch.arange(column_count, device=device)
        first = row_index[None, :, None] == row_a[:, None, None]
        second = row_index[None, :, None] == row_b[:, None, None]
        selected_column = (
            column_index[None, None, :] == column[:, None, None]
        )
        halt = action_kind == ACTION_TO_INDEX[ACTION_HALT]
        first = first.expand(-1, -1, column_count) & ~halt[:, None, None]
        second = (
            second.expand(-1, -1, column_count)
            & ~halt[:, None, None]
            & (
                (action_kind == ACTION_TO_INDEX[ACTION_ELIMINATE])
                | (action_kind == ACTION_TO_INDEX[ACTION_SWAP])
            )[:, None, None]
        )
        selected_column = (
            selected_column.expand(-1, row_count, -1)
            & ~halt[:, None, None]
            & (action_kind != ACTION_TO_INDEX[ACTION_SWAP])[:, None, None]
        )
        any_selected_row = first | second
        return torch.stack(
            (
                first,
                second,
                any_selected_row,
                selected_column,
                first & selected_column,
                second & selected_column,
                ~any_selected_row,
                ~selected_column,
            ),
            dim=-1,
        ).to(dtype=dtype)

    @staticmethod
    def _masked_mean(cells: Tensor, mask: Tensor) -> Tensor:
        weights = mask.to(cells.dtype)
        denominator = weights.sum(dim=(1, 2)).clamp_min(1.0)
        return (cells * weights[..., None]).sum(dim=(1, 2)) / denominator[:, None]

    def _encode_counterfactuals(
        self,
        current_values: Tensor,
        successor_values: Tensor,
        action_kind: Tensor,
        row_a: Tensor,
        row_b: Tensor,
        column: Tensor,
        row_codes: Tensor,
        column_codes: Tensor,
    ) -> Tensor:
        if current_values.ndim != 2 or successor_values.ndim != 3:
            raise SuccessorValueError("matrix tensors have the wrong rank")
        action_count, row_count, column_count = successor_values.shape
        if tuple(current_values.shape) != (row_count, column_count):
            raise SuccessorValueError("current and successor geometries differ")
        for name, tensor in (
            ("action_kind", action_kind),
            ("row_a", row_a),
            ("row_b", row_b),
            ("column", column),
        ):
            if tensor.shape != (action_count,):
                raise SuccessorValueError(f"{name} has the wrong shape")
        if row_codes.shape != (row_count,) or column_codes.shape != (column_count,):
            raise SuccessorValueError("coordinate labels have the wrong shape")
        current = self.field_embedding(current_values)
        successor = self.field_embedding(successor_values)
        current = current[None, :, :, :].expand(action_count, -1, -1, -1)
        coordinate = self._coordinate_features(
            row_codes,
            column_codes,
            dtype=current.dtype,
        )
        coordinate = coordinate[None, :, :, :].expand(action_count, -1, -1, -1)
        roles = self._role_features(
            action_kind,
            row_a,
            row_b,
            column,
            row_count=row_count,
            column_count=column_count,
            dtype=current.dtype,
        )
        kinds = F.one_hot(
            action_kind,
            num_classes=KIND_FEATURES,
        ).to(current.dtype)
        kind_grid = kinds[:, None, None, :].expand(
            action_count,
            row_count,
            column_count,
            KIND_FEATURES,
        )
        cells = self.cell_projection(
            torch.cat(
                (
                    current,
                    successor,
                    coordinate,
                    roles,
                    kind_grid,
                ),
                dim=-1,
            )
        )
        for layer in self.grid_layers:
            cells = layer(cells)
        first_mask = roles[..., 0].bool()
        second_mask = roles[..., 1].bool()
        column_mask = roles[..., 3].bool()
        first_column_mask = roles[..., 4].bool()
        second_column_mask = roles[..., 5].bool()
        global_mean = cells.mean(dim=(1, 2))
        global_max = cells.amax(dim=(1, 2))
        pooled = torch.cat(
            (
                global_mean,
                global_max,
                self._masked_mean(cells, first_mask),
                self._masked_mean(cells, second_mask),
                self._masked_mean(cells, column_mask),
                self._masked_mean(cells, first_column_mask),
                self._masked_mean(cells, second_column_mask),
                kinds,
            ),
            dim=-1,
        )
        return self.action_projection(pooled)

    def forward(
        self,
        current_values: Tensor,
        successor_values: Tensor,
        action_kind: Tensor,
        row_a: Tensor,
        row_b: Tensor,
        column: Tensor,
        row_codes: Tensor,
        column_codes: Tensor,
        planner_iterations: int,
    ) -> Tensor:
        """Score one variable-size action set using fixed shared recurrence."""

        iterations = _nonnegative_int(
            planner_iterations,
            label="planner_iterations",
        )
        hidden = self._encode_counterfactuals(
            current_values,
            successor_values,
            action_kind,
            row_a,
            row_b,
            column,
            row_codes,
            column_codes,
        )
        local = hidden
        for _ in range(iterations):
            mean = hidden.mean(dim=0, keepdim=True).expand_as(hidden)
            maximum = hidden.amax(dim=0, keepdim=True).expand_as(hidden)
            planner_input = self.planner_input(
                torch.cat((local, hidden, mean, maximum), dim=-1)
            )
            recurrent = self.shared_planner_cell(planner_input, hidden)
            recall_gate = self.raw_recall_projection(hidden)
            hidden = recurrent + recall_gate * local
        return self.value_head(hidden).squeeze(-1)

    def score_actions(
        self,
        rows: Iterable[Iterable[int]],
        *,
        mode: str,
        binding_seed: int,
        planner_iterations: int,
        resources: MutableResourceCounts,
        actions: Sequence[SuccessorAction] | None = None,
        row_codes: Sequence[int] | None = None,
        column_codes: Sequence[int] | None = None,
    ) -> SuccessorScores:
        rendered = render_counterfactuals(
            rows,
            mode=mode,
            binding_seed=binding_seed,
            resources=resources,
            actions=actions,
        )
        reference = next(self.parameters())
        device = reference.device
        matrix = rendered.matrix
        action_kind = torch.tensor(
            [ACTION_TO_INDEX[action.kind] for action in rendered.actions],
            dtype=torch.long,
            device=device,
        )
        row_a = torch.tensor(
            [action.row_a for action in rendered.actions],
            dtype=torch.long,
            device=device,
        )
        row_b = torch.tensor(
            [action.row_b for action in rendered.actions],
            dtype=torch.long,
            device=device,
        )
        column = torch.tensor(
            [action.column for action in rendered.actions],
            dtype=torch.long,
            device=device,
        )
        row_labels = (
            tuple(range(len(matrix))) if row_codes is None else tuple(row_codes)
        )
        column_labels = (
            tuple(range(len(matrix[0])))
            if column_codes is None
            else tuple(column_codes)
        )
        logits = self(
            torch.tensor(matrix, dtype=torch.long, device=device),
            torch.tensor(
                rendered.visible_successors,
                dtype=torch.long,
                device=device,
            ),
            action_kind,
            row_a,
            row_b,
            column,
            torch.tensor(row_labels, dtype=torch.float32, device=device),
            torch.tensor(column_labels, dtype=torch.float32, device=device),
            planner_iterations,
        )
        resources.model_forward_calls += 1
        resources.action_candidates_scored += len(rendered.actions)
        resources.planner_iterations += planner_iterations
        resources.recurrent_action_updates += (
            planner_iterations * len(rendered.actions)
        )
        return SuccessorScores(
            actions=rendered.actions,
            logits=logits,
            binding_manifest_sha256=rendered.binding_manifest_sha256,
            planner_iterations=planner_iterations,
        )


def model_state_sha256(model: SuccessorValueController) -> str:
    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _target_index(
    actions: Sequence[SuccessorAction],
    target: SuccessorAction,
) -> int:
    try:
        return tuple(actions).index(target)
    except ValueError as error:
        raise SuccessorValueError("target is absent from candidate actions") from error


@dataclass(frozen=True, slots=True)
class TrainingResult:
    optimizer_updates: int
    mean_loss: float
    final_loss: float
    resources: ResourceCounts


def _autocast(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def progressive_paired_depths(
    *,
    update_index: int,
    total_updates: int,
    batch_size: int,
    fixed_depth: int,
    rng: random.Random,
) -> tuple[int, ...]:
    """Return paired random depths with exactly the fixed-depth compute sum."""

    update = _nonnegative_int(update_index, label="update_index")
    updates = _positive_int(total_updates, label="total_updates")
    batch = _positive_int(batch_size, label="batch_size")
    depth = _nonnegative_int(fixed_depth, label="fixed_depth")
    if update >= updates:
        raise SuccessorValueError("update_index leaves the training schedule")
    if batch % 2:
        raise SuccessorValueError(
            "progressive paired-depth training requires an even batch size"
        )
    span = min(depth, math.floor(depth * (update + 1) / updates))
    result: list[int] = []
    for _ in range(batch // 2):
        delta = rng.randint(0, span)
        if rng.randrange(2):
            result.extend((depth - delta, depth + delta))
        else:
            result.extend((depth + delta, depth - delta))
    if sum(result) != batch * depth:
        raise SuccessorValueError("paired depth schedule changed compute budget")
    return tuple(result)


def train_controller(
    controller: SuccessorValueController,
    states: Sequence[LabeledSuccessorState],
    *,
    input_mode: str,
    depth_regime: str,
    binding_seed: int,
    optimizer_updates: int,
    batch_size: int,
    learning_rate: float,
    shuffle_seed: int,
    amp_bfloat16: bool,
) -> TrainingResult:
    """Run exactly the requested number of matched optimizer updates."""

    updates = _positive_int(optimizer_updates, label="optimizer_updates")
    batch = _positive_int(batch_size, label="batch_size")
    if not states:
        raise SuccessorValueError("training states must be nonempty")
    if depth_regime not in DEPTH_REGIMES:
        raise SuccessorValueError(f"unknown depth regime {depth_regime!r}")
    if depth_regime == DEPTH_PROGRESSIVE and batch % 2:
        raise SuccessorValueError(
            "progressive depth regime requires an even batch size"
        )
    if not isinstance(learning_rate, float) or learning_rate <= 0.0:
        raise SuccessorValueError("learning_rate must be a positive float")
    device = next(controller.parameters()).device
    optimizer = torch.optim.AdamW(
        controller.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        fused=device.type == "cuda",
    )
    rng = random.Random(shuffle_seed)
    order = list(range(len(states)))
    cursor = len(order)
    resources = MutableResourceCounts()
    losses: list[float] = []
    controller.train()
    depth_rng = random.Random(shuffle_seed + 1_000_003)
    fixed_depth = controller.config.planner_iterations
    for update_index in range(updates):
        if cursor + batch > len(order):
            rng.shuffle(order)
            cursor = 0
        indices = order[cursor : cursor + batch]
        cursor += batch
        if depth_regime == DEPTH_FIXED:
            depths = (fixed_depth,) * len(indices)
        elif depth_regime == DEPTH_DISABLED:
            depths = (0,) * len(indices)
        else:
            depths = progressive_paired_depths(
                update_index=update_index,
                total_updates=updates,
                batch_size=len(indices),
                fixed_depth=fixed_depth,
                rng=depth_rng,
            )
        optimizer.zero_grad(set_to_none=True)
        state_losses = []
        with _autocast(device, amp_bfloat16):
            for index, planner_depth in zip(indices, depths, strict=True):
                state = states[index]
                scored = controller.score_actions(
                    state.rows,
                    mode=input_mode,
                    binding_seed=binding_seed,
                    planner_iterations=planner_depth,
                    resources=resources,
                )
                target = torch.tensor(
                    [_target_index(scored.actions, state.target_action)],
                    dtype=torch.long,
                    device=device,
                )
                state_losses.append(
                    F.cross_entropy(scored.logits.float()[None, :], target)
                )
            loss = torch.stack(state_losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return TrainingResult(
        optimizer_updates=updates,
        mean_loss=sum(losses) / len(losses),
        final_loss=losses[-1],
        resources=resources.freeze(),
    )


@torch.no_grad()
def label_accuracy(
    controller: SuccessorValueController,
    states: Sequence[LabeledSuccessorState],
    *,
    input_mode: str,
    planner_iterations: int,
    binding_seed: int,
) -> tuple[int, int, ResourceCounts]:
    controller.eval()
    correct = 0
    resources = MutableResourceCounts()
    for state in states:
        scored = controller.score_actions(
            state.rows,
            mode=input_mode,
            binding_seed=binding_seed,
            planner_iterations=planner_iterations,
            resources=resources,
        )
        predicted = scored.actions[int(scored.logits.argmax().item())]
        correct += predicted == state.target_action
    return correct, len(states), resources.freeze()


@dataclass(frozen=True, slots=True)
class CandidateRollout:
    halted: bool
    invalid: bool
    overlong: bool
    actions: tuple[SuccessorAction, ...]
    output_rows: tuple[tuple[int, ...], ...]
    resources: ResourceCounts


@torch.no_grad()
def candidate_successor_only_rollout(
    controller: SuccessorValueController,
    rows: Iterable[Iterable[int]],
    *,
    input_mode: str,
    planner_iterations: int,
    binding_seed: int,
    maximum_steps: int,
) -> CandidateRollout:
    """Run without oracle, search, verifier, rank, energy, or frontier."""

    if not _PREPARATION_LOCKED:
        raise SuccessorValueError("preparation oracle must be locked before rollout")
    limit = _positive_int(maximum_steps, label="maximum_steps")
    matrix = canonical_matrix(rows)
    actions: list[SuccessorAction] = []
    resources = MutableResourceCounts()
    controller.eval()
    for _ in range(limit):
        scored = controller.score_actions(
            matrix,
            mode=input_mode,
            binding_seed=binding_seed,
            planner_iterations=planner_iterations,
            resources=resources,
        )
        action = scored.actions[int(scored.logits.argmax().item())]
        actions.append(action)
        if action.kind == ACTION_HALT:
            return CandidateRollout(
                halted=True,
                invalid=False,
                overlong=False,
                actions=tuple(actions),
                output_rows=matrix,
                resources=resources.freeze(),
            )
        try:
            matrix = apply_action(matrix, action)
        except SuccessorValueError:
            return CandidateRollout(
                halted=False,
                invalid=True,
                overlong=False,
                actions=tuple(actions),
                output_rows=matrix,
                resources=resources.freeze(),
            )
    return CandidateRollout(
        halted=False,
        invalid=False,
        overlong=True,
        actions=tuple(actions),
        output_rows=matrix,
        resources=resources.freeze(),
    )


@dataclass(frozen=True, slots=True)
class AssessedRollout:
    strict_canonical_certified: bool
    invalid: bool
    overlong: bool
    posthoc_verifier_calls: int


def assess_rollout_posthoc(
    input_rows: Iterable[Iterable[int]],
    rollout: CandidateRollout,
) -> AssessedRollout:
    """Call the original strict verifier only after candidate termination."""

    source = canonical_matrix(input_rows)
    if rollout.overlong:
        return AssessedRollout(False, False, True, 0)
    if rollout.invalid or not rollout.halted:
        return AssessedRollout(False, True, False, 0)
    try:
        program = compile_trace_to_primitives(source, rollout.actions)
        state = execute_program(
            source,
            program,
            register_count=DEFAULT_REGISTER_COUNT,
        )
        receipt = verify_reduction_program(source, state)
    except (AlgebraMachineError, SuccessorValueError):
        return AssessedRollout(False, True, False, 1)
    return AssessedRollout(
        strict_canonical_certified=receipt.passed,
        invalid=not receipt.passed,
        overlong=False,
        posthoc_verifier_calls=1,
    )


def _sum_resources(values: Sequence[ResourceCounts]) -> ResourceCounts:
    return ResourceCounts(
        schema=RESOURCE_SCHEMA,
        successor_evaluations=sum(item.successor_evaluations for item in values),
        successor_matrix_cells=sum(item.successor_matrix_cells for item in values),
        model_forward_calls=sum(item.model_forward_calls for item in values),
        action_candidates_scored=sum(item.action_candidates_scored for item in values),
        planner_iterations=sum(item.planner_iterations for item in values),
        recurrent_action_updates=sum(
            item.recurrent_action_updates for item in values
        ),
        oracle_calls=sum(item.oracle_calls for item in values),
        search_calls=sum(item.search_calls for item in values),
        verifier_calls=sum(item.verifier_calls for item in values),
    )


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    strict_canonical_certified: int
    total: int
    invalid: int
    overlong: int
    posthoc_verifier_calls: int
    resources: ResourceCounts

    @property
    def certification_rate(self) -> float:
        return self.strict_canonical_certified / self.total if self.total else 0.0


def evaluate_controller(
    controller: SuccessorValueController,
    matrices: Sequence[tuple[tuple[int, ...], ...]],
    *,
    input_mode: str,
    planner_iterations: int,
    binding_seed: int,
    maximum_steps: int,
) -> EvaluationResult:
    certified = invalid = overlong = posthoc = 0
    resources = []
    for matrix in matrices:
        rollout = candidate_successor_only_rollout(
            controller,
            matrix,
            input_mode=input_mode,
            planner_iterations=planner_iterations,
            binding_seed=binding_seed,
            maximum_steps=maximum_steps,
        )
        assessment = assess_rollout_posthoc(matrix, rollout)
        certified += int(assessment.strict_canonical_certified)
        invalid += int(assessment.invalid)
        overlong += int(assessment.overlong)
        posthoc += assessment.posthoc_verifier_calls
        resources.append(rollout.resources)
    return EvaluationResult(
        strict_canonical_certified=certified,
        total=len(matrices),
        invalid=invalid,
        overlong=overlong,
        posthoc_verifier_calls=posthoc,
        resources=_sum_resources(resources),
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
    """Generate exact-disjoint sparse matrices without rank filtering."""

    target = _positive_int(count, label="count")
    min_rows = _positive_int(minimum_rows, label="minimum_rows")
    max_rows = _positive_int(maximum_rows, label="maximum_rows")
    min_columns = _positive_int(minimum_columns, label="minimum_columns")
    max_columns = _positive_int(maximum_columns, label="maximum_columns")
    if min_rows > max_rows or min_columns > max_columns:
        raise SuccessorValueError("matrix generation bounds are inverted")
    if max_rows > MAX_ROWS or max_columns > MAX_COLUMNS:
        raise SuccessorValueError("matrix generation exceeds mechanics bounds")
    rng = random.Random(seed)
    seen = set() if excluded is None else set(excluded)
    result = []
    attempts = 0
    while len(result) < target and attempts < target * 10_000:
        attempts += 1
        row_count = rng.randint(min_rows, max_rows)
        column_count = rng.randint(max(row_count, min_columns), max_columns)
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
        raise SuccessorValueError("matrix generator exhausted bounded attempts")
    return tuple(result)


def matrix_manifest(
    matrices: Iterable[Iterable[Iterable[int]]],
) -> str:
    return sha256(
        ("\n".join(matrix_sha256(matrix) for matrix in matrices) + "\n").encode(
            "ascii"
        )
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ArmSpec:
    name: str
    input_mode: str
    depth_regime: str
    random_labels: bool


ARM_SPECS = (
    ArmSpec(ARM_TREATMENT, MODE_RAW, DEPTH_FIXED, False),
    ArmSpec(ARM_PROGRESSIVE, MODE_RAW, DEPTH_PROGRESSIVE, False),
    ArmSpec(ARM_ZERO, MODE_ZERO, DEPTH_FIXED, False),
    ArmSpec(ARM_SHUFFLED, MODE_SHUFFLED, DEPTH_FIXED, False),
    ArmSpec(ARM_NO_RECURRENCE, MODE_RAW, DEPTH_DISABLED, False),
    ArmSpec(ARM_RANDOM_LABELS, MODE_RAW, DEPTH_FIXED, True),
)


@dataclass(frozen=True, slots=True)
class ArmReport:
    name: str
    input_mode: str
    depth_regime: str
    random_labels: bool
    explicit_raw_input_recall_path: bool
    trained_evaluation_depth: int
    longer_evaluation_depth: int
    controller_parameters: int
    complete_system_parameters: int
    parameter_count_breakdown: Mapping[str, int]
    optimizer_updates: int
    mean_training_loss: float
    final_training_loss: float
    assigned_train_label_correct: int
    assigned_train_label_total: int
    true_train_label_correct: int
    true_train_label_total: int
    strict_canonical_certified: int
    evaluation_total: int
    certification_rate: float
    invalid: int
    overlong: int
    posthoc_verifier_calls: int
    longer_strict_canonical_certified: int
    longer_evaluation_total: int
    longer_certification_rate: float
    longer_invalid: int
    longer_overlong: int
    longer_posthoc_verifier_calls: int
    overthinking_accuracy_delta: float
    training_resources: ResourceCounts
    assigned_label_diagnostic_resources: ResourceCounts
    true_label_diagnostic_resources: ResourceCounts
    evaluation_resources: ResourceCounts
    longer_evaluation_resources: ResourceCounts
    model_state_sha256: str
    model_file_sha256: str | None


@dataclass(frozen=True, slots=True)
class SuccessorExperimentConfig:
    seed: int = 20260724
    train_matrices: int = 128
    evaluation_matrices: int = 96
    train_maximum_rows: int = 3
    train_maximum_columns: int = 4
    evaluation_minimum_rows: int = 4
    evaluation_minimum_columns: int = 5
    evaluation_maximum_rows: int = 4
    evaluation_maximum_columns: int = 6
    maximum_preparation_steps: int = 96
    maximum_rollout_steps: int = 192
    optimizer_updates: int = 1_200
    batch_size: int = 4
    learning_rate: float = 6e-4
    amp_bfloat16: bool = True
    material_minimum_cases: int = 96
    material_minimum_rate: float = 0.80
    material_minimum_control_gap: float = 0.10
    device: str = "cpu"
    controller: SuccessorValueConfig = SuccessorValueConfig()

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
            ("optimizer_updates", self.optimizer_updates),
            ("batch_size", self.batch_size),
            ("material_minimum_cases", self.material_minimum_cases),
        ):
            _positive_int(value, label=label)
        if self.evaluation_minimum_rows <= self.train_maximum_rows:
            raise SuccessorValueError("evaluation rows must be strictly larger")
        if self.evaluation_minimum_columns <= self.train_maximum_columns:
            raise SuccessorValueError("evaluation columns must be strictly larger")
        if self.evaluation_minimum_rows > self.evaluation_maximum_rows:
            raise SuccessorValueError("evaluation row bounds are inverted")
        if self.evaluation_minimum_columns > self.evaluation_maximum_columns:
            raise SuccessorValueError("evaluation column bounds are inverted")
        if not isinstance(self.learning_rate, float) or self.learning_rate <= 0.0:
            raise SuccessorValueError("learning_rate must be positive")
        for label, value in (
            ("material_minimum_rate", self.material_minimum_rate),
            ("material_minimum_control_gap", self.material_minimum_control_gap),
        ):
            if not isinstance(value, float) or not 0.0 <= value <= 1.0:
                raise SuccessorValueError(f"{label} must be in [0, 1]")
        if self.device not in ("cpu", "cuda"):
            raise SuccessorValueError("device must be cpu or cuda")


@dataclass(frozen=True, slots=True)
class SuccessorExperimentReport:
    schema: str
    status: str
    claim_classification: str
    seed: int
    device: str
    amp_bfloat16: bool
    source_sha256: str
    candidate_input_fields: tuple[str, ...]
    forbidden_candidate_inputs: tuple[str, ...]
    candidate_has_exact_energy: bool
    candidate_has_rank: bool
    candidate_has_frontier: bool
    candidate_has_reference_schedule: bool
    candidate_has_search: bool
    candidate_has_verifier: bool
    candidate_has_oracle: bool
    fixed_shared_weight_recurrence: bool
    explicit_raw_input_recall_path: bool
    fixed_depth_training_preregistered: bool
    progressive_random_depth_training_preregistered: bool
    longer_depth_overthinking_evaluation_preregistered: bool
    geometry_general_parameters: bool
    learned_absolute_geometry_tables: bool
    preparation_oracle_locked_before_training_and_eval: bool
    preparation_oracle_source_sha256: str
    preparation_oracle_calls: int
    train_matrices: int
    train_states: int
    evaluation_matrices: int
    strict_geometry_disjoint: bool
    train_matrix_manifest_sha256: str
    evaluation_matrix_manifest_sha256: str
    true_state_manifest_sha256: str
    random_label_state_manifest_sha256: str
    controls_equal_parameters: bool
    controls_equal_optimizer_updates: bool
    controls_equal_training_successor_evaluations: bool
    fixed_and_progressive_equal_training_planner_iterations: bool
    parameter_budget: int
    parameter_budget_passed: bool
    material_minimum_cases: int
    material_minimum_rate: float
    material_minimum_control_gap: float
    material_gate_passed: bool
    best_successor_arm: str
    best_successor_rate: float
    strongest_control_rate: float
    treatment_control_gap: float
    arms: tuple[ArmReport, ...]

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(asdict(self)) + b"\n"


def _save_model(
    controller: SuccessorValueController,
    *,
    model_dir: Path | None,
    arm: str,
    seed: int,
) -> str | None:
    if model_dir is None:
        return None
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{arm}_seed{seed}.pt"
    if path.exists():
        raise SuccessorValueError(f"model output already exists: {path}")
    torch.save(
        {
            "schema": ARCHITECTURE_SCHEMA,
            "arm": arm,
            "seed": seed,
            "state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in controller.state_dict().items()
            },
            "config": asdict(controller.config),
        },
        path,
    )
    return sha256(path.read_bytes()).hexdigest()


def run_bounded_experiment(
    config: SuccessorExperimentConfig,
    *,
    model_dir: Path | None = None,
) -> SuccessorExperimentReport:
    """Run the treatment and all four matched controls."""

    global _PREPARATION_LOCKED
    _PREPARATION_LOCKED = False
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SuccessorValueError("CUDA requested but unavailable")
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
    preparation = build_preparation_states(
        train,
        maximum_steps=config.maximum_preparation_steps,
    )
    true_states = preparation.states
    random_states = make_random_label_control(
        true_states,
        seed=config.seed + 2,
    )
    lock_preparation_oracle()

    torch.manual_seed(config.seed + 3)
    initial = SuccessorValueController(config.controller)
    initial_state = {
        name: tensor.detach().clone() for name, tensor in initial.state_dict().items()
    }
    arm_reports = []
    for spec in ARM_SPECS:
        controller = SuccessorValueController(config.controller)
        controller.load_state_dict(initial_state)
        controller.to(device)
        assigned_states = random_states if spec.random_labels else true_states
        training = train_controller(
            controller,
            assigned_states,
            input_mode=spec.input_mode,
            depth_regime=spec.depth_regime,
            binding_seed=config.seed + 5,
            optimizer_updates=config.optimizer_updates,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            shuffle_seed=config.seed + 4,
            amp_bfloat16=config.amp_bfloat16,
        )
        trained_depth = (
            0
            if spec.depth_regime == DEPTH_DISABLED
            else config.controller.planner_iterations
        )
        longer_depth = (
            config.controller.planner_iterations
            if spec.depth_regime == DEPTH_DISABLED
            else 2 * config.controller.planner_iterations
        )
        assigned_correct, assigned_total, assigned_resources = label_accuracy(
            controller,
            assigned_states,
            input_mode=spec.input_mode,
            planner_iterations=trained_depth,
            binding_seed=config.seed + 5,
        )
        true_correct, true_total, true_resources = label_accuracy(
            controller,
            true_states,
            input_mode=spec.input_mode,
            planner_iterations=trained_depth,
            binding_seed=config.seed + 5,
        )
        evaluated = evaluate_controller(
            controller,
            evaluation,
            input_mode=spec.input_mode,
            planner_iterations=trained_depth,
            binding_seed=config.seed + 5,
            maximum_steps=config.maximum_rollout_steps,
        )
        longer_evaluated = evaluate_controller(
            controller,
            evaluation,
            input_mode=spec.input_mode,
            planner_iterations=longer_depth,
            binding_seed=config.seed + 5,
            maximum_steps=config.maximum_rollout_steps,
        )
        candidate_forbidden_calls = (
            evaluated.resources.oracle_calls
            + evaluated.resources.search_calls
            + evaluated.resources.verifier_calls
            + longer_evaluated.resources.oracle_calls
            + longer_evaluated.resources.search_calls
            + longer_evaluated.resources.verifier_calls
        )
        if candidate_forbidden_calls:
            raise SuccessorValueError("candidate crossed a forbidden eval boundary")
        model_file_sha = _save_model(
            controller,
            model_dir=model_dir,
            arm=spec.name,
            seed=config.seed,
        )
        arm_reports.append(
            ArmReport(
                name=spec.name,
                input_mode=spec.input_mode,
                depth_regime=spec.depth_regime,
                random_labels=spec.random_labels,
                explicit_raw_input_recall_path=True,
                trained_evaluation_depth=trained_depth,
                longer_evaluation_depth=longer_depth,
                controller_parameters=controller.parameter_count,
                complete_system_parameters=(
                    controller.complete_system_parameter_count
                ),
                parameter_count_breakdown=(
                    controller.parameter_count_breakdown()
                ),
                optimizer_updates=training.optimizer_updates,
                mean_training_loss=training.mean_loss,
                final_training_loss=training.final_loss,
                assigned_train_label_correct=assigned_correct,
                assigned_train_label_total=assigned_total,
                true_train_label_correct=true_correct,
                true_train_label_total=true_total,
                strict_canonical_certified=(
                    evaluated.strict_canonical_certified
                ),
                evaluation_total=evaluated.total,
                certification_rate=evaluated.certification_rate,
                invalid=evaluated.invalid,
                overlong=evaluated.overlong,
                posthoc_verifier_calls=evaluated.posthoc_verifier_calls,
                longer_strict_canonical_certified=(
                    longer_evaluated.strict_canonical_certified
                ),
                longer_evaluation_total=longer_evaluated.total,
                longer_certification_rate=(
                    longer_evaluated.certification_rate
                ),
                longer_invalid=longer_evaluated.invalid,
                longer_overlong=longer_evaluated.overlong,
                longer_posthoc_verifier_calls=(
                    longer_evaluated.posthoc_verifier_calls
                ),
                overthinking_accuracy_delta=(
                    longer_evaluated.certification_rate
                    - evaluated.certification_rate
                ),
                training_resources=training.resources,
                assigned_label_diagnostic_resources=assigned_resources,
                true_label_diagnostic_resources=true_resources,
                evaluation_resources=evaluated.resources,
                longer_evaluation_resources=longer_evaluated.resources,
                model_state_sha256=model_state_sha256(controller),
                model_file_sha256=model_file_sha,
            )
        )
        del controller
        if device.type == "cuda":
            torch.cuda.empty_cache()
    reports = tuple(arm_reports)
    by_name = {report.name: report for report in reports}
    successor_arms = (ARM_TREATMENT, ARM_PROGRESSIVE)
    best_successor_name = max(
        successor_arms,
        key=lambda name: by_name[name].certification_rate,
    )
    treatment_rate = by_name[best_successor_name].certification_rate
    strongest_control = max(
        by_name[name].certification_rate
        for name in (ARM_ZERO, ARM_SHUFFLED, ARM_NO_RECURRENCE, ARM_RANDOM_LABELS)
    )
    gap = treatment_rate - strongest_control
    material_gate = (
        len(evaluation) >= config.material_minimum_cases
        and treatment_rate >= config.material_minimum_rate
        and gap >= config.material_minimum_control_gap
    )
    if material_gate:
        claim = CLAIM_MATERIAL
    elif treatment_rate > strongest_control:
        claim = CLAIM_SUGGESTIVE
    else:
        claim = CLAIM_NO_GO
    parameter_counts = {report.controller_parameters for report in reports}
    update_counts = {report.optimizer_updates for report in reports}
    training_successor_counts = {
        report.training_resources.successor_evaluations for report in reports
    }
    fixed_progressive_iteration_budget_equal = (
        by_name[ARM_TREATMENT].training_resources.planner_iterations
        == by_name[ARM_PROGRESSIVE].training_resources.planner_iterations
    )
    source_sha = sha256(Path(__file__).read_bytes()).hexdigest()
    return SuccessorExperimentReport(
        schema=EXPERIMENT_SCHEMA,
        status=STATUS,
        claim_classification=claim,
        seed=config.seed,
        device=str(device),
        amp_bfloat16=config.amp_bfloat16,
        source_sha256=source_sha,
        candidate_input_fields=(
            "current_raw_field_matrix",
            "raw_one_step_successor_matrices",
            "local_action_kind_and_role_masks",
            "deterministic_row_and_column_coordinate_labels",
        ),
        forbidden_candidate_inputs=(
            "exact_energy",
            "rank",
            "frontier",
            "reference_schedule",
            "search",
            "verifier",
            "oracle",
        ),
        candidate_has_exact_energy=False,
        candidate_has_rank=False,
        candidate_has_frontier=False,
        candidate_has_reference_schedule=False,
        candidate_has_search=False,
        candidate_has_verifier=False,
        candidate_has_oracle=False,
        fixed_shared_weight_recurrence=True,
        explicit_raw_input_recall_path=True,
        fixed_depth_training_preregistered=True,
        progressive_random_depth_training_preregistered=True,
        longer_depth_overthinking_evaluation_preregistered=True,
        geometry_general_parameters=True,
        learned_absolute_geometry_tables=False,
        preparation_oracle_locked_before_training_and_eval=_PREPARATION_LOCKED,
        preparation_oracle_source_sha256=(
            preparation.oracle_source_sha256
        ),
        preparation_oracle_calls=preparation.oracle_calls,
        train_matrices=len(train),
        train_states=len(true_states),
        evaluation_matrices=len(evaluation),
        strict_geometry_disjoint=not bool(set(train) & set(evaluation)),
        train_matrix_manifest_sha256=matrix_manifest(train),
        evaluation_matrix_manifest_sha256=matrix_manifest(evaluation),
        true_state_manifest_sha256=state_manifest(true_states),
        random_label_state_manifest_sha256=state_manifest(random_states),
        controls_equal_parameters=len(parameter_counts) == 1,
        controls_equal_optimizer_updates=len(update_counts) == 1,
        controls_equal_training_successor_evaluations=(
            len(training_successor_counts) == 1
        ),
        fixed_and_progressive_equal_training_planner_iterations=(
            fixed_progressive_iteration_budget_equal
        ),
        parameter_budget=TOTAL_PARAMETER_BUDGET,
        parameter_budget_passed=(
            max(report.complete_system_parameters for report in reports)
            < TOTAL_PARAMETER_BUDGET
        ),
        material_minimum_cases=config.material_minimum_cases,
        material_minimum_rate=config.material_minimum_rate,
        material_minimum_control_gap=config.material_minimum_control_gap,
        material_gate_passed=material_gate,
        best_successor_arm=best_successor_name,
        best_successor_rate=treatment_rate,
        strongest_control_rate=strongest_control,
        treatment_control_gap=gap,
        arms=reports,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--train-matrices", type=int, default=128)
    parser.add_argument("--evaluation-matrices", type=int, default=96)
    parser.add_argument("--train-maximum-rows", type=int, default=3)
    parser.add_argument("--train-maximum-columns", type=int, default=4)
    parser.add_argument("--evaluation-minimum-rows", type=int, default=4)
    parser.add_argument("--evaluation-minimum-columns", type=int, default=5)
    parser.add_argument("--evaluation-maximum-rows", type=int, default=4)
    parser.add_argument("--evaluation-maximum-columns", type=int, default=6)
    parser.add_argument("--maximum-preparation-steps", type=int, default=96)
    parser.add_argument("--maximum-rollout-steps", type=int, default=192)
    parser.add_argument("--optimizer-updates", type=int, default=1_200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--field-width", type=int, default=64)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--cell-hidden", type=int, default=384)
    parser.add_argument("--matrix-layers", type=int, default=4)
    parser.add_argument("--planner-hidden", type=int, default=512)
    parser.add_argument("--planner-iterations", type=int, default=8)
    parser.add_argument("--coordinate-harmonics", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--no-amp-bfloat16", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output.exists():
        raise SuccessorValueError(f"output already exists: {args.output}")
    report = run_bounded_experiment(
        SuccessorExperimentConfig(
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
            optimizer_updates=args.optimizer_updates,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            amp_bfloat16=not args.no_amp_bfloat16,
            device=args.device,
            controller=SuccessorValueConfig(
                field_width=args.field_width,
                width=args.width,
                cell_hidden=args.cell_hidden,
                matrix_layers=args.matrix_layers,
                planner_hidden=args.planner_hidden,
                planner_iterations=args.planner_iterations,
                coordinate_harmonics=args.coordinate_harmonics,
            ),
        ),
        model_dir=args.model_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(report.canonical_bytes())
    print(report.canonical_bytes().decode("ascii"), end="")


if __name__ == "__main__":
    main()
