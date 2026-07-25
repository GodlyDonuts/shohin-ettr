#!/usr/bin/env python3
"""Search-distilled autonomous SSQAC controller falsifier.

Preparation is the only phase allowed to import the verifier-guided search
teacher or the ordinary reference scheduler.  The teacher programs and search
receipts are written to an ephemeral directory, reduced to labeled
state/action preferences plus aggregate hashes, and then deleted before any
model is initialized.

Three otherwise identical geometry-equivariant feed-forward policies are fit:

* ``search_teacher`` imitates the selected action on a bounded beam-search
  path.
* ``ordinary_oracle`` imitates the ordinary deterministic RREF scheduler on
  the exact same observed states.
* ``randomized_label`` receives a seeded legal alternative on those states.

Final rollout is greedy model inference plus the public primitive row VM only.
It has no search, oracle, structural-potential, verifier, or callback input.
A separate assessor invokes ``verify_reduction_program`` after rollout has
terminated.  This experiment is a mechanics falsifier, not a reasoning claim.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import random
import shutil
import tempfile
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
    OP_SWAP,
    AlgebraInstruction,
    AlgebraMachineError,
    execute_program,
    verify_reduction_program,
)


SCHEMA = "ssqac_search_distillation_pilot_v1"
PREPARATION_SCHEMA = "ssqac_search_distillation_preparation_v1"
MODEL_SCHEMA = "ssqac_search_distilled_equivariant_policy_v1"
STATUS = "mechanics_falsifier_only_not_reasoning"
ARM_SEARCH = "search_teacher"
ARM_ORACLE = "ordinary_oracle"
ARM_RANDOM = "randomized_label"
ARMS = (ARM_SEARCH, ARM_ORACLE, ARM_RANDOM)
ACTION_ELIMINATE = "ELIMINATE"
ACTION_HALT = "HALT"
ACTION_NORMALIZE = "NORMALIZE"
ACTION_SWAP = "SWAP"
ACTION_TYPES = (
    ACTION_ELIMINATE,
    ACTION_HALT,
    ACTION_NORMALIZE,
    ACTION_SWAP,
)


class SearchDistillationError(ValueError):
    """The isolated distillation protocol failed closed."""


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
        raise SearchDistillationError(
            "value is not canonical ASCII JSON data"
        ) from error


def _digest(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _plain_positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SearchDistillationError(f"{label} must be a positive integer")
    return value


def canonical_matrix(
    rows: Iterable[Iterable[int]],
) -> tuple[tuple[int, ...], ...]:
    frozen = tuple(tuple(int(value) % FIELD_MODULUS for value in row) for row in rows)
    if not frozen or not frozen[0]:
        raise SearchDistillationError("matrix must be nonempty")
    if any(len(row) != len(frozen[0]) for row in frozen):
        raise SearchDistillationError("matrix must be rectangular")
    try:
        execute_program(frozen, ())
    except AlgebraMachineError as error:
        raise SearchDistillationError("matrix violates primitive VM bounds") from error
    return frozen


def matrix_sha256(rows: Iterable[Iterable[int]]) -> str:
    return _digest([list(row) for row in canonical_matrix(rows)])


def strict_rref_terminal(rows: Iterable[Iterable[int]]) -> bool:
    """Check the local terminal form without invoking the strict assessor."""

    matrix = canonical_matrix(rows)
    pivots: list[int] = []
    zero_seen = False
    for row in matrix:
        pivot = next((column for column, value in enumerate(row) if value), None)
        if pivot is None:
            zero_seen = True
            continue
        if zero_seen or row[pivot] != 1:
            return False
        if pivots and pivot <= pivots[-1]:
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
class PolicyAction:
    """One geometry-relative legal row-repair macro."""

    kind: str
    row_a: int = 0
    row_b: int = 0
    column: int = 0

    def __post_init__(self) -> None:
        if self.kind not in ACTION_TYPES:
            raise SearchDistillationError(f"unknown action kind {self.kind!r}")
        for name, value in (
            ("row_a", self.row_a),
            ("row_b", self.row_b),
            ("column", self.column),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise SearchDistillationError(f"{name} must be an integer")

    def canonical_data(self) -> list[object]:
        return [self.kind, self.row_a, self.row_b, self.column]

    @property
    def sha256(self) -> str:
        return _digest(self.canonical_data())


def enumerate_policy_actions(
    rows: Iterable[Iterable[int]],
) -> tuple[PolicyAction, ...]:
    """Enumerate legal candidate actions from the current sealed matrix."""

    matrix = canonical_matrix(rows)
    if strict_rref_terminal(matrix):
        return (PolicyAction(ACTION_HALT),)
    leading = tuple(
        next((column for column, value in enumerate(row) if value), None)
        for row in matrix
    )
    actions: list[PolicyAction] = []
    for pivot_row, pivot_column in enumerate(leading):
        if pivot_column is None:
            continue
        if matrix[pivot_row][pivot_column] != 1:
            actions.append(
                PolicyAction(
                    ACTION_NORMALIZE,
                    row_a=pivot_row,
                    column=pivot_column,
                )
            )
            continue
        for target_row, target in enumerate(matrix):
            if target_row != pivot_row and target[pivot_column] % FIELD_MODULUS != 0:
                actions.append(
                    PolicyAction(
                        ACTION_ELIMINATE,
                        row_a=target_row,
                        row_b=pivot_row,
                        column=pivot_column,
                    )
                )
    for left in range(len(matrix)):
        for right in range(left + 1, len(matrix)):
            if matrix[left] != matrix[right]:
                actions.append(
                    PolicyAction(
                        ACTION_SWAP,
                        row_a=left,
                        row_b=right,
                    )
                )
    if not actions:
        raise SearchDistillationError("nonterminal matrix has no legal repair action")
    return tuple(
        sorted(
            actions,
            key=lambda action: (
                action.kind,
                action.row_a,
                action.row_b,
                action.column,
                action.sha256,
            ),
        )
    )


def enumerate_candidate_action_universe(
    row_count: int,
    column_count: int,
) -> tuple[PolicyAction, ...]:
    """Return fixed geometry-derived slots without inspecting matrix values."""

    _plain_positive_int(row_count, label="row_count")
    _plain_positive_int(column_count, label="column_count")
    actions: list[PolicyAction] = [PolicyAction(ACTION_HALT)]
    actions.extend(
        PolicyAction(
            ACTION_NORMALIZE,
            row_a=row,
            column=column,
        )
        for row in range(row_count)
        for column in range(column_count)
    )
    actions.extend(
        PolicyAction(
            ACTION_ELIMINATE,
            row_a=target,
            row_b=source,
            column=column,
        )
        for target in range(row_count)
        for source in range(row_count)
        if source != target
        for column in range(column_count)
    )
    actions.extend(
        PolicyAction(ACTION_SWAP, row_a=left, row_b=right)
        for left in range(row_count)
        for right in range(left + 1, row_count)
    )
    return tuple(
        sorted(
            actions,
            key=lambda action: (
                action.kind,
                action.row_a,
                action.row_b,
                action.column,
            ),
        )
    )


def compile_policy_action(
    rows: Iterable[Iterable[int]],
    action: PolicyAction,
) -> tuple[AlgebraInstruction, ...]:
    """Compile one legal macro to the public branch-free primitive VM."""

    matrix = canonical_matrix(rows)
    if action not in enumerate_policy_actions(matrix):
        raise SearchDistillationError("policy action is not legal in this state")
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


def compile_candidate_action(
    rows: Iterable[Iterable[int]],
    action: PolicyAction,
) -> tuple[AlgebraInstruction, ...]:
    """Compile a greedy action slot without a state-dependent legality oracle."""

    matrix = canonical_matrix(rows)
    if action not in enumerate_candidate_action_universe(
        len(matrix),
        len(matrix[0]),
    ):
        raise SearchDistillationError("candidate action leaves geometry bounds")
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


def apply_policy_action(
    rows: Iterable[Iterable[int]],
    action: PolicyAction,
) -> tuple[tuple[int, ...], ...]:
    matrix = canonical_matrix(rows)
    try:
        state = execute_program(matrix, compile_policy_action(matrix, action))
    except AlgebraMachineError as error:
        raise SearchDistillationError("legal macro failed primitive replay") from error
    return state.rows


def decode_macro_program(
    rows: Iterable[Iterable[int]],
    program: Sequence[AlgebraInstruction],
) -> tuple[PolicyAction, ...]:
    """Decode a preparation program by exact legal-prefix matching."""

    matrix = canonical_matrix(rows)
    frozen = tuple(program)
    offset = 0
    decoded: list[PolicyAction] = []
    while offset < len(frozen):
        matches = tuple(
            action
            for action in enumerate_policy_actions(matrix)
            if frozen[offset : offset + len(compile_policy_action(matrix, action))]
            == compile_policy_action(matrix, action)
        )
        if len(matches) != 1:
            raise SearchDistillationError(
                "teacher program does not have one legal macro decoding"
            )
        action = matches[0]
        decoded.append(action)
        offset += len(compile_policy_action(matrix, action))
        matrix = apply_policy_action(matrix, action)
        if action.kind == ACTION_HALT and offset != len(frozen):
            raise SearchDistillationError("teacher program continues after HALT")
    if not decoded or decoded[-1].kind != ACTION_HALT:
        raise SearchDistillationError("teacher program does not terminate in HALT")
    return tuple(decoded)


@dataclass(frozen=True, slots=True)
class LabeledPolicyState:
    """One matrix-only observation and one legal action preference."""

    rows: tuple[tuple[int, ...], ...]
    target: PolicyAction

    def __post_init__(self) -> None:
        matrix = canonical_matrix(self.rows)
        if matrix != self.rows:
            raise SearchDistillationError("labeled rows are not canonical")
        if self.target not in enumerate_policy_actions(matrix):
            raise SearchDistillationError("target action is not legal")

    @property
    def observation_sha256(self) -> str:
        return matrix_sha256(self.rows)

    @property
    def label_sha256(self) -> str:
        return _digest(
            {
                "observation_sha256": self.observation_sha256,
                "target": self.target.canonical_data(),
            }
        )


def _observation_manifest(states: Sequence[LabeledPolicyState]) -> str:
    return _digest([state.observation_sha256 for state in states])


def _label_manifest(states: Sequence[LabeledPolicyState]) -> str:
    return _digest([state.label_sha256 for state in states])


@dataclass(frozen=True, slots=True)
class SearchPreparationConfig:
    max_nodes_expanded: int = 2_048
    max_edges_considered: int = 50_000
    max_depth: int = 36
    max_frontier: int = 96
    beam_width: int = 96
    max_program_instructions: int = 160
    policy_noise_scale: float = 8.0

    def __post_init__(self) -> None:
        for name in (
            "max_nodes_expanded",
            "max_edges_considered",
            "max_depth",
            "max_frontier",
            "beam_width",
            "max_program_instructions",
        ):
            _plain_positive_int(getattr(self, name), label=name)
        if self.beam_width > self.max_frontier:
            raise SearchDistillationError("beam width exceeds the hard frontier cap")
        if not isinstance(self.policy_noise_scale, float):
            raise SearchDistillationError("policy noise scale must be a float")
        if self.policy_noise_scale < 0.0:
            raise SearchDistillationError("policy noise scale cannot be negative")


@dataclass(frozen=True, slots=True)
class PreparationReceipt:
    schema: str
    requested_cases: int
    completed_search_cases: int
    failed_search_cases: int
    retained_states: int
    states_per_case_cap: int
    preparation_search_calls: int
    preparation_oracle_calls: int
    source_matrix_manifest_sha256: str
    observation_manifest_sha256: str
    search_label_manifest_sha256: str
    ordinary_label_manifest_sha256: str
    randomized_label_manifest_sha256: str
    search_receipt_manifest_sha256: str
    deleted_trace_manifest_sha256: str
    trace_directory_deleted: bool
    retained_search_trace_files: int
    raw_search_programs_retained: bool
    arm_state_budgets_matched: bool


@dataclass(frozen=True, slots=True)
class PreparedMatchedDatasets:
    search_teacher: tuple[LabeledPolicyState, ...]
    ordinary_oracle: tuple[LabeledPolicyState, ...]
    randomized_label: tuple[LabeledPolicyState, ...]
    receipt: PreparationReceipt

    def arm(self, name: str) -> tuple[LabeledPolicyState, ...]:
        if name == ARM_SEARCH:
            return self.search_teacher
        if name == ARM_ORACLE:
            return self.ordinary_oracle
        if name == ARM_RANDOM:
            return self.randomized_label
        raise SearchDistillationError(f"unknown arm {name!r}")


def _randomized_alternative(
    rows: tuple[tuple[int, ...], ...],
    search_target: PolicyAction,
    *,
    seed: int,
) -> PolicyAction:
    legal = enumerate_policy_actions(rows)
    alternatives = tuple(action for action in legal if action != search_target)
    if not alternatives:
        return search_target
    material = f"{seed}:{matrix_sha256(rows)}".encode("ascii")
    index = int.from_bytes(sha256(material).digest()[:8], "big") % len(alternatives)
    return alternatives[index]


def _first_ordinary_oracle_action(
    rows: tuple[tuple[int, ...], ...],
    *,
    compile_reference_program: object,
) -> PolicyAction:
    if not callable(compile_reference_program):
        raise SearchDistillationError("ordinary oracle is not callable")
    program = compile_reference_program(rows)
    return decode_macro_program(rows, program)[0]


def prepare_matched_datasets(
    matrices: Sequence[Iterable[Iterable[int]]],
    *,
    seed: int,
    states_per_case: int,
    search_config: SearchPreparationConfig,
    scratch_root: Path | None = None,
) -> PreparedMatchedDatasets:
    """Run preparation-only search and delete every raw trace before return."""

    _plain_positive_int(states_per_case, label="states_per_case")
    frozen_matrices = tuple(canonical_matrix(matrix) for matrix in matrices)
    if not frozen_matrices:
        raise SearchDistillationError("preparation needs at least one matrix")

    # These imports are deliberately local.  No autonomous inference function
    # references either preparation dependency.
    from ssqac_controller_trace_pilot import compile_reference_program
    from ssqac_verifier_guided_search import (
        SearchBudget,
        SealedAlgebraPacket,
        WeakLocalScorer,
        bounded_beam_candidate_search,
    )

    budget = SearchBudget(
        max_nodes_expanded=search_config.max_nodes_expanded,
        max_edges_considered=search_config.max_edges_considered,
        max_depth=search_config.max_depth,
        max_frontier=search_config.max_frontier,
        beam_width=search_config.beam_width,
        max_program_instructions=search_config.max_program_instructions,
    )
    root = None if scratch_root is None else str(scratch_root)
    trace_directory = Path(tempfile.mkdtemp(prefix="ssqac-search-traces-", dir=root))
    trace_hashes: list[str] = []
    receipt_hashes: list[str] = []
    search_calls = 0
    oracle_calls = 0
    completed_cases = 0
    observations: dict[
        str,
        tuple[
            tuple[tuple[int, ...], ...],
            PolicyAction,
            PolicyAction,
            PolicyAction,
        ],
    ] = {}
    try:
        for case_index, matrix in enumerate(frozen_matrices):
            packet = SealedAlgebraPacket.from_rows(matrix, register_count=4)
            scorer = WeakLocalScorer(
                seed=seed + case_index,
                noise_scale=search_config.policy_noise_scale,
            )
            result = bounded_beam_candidate_search(
                packet,
                scorer,
                budget,
            )
            search_calls += 1
            trace_payload = {
                "case_index": case_index,
                "matrix_sha256": matrix_sha256(matrix),
                "program": None
                if result.program is None
                else [instruction.canonical_data() for instruction in result.program],
                "receipt": asdict(result.receipt),
            }
            trace_bytes = _canonical_bytes(trace_payload) + b"\n"
            trace_path = trace_directory / f"trace_{case_index:06d}.json"
            trace_path.write_bytes(trace_bytes)
            trace_hashes.append(sha256(trace_bytes).hexdigest())
            receipt_hashes.append(sha256(result.receipt.canonical_bytes()).hexdigest())
            if result.program is None:
                continue
            actions = decode_macro_program(matrix, result.program)
            completed_cases += 1
            current = matrix
            retained_for_case = 0
            for search_target in actions:
                if retained_for_case >= states_per_case:
                    break
                ordinary_target = _first_ordinary_oracle_action(
                    current,
                    compile_reference_program=compile_reference_program,
                )
                oracle_calls += 1
                random_target = _randomized_alternative(
                    current,
                    search_target,
                    seed=seed ^ 0x5EED,
                )
                key = matrix_sha256(current)
                value = (
                    current,
                    search_target,
                    ordinary_target,
                    random_target,
                )
                prior = observations.get(key)
                if prior is not None and prior != value:
                    raise SearchDistillationError(
                        "preparation produced conflicting labels"
                    )
                observations[key] = value
                retained_for_case += 1
                current = apply_policy_action(current, search_target)
            del actions, result
        files_before_delete = tuple(sorted(trace_directory.iterdir()))
        deleted_trace_manifest = _digest(
            [
                {
                    "name": path.name,
                    "sha256": sha256(path.read_bytes()).hexdigest(),
                }
                for path in files_before_delete
            ]
        )
    finally:
        shutil.rmtree(trace_directory, ignore_errors=False)
    trace_deleted = not trace_directory.exists()
    if not trace_deleted:
        raise RuntimeError("preparation search trace directory survived deletion")
    if not observations:
        raise SearchDistillationError("bounded search produced no distillation states")

    ordered = tuple(observations[key] for key in sorted(observations))
    search_states = tuple(
        LabeledPolicyState(rows=rows, target=search_target)
        for rows, search_target, _, _ in ordered
    )
    ordinary_states = tuple(
        LabeledPolicyState(rows=rows, target=ordinary_target)
        for rows, _, ordinary_target, _ in ordered
    )
    random_states = tuple(
        LabeledPolicyState(rows=rows, target=random_target)
        for rows, _, _, random_target in ordered
    )
    manifests = {
        _observation_manifest(search_states),
        _observation_manifest(ordinary_states),
        _observation_manifest(random_states),
    }
    if len(manifests) != 1:
        raise RuntimeError("matched arms do not contain the same observations")
    state_budgets_matched = (
        len(search_states) == len(ordinary_states) == len(random_states)
    )
    if not state_budgets_matched:
        raise RuntimeError("matched arm state budgets differ")
    receipt = PreparationReceipt(
        schema=PREPARATION_SCHEMA,
        requested_cases=len(frozen_matrices),
        completed_search_cases=completed_cases,
        failed_search_cases=len(frozen_matrices) - completed_cases,
        retained_states=len(search_states),
        states_per_case_cap=states_per_case,
        preparation_search_calls=search_calls,
        preparation_oracle_calls=oracle_calls,
        source_matrix_manifest_sha256=_digest(
            [matrix_sha256(matrix) for matrix in frozen_matrices]
        ),
        observation_manifest_sha256=next(iter(manifests)),
        search_label_manifest_sha256=_label_manifest(search_states),
        ordinary_label_manifest_sha256=_label_manifest(ordinary_states),
        randomized_label_manifest_sha256=_label_manifest(random_states),
        search_receipt_manifest_sha256=_digest(receipt_hashes),
        deleted_trace_manifest_sha256=deleted_trace_manifest,
        trace_directory_deleted=trace_deleted,
        retained_search_trace_files=0,
        raw_search_programs_retained=False,
        arm_state_budgets_matched=state_budgets_matched,
    )
    # Drop the remaining aggregate preparation-only containers before fitting.
    del ordered, observations, trace_hashes, receipt_hashes
    return PreparedMatchedDatasets(
        search_teacher=search_states,
        ordinary_oracle=ordinary_states,
        randomized_label=random_states,
        receipt=receipt,
    )


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    maximum_rows: int = 6
    maximum_columns: int = 8
    width: int = 256
    blocks: int = 4
    feedforward: int = 768
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "maximum_rows",
            "maximum_columns",
            "width",
            "blocks",
            "feedforward",
        ):
            _plain_positive_int(getattr(self, name), label=name)
        if self.maximum_columns < self.maximum_rows:
            raise SearchDistillationError(
                "maximum columns must admit every row geometry"
            )
        if not isinstance(self.dropout, float) or not 0.0 <= self.dropout < 1.0:
            raise SearchDistillationError("dropout must be a float in [0, 1)")

    @property
    def sha256(self) -> str:
        return _digest(asdict(self))


def _masked_mean(values: Tensor, mask: Tensor, dimension: int) -> Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    numerator = (values * weights).sum(dim=dimension)
    denominator = weights.sum(dim=dimension).clamp_min(1.0)
    return numerator / denominator


class _EquivariantMixBlock(nn.Module):
    def __init__(self, width: int, feedforward: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.cell = nn.Linear(width, width, bias=False)
        self.row = nn.Linear(width, width, bias=False)
        self.column = nn.Linear(width, width, bias=False)
        self.global_projection = nn.Linear(width, width, bias=False)
        self.update = nn.Sequential(
            nn.GELU(),
            nn.Linear(width, feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward, width),
        )

    def forward(self, cells: Tensor, visible: Tensor) -> Tensor:
        normalized = self.norm(cells)
        row_summary = _masked_mean(normalized, visible, 2)
        column_summary = _masked_mean(normalized, visible, 1)
        global_summary = _masked_mean(
            row_summary,
            visible.any(dim=2),
            1,
        )
        message = (
            self.cell(normalized)
            + self.row(row_summary)[:, :, None, :]
            + self.column(column_summary)[:, None, :, :]
            + self.global_projection(global_summary)[:, None, None, :]
        )
        result = cells + self.update(message)
        return result * visible.unsqueeze(-1).to(result.dtype)


@dataclass(frozen=True, slots=True)
class PolicyBatch:
    rows: Tensor
    row_mask: Tensor
    column_mask: Tensor
    action_kind: Tensor
    action_row_a: Tensor
    action_row_b: Tensor
    action_column: Tensor
    action_presence: Tensor
    action_mask: Tensor
    targets: Tensor
    examples: int
    observation_manifest_sha256: str
    label_manifest_sha256: str
    tensor_manifest_sha256: str

    def to(self, device: torch.device) -> "PolicyBatch":
        return PolicyBatch(
            rows=self.rows.to(device, non_blocking=True),
            row_mask=self.row_mask.to(device, non_blocking=True),
            column_mask=self.column_mask.to(device, non_blocking=True),
            action_kind=self.action_kind.to(device, non_blocking=True),
            action_row_a=self.action_row_a.to(device, non_blocking=True),
            action_row_b=self.action_row_b.to(device, non_blocking=True),
            action_column=self.action_column.to(device, non_blocking=True),
            action_presence=self.action_presence.to(
                device,
                non_blocking=True,
            ),
            action_mask=self.action_mask.to(device, non_blocking=True),
            targets=self.targets.to(device, non_blocking=True),
            examples=self.examples,
            observation_manifest_sha256=self.observation_manifest_sha256,
            label_manifest_sha256=self.label_manifest_sha256,
            tensor_manifest_sha256=self.tensor_manifest_sha256,
        )

    def select(self, indices: Tensor) -> Mapping[str, Tensor]:
        return {
            "rows": self.rows[indices],
            "row_mask": self.row_mask[indices],
            "column_mask": self.column_mask[indices],
            "action_kind": self.action_kind[indices],
            "action_row_a": self.action_row_a[indices],
            "action_row_b": self.action_row_b[indices],
            "action_column": self.action_column[indices],
            "action_presence": self.action_presence[indices],
            "action_mask": self.action_mask[indices],
        }


def _tensor_digest(tensors: Mapping[str, Tensor]) -> str:
    digest = sha256()
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(_canonical_bytes(list(tensor.shape)))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def tensorize_labeled_states(
    states: Sequence[LabeledPolicyState],
    config: PolicyConfig,
) -> PolicyBatch:
    if not states:
        raise SearchDistillationError("cannot tensorize an empty state set")
    examples = len(states)
    maximum_actions = max(
        len(
            enumerate_candidate_action_universe(
                len(state.rows),
                len(state.rows[0]),
            )
        )
        for state in states
    )
    rows = torch.zeros(
        examples,
        config.maximum_rows,
        config.maximum_columns,
        dtype=torch.long,
    )
    row_mask = torch.zeros(
        examples,
        config.maximum_rows,
        dtype=torch.bool,
    )
    column_mask = torch.zeros(
        examples,
        config.maximum_columns,
        dtype=torch.bool,
    )
    action_kind = torch.zeros(examples, maximum_actions, dtype=torch.long)
    action_row_a = torch.zeros(examples, maximum_actions, dtype=torch.long)
    action_row_b = torch.zeros(examples, maximum_actions, dtype=torch.long)
    action_column = torch.zeros(examples, maximum_actions, dtype=torch.long)
    action_presence = torch.zeros(
        examples,
        maximum_actions,
        3,
        dtype=torch.float32,
    )
    action_mask = torch.zeros(examples, maximum_actions, dtype=torch.bool)
    targets = torch.zeros(examples, dtype=torch.long)
    kind_index = {kind: index for index, kind in enumerate(ACTION_TYPES)}
    for example, state in enumerate(states):
        row_count = len(state.rows)
        column_count = len(state.rows[0])
        if row_count > config.maximum_rows:
            raise SearchDistillationError("state exceeds configured row bound")
        if column_count > config.maximum_columns:
            raise SearchDistillationError("state exceeds configured column bound")
        rows[example, :row_count, :column_count] = torch.tensor(
            state.rows,
            dtype=torch.long,
        )
        row_mask[example, :row_count] = True
        column_mask[example, :column_count] = True
        actions = enumerate_candidate_action_universe(
            row_count,
            column_count,
        )
        for action_index, action in enumerate(actions):
            action_kind[example, action_index] = kind_index[action.kind]
            action_row_a[example, action_index] = action.row_a
            action_row_b[example, action_index] = action.row_b
            action_column[example, action_index] = action.column
            action_presence[example, action_index] = torch.tensor(
                (
                    0.0 if action.kind == ACTION_HALT else 1.0,
                    1.0 if action.kind in (ACTION_ELIMINATE, ACTION_SWAP) else 0.0,
                    1.0 if action.kind in (ACTION_ELIMINATE, ACTION_NORMALIZE) else 0.0,
                )
            )
            action_mask[example, action_index] = True
        targets[example] = actions.index(state.target)
    tensors = {
        "action_column": action_column,
        "action_kind": action_kind,
        "action_mask": action_mask,
        "action_presence": action_presence,
        "action_row_a": action_row_a,
        "action_row_b": action_row_b,
        "column_mask": column_mask,
        "row_mask": row_mask,
        "rows": rows,
        "targets": targets,
    }
    return PolicyBatch(
        rows=rows,
        row_mask=row_mask,
        column_mask=column_mask,
        action_kind=action_kind,
        action_row_a=action_row_a,
        action_row_b=action_row_b,
        action_column=action_column,
        action_presence=action_presence,
        action_mask=action_mask,
        targets=targets,
        examples=examples,
        observation_manifest_sha256=_observation_manifest(states),
        label_manifest_sha256=_label_manifest(states),
        tensor_manifest_sha256=_tensor_digest(tensors),
    )


class EquivariantActionPolicy(nn.Module):
    """Shared feed-forward action scorer with no absolute position tables."""

    def __init__(self, config: PolicyConfig = PolicyConfig()) -> None:
        super().__init__()
        self.config = config
        width = config.width
        self.coefficient_embedding = nn.Embedding(FIELD_MODULUS, width)
        self.row_coordinate_projection = nn.Linear(4, width, bias=False)
        self.column_coordinate_projection = nn.Linear(4, width, bias=False)
        self.input_norm = nn.LayerNorm(width)
        self.blocks = nn.ModuleList(
            _EquivariantMixBlock(
                width,
                config.feedforward,
                config.dropout,
            )
            for _ in range(config.blocks)
        )
        self.kind_embedding = nn.Embedding(len(ACTION_TYPES), width)
        self.geometry_projection = nn.Linear(4, width, bias=False)
        self.action_projection = nn.Linear(
            6 * width + 14,
            width,
            bias=False,
        )
        self.action_scorer = nn.Sequential(
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, config.feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward, 1),
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _gather(tokens: Tensor, indices: Tensor) -> Tensor:
        return torch.gather(
            tokens,
            1,
            indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]),
        )

    def forward(
        self,
        *,
        rows: Tensor,
        row_mask: Tensor,
        column_mask: Tensor,
        action_kind: Tensor,
        action_row_a: Tensor,
        action_row_b: Tensor,
        action_column: Tensor,
        action_presence: Tensor,
        action_mask: Tensor,
    ) -> Tensor:
        if rows.ndim != 3:
            raise SearchDistillationError("rows must be [batch, rows, columns]")
        batch, row_count, column_count = rows.shape
        if row_count > self.config.maximum_rows:
            raise SearchDistillationError("row tensor exceeds configured bound")
        if column_count > self.config.maximum_columns:
            raise SearchDistillationError("column tensor exceeds configured bound")
        if tuple(row_mask.shape) != (batch, row_count):
            raise SearchDistillationError("row mask shape differs")
        if tuple(column_mask.shape) != (batch, column_count):
            raise SearchDistillationError("column mask shape differs")
        if action_kind.shape != action_mask.shape:
            raise SearchDistillationError("action kind/mask shapes differ")
        if action_presence.shape != (*action_mask.shape, 3):
            raise SearchDistillationError("action presence shape differs")
        if torch.any(rows < 0) or torch.any(rows >= FIELD_MODULUS):
            raise SearchDistillationError("matrix coefficient leaves F_257")
        visible = row_mask[:, :, None] & column_mask[:, None, :]
        row_lengths = row_mask.sum(dim=1).to(torch.float32)
        column_lengths = column_mask.sum(dim=1).to(torch.float32)
        row_positions = torch.arange(
            row_count,
            device=rows.device,
            dtype=torch.float32,
        )[None, :].expand(batch, -1)
        column_positions = torch.arange(
            column_count,
            device=rows.device,
            dtype=torch.float32,
        )[None, :].expand(batch, -1)
        row_fraction = row_positions / (row_lengths[:, None] - 1.0).clamp_min(1.0)
        column_fraction = column_positions / (column_lengths[:, None] - 1.0).clamp_min(
            1.0
        )
        row_coordinates = torch.stack(
            (
                row_fraction,
                torch.sin(torch.pi * row_fraction),
                torch.cos(torch.pi * row_fraction),
                row_positions == 0,
            ),
            dim=-1,
        ).to(self.coefficient_embedding.weight.dtype)
        column_coordinates = torch.stack(
            (
                column_fraction,
                torch.sin(torch.pi * column_fraction),
                torch.cos(torch.pi * column_fraction),
                column_positions == 0,
            ),
            dim=-1,
        ).to(self.coefficient_embedding.weight.dtype)
        cells = self.input_norm(
            self.coefficient_embedding(rows.long())
            + self.row_coordinate_projection(row_coordinates)[:, :, None, :]
            + self.column_coordinate_projection(column_coordinates)[:, None, :, :]
        )
        cells = cells * visible.unsqueeze(-1).to(cells.dtype)
        for block in self.blocks:
            cells = block(cells, visible)
        row_tokens = _masked_mean(cells, visible, 2)
        column_tokens = _masked_mean(cells, visible, 1)
        global_token = _masked_mean(row_tokens, row_mask, 1)
        row_a = self._gather(row_tokens, action_row_a.long())
        row_b = self._gather(row_tokens, action_row_b.long())
        column = self._gather(column_tokens, action_column.long())
        action_count = action_kind.shape[1]
        global_actions = global_token[:, None, :].expand(
            -1,
            action_count,
            -1,
        )
        geometry = torch.stack(
            (
                torch.log1p(row_lengths),
                torch.log1p(column_lengths),
                row_lengths / column_lengths.clamp_min(1.0),
                column_lengths / row_lengths.clamp_min(1.0),
            ),
            dim=-1,
        ).to(cells.dtype)
        geometry_token = self.geometry_projection(geometry)[:, None, :].expand(
            -1,
            action_count,
            -1,
        )
        batch_index = torch.arange(batch, device=rows.device)[:, None]
        coefficient_a = rows[
            batch_index,
            action_row_a.long(),
            action_column.long(),
        ].to(cells.dtype)
        coefficient_b = rows[
            batch_index,
            action_row_b.long(),
            action_column.long(),
        ].to(cells.dtype)
        row_a_fraction = torch.gather(
            row_fraction,
            1,
            action_row_a.long(),
        ).to(cells.dtype)
        row_b_fraction = torch.gather(
            row_fraction,
            1,
            action_row_b.long(),
        ).to(cells.dtype)
        action_column_fraction = torch.gather(
            column_fraction,
            1,
            action_column.long(),
        ).to(cells.dtype)
        scalar = torch.cat(
            (
                action_presence.to(cells.dtype),
                (coefficient_a / (FIELD_MODULUS - 1)).unsqueeze(-1),
                (coefficient_b / (FIELD_MODULUS - 1)).unsqueeze(-1),
                (coefficient_a == 0).to(cells.dtype).unsqueeze(-1),
                (coefficient_a == 1).to(cells.dtype).unsqueeze(-1),
                (coefficient_b == 0).to(cells.dtype).unsqueeze(-1),
                (coefficient_b == 1).to(cells.dtype).unsqueeze(-1),
                row_a_fraction.unsqueeze(-1),
                row_b_fraction.unsqueeze(-1),
                action_column_fraction.unsqueeze(-1),
                (row_a_fraction - row_b_fraction).unsqueeze(-1),
                (row_a_fraction - row_b_fraction).abs().unsqueeze(-1),
            ),
            dim=-1,
        )
        features = torch.cat(
            (
                global_actions + self.kind_embedding(action_kind.long()),
                geometry_token,
                row_a,
                row_b,
                column,
                row_a * row_b,
                scalar,
            ),
            dim=-1,
        )
        logits = self.action_scorer(self.action_projection(features)).squeeze(-1)
        return logits.masked_fill(~action_mask, -torch.inf)


def model_state_sha256(model: nn.Module) -> str:
    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        frozen = tensor.detach().cpu().contiguous()
        digest.update(name.encode("ascii"))
        digest.update(str(frozen.dtype).encode("ascii"))
        digest.update(_canonical_bytes(list(frozen.shape)))
        digest.update(frozen.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 24
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    amp_bfloat16: bool = True
    torch_compile: bool = False

    def __post_init__(self) -> None:
        _plain_positive_int(self.epochs, label="epochs")
        _plain_positive_int(self.batch_size, label="batch_size")
        if self.learning_rate <= 0.0:
            raise SearchDistillationError("learning rate must be positive")
        if self.weight_decay < 0.0:
            raise SearchDistillationError("weight decay cannot be negative")


@dataclass(frozen=True, slots=True)
class TrainingReceipt:
    optimizer_updates: int
    mean_loss: float
    final_loss: float
    batch_schedule_sha256: str


def build_batch_schedule(
    *,
    examples: int,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    _plain_positive_int(examples, label="examples")
    _plain_positive_int(epochs, label="epochs")
    _plain_positive_int(batch_size, label="batch_size")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    schedule: list[tuple[int, ...]] = []
    for _ in range(epochs):
        order = torch.randperm(examples, generator=generator).tolist()
        schedule.extend(
            tuple(order[offset : offset + batch_size])
            for offset in range(0, examples, batch_size)
        )
    return tuple(schedule)


def _autocast(device: torch.device, enabled: bool) -> object:
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def train_policy(
    model: EquivariantActionPolicy,
    dataset: PolicyBatch,
    *,
    schedule: Sequence[Sequence[int]],
    config: TrainingConfig,
    device: torch.device,
) -> TrainingReceipt:
    resident = dataset.to(device)
    model.to(device)
    model.train()
    candidate: nn.Module = model
    if config.torch_compile:
        compiler = getattr(torch, "compile", None)
        if compiler is None:
            raise SearchDistillationError("torch.compile is unavailable")
        candidate = compiler(model, mode="reduce-overhead")
    optimizer_arguments: dict[str, object] = {
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
    }
    if device.type == "cuda":
        optimizer_arguments["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_arguments)
    losses: list[float] = []
    for raw_indices in schedule:
        indices = torch.tensor(raw_indices, device=device, dtype=torch.long)
        inputs = resident.select(indices)
        targets = resident.targets[indices]
        with _autocast(device, config.amp_bfloat16):
            logits = candidate(**inputs)
            loss = nn.functional.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("training loss became nonfinite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    if not losses:
        raise RuntimeError("training schedule performed no updates")
    return TrainingReceipt(
        optimizer_updates=len(losses),
        mean_loss=sum(losses) / len(losses),
        final_loss=losses[-1],
        batch_schedule_sha256=_digest([list(indices) for indices in schedule]),
    )


def label_accuracy(
    model: EquivariantActionPolicy,
    dataset: PolicyBatch,
    *,
    device: torch.device,
    amp_bfloat16: bool,
) -> float:
    resident = dataset.to(device)
    model.eval()
    with torch.no_grad(), _autocast(device, amp_bfloat16):
        logits = model(**resident.select(torch.arange(dataset.examples, device=device)))
    correct = int(
        (logits.argmax(dim=-1) == resident.targets).sum().detach().cpu().item()
    )
    return correct / dataset.examples


@dataclass(slots=True)
class RuntimeAccessCounter:
    search_calls: int = 0
    oracle_calls: int = 0
    verifier_calls: int = 0
    model_decisions: int = 0
    vm_calls: int = 0


@dataclass(frozen=True, slots=True)
class CandidateProgram:
    program: tuple[AlgebraInstruction, ...]
    termination: str
    model_decisions: int
    vm_calls: int

    @property
    def program_sha256(self) -> str:
        return _digest([instruction.canonical_data() for instruction in self.program])


def _single_observation_batch(
    rows: tuple[tuple[int, ...], ...],
    config: PolicyConfig,
) -> tuple[Mapping[str, Tensor], tuple[PolicyAction, ...]]:
    matrix = canonical_matrix(rows)
    row_count = len(matrix)
    column_count = len(matrix[0])
    if row_count > config.maximum_rows or column_count > config.maximum_columns:
        raise SearchDistillationError("observation exceeds configured model geometry")
    actions = enumerate_candidate_action_universe(row_count, column_count)
    action_count = len(actions)
    kind_index = {kind: index for index, kind in enumerate(ACTION_TYPES)}
    row_tensor = torch.zeros(
        1,
        config.maximum_rows,
        config.maximum_columns,
        dtype=torch.long,
    )
    row_tensor[0, :row_count, :column_count] = torch.tensor(
        matrix,
        dtype=torch.long,
    )
    row_mask = torch.zeros(1, config.maximum_rows, dtype=torch.bool)
    row_mask[0, :row_count] = True
    column_mask = torch.zeros(1, config.maximum_columns, dtype=torch.bool)
    column_mask[0, :column_count] = True
    action_kind = torch.zeros(1, action_count, dtype=torch.long)
    action_row_a = torch.zeros(1, action_count, dtype=torch.long)
    action_row_b = torch.zeros(1, action_count, dtype=torch.long)
    action_column = torch.zeros(1, action_count, dtype=torch.long)
    action_presence = torch.zeros(1, action_count, 3, dtype=torch.float32)
    action_mask = torch.ones(1, action_count, dtype=torch.bool)
    for index, action in enumerate(actions):
        action_kind[0, index] = kind_index[action.kind]
        action_row_a[0, index] = action.row_a
        action_row_b[0, index] = action.row_b
        action_column[0, index] = action.column
        action_presence[0, index] = torch.tensor(
            (
                0.0 if action.kind == ACTION_HALT else 1.0,
                1.0 if action.kind in (ACTION_ELIMINATE, ACTION_SWAP) else 0.0,
                1.0 if action.kind in (ACTION_ELIMINATE, ACTION_NORMALIZE) else 0.0,
            )
        )
    return (
        {
            "rows": row_tensor,
            "row_mask": row_mask,
            "column_mask": column_mask,
            "action_kind": action_kind,
            "action_row_a": action_row_a,
            "action_row_b": action_row_b,
            "action_column": action_column,
            "action_presence": action_presence,
            "action_mask": action_mask,
        },
        actions,
    )


def greedy_model_vm_rollout(
    model: EquivariantActionPolicy,
    matrix: Iterable[Iterable[int]],
    *,
    device: torch.device,
    amp_bfloat16: bool,
    maximum_macros: int,
    maximum_instructions: int,
    counter: RuntimeAccessCounter | None = None,
) -> CandidateProgram:
    """Autonomous candidate runtime: greedy model plus primitive VM only."""

    _plain_positive_int(maximum_macros, label="maximum_macros")
    _plain_positive_int(maximum_instructions, label="maximum_instructions")
    source = canonical_matrix(matrix)
    access = counter if counter is not None else RuntimeAccessCounter()
    if access.search_calls or access.oracle_calls or access.verifier_calls:
        raise SearchDistillationError(
            "autonomous runtime received a contaminated access counter"
        )
    program: list[AlgebraInstruction] = []
    visited: set[tuple[tuple[int, ...], ...]] = set()
    model.eval()
    for _ in range(maximum_macros):
        try:
            state = execute_program(
                source,
                program,
                maximum_instructions=maximum_instructions,
            )
        except AlgebraMachineError:
            return CandidateProgram(
                tuple(program),
                "vm_error",
                access.model_decisions,
                access.vm_calls,
            )
        access.vm_calls += 1
        rows = state.rows
        if rows in visited:
            return CandidateProgram(
                tuple(program),
                "cycle",
                access.model_decisions,
                access.vm_calls,
            )
        visited.add(rows)
        inputs, actions = _single_observation_batch(rows, model.config)
        resident = {name: tensor.to(device) for name, tensor in inputs.items()}
        with torch.no_grad(), _autocast(device, amp_bfloat16):
            logits = model(**resident)
        selected = int(logits[0].argmax().detach().cpu().item())
        action = actions[selected]
        access.model_decisions += 1
        primitives = compile_candidate_action(rows, action)
        if len(program) + len(primitives) > maximum_instructions:
            return CandidateProgram(
                tuple(program),
                "instruction_budget",
                access.model_decisions,
                access.vm_calls,
            )
        program.extend(primitives)
        if action.kind == ACTION_HALT:
            try:
                execute_program(
                    source,
                    program,
                    maximum_instructions=maximum_instructions,
                )
            except AlgebraMachineError:
                return CandidateProgram(
                    tuple(program),
                    "vm_error",
                    access.model_decisions,
                    access.vm_calls,
                )
            access.vm_calls += 1
            return CandidateProgram(
                tuple(program),
                "halted",
                access.model_decisions,
                access.vm_calls,
            )
    return CandidateProgram(
        tuple(program),
        "macro_budget",
        access.model_decisions,
        access.vm_calls,
    )


@dataclass(frozen=True, slots=True)
class AssessmentResult:
    passed: bool
    reason: str
    program_sha256: str
    output_sha256: str


def assess_candidate_program(
    matrix: Iterable[Iterable[int]],
    candidate: CandidateProgram,
    *,
    counter: RuntimeAccessCounter | None = None,
) -> AssessmentResult:
    """Separate strict assessor; never called by autonomous rollout."""

    source = canonical_matrix(matrix)
    access = counter if counter is not None else RuntimeAccessCounter()
    access.verifier_calls += 1
    try:
        state = execute_program(source, candidate.program)
        receipt = verify_reduction_program(source, state)
    except AlgebraMachineError as error:
        return AssessmentResult(
            passed=False,
            reason=type(error).__name__,
            program_sha256=candidate.program_sha256,
            output_sha256="",
        )
    return AssessmentResult(
        passed=receipt.passed,
        reason="certified" if receipt.passed else "verifier_rejected",
        program_sha256=candidate.program_sha256,
        output_sha256=receipt.output_sha256,
    )


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    matrix_sha256: str
    termination: str
    passed: bool
    program_sha256: str
    model_decisions: int
    vm_calls: int


@dataclass(frozen=True, slots=True)
class EvaluationReceipt:
    certified: int
    invalid: int
    overlong: int
    final_search_calls: int
    final_oracle_calls: int
    assessor_verifier_calls: int
    model_decisions: int
    vm_calls: int
    cases: tuple[EvaluationCase, ...]

    @property
    def total(self) -> int:
        return self.certified + self.invalid + self.overlong

    @property
    def certification_rate(self) -> float:
        return self.certified / self.total if self.total else 0.0


def evaluate_autonomous_policy(
    model: EquivariantActionPolicy,
    matrices: Sequence[Iterable[Iterable[int]]],
    *,
    device: torch.device,
    amp_bfloat16: bool,
    maximum_macros: int,
    maximum_instructions: int,
) -> EvaluationReceipt:
    cases: list[EvaluationCase] = []
    certified = 0
    invalid = 0
    overlong = 0
    final_search_calls = 0
    final_oracle_calls = 0
    verifier_calls = 0
    model_decisions = 0
    vm_calls = 0
    for raw in matrices:
        matrix = canonical_matrix(raw)
        runtime_counter = RuntimeAccessCounter()
        candidate = greedy_model_vm_rollout(
            model,
            matrix,
            device=device,
            amp_bfloat16=amp_bfloat16,
            maximum_macros=maximum_macros,
            maximum_instructions=maximum_instructions,
            counter=runtime_counter,
        )
        if runtime_counter.search_calls or runtime_counter.oracle_calls:
            raise RuntimeError("autonomous rollout touched preparation machinery")
        assessor_counter = RuntimeAccessCounter()
        assessment = assess_candidate_program(
            matrix,
            candidate,
            counter=assessor_counter,
        )
        if assessment.passed:
            certified += 1
        elif candidate.termination in ("macro_budget", "instruction_budget"):
            overlong += 1
        else:
            invalid += 1
        final_search_calls += runtime_counter.search_calls
        final_oracle_calls += runtime_counter.oracle_calls
        verifier_calls += assessor_counter.verifier_calls
        model_decisions += runtime_counter.model_decisions
        vm_calls += runtime_counter.vm_calls
        cases.append(
            EvaluationCase(
                matrix_sha256=matrix_sha256(matrix),
                termination=candidate.termination,
                passed=assessment.passed,
                program_sha256=assessment.program_sha256,
                model_decisions=candidate.model_decisions,
                vm_calls=candidate.vm_calls,
            )
        )
    return EvaluationReceipt(
        certified=certified,
        invalid=invalid,
        overlong=overlong,
        final_search_calls=final_search_calls,
        final_oracle_calls=final_oracle_calls,
        assessor_verifier_calls=verifier_calls,
        model_decisions=model_decisions,
        vm_calls=vm_calls,
        cases=tuple(cases),
    )


def generate_matrix_cases(
    *,
    seed: int,
    count: int,
    minimum_rows: int,
    maximum_rows: int,
    minimum_columns: int,
    maximum_columns: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    _plain_positive_int(count, label="count")
    if not 2 <= minimum_rows <= maximum_rows:
        raise SearchDistillationError("row geometry bounds differ")
    if not 2 <= minimum_columns <= maximum_columns:
        raise SearchDistillationError("column geometry bounds differ")
    if maximum_columns < maximum_rows:
        raise SearchDistillationError("columns must admit every requested row geometry")
    rng = random.Random(seed)
    matrices: list[tuple[tuple[int, ...], ...]] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    while len(matrices) < count:
        row_count = rng.randint(minimum_rows, maximum_rows)
        column_count = rng.randint(
            max(row_count, minimum_columns),
            maximum_columns,
        )
        matrix = tuple(
            tuple(
                0 if rng.random() < 0.48 else rng.randrange(1, FIELD_MODULUS)
                for _ in range(column_count)
            )
            for _ in range(row_count)
        )
        if matrix in seen or not any(value for row in matrix for value in row):
            continue
        seen.add(matrix)
        matrices.append(matrix)
    return tuple(matrices)


@dataclass(frozen=True, slots=True)
class ArmReport:
    name: str
    examples: int
    parameters: int
    initial_model_sha256: str
    final_model_sha256: str
    optimizer_updates: int
    batch_schedule_sha256: str
    observation_manifest_sha256: str
    label_manifest_sha256: str
    tensor_manifest_sha256: str
    mean_loss: float
    final_loss: float
    teacher_forced_label_accuracy: float
    certified: int
    invalid: int
    overlong: int
    certification_rate: float
    final_search_calls: int
    final_oracle_calls: int
    assessor_verifier_calls: int
    model_decisions: int
    vm_calls: int
    cases: tuple[EvaluationCase, ...]


@dataclass(frozen=True, slots=True)
class PilotReport:
    schema: str
    status: str
    reasoning_claim_authorized: bool
    candidate_runtime: str
    assessor_boundary: str
    autonomous_input_fields: tuple[str, ...]
    forbidden_autonomous_capabilities: tuple[str, ...]
    seed: int
    device: str
    train_cases: int
    evaluation_cases: int
    fit_maximum_rows: int
    fit_maximum_columns: int
    evaluation_minimum_rows: int
    evaluation_minimum_columns: int
    evaluation_maximum_rows: int
    evaluation_maximum_columns: int
    strict_geometry_disjoint: bool
    model_schema: str
    model_config_sha256: str
    controller_parameters: int
    matched_parameter_budget: bool
    matched_update_budget: bool
    matched_data_budget: bool
    identical_initial_weights: bool
    final_search_calls: int
    final_oracle_calls: int
    evaluation_manifest_sha256: str
    preparation: PreparationReceipt
    arms: tuple[ArmReport, ...]
    minimum_material_evaluation_cases: int
    minimum_material_certification_rate: float
    search_beats_oracle_margin: float
    search_beats_random_margin: float
    material_mechanics_gate_passed: bool

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(asdict(self)) + b"\n"


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SearchDistillationError("CUDA device requested but unavailable")
    return device


def _validate_geometry(args: argparse.Namespace) -> bool:
    strict = (
        args.fit_maximum_rows < args.evaluation_minimum_rows
        and args.fit_maximum_columns < args.evaluation_minimum_columns
    )
    if not strict:
        raise SearchDistillationError(
            "evaluation geometry must be strictly larger on both axes"
        )
    if args.evaluation_maximum_rows > args.maximum_rows:
        raise SearchDistillationError("evaluation rows exceed model capacity")
    if args.evaluation_maximum_columns > args.maximum_columns:
        raise SearchDistillationError("evaluation columns exceed model capacity")
    return strict


def run_pilot(args: argparse.Namespace) -> PilotReport:
    strict_geometry = _validate_geometry(args)
    device = _resolve_device(args.device)
    train_matrices = generate_matrix_cases(
        seed=args.seed,
        count=args.train_cases,
        minimum_rows=2,
        maximum_rows=args.fit_maximum_rows,
        minimum_columns=2,
        maximum_columns=args.fit_maximum_columns,
    )
    evaluation_matrices = generate_matrix_cases(
        seed=args.seed ^ 0xE7A1,
        count=args.evaluation_cases,
        minimum_rows=args.evaluation_minimum_rows,
        maximum_rows=args.evaluation_maximum_rows,
        minimum_columns=args.evaluation_minimum_columns,
        maximum_columns=args.evaluation_maximum_columns,
    )
    prepared = prepare_matched_datasets(
        train_matrices,
        seed=args.seed,
        states_per_case=args.states_per_case,
        search_config=SearchPreparationConfig(
            max_nodes_expanded=args.search_max_nodes,
            max_edges_considered=args.search_max_edges,
            max_depth=args.search_max_depth,
            max_frontier=args.search_max_frontier,
            beam_width=args.search_beam_width,
            max_program_instructions=args.search_max_instructions,
            policy_noise_scale=float(args.search_policy_noise),
        ),
        scratch_root=args.scratch_root,
    )
    policy_config = PolicyConfig(
        maximum_rows=args.maximum_rows,
        maximum_columns=args.maximum_columns,
        width=args.width,
        blocks=args.blocks,
        feedforward=args.feedforward,
        dropout=0.0,
    )
    datasets = {
        arm: tensorize_labeled_states(prepared.arm(arm), policy_config) for arm in ARMS
    }
    observation_manifests = {
        dataset.observation_manifest_sha256 for dataset in datasets.values()
    }
    if len(observation_manifests) != 1:
        raise RuntimeError("arm tensor datasets do not share observations")
    torch.manual_seed(args.seed)
    template = EquivariantActionPolicy(policy_config)
    initial_state = deepcopy(template.state_dict())
    models = {arm: EquivariantActionPolicy(policy_config) for arm in ARMS}
    for model in models.values():
        model.load_state_dict(initial_state)
    initial_hashes = {arm: model_state_sha256(model) for arm, model in models.items()}
    if len(set(initial_hashes.values())) != 1:
        raise RuntimeError("arm initial weights differ")
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        amp_bfloat16=args.amp_bfloat16,
        torch_compile=args.torch_compile,
    )
    examples = next(iter(datasets.values())).examples
    schedule = build_batch_schedule(
        examples=examples,
        epochs=training_config.epochs,
        batch_size=training_config.batch_size,
        seed=args.seed ^ 0xBA7C,
    )
    arm_reports: list[ArmReport] = []
    training_receipts: dict[str, TrainingReceipt] = {}
    evaluation_receipts: dict[str, EvaluationReceipt] = {}
    for arm in ARMS:
        training = train_policy(
            models[arm],
            datasets[arm],
            schedule=schedule,
            config=training_config,
            device=device,
        )
        training_receipts[arm] = training
        accuracy = label_accuracy(
            models[arm],
            datasets[arm],
            device=device,
            amp_bfloat16=training_config.amp_bfloat16,
        )
        evaluation = evaluate_autonomous_policy(
            models[arm],
            evaluation_matrices,
            device=device,
            amp_bfloat16=training_config.amp_bfloat16,
            maximum_macros=args.maximum_rollout_macros,
            maximum_instructions=args.maximum_rollout_instructions,
        )
        evaluation_receipts[arm] = evaluation
        arm_reports.append(
            ArmReport(
                name=arm,
                examples=datasets[arm].examples,
                parameters=models[arm].parameter_count,
                initial_model_sha256=initial_hashes[arm],
                final_model_sha256=model_state_sha256(models[arm]),
                optimizer_updates=training.optimizer_updates,
                batch_schedule_sha256=training.batch_schedule_sha256,
                observation_manifest_sha256=datasets[arm].observation_manifest_sha256,
                label_manifest_sha256=datasets[arm].label_manifest_sha256,
                tensor_manifest_sha256=datasets[arm].tensor_manifest_sha256,
                mean_loss=training.mean_loss,
                final_loss=training.final_loss,
                teacher_forced_label_accuracy=accuracy,
                certified=evaluation.certified,
                invalid=evaluation.invalid,
                overlong=evaluation.overlong,
                certification_rate=evaluation.certification_rate,
                final_search_calls=evaluation.final_search_calls,
                final_oracle_calls=evaluation.final_oracle_calls,
                assessor_verifier_calls=evaluation.assessor_verifier_calls,
                model_decisions=evaluation.model_decisions,
                vm_calls=evaluation.vm_calls,
                cases=evaluation.cases,
            )
        )
    parameter_counts = {report.parameters for report in arm_reports}
    update_counts = {report.optimizer_updates for report in arm_reports}
    data_counts = {report.examples for report in arm_reports}
    schedule_hashes = {report.batch_schedule_sha256 for report in arm_reports}
    search_evaluation = evaluation_receipts[ARM_SEARCH]
    oracle_evaluation = evaluation_receipts[ARM_ORACLE]
    random_evaluation = evaluation_receipts[ARM_RANDOM]
    search_rate = search_evaluation.certification_rate
    oracle_rate = oracle_evaluation.certification_rate
    random_rate = random_evaluation.certification_rate
    material_gate = (
        args.evaluation_cases >= args.minimum_material_evaluation_cases
        and search_rate >= args.minimum_material_certification_rate
        and search_rate - oracle_rate >= args.search_beats_oracle_margin
        and search_rate - random_rate >= args.search_beats_random_margin
        and all(
            report.final_search_calls == 0 and report.final_oracle_calls == 0
            for report in arm_reports
        )
    )
    return PilotReport(
        schema=SCHEMA,
        status=STATUS,
        reasoning_claim_authorized=False,
        candidate_runtime="greedy_feed_forward_model_plus_primitive_vm_only",
        assessor_boundary=(
            "strict_verify_reduction_program_after_candidate_termination"
        ),
        autonomous_input_fields=(
            "current_matrix",
            "geometry_derived_unmasked_action_slots",
        ),
        forbidden_autonomous_capabilities=(
            "beam_search",
            "bounded_search",
            "oracle",
            "search_callback",
            "structural_potential",
            "verifier",
        ),
        seed=args.seed,
        device=str(device),
        train_cases=args.train_cases,
        evaluation_cases=args.evaluation_cases,
        fit_maximum_rows=args.fit_maximum_rows,
        fit_maximum_columns=args.fit_maximum_columns,
        evaluation_minimum_rows=args.evaluation_minimum_rows,
        evaluation_minimum_columns=args.evaluation_minimum_columns,
        evaluation_maximum_rows=args.evaluation_maximum_rows,
        evaluation_maximum_columns=args.evaluation_maximum_columns,
        strict_geometry_disjoint=strict_geometry,
        model_schema=MODEL_SCHEMA,
        model_config_sha256=policy_config.sha256,
        controller_parameters=next(iter(parameter_counts)),
        matched_parameter_budget=len(parameter_counts) == 1,
        matched_update_budget=(len(update_counts) == 1 and len(schedule_hashes) == 1),
        matched_data_budget=(len(data_counts) == 1 and len(observation_manifests) == 1),
        identical_initial_weights=len(set(initial_hashes.values())) == 1,
        final_search_calls=sum(report.final_search_calls for report in arm_reports),
        final_oracle_calls=sum(report.final_oracle_calls for report in arm_reports),
        evaluation_manifest_sha256=_digest(
            [matrix_sha256(matrix) for matrix in evaluation_matrices]
        ),
        preparation=prepared.receipt,
        arms=tuple(arm_reports),
        minimum_material_evaluation_cases=args.minimum_material_evaluation_cases,
        minimum_material_certification_rate=(args.minimum_material_certification_rate),
        search_beats_oracle_margin=args.search_beats_oracle_margin,
        search_beats_random_margin=args.search_beats_random_margin,
        material_mechanics_gate_passed=material_gate,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-cases", type=int, default=128)
    parser.add_argument("--evaluation-cases", type=int, default=128)
    parser.add_argument("--states-per-case", type=int, default=16)
    parser.add_argument("--fit-maximum-rows", type=int, default=3)
    parser.add_argument("--fit-maximum-columns", type=int, default=4)
    parser.add_argument("--evaluation-minimum-rows", type=int, default=4)
    parser.add_argument("--evaluation-minimum-columns", type=int, default=5)
    parser.add_argument("--evaluation-maximum-rows", type=int, default=4)
    parser.add_argument("--evaluation-maximum-columns", type=int, default=6)
    parser.add_argument("--maximum-rows", type=int, default=6)
    parser.add_argument("--maximum-columns", type=int, default=8)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--feedforward", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--amp-bfloat16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--torch-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--search-max-nodes", type=int, default=2_048)
    parser.add_argument("--search-max-edges", type=int, default=50_000)
    parser.add_argument("--search-max-depth", type=int, default=36)
    parser.add_argument("--search-max-frontier", type=int, default=96)
    parser.add_argument("--search-beam-width", type=int, default=96)
    parser.add_argument("--search-max-instructions", type=int, default=160)
    parser.add_argument("--search-policy-noise", type=float, default=8.0)
    parser.add_argument("--maximum-rollout-macros", type=int, default=96)
    parser.add_argument(
        "--maximum-rollout-instructions",
        type=int,
        default=320,
    )
    parser.add_argument(
        "--minimum-material-evaluation-cases",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--minimum-material-certification-rate",
        type=float,
        default=0.80,
    )
    parser.add_argument("--search-beats-oracle-margin", type=float, default=0.05)
    parser.add_argument("--search-beats-random-margin", type=float, default=0.20)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        args.device = "cpu"
        args.train_cases = 4
        args.evaluation_cases = 8
        args.states_per_case = 4
        args.width = 32
        args.blocks = 1
        args.feedforward = 64
        args.epochs = 1
        args.batch_size = 8
        args.amp_bfloat16 = False
        args.torch_compile = False
        args.search_max_nodes = 512
        args.search_max_edges = 10_000
        args.search_max_depth = 24
        args.search_max_frontier = 32
        args.search_beam_width = 32
        args.search_max_instructions = 96
        args.maximum_rollout_macros = 48
        args.maximum_rollout_instructions = 160
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_pilot(args)
    payload = report.canonical_bytes()
    if args.output is None:
        print(payload.decode("ascii"), end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
