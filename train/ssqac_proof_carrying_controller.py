#!/usr/bin/env python3
"""Proof-carrying local-action controller falsifier for SSQAC.

The candidate scores every legal ``(current, action, one-step successor)``
triple using four independently supervised contracts:

* expert action preference;
* current/successor remaining-step classes;
* successor terminality; and
* consistency between the predicted current and successor progress.

Only raw field matrices and geometry-relative action identity are candidate
inputs.  Canonical traces and remaining-step labels exist only in a sealed CPU
preparation artifact.  Final rollout is autonomous and search/oracle/verifier
free; the strict verifier is invoked only by a posthoc assessor.

This is an isolated mechanics falsifier.  It neither loads nor modifies the
protected Shohin checkpoint and cannot authorize continuation pretraining.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import resource
import time
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from episode_functor_algebra_machine import FIELD_MODULUS
from ssqac_successor_value_controller import (
    ACTION_ELIMINATE,
    ACTION_HALT,
    ACTION_SWAP,
    ACTION_TO_INDEX,
    ACTION_TYPES,
    PROTECTED_FLAGSHIP_PARAMETERS,
    TOTAL_PARAMETER_BUDGET,
    CandidateRollout,
    ResourceCounts,
    SuccessorAction,
    SuccessorValueError,
    apply_action,
    assess_rollout_posthoc,
    canonical_matrix,
    enumerate_legal_actions,
    generate_matrices,
    matrix_manifest,
    matrix_sha256,
)


PREPARATION_SCHEMA = "ssqac_proof_carrying_preparation_v1"
MODEL_SCHEMA = "ssqac_proof_carrying_action_controller_v1"
REPORT_SCHEMA = "ssqac_proof_carrying_experiment_v1"
RESOURCE_SCHEMA = "ssqac_proof_carrying_resource_counts_v1"
STATUS = "isolated_proof_carrying_action_falsifier_not_reasoning"

ARM_TREATMENT = "proof_carrying"
ARM_CLASSIFIER = "classifier_only"
ARM_ZERO_PROOF = "proof_heads_zeroed_at_inference"
ARM_SHUFFLED_BINDING = "shuffled_action_successor_binding"
ARM_SHUFFLED_PROGRESS = "shuffled_progress_labels"
ARM_RANDOM = "random_labels"
ARMS = (
    ARM_TREATMENT,
    ARM_CLASSIFIER,
    ARM_ZERO_PROOF,
    ARM_SHUFFLED_BINDING,
    ARM_SHUFFLED_PROGRESS,
    ARM_RANDOM,
)

MAX_PROGRESS_CLASS = 63
ROLE_FEATURES = 8
KIND_FEATURES = len(ACTION_TYPES)


class ProofCarryingError(ValueError):
    """The proof-carrying experiment failed closed."""


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
        raise ProofCarryingError("value is not canonical ASCII JSON") from error


def _digest(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ProofCarryingError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProofCarryingError(f"{label} must be a nonnegative integer")
    return value


def _source_sha256(path: Path) -> str:
    if not path.is_file():
        raise ProofCarryingError(f"source is missing: {path}")
    return sha256(path.read_bytes()).hexdigest()


def _action_data(action: SuccessorAction) -> list[object]:
    return action.canonical_data()


def _action_from_data(value: Sequence[object]) -> SuccessorAction:
    if len(value) != 4:
        raise ProofCarryingError("action record has the wrong width")
    return SuccessorAction(
        str(value[0]),
        row_a=int(value[1]),
        row_b=int(value[2]),
        column=int(value[3]),
    )


@dataclass(frozen=True, slots=True)
class ContractCandidate:
    action: SuccessorAction
    successor: tuple[tuple[int, ...], ...]
    expert_preference: int
    current_remaining: int
    successor_remaining: int
    successor_terminal: int
    progress_consistent: int

    def __post_init__(self) -> None:
        for label, value in (
            ("expert_preference", self.expert_preference),
            ("successor_terminal", self.successor_terminal),
            ("progress_consistent", self.progress_consistent),
        ):
            if value not in (0, 1):
                raise ProofCarryingError(f"{label} must be binary")
        for label, value in (
            ("current_remaining", self.current_remaining),
            ("successor_remaining", self.successor_remaining),
        ):
            _nonnegative_int(value, label=label)
            if value > MAX_PROGRESS_CLASS:
                raise ProofCarryingError(f"{label} exceeds class bound")

    def canonical_data(self) -> Mapping[str, object]:
        return {
            "action": _action_data(self.action),
            "successor": [list(row) for row in self.successor],
            "expert_preference": self.expert_preference,
            "current_remaining": self.current_remaining,
            "successor_remaining": self.successor_remaining,
            "successor_terminal": self.successor_terminal,
            "progress_consistent": self.progress_consistent,
        }


@dataclass(frozen=True, slots=True)
class ContractState:
    rows: tuple[tuple[int, ...], ...]
    candidates: tuple[ContractCandidate, ...]

    def __post_init__(self) -> None:
        legal = enumerate_legal_actions(self.rows)
        if tuple(candidate.action for candidate in self.candidates) != legal:
            raise ProofCarryingError("contract candidates do not match legal actions")
        if sum(candidate.expert_preference for candidate in self.candidates) != 1:
            raise ProofCarryingError("state must have exactly one expert action")
        for candidate in self.candidates:
            if apply_action(self.rows, candidate.action) != candidate.successor:
                raise ProofCarryingError("prepared successor does not match action")

    @property
    def expert_index(self) -> int:
        return next(
            index
            for index, candidate in enumerate(self.candidates)
            if candidate.expert_preference
        )

    def canonical_data(self) -> Mapping[str, object]:
        return {
            "rows": [list(row) for row in self.rows],
            "candidates": [
                candidate.canonical_data() for candidate in self.candidates
            ],
        }

    @property
    def sha256(self) -> str:
        return _digest(self.canonical_data())


@dataclass(frozen=True, slots=True)
class PreparationArtifact:
    schema: str
    seed: int
    source_sha256: str
    oracle_source_sha256: str
    successor_source_sha256: str
    train_matrix_manifest_sha256: str
    evaluation_matrix_manifest_sha256: str
    state_manifest_sha256: str
    train_matrices: tuple[tuple[tuple[int, ...], ...], ...]
    evaluation_matrices: tuple[tuple[tuple[int, ...], ...], ...]
    states: tuple[ContractState, ...]
    oracle_calls: int
    canonical_distance_cache_entries: int
    legal_triples: int
    matched_legal_negative_triples: int
    train_maximum_rows: int
    train_maximum_columns: int
    evaluation_minimum_rows: int
    evaluation_minimum_columns: int
    evaluation_maximum_rows: int
    evaluation_maximum_columns: int
    maximum_preparation_steps: int
    wall_seconds: float

    def canonical_data(self) -> Mapping[str, object]:
        return {
            "schema": self.schema,
            "seed": self.seed,
            "source_sha256": self.source_sha256,
            "oracle_source_sha256": self.oracle_source_sha256,
            "successor_source_sha256": self.successor_source_sha256,
            "train_matrix_manifest_sha256": self.train_matrix_manifest_sha256,
            "evaluation_matrix_manifest_sha256": (
                self.evaluation_matrix_manifest_sha256
            ),
            "state_manifest_sha256": self.state_manifest_sha256,
            "train_matrices": [
                [list(row) for row in matrix] for matrix in self.train_matrices
            ],
            "evaluation_matrices": [
                [list(row) for row in matrix] for matrix in self.evaluation_matrices
            ],
            "states": [state.canonical_data() for state in self.states],
            "oracle_calls": self.oracle_calls,
            "canonical_distance_cache_entries": (
                self.canonical_distance_cache_entries
            ),
            "legal_triples": self.legal_triples,
            "matched_legal_negative_triples": (
                self.matched_legal_negative_triples
            ),
            "train_maximum_rows": self.train_maximum_rows,
            "train_maximum_columns": self.train_maximum_columns,
            "evaluation_minimum_rows": self.evaluation_minimum_rows,
            "evaluation_minimum_columns": self.evaluation_minimum_columns,
            "evaluation_maximum_rows": self.evaluation_maximum_rows,
            "evaluation_maximum_columns": self.evaluation_maximum_columns,
            "maximum_preparation_steps": self.maximum_preparation_steps,
            "wall_seconds": self.wall_seconds,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.canonical_data()) + b"\n"


def _state_manifest(states: Sequence[ContractState]) -> str:
    return sha256(
        ("\n".join(state.sha256 for state in states) + "\n").encode("ascii")
    ).hexdigest()


def _load_preparation(path: Path) -> PreparationArtifact:
    raw_bytes = path.read_bytes()
    try:
        value = json.loads(raw_bytes)
    except json.JSONDecodeError as error:
        raise ProofCarryingError("preparation artifact is invalid JSON") from error
    if value.get("schema") != PREPARATION_SCHEMA:
        raise ProofCarryingError("preparation schema mismatch")
    train = tuple(
        canonical_matrix(matrix) for matrix in value["train_matrices"]
    )
    evaluation = tuple(
        canonical_matrix(matrix) for matrix in value["evaluation_matrices"]
    )
    states = []
    for raw_state in value["states"]:
        rows = canonical_matrix(raw_state["rows"])
        candidates = tuple(
            ContractCandidate(
                action=_action_from_data(item["action"]),
                successor=canonical_matrix(item["successor"]),
                expert_preference=int(item["expert_preference"]),
                current_remaining=int(item["current_remaining"]),
                successor_remaining=int(item["successor_remaining"]),
                successor_terminal=int(item["successor_terminal"]),
                progress_consistent=int(item["progress_consistent"]),
            )
            for item in raw_state["candidates"]
        )
        states.append(ContractState(rows=rows, candidates=candidates))
    artifact = PreparationArtifact(
        schema=value["schema"],
        seed=int(value["seed"]),
        source_sha256=str(value["source_sha256"]),
        oracle_source_sha256=str(value["oracle_source_sha256"]),
        successor_source_sha256=str(value["successor_source_sha256"]),
        train_matrix_manifest_sha256=str(value["train_matrix_manifest_sha256"]),
        evaluation_matrix_manifest_sha256=str(
            value["evaluation_matrix_manifest_sha256"]
        ),
        state_manifest_sha256=str(value["state_manifest_sha256"]),
        train_matrices=train,
        evaluation_matrices=evaluation,
        states=tuple(states),
        oracle_calls=int(value["oracle_calls"]),
        canonical_distance_cache_entries=int(
            value["canonical_distance_cache_entries"]
        ),
        legal_triples=int(value["legal_triples"]),
        matched_legal_negative_triples=int(
            value["matched_legal_negative_triples"]
        ),
        train_maximum_rows=int(value["train_maximum_rows"]),
        train_maximum_columns=int(value["train_maximum_columns"]),
        evaluation_minimum_rows=int(value["evaluation_minimum_rows"]),
        evaluation_minimum_columns=int(value["evaluation_minimum_columns"]),
        evaluation_maximum_rows=int(value["evaluation_maximum_rows"]),
        evaluation_maximum_columns=int(value["evaluation_maximum_columns"]),
        maximum_preparation_steps=int(value["maximum_preparation_steps"]),
        wall_seconds=float(value["wall_seconds"]),
    )
    if artifact.canonical_bytes() != raw_bytes:
        raise ProofCarryingError("preparation artifact is not canonical")
    if matrix_manifest(train) != artifact.train_matrix_manifest_sha256:
        raise ProofCarryingError("train matrix manifest mismatch")
    if matrix_manifest(evaluation) != artifact.evaluation_matrix_manifest_sha256:
        raise ProofCarryingError("evaluation matrix manifest mismatch")
    if _state_manifest(artifact.states) != artifact.state_manifest_sha256:
        raise ProofCarryingError("state manifest mismatch")
    if set(train) & set(evaluation):
        raise ProofCarryingError("train and evaluation matrices overlap")
    if artifact.evaluation_minimum_rows <= artifact.train_maximum_rows:
        raise ProofCarryingError("evaluation row geometry is not held out")
    if artifact.evaluation_minimum_columns <= artifact.train_maximum_columns:
        raise ProofCarryingError("evaluation column geometry is not held out")
    return artifact


def _canonical_remaining_distance(
    rows: tuple[tuple[int, ...], ...],
    *,
    oracle_module: object,
    counter: object,
    maximum_steps: int,
    cache: dict[str, int],
) -> int:
    key = matrix_sha256(rows)
    if key in cache:
        return cache[key]
    matrix = rows
    trail: list[tuple[tuple[int, ...], ...]] = []
    for _ in range(maximum_steps):
        trail.append(matrix)
        raw_action = oracle_module.next_preparation_macro(matrix, counter=counter)
        action = SuccessorAction(
            raw_action.kind,
            row_a=raw_action.row_a,
            row_b=raw_action.row_b,
            column=raw_action.column,
        )
        if action.kind == ACTION_HALT:
            distance = 0
            break
        matrix = apply_action(matrix, action)
    else:
        raise ProofCarryingError("canonical distance exceeded preparation bound")
    for reverse_index, seen in enumerate(reversed(trail)):
        value = distance + reverse_index
        if value > MAX_PROGRESS_CLASS:
            raise ProofCarryingError("canonical distance exceeds class vocabulary")
        cache[matrix_sha256(seen)] = value
    return cache[key]


def prepare_artifact(
    *,
    seed: int,
    train_matrices: int,
    evaluation_matrices: int,
    train_maximum_rows: int,
    train_maximum_columns: int,
    evaluation_minimum_rows: int,
    evaluation_minimum_columns: int,
    evaluation_maximum_rows: int,
    evaluation_maximum_columns: int,
    maximum_preparation_steps: int,
) -> PreparationArtifact:
    """Generate and seal all oracle-derived data before GPU fitting."""

    started = time.perf_counter()
    import inspect
    import ssqac_soft_value_iteration_controller as oracle_module
    import ssqac_successor_value_controller as successor_module

    oracle_path = Path(inspect.getsourcefile(oracle_module) or "")
    successor_path = Path(inspect.getsourcefile(successor_module) or "")
    train = generate_matrices(
        seed=seed,
        count=train_matrices,
        minimum_rows=2,
        maximum_rows=train_maximum_rows,
        minimum_columns=2,
        maximum_columns=train_maximum_columns,
    )
    evaluation = generate_matrices(
        seed=seed + 1,
        count=evaluation_matrices,
        minimum_rows=evaluation_minimum_rows,
        maximum_rows=evaluation_maximum_rows,
        minimum_columns=evaluation_minimum_columns,
        maximum_columns=evaluation_maximum_columns,
        excluded=set(train),
    )
    counter = oracle_module.PreparationOracleCounter()
    cache: dict[str, int] = {}
    states_by_hash: dict[str, ContractState] = {}
    for initial in train:
        matrix = initial
        for _ in range(maximum_preparation_steps):
            raw_expert = oracle_module.next_preparation_macro(
                matrix,
                counter=counter,
            )
            expert = SuccessorAction(
                raw_expert.kind,
                row_a=raw_expert.row_a,
                row_b=raw_expert.row_b,
                column=raw_expert.column,
            )
            legal = enumerate_legal_actions(matrix)
            if expert not in legal:
                raise ProofCarryingError("canonical oracle emitted illegal action")
            current_remaining = _canonical_remaining_distance(
                matrix,
                oracle_module=oracle_module,
                counter=counter,
                maximum_steps=maximum_preparation_steps,
                cache=cache,
            )
            candidates = []
            for action in legal:
                successor = apply_action(matrix, action)
                successor_remaining = _canonical_remaining_distance(
                    successor,
                    oracle_module=oracle_module,
                    counter=counter,
                    maximum_steps=maximum_preparation_steps,
                    cache=cache,
                )
                candidates.append(
                    ContractCandidate(
                        action=action,
                        successor=successor,
                        expert_preference=int(action == expert),
                        current_remaining=current_remaining,
                        successor_remaining=successor_remaining,
                        successor_terminal=int(successor_remaining == 0),
                        progress_consistent=int(
                            successor_remaining
                            == max(current_remaining - 1, 0)
                        ),
                    )
                )
            state = ContractState(matrix, tuple(candidates))
            prior = states_by_hash.get(matrix_sha256(matrix))
            if prior is not None and prior != state:
                raise ProofCarryingError("preparation state labels conflict")
            states_by_hash[matrix_sha256(matrix)] = state
            if expert.kind == ACTION_HALT:
                break
            matrix = apply_action(matrix, expert)
        else:
            raise ProofCarryingError("canonical trace exceeded preparation bound")
    states = tuple(states_by_hash[key] for key in sorted(states_by_hash))
    legal_triples = sum(len(state.candidates) for state in states)
    negatives = legal_triples - len(states)
    return PreparationArtifact(
        schema=PREPARATION_SCHEMA,
        seed=seed,
        source_sha256=_source_sha256(Path(__file__)),
        oracle_source_sha256=_source_sha256(oracle_path),
        successor_source_sha256=_source_sha256(successor_path),
        train_matrix_manifest_sha256=matrix_manifest(train),
        evaluation_matrix_manifest_sha256=matrix_manifest(evaluation),
        state_manifest_sha256=_state_manifest(states),
        train_matrices=train,
        evaluation_matrices=evaluation,
        states=states,
        oracle_calls=int(counter.calls),
        canonical_distance_cache_entries=len(cache),
        legal_triples=legal_triples,
        matched_legal_negative_triples=negatives,
        train_maximum_rows=train_maximum_rows,
        train_maximum_columns=train_maximum_columns,
        evaluation_minimum_rows=evaluation_minimum_rows,
        evaluation_minimum_columns=evaluation_minimum_columns,
        evaluation_maximum_rows=evaluation_maximum_rows,
        evaluation_maximum_columns=evaluation_maximum_columns,
        maximum_preparation_steps=maximum_preparation_steps,
        wall_seconds=time.perf_counter() - started,
    )


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    field_width: int = 64
    width: int = 384
    cell_hidden: int = 512
    matrix_layers: int = 4
    contract_hidden: int = 512
    coordinate_harmonics: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for label, value in (
            ("field_width", self.field_width),
            ("width", self.width),
            ("cell_hidden", self.cell_hidden),
            ("matrix_layers", self.matrix_layers),
            ("contract_hidden", self.contract_hidden),
            ("coordinate_harmonics", self.coordinate_harmonics),
        ):
            _positive_int(value, label=label)
        if not isinstance(self.dropout, float) or not 0.0 <= self.dropout < 1.0:
            raise ProofCarryingError("dropout must be in [0, 1)")


class EquivariantGridLayer(nn.Module):
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
        return self.norm(
            cells
            + self.update(
                torch.cat((cells, row_mean, column_mean, global_mean), dim=-1)
            )
        )


@dataclass(frozen=True, slots=True)
class ContractOutputs:
    preference_logits: Tensor
    current_progress_logits: Tensor
    successor_progress_logits: Tensor
    terminal_logits: Tensor
    consistency_logits: Tensor
    contract_logits: Tensor
    action_hidden: Tensor


class ProofCarryingController(nn.Module):
    """Geometry-general raw-triple encoder with separate learned contracts."""

    def __init__(self, config: ControllerConfig = ControllerConfig()) -> None:
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
            EquivariantGridLayer(
                config.width,
                config.cell_hidden,
                config.dropout,
            )
            for _ in range(config.matrix_layers)
        )
        pooled_width = config.width * 7 + KIND_FEATURES
        self.triple_projection = nn.Sequential(
            nn.Linear(pooled_width, config.contract_hidden),
            nn.GELU(),
            nn.LayerNorm(config.contract_hidden),
        )
        self.contract_recurrence = nn.GRUCell(
            config.contract_hidden,
            config.contract_hidden,
        )
        self.preference_head = nn.Linear(config.contract_hidden, 1)
        self.current_progress_head = nn.Linear(
            config.contract_hidden,
            MAX_PROGRESS_CLASS + 1,
        )
        self.successor_progress_head = nn.Linear(
            config.contract_hidden,
            MAX_PROGRESS_CLASS + 1,
        )
        self.terminal_head = nn.Linear(config.contract_hidden, 1)
        self.consistency_head = nn.Sequential(
            nn.Linear(2 * (MAX_PROGRESS_CLASS + 1), config.contract_hidden),
            nn.GELU(),
            nn.Linear(config.contract_hidden, 1),
        )
        self.contract_aggregation = nn.Sequential(
            nn.Linear(5, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        if self.complete_system_parameters >= TOTAL_PARAMETER_BUDGET:
            raise ProofCarryingError("complete system exceeds 200M parameters")

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def complete_system_parameters(self) -> int:
        return PROTECTED_FLAGSHIP_PARAMETERS + self.parameter_count

    def parameter_count_breakdown(self) -> Mapping[str, int]:
        names = (
            "field_embedding",
            "cell_projection",
            "grid_layers",
            "triple_projection",
            "contract_recurrence",
            "preference_head",
            "current_progress_head",
            "successor_progress_head",
            "terminal_head",
            "consistency_head",
            "contract_aggregation",
        )
        values = {
            name: sum(
                parameter.numel()
                for parameter in getattr(self, name).parameters()
            )
            for name in names
        }
        values["total"] = sum(values.values())
        return values

    def _coordinate_features(
        self,
        row_count: int,
        column_count: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        row = (
            torch.arange(row_count, device=device, dtype=torch.float32) + 0.5
        ) / row_count
        column = (
            torch.arange(column_count, device=device, dtype=torch.float32) + 0.5
        ) / column_count
        row_grid = row[:, None].expand(row_count, column_count)
        column_grid = column[None, :].expand(row_count, column_count)
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
        rows = torch.arange(row_count, device=device)
        columns = torch.arange(column_count, device=device)
        first = rows[None, :, None] == row_a[:, None, None]
        second = rows[None, :, None] == row_b[:, None, None]
        selected_column = columns[None, None, :] == column[:, None, None]
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
        any_row = first | second
        return torch.stack(
            (
                first,
                second,
                any_row,
                selected_column,
                first & selected_column,
                second & selected_column,
                ~any_row,
                ~selected_column,
            ),
            dim=-1,
        ).to(dtype=dtype)

    @staticmethod
    def _masked_mean(cells: Tensor, mask: Tensor) -> Tensor:
        weights = mask.to(cells.dtype)
        denominator = weights.sum(dim=(1, 2)).clamp_min(1.0)
        return (cells * weights[..., None]).sum(dim=(1, 2)) / denominator[:, None]

    def forward(
        self,
        current_values: Tensor,
        successor_values: Tensor,
        action_kind: Tensor,
        row_a: Tensor,
        row_b: Tensor,
        column: Tensor,
        recurrent_state: Tensor | None = None,
    ) -> ContractOutputs:
        """Score raw triples without host-computed structural features."""

        action_count, row_count, column_count = successor_values.shape
        if current_values.shape != (row_count, column_count):
            raise ProofCarryingError("current/successor geometry mismatch")
        current = self.field_embedding(current_values)
        successor = self.field_embedding(successor_values)
        current = current[None].expand(action_count, -1, -1, -1)
        coordinates = self._coordinate_features(
            row_count,
            column_count,
            device=current.device,
            dtype=current.dtype,
        )[None].expand(action_count, -1, -1, -1)
        roles = self._role_features(
            action_kind,
            row_a,
            row_b,
            column,
            row_count=row_count,
            column_count=column_count,
            dtype=current.dtype,
        )
        kinds = F.one_hot(action_kind, num_classes=KIND_FEATURES).to(current.dtype)
        kind_grid = kinds[:, None, None].expand(
            action_count,
            row_count,
            column_count,
            KIND_FEATURES,
        )
        cells = self.cell_projection(
            torch.cat(
                (current, successor, coordinates, roles, kind_grid),
                dim=-1,
            )
        )
        for layer in self.grid_layers:
            cells = layer(cells)
        masks = [roles[..., index].bool() for index in (0, 1, 3, 4, 5)]
        pooled = torch.cat(
            (
                cells.mean(dim=(1, 2)),
                cells.amax(dim=(1, 2)),
                *(self._masked_mean(cells, mask) for mask in masks),
                kinds,
            ),
            dim=-1,
        )
        hidden = self.triple_projection(pooled)
        if recurrent_state is None:
            recurrent_state = torch.zeros(
                self.config.contract_hidden,
                device=hidden.device,
                dtype=hidden.dtype,
            )
        if recurrent_state.shape != (self.config.contract_hidden,):
            raise ProofCarryingError("recurrent contract state has wrong shape")
        hidden = self.contract_recurrence(
            hidden,
            recurrent_state[None].expand_as(hidden),
        )
        preference = self.preference_head(hidden).squeeze(-1)
        current_progress = self.current_progress_head(hidden)
        successor_progress = self.successor_progress_head(hidden)
        consistency = self.consistency_head(
            torch.cat(
                (
                    F.softmax(current_progress.float(), dim=-1).to(hidden.dtype),
                    F.softmax(successor_progress.float(), dim=-1).to(hidden.dtype),
                ),
                dim=-1,
            )
        ).squeeze(-1)
        terminal = self.terminal_head(hidden).squeeze(-1)
        classes = torch.arange(
            MAX_PROGRESS_CLASS + 1,
            device=hidden.device,
            dtype=torch.float32,
        )
        expected_current = (
            F.softmax(current_progress.float(), dim=-1) * classes
        ).sum(dim=-1) / MAX_PROGRESS_CLASS
        expected_successor = (
            F.softmax(successor_progress.float(), dim=-1) * classes
        ).sum(dim=-1) / MAX_PROGRESS_CLASS
        contract_features = torch.stack(
            (
                expected_current,
                expected_successor,
                expected_current - expected_successor,
                torch.sigmoid(terminal.float()),
                torch.sigmoid(consistency.float()),
            ),
            dim=-1,
        ).to(hidden.dtype)
        contract = self.contract_aggregation(contract_features).squeeze(-1)
        return ContractOutputs(
            preference_logits=preference,
            current_progress_logits=current_progress,
            successor_progress_logits=successor_progress,
            terminal_logits=terminal,
            consistency_logits=consistency,
            contract_logits=contract,
            action_hidden=hidden,
        )


@dataclass(slots=True)
class MutableProofResources:
    model_forward_calls: int = 0
    legal_triples_scored: int = 0
    raw_current_matrix_cells: int = 0
    raw_successor_matrix_cells: int = 0
    selected_actions: int = 0
    candidate_oracle_calls: int = 0
    candidate_search_calls: int = 0
    candidate_verifier_calls: int = 0

    def freeze(self) -> Mapping[str, int | str]:
        return {
            "schema": RESOURCE_SCHEMA,
            "model_forward_calls": self.model_forward_calls,
            "legal_triples_scored": self.legal_triples_scored,
            "raw_current_matrix_cells": self.raw_current_matrix_cells,
            "raw_successor_matrix_cells": self.raw_successor_matrix_cells,
            "selected_actions": self.selected_actions,
            "candidate_oracle_calls": self.candidate_oracle_calls,
            "candidate_search_calls": self.candidate_search_calls,
            "candidate_verifier_calls": self.candidate_verifier_calls,
        }


def _binding_permutation(
    rows: tuple[tuple[int, ...], ...],
    actions: Sequence[SuccessorAction],
    *,
    seed: int,
) -> tuple[int, ...]:
    if len(actions) <= 1:
        return tuple(range(len(actions)))
    digest = sha256(
        _canonical_bytes(
            {
                "rows": [list(row) for row in rows],
                "actions": [_action_data(action) for action in actions],
                "seed": seed,
            }
        )
    ).digest()
    shift = 1 + int.from_bytes(digest[:8], "big") % (len(actions) - 1)
    return tuple((index + shift) % len(actions) for index in range(len(actions)))


def _visible_successors(
    rows: tuple[tuple[int, ...], ...],
    actions: Sequence[SuccessorAction],
    *,
    shuffled_binding: bool,
    binding_seed: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    successors = tuple(apply_action(rows, action) for action in actions)
    if not shuffled_binding:
        return successors
    permutation = _binding_permutation(rows, actions, seed=binding_seed)
    return tuple(successors[index] for index in permutation)


def _model_inputs(
    model: ProofCarryingController,
    rows: tuple[tuple[int, ...], ...],
    actions: Sequence[SuccessorAction],
    successors: Sequence[tuple[tuple[int, ...], ...]],
    recurrent_state: Tensor | None,
    resources: MutableProofResources,
) -> ContractOutputs:
    reference = next(model.parameters())
    device = reference.device
    outputs = model(
        torch.tensor(rows, dtype=torch.long, device=device),
        torch.tensor(successors, dtype=torch.long, device=device),
        torch.tensor(
            [ACTION_TO_INDEX[action.kind] for action in actions],
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            [action.row_a for action in actions],
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            [action.row_b for action in actions],
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            [action.column for action in actions],
            dtype=torch.long,
            device=device,
        ),
        recurrent_state,
    )
    cells = len(rows) * len(rows[0])
    resources.model_forward_calls += 1
    resources.legal_triples_scored += len(actions)
    resources.raw_current_matrix_cells += cells
    resources.raw_successor_matrix_cells += cells * len(actions)
    return outputs


def _selection_logits(
    outputs: ContractOutputs,
    *,
    use_contract: bool,
    zero_proof: bool,
) -> Tensor:
    if not use_contract or zero_proof:
        return outputs.preference_logits
    return outputs.preference_logits + outputs.contract_logits


@dataclass(frozen=True, slots=True)
class ArmSpec:
    name: str
    train_contract: bool
    inference_contract: bool
    inference_zero_proof: bool
    shuffled_binding: bool
    shuffled_progress_labels: bool
    random_labels: bool


ARM_SPECS = (
    ArmSpec(ARM_TREATMENT, True, True, False, False, False, False),
    ArmSpec(ARM_CLASSIFIER, False, False, False, False, False, False),
    ArmSpec(ARM_ZERO_PROOF, True, True, True, False, False, False),
    ArmSpec(ARM_SHUFFLED_BINDING, True, True, False, True, False, False),
    ArmSpec(ARM_SHUFFLED_PROGRESS, True, True, False, False, True, False),
    ArmSpec(ARM_RANDOM, True, True, False, False, False, True),
)


def _proof_targets(
    state: ContractState,
    *,
    spec: ArmSpec,
    seed: int,
) -> tuple[int, Tensor, Tensor, Tensor, Tensor]:
    rng = random.Random(
        int.from_bytes(
            sha256(
                _canonical_bytes(
                    {"state": state.sha256, "arm": spec.name, "seed": seed}
                )
            ).digest()[:8],
            "big",
        )
    )
    expert_index = state.expert_index
    current = [candidate.current_remaining for candidate in state.candidates]
    successor = [candidate.successor_remaining for candidate in state.candidates]
    terminal = [candidate.successor_terminal for candidate in state.candidates]
    consistent = [candidate.progress_consistent for candidate in state.candidates]
    if spec.shuffled_progress_labels and len(state.candidates) > 1:
        permutation = list(range(len(state.candidates)))
        rng.shuffle(permutation)
        current = [current[index] for index in permutation]
        successor = [successor[index] for index in permutation]
        terminal = [terminal[index] for index in permutation]
        consistent = [consistent[index] for index in permutation]
    if spec.random_labels:
        alternatives = [
            index for index in range(len(state.candidates)) if index != expert_index
        ]
        if alternatives:
            expert_index = rng.choice(alternatives)
        current = [
            rng.randrange(MAX_PROGRESS_CLASS + 1) for _ in state.candidates
        ]
        successor = [
            rng.randrange(MAX_PROGRESS_CLASS + 1) for _ in state.candidates
        ]
        terminal = [rng.randrange(2) for _ in state.candidates]
        consistent = [rng.randrange(2) for _ in state.candidates]
    return (
        expert_index,
        torch.tensor(current, dtype=torch.long),
        torch.tensor(successor, dtype=torch.long),
        torch.tensor(terminal, dtype=torch.float32),
        torch.tensor(consistent, dtype=torch.float32),
    )


@dataclass(frozen=True, slots=True)
class TrainingResult:
    optimizer_updates: int
    examples_seen: int
    legal_triples_seen: int
    mean_total_loss: float
    final_total_loss: float
    mean_preference_loss: float
    mean_progress_loss: float
    mean_terminal_loss: float
    mean_consistency_loss: float
    train_label_correct: int
    train_label_total: int
    wall_seconds: float
    peak_cuda_memory_bytes: int
    resources: Mapping[str, int | str]


def _autocast(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def train_arm(
    model: ProofCarryingController,
    states: Sequence[ContractState],
    *,
    spec: ArmSpec,
    optimizer_updates: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    amp_bfloat16: bool,
    binding_seed: int,
) -> TrainingResult:
    """Fit one matched arm for exactly the requested optimizer updates."""

    updates = _positive_int(optimizer_updates, label="optimizer_updates")
    batch = _positive_int(batch_size, label="batch_size")
    if not states:
        raise ProofCarryingError("training states are empty")
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        fused=device.type == "cuda",
    )
    rng = random.Random(seed)
    order = list(range(len(states)))
    cursor = len(order)
    total_losses = []
    preference_losses = []
    progress_losses = []
    terminal_losses = []
    consistency_losses = []
    resources = MutableProofResources()
    triples = examples = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        batch_loss = torch.zeros((), device=device)
        for _ in range(batch):
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            state = states[order[cursor]]
            cursor += 1
            actions = tuple(candidate.action for candidate in state.candidates)
            successors = _visible_successors(
                state.rows,
                actions,
                shuffled_binding=spec.shuffled_binding,
                binding_seed=binding_seed,
            )
            targets = _proof_targets(state, spec=spec, seed=seed)
            target_index, current, successor, terminal, consistent = targets
            current = current.to(device)
            successor = successor.to(device)
            terminal = terminal.to(device)
            consistent = consistent.to(device)
            with _autocast(device, amp_bfloat16):
                outputs = _model_inputs(
                    model,
                    state.rows,
                    actions,
                    successors,
                    None,
                    resources,
                )
                preference_loss = F.cross_entropy(
                    outputs.preference_logits[None],
                    torch.tensor([target_index], device=device),
                )
                if spec.train_contract:
                    current_loss = F.cross_entropy(
                        outputs.current_progress_logits,
                        current,
                    )
                    successor_loss = F.cross_entropy(
                        outputs.successor_progress_logits,
                        successor,
                    )
                    progress_loss = 0.5 * (current_loss + successor_loss)
                    terminal_loss = F.binary_cross_entropy_with_logits(
                        outputs.terminal_logits,
                        terminal,
                    )
                    consistency_loss = F.binary_cross_entropy_with_logits(
                        outputs.consistency_logits,
                        consistent,
                    )
                    selection_loss = F.cross_entropy(
                        _selection_logits(
                            outputs,
                            use_contract=True,
                            zero_proof=False,
                        )[None],
                        torch.tensor([target_index], device=device),
                    )
                    loss = (
                        selection_loss
                        + 0.5 * preference_loss
                        + 0.25 * progress_loss
                        + 0.15 * terminal_loss
                        + 0.15 * consistency_loss
                    )
                else:
                    progress_loss = torch.zeros((), device=device)
                    terminal_loss = torch.zeros((), device=device)
                    consistency_loss = torch.zeros((), device=device)
                    loss = 1.5 * preference_loss
            batch_loss = batch_loss + loss / batch
            total_losses.append(float(loss.detach()))
            preference_losses.append(float(preference_loss.detach()))
            progress_losses.append(float(progress_loss.detach()))
            terminal_losses.append(float(terminal_loss.detach()))
            consistency_losses.append(float(consistency_loss.detach()))
            examples += 1
            triples += len(actions)
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    correct = 0
    diagnostic_resources = MutableProofResources()
    model.eval()
    with torch.no_grad():
        for state in states:
            actions = tuple(candidate.action for candidate in state.candidates)
            successors = _visible_successors(
                state.rows,
                actions,
                shuffled_binding=spec.shuffled_binding,
                binding_seed=binding_seed,
            )
            outputs = _model_inputs(
                model,
                state.rows,
                actions,
                successors,
                None,
                diagnostic_resources,
            )
            predicted = int(
                _selection_logits(
                    outputs,
                    use_contract=spec.inference_contract,
                    zero_proof=spec.inference_zero_proof,
                ).argmax()
            )
            correct += predicted == state.expert_index
    peak = (
        torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0
    )
    return TrainingResult(
        optimizer_updates=updates,
        examples_seen=examples,
        legal_triples_seen=triples,
        mean_total_loss=sum(total_losses) / len(total_losses),
        final_total_loss=total_losses[-1],
        mean_preference_loss=sum(preference_losses) / len(preference_losses),
        mean_progress_loss=sum(progress_losses) / len(progress_losses),
        mean_terminal_loss=sum(terminal_losses) / len(terminal_losses),
        mean_consistency_loss=(
            sum(consistency_losses) / len(consistency_losses)
        ),
        train_label_correct=correct,
        train_label_total=len(states),
        wall_seconds=time.perf_counter() - started,
        peak_cuda_memory_bytes=peak,
        resources={
            "training": resources.freeze(),
            "diagnostic": diagnostic_resources.freeze(),
        },
    )


@dataclass(frozen=True, slots=True)
class ProofRollout:
    halted: bool
    invalid: bool
    overlong: bool
    actions: tuple[SuccessorAction, ...]
    output_rows: tuple[tuple[int, ...], ...]
    resources: Mapping[str, int | str]


@torch.no_grad()
def autonomous_rollout(
    model: ProofCarryingController,
    rows: Iterable[Iterable[int]],
    *,
    spec: ArmSpec,
    binding_seed: int,
    maximum_steps: int,
) -> ProofRollout:
    """Roll out using only raw triples and model-owned contract state."""

    matrix = canonical_matrix(rows)
    actions_taken = []
    resources = MutableProofResources()
    recurrent_state: Tensor | None = None
    model.eval()
    for _ in range(maximum_steps):
        actions = enumerate_legal_actions(matrix)
        successors = _visible_successors(
            matrix,
            actions,
            shuffled_binding=spec.shuffled_binding,
            binding_seed=binding_seed,
        )
        outputs = _model_inputs(
            model,
            matrix,
            actions,
            successors,
            recurrent_state,
            resources,
        )
        selected_index = int(
            _selection_logits(
                outputs,
                use_contract=spec.inference_contract,
                zero_proof=spec.inference_zero_proof,
            ).argmax()
        )
        selected = actions[selected_index]
        resources.selected_actions += 1
        actions_taken.append(selected)
        recurrent_state = outputs.action_hidden[selected_index].detach()
        if selected.kind == ACTION_HALT:
            return ProofRollout(
                True,
                False,
                False,
                tuple(actions_taken),
                matrix,
                resources.freeze(),
            )
        try:
            matrix = apply_action(matrix, selected)
        except SuccessorValueError:
            return ProofRollout(
                False,
                True,
                False,
                tuple(actions_taken),
                matrix,
                resources.freeze(),
            )
    return ProofRollout(
        False,
        False,
        True,
        tuple(actions_taken),
        matrix,
        resources.freeze(),
    )


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    strict_certified: int
    total: int
    invalid: int
    overlong: int
    halted: int
    posthoc_verifier_calls: int
    candidate_resources: Mapping[str, int | str]
    wall_seconds: float

    @property
    def rate(self) -> float:
        return self.strict_certified / self.total if self.total else 0.0


def evaluate_arm(
    model: ProofCarryingController,
    matrices: Sequence[tuple[tuple[int, ...], ...]],
    *,
    spec: ArmSpec,
    binding_seed: int,
    maximum_steps: int,
) -> EvaluationResult:
    certified = invalid = overlong = halted = posthoc = 0
    resource_totals = MutableProofResources()
    started = time.perf_counter()
    for matrix in matrices:
        rollout = autonomous_rollout(
            model,
            matrix,
            spec=spec,
            binding_seed=binding_seed,
            maximum_steps=maximum_steps,
        )
        for field in (
            "model_forward_calls",
            "legal_triples_scored",
            "raw_current_matrix_cells",
            "raw_successor_matrix_cells",
            "selected_actions",
            "candidate_oracle_calls",
            "candidate_search_calls",
            "candidate_verifier_calls",
        ):
            setattr(
                resource_totals,
                field,
                getattr(resource_totals, field) + int(rollout.resources[field]),
            )
        compatible = CandidateRollout(
            halted=rollout.halted,
            invalid=rollout.invalid,
            overlong=rollout.overlong,
            actions=rollout.actions,
            output_rows=rollout.output_rows,
            resources=ResourceCounts(
                schema="posthoc_adapter",
                successor_evaluations=0,
                successor_matrix_cells=0,
                model_forward_calls=0,
                action_candidates_scored=0,
                planner_iterations=0,
                recurrent_action_updates=0,
                oracle_calls=0,
                search_calls=0,
                verifier_calls=0,
            ),
        )
        assessment = assess_rollout_posthoc(matrix, compatible)
        certified += int(assessment.strict_canonical_certified)
        invalid += int(assessment.invalid)
        overlong += int(assessment.overlong)
        halted += int(rollout.halted)
        posthoc += assessment.posthoc_verifier_calls
    candidate_resources = resource_totals.freeze()
    if any(
        int(candidate_resources[name])
        for name in (
            "candidate_oracle_calls",
            "candidate_search_calls",
            "candidate_verifier_calls",
        )
    ):
        raise ProofCarryingError("candidate crossed a forbidden boundary")
    return EvaluationResult(
        strict_certified=certified,
        total=len(matrices),
        invalid=invalid,
        overlong=overlong,
        halted=halted,
        posthoc_verifier_calls=posthoc,
        candidate_resources=candidate_resources,
        wall_seconds=time.perf_counter() - started,
    )


def _model_state_sha256(model: ProofCarryingController) -> str:
    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _save_model(
    model: ProofCarryingController,
    *,
    path: Path,
    arm: str,
    seed: int,
) -> str:
    if path.exists():
        raise ProofCarryingError(f"model output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": MODEL_SCHEMA,
            "arm": arm,
            "seed": seed,
            "config": asdict(model.config),
            "state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            },
        },
        path,
    )
    return sha256(path.read_bytes()).hexdigest()


def run_experiment(
    *,
    preparation_path: Path,
    output: Path,
    model_dir: Path,
    seed: int,
    optimizer_updates: int,
    batch_size: int,
    learning_rate: float,
    maximum_rollout_steps: int,
    device_name: str,
    amp_bfloat16: bool,
    controller_config: ControllerConfig,
) -> Mapping[str, object]:
    """Fit and evaluate every arm from one sealed preparation artifact."""

    if output.exists() or model_dir.exists():
        raise ProofCarryingError("isolated output already exists")
    preparation_bytes = preparation_path.read_bytes()
    preparation_sha = sha256(preparation_bytes).hexdigest()
    preparation = _load_preparation(preparation_path)
    import inspect
    import ssqac_soft_value_iteration_controller as oracle_module
    import ssqac_successor_value_controller as successor_module

    live_hashes = {
        "controller": _source_sha256(Path(__file__)),
        "oracle": _source_sha256(
            Path(inspect.getsourcefile(oracle_module) or "")
        ),
        "successor": _source_sha256(
            Path(inspect.getsourcefile(successor_module) or "")
        ),
    }
    expected_hashes = {
        "controller": preparation.source_sha256,
        "oracle": preparation.oracle_source_sha256,
        "successor": preparation.successor_source_sha256,
    }
    if live_hashes != expected_hashes:
        raise ProofCarryingError("live sources differ from sealed preparation")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ProofCarryingError("CUDA was requested but is unavailable")
    torch.manual_seed(seed)
    initial = ProofCarryingController(controller_config)
    initial_state = {
        name: tensor.detach().clone() for name, tensor in initial.state_dict().items()
    }
    reports = []
    total_started = time.perf_counter()
    for arm_index, spec in enumerate(ARM_SPECS):
        torch.manual_seed(seed + arm_index + 1)
        model = ProofCarryingController(controller_config)
        model.load_state_dict(initial_state)
        model.to(device)
        training = train_arm(
            model,
            preparation.states,
            spec=spec,
            optimizer_updates=optimizer_updates,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed + 10_000,
            amp_bfloat16=amp_bfloat16,
            binding_seed=seed + 20_000,
        )
        evaluation = evaluate_arm(
            model,
            preparation.evaluation_matrices,
            spec=spec,
            binding_seed=seed + 20_000,
            maximum_steps=maximum_rollout_steps,
        )
        model_path = model_dir / f"{spec.name}_seed{seed}.pt"
        model_file_sha = _save_model(
            model,
            path=model_path,
            arm=spec.name,
            seed=seed,
        )
        reports.append(
            {
                "name": spec.name,
                "train_contract": spec.train_contract,
                "inference_contract": spec.inference_contract,
                "inference_zero_proof": spec.inference_zero_proof,
                "shuffled_binding": spec.shuffled_binding,
                "shuffled_progress_labels": spec.shuffled_progress_labels,
                "random_labels": spec.random_labels,
                "controller_parameters": model.parameter_count,
                "complete_system_parameters": model.complete_system_parameters,
                "parameter_count_breakdown": model.parameter_count_breakdown(),
                "training": asdict(training),
                "evaluation": {
                    **asdict(evaluation),
                    "certification_rate": evaluation.rate,
                },
                "model_state_sha256": _model_state_sha256(model),
                "model_file_sha256": model_file_sha,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    by_name = {report["name"]: report for report in reports}
    treatment_score = int(
        by_name[ARM_TREATMENT]["evaluation"]["strict_certified"]
    )
    classifier_score = int(
        by_name[ARM_CLASSIFIER]["evaluation"]["strict_certified"]
    )
    zero_score = int(
        by_name[ARM_ZERO_PROOF]["evaluation"]["strict_certified"]
    )
    total = len(preparation.evaluation_matrices)
    proof_beats_classifier = treatment_score > classifier_score
    proof_beats_zeroed = treatment_score > zero_score
    counts = {int(report["controller_parameters"]) for report in reports}
    update_counts = {
        int(report["training"]["optimizer_updates"]) for report in reports
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": STATUS,
        "seed": seed,
        "source_sha256": _source_sha256(Path(__file__)),
        "preparation_path_name": preparation_path.name,
        "preparation_sha256": preparation_sha,
        "preparation_source_sha256": preparation.source_sha256,
        "preparation_oracle_source_sha256": (
            preparation.oracle_source_sha256
        ),
        "preparation_successor_source_sha256": (
            preparation.successor_source_sha256
        ),
        "preparation_oracle_calls": preparation.oracle_calls,
        "preparation_wall_seconds": preparation.wall_seconds,
        "train_matrix_manifest_sha256": (
            preparation.train_matrix_manifest_sha256
        ),
        "evaluation_matrix_manifest_sha256": (
            preparation.evaluation_matrix_manifest_sha256
        ),
        "state_manifest_sha256": preparation.state_manifest_sha256,
        "train_matrices": len(preparation.train_matrices),
        "train_states": len(preparation.states),
        "legal_training_triples": preparation.legal_triples,
        "matched_legal_negative_triples": (
            preparation.matched_legal_negative_triples
        ),
        "evaluation_matrices": total,
        "strict_larger_geometry_holdout": (
            preparation.evaluation_minimum_rows
            > preparation.train_maximum_rows
            and preparation.evaluation_minimum_columns
            > preparation.train_maximum_columns
        ),
        "candidate_inputs": (
            "raw_current_field_matrix",
            "raw_deterministic_one_step_successor_matrix",
            "action_identity_and_geometry",
            "model_owned_recurrent_contract_state",
        ),
        "forbidden_candidate_inputs": (
            "rank",
            "frontier",
            "energy",
            "correctness_features",
            "oracle",
            "search",
            "verifier",
        ),
        "candidate_oracle_calls": 0,
        "candidate_search_calls": 0,
        "candidate_verifier_calls": 0,
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "amp_bfloat16": amp_bfloat16,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "allocated_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
        "maximum_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "optimizer_updates_per_arm": optimizer_updates,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "maximum_rollout_steps": maximum_rollout_steps,
        "controls_equal_parameters": len(counts) == 1,
        "controls_equal_optimizer_updates": len(update_counts) == 1,
        "parameter_budget": TOTAL_PARAMETER_BUDGET,
        "parameter_budget_passed": max(
            int(item["complete_system_parameters"]) for item in reports
        )
        < TOTAL_PARAMETER_BUDGET,
        "treatment_strict_certified": treatment_score,
        "classifier_only_strict_certified": classifier_score,
        "proof_zeroed_strict_certified": zero_score,
        "proof_heads_beat_classifier_only": proof_beats_classifier,
        "proof_heads_beat_zeroed_inference": proof_beats_zeroed,
        "proof_heads_causal_gate_passed": (
            proof_beats_classifier and proof_beats_zeroed
        ),
        "treatment_minus_classifier_rate": (
            treatment_score - classifier_score
        )
        / total,
        "treatment_minus_zeroed_rate": (treatment_score - zero_score) / total,
        "total_wall_seconds": time.perf_counter() - total_started,
        "arms": reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_bytes(report) + b"\n")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--seed", type=int, required=True)
    prepare.add_argument("--train-matrices", type=int, default=256)
    prepare.add_argument("--evaluation-matrices", type=int, default=128)
    prepare.add_argument("--train-maximum-rows", type=int, default=3)
    prepare.add_argument("--train-maximum-columns", type=int, default=4)
    prepare.add_argument("--evaluation-minimum-rows", type=int, default=4)
    prepare.add_argument("--evaluation-minimum-columns", type=int, default=5)
    prepare.add_argument("--evaluation-maximum-rows", type=int, default=4)
    prepare.add_argument("--evaluation-maximum-columns", type=int, default=6)
    prepare.add_argument("--maximum-preparation-steps", type=int, default=96)
    prepare.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--preparation", type=Path, required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--optimizer-updates", type=int, default=1_500)
    run.add_argument("--batch-size", type=int, default=4)
    run.add_argument("--learning-rate", type=float, default=6e-4)
    run.add_argument("--maximum-rollout-steps", type=int, default=192)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument("--no-amp-bfloat16", action="store_true")
    run.add_argument("--field-width", type=int, default=64)
    run.add_argument("--width", type=int, default=384)
    run.add_argument("--cell-hidden", type=int, default=512)
    run.add_argument("--matrix-layers", type=int, default=4)
    run.add_argument("--contract-hidden", type=int, default=512)
    run.add_argument("--coordinate-harmonics", type=int, default=4)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--model-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "prepare":
        if args.output.exists():
            raise ProofCarryingError(
                f"preparation output already exists: {args.output}"
            )
        artifact = prepare_artifact(
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
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(artifact.canonical_bytes())
        print(
            _canonical_bytes(
                {
                    "output": str(args.output),
                    "sha256": sha256(args.output.read_bytes()).hexdigest(),
                    "states": len(artifact.states),
                    "legal_triples": artifact.legal_triples,
                    "oracle_calls": artifact.oracle_calls,
                    "wall_seconds": artifact.wall_seconds,
                }
            ).decode("ascii")
        )
        return
    report = run_experiment(
        preparation_path=args.preparation,
        output=args.output,
        model_dir=args.model_dir,
        seed=args.seed,
        optimizer_updates=args.optimizer_updates,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        maximum_rollout_steps=args.maximum_rollout_steps,
        device_name=args.device,
        amp_bfloat16=not args.no_amp_bfloat16,
        controller_config=ControllerConfig(
            field_width=args.field_width,
            width=args.width,
            cell_hidden=args.cell_hidden,
            matrix_layers=args.matrix_layers,
            contract_hidden=args.contract_hidden,
            coordinate_harmonics=args.coordinate_harmonics,
        ),
    )
    print(_canonical_bytes(report).decode("ascii"))


if __name__ == "__main__":
    main()
