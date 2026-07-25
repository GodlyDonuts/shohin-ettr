#!/usr/bin/env python3
"""Learned Lyapunov/Bellman potential falsifier for SSQAC.

This lane is deliberately disjoint from action imitation.  Preparation may
apply the deterministic canonical scheduler to a raw matrix in order to label
the source-independent remaining distance of that matrix.  The treatment is
never trained against an expert action.  It learns a scalar state potential
with:

* remaining-distance regression;
* pairwise monotonic ordering across raw one-step transitions;
* Bellman consistency against the minimum learned successor potential; and
* a separate learned terminal/HALT head.

At candidate inference the treatment receives only the raw current matrix,
the descriptors of all legal local actions, and the raw matrices produced by
those actions.  A shared geometry-general encoder evaluates every successor.
The candidate emits HALT when its terminal head fires; otherwise it emits the
hard action whose successor has minimum learned potential.  The strict
canonical verifier runs only in a posthoc assessor.

Matched controls are:

* equal-data/update direct action classification with the same model;
* deterministically shuffled nonterminal distance labels;
* distance regression with monotonic and Bellman losses disabled;
* deterministic action-to-successor binding shuffling at candidate time; and
* fully random potential and terminal labels.

Preparation transcripts are serialized and hashed, distilled into training
objects, and deleted.  Training labels and tensors are destroyed before an
evaluation token can be minted.  Candidate resource receipts separately
assert zero oracle, search, or verifier calls.

This is an isolated falsifier.  It does not load or modify Shohin, any
checkpoint, pretraining, EFC, SVI, successor-planner, canonical-energy, or
on-policy artifact.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import gc
from hashlib import sha256
import inspect
import json
import math
from pathlib import Path
import random
import shutil
import tempfile
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from episode_functor_algebra_machine import (
    execute_program,
    verify_reduction_program,
)
from ssqac_successor_value_controller import (
    ACTION_ELIMINATE,
    ACTION_HALT,
    ACTION_NORMALIZE,
    ACTION_SWAP,
    ACTION_TO_INDEX,
    ACTION_TYPES,
    DEFAULT_REGISTER_COUNT,
    FIELD_MODULUS,
    PROTECTED_FLAGSHIP_PARAMETERS,
    TOTAL_PARAMETER_BUDGET,
    SuccessorAction,
    apply_action,
    canonical_matrix,
    compile_trace_to_primitives,
    enumerate_legal_actions,
    generate_matrices,
    matrix_manifest,
    matrix_sha256,
)


ARCHITECTURE_SCHEMA = "ssqac_lyapunov_value_controller_v1"
EXPERIMENT_SCHEMA = "ssqac_lyapunov_value_experiment_v1"
RESOURCE_SCHEMA = "ssqac_lyapunov_value_resources_v1"
PREPARATION_SCHEMA = "ssqac_lyapunov_preparation_source_v1"
BOUNDARY_SCHEMA = "ssqac_lyapunov_evaluation_boundary_v1"
PACKET_SCHEMA = "ssqac_lyapunov_prepared_packet_v1"

STATUS = "isolated_lyapunov_bellman_falsifier_not_reasoning"
CLAIM_NO_GO = "lyapunov_bellman_mechanism_falsified_or_below_gate"
CLAIM_SUGGESTIVE = "replicated_lyapunov_signal_below_native_reasoning_gate"
CLAIM_MATERIAL = (
    "material_lyapunov_mechanics_pass_replication_required_not_reasoning"
)

ARM_TREATMENT = "lyapunov_bellman"
ARM_CLASSIFICATION = "action_classification_only"
ARM_SHUFFLED_DISTANCE = "shuffled_potential_distance"
ARM_ZERO_STRUCTURE = "zero_bellman_monotonic"
ARM_RANDOM_LABELS = "random_labels"
ARM_SHUFFLED_BINDINGS = "shuffled_successor_bindings"
TRAINED_ARMS = (
    ARM_TREATMENT,
    ARM_CLASSIFICATION,
    ARM_SHUFFLED_DISTANCE,
    ARM_ZERO_STRUCTURE,
    ARM_RANDOM_LABELS,
)
REPORTED_ARMS = TRAINED_ARMS + (ARM_SHUFFLED_BINDINGS,)

INFERENCE_POTENTIAL = "hard_minimum_successor_potential"
INFERENCE_CLASSIFICATION = "direct_action_classification"
INFERENCE_MODES = (INFERENCE_POTENTIAL, INFERENCE_CLASSIFICATION)

LABEL_TRUE = "true_source_independent_distance"
LABEL_SHUFFLED = "shuffled_nonterminal_distance"
LABEL_RANDOM = "random_potential_and_terminal"
LABEL_MODES = (LABEL_TRUE, LABEL_SHUFFLED, LABEL_RANDOM)

BINDING_RAW = "raw_action_successor_binding"
BINDING_SHUFFLED = "deterministically_shuffled_successor_binding"
BINDING_MODES = (BINDING_RAW, BINDING_SHUFFLED)

ACTION_FEATURE_WIDTH = len(ACTION_TYPES) + 9
MAX_MECHANICS_ROWS = 32
MAX_MECHANICS_COLUMNS = 32


class LyapunovValueError(ValueError):
    """A falsifier contract was violated."""


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise LyapunovValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LyapunovValueError(f"{label} must be a nonnegative integer")
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
        raise LyapunovValueError("value is not canonical ASCII JSON") from error


def _digest(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DistanceLabel:
    """A state-only preparation label."""

    rows: tuple[tuple[int, ...], ...]
    remaining_distance: int
    terminal: bool
    canonical_next_action: SuccessorAction

    def __post_init__(self) -> None:
        canonical = canonical_matrix(self.rows)
        if canonical != self.rows:
            raise LyapunovValueError("distance label matrix is not canonical")
        _nonnegative_int(self.remaining_distance, label="remaining_distance")
        if self.terminal != (self.remaining_distance == 0):
            raise LyapunovValueError("terminal and distance labels disagree")
        if self.terminal != (self.canonical_next_action.kind == ACTION_HALT):
            raise LyapunovValueError("terminal and canonical action disagree")

    @property
    def sha256(self) -> str:
        return _digest(
            {
                "rows": self.rows,
                "remaining_distance": self.remaining_distance,
                "terminal": self.terminal,
                "canonical_next_action": self.canonical_next_action.canonical_data(),
            }
        )


@dataclass(frozen=True, slots=True)
class DecisionTransition:
    action: SuccessorAction
    successor_rows: tuple[tuple[int, ...], ...]

    @property
    def sha256(self) -> str:
        return _digest(
            {
                "action": self.action.canonical_data(),
                "successor_rows": self.successor_rows,
            }
        )


@dataclass(frozen=True, slots=True)
class DecisionState:
    rows: tuple[tuple[int, ...], ...]
    canonical_next_action: SuccessorAction
    transitions: tuple[DecisionTransition, ...]

    def __post_init__(self) -> None:
        legal = enumerate_legal_actions(self.rows)
        if tuple(item.action for item in self.transitions) != legal:
            raise LyapunovValueError("decision transitions do not match legal actions")
        if self.canonical_next_action not in legal:
            raise LyapunovValueError("canonical action is not legal")
        for transition in self.transitions:
            if apply_action(self.rows, transition.action) != transition.successor_rows:
                raise LyapunovValueError("transition successor is not raw one-step output")

    @property
    def sha256(self) -> str:
        return _digest(
            {
                "rows": self.rows,
                "canonical_next_action": self.canonical_next_action.canonical_data(),
                "transitions": [item.sha256 for item in self.transitions],
            }
        )


@dataclass(frozen=True, slots=True)
class PotentialDataset:
    labels: tuple[DistanceLabel, ...]
    decisions: tuple[DecisionState, ...]
    label_index_by_matrix_sha256: Mapping[str, int]
    maximum_distance: int
    manifest_sha256: str

    def __post_init__(self) -> None:
        if not self.labels or not self.decisions:
            raise LyapunovValueError("potential dataset must be nonempty")
        _positive_int(self.maximum_distance, label="maximum_distance")
        if len(self.label_index_by_matrix_sha256) != len(self.labels):
            raise LyapunovValueError("label index cardinality mismatch")


@dataclass(frozen=True, slots=True)
class PreparationReceipt:
    schema: str
    oracle_calls: int
    canonical_trace_states: int
    unique_labeled_states: int
    decision_states: int
    one_step_successors_labeled: int
    source_module_path: str
    source_module_sha256: str
    serialized_source_sha256: str
    serialized_source_bytes: int
    serialized_source_destroyed: bool
    serialized_source_exists_after_destroy: bool
    source_independence_conflicts: int
    strict_preparation_verifier_calls: int
    dataset_manifest_sha256: str


class CanonicalDistanceOracle:
    """Preparation-only state-distance oracle with exact call accounting."""

    def __init__(self, maximum_steps: int) -> None:
        self.maximum_steps = _positive_int(
            maximum_steps,
            label="maximum_steps",
        )
        self.calls = 0
        self.strict_verifier_calls = 0
        self.source_independence_conflicts = 0
        self._cache: dict[str, DistanceLabel] = {}
        self._locked = False

        import ssqac_soft_value_iteration_controller as preparation_module

        self._module = preparation_module
        source = Path(inspect.getsourcefile(preparation_module) or "")
        if not source.is_file():
            raise LyapunovValueError("canonical preparation source is unavailable")
        self.source_path = source
        self.source_sha256 = _file_sha256(source)

    def lock(self) -> None:
        self._locked = True
        self._module = None

    def _convert_action(self, action: object) -> SuccessorAction:
        return SuccessorAction(
            str(getattr(action, "kind")),
            row_a=int(getattr(action, "row_a")),
            row_b=int(getattr(action, "row_b")),
            column=int(getattr(action, "column")),
        )

    def label(self, rows: Iterable[Iterable[int]]) -> DistanceLabel:
        if self._locked:
            raise LyapunovValueError("preparation oracle is locked")
        matrix = canonical_matrix(rows)
        key = matrix_sha256(matrix)
        cached = self._cache.get(key)
        if cached is not None:
            if cached.rows != matrix:
                raise LyapunovValueError("matrix hash collision")
            return cached

        module = self._module
        if module is None:
            raise LyapunovValueError("preparation module was destroyed")
        counter = module.PreparationOracleCounter()
        path: list[
            tuple[tuple[tuple[int, ...], ...], SuccessorAction]
        ] = []
        seen: set[str] = set()
        current = matrix
        base_distance: int | None = None
        for _ in range(self.maximum_steps):
            current_key = matrix_sha256(current)
            prior = self._cache.get(current_key)
            if prior is not None:
                base_distance = prior.remaining_distance
                break
            if current_key in seen:
                raise LyapunovValueError("canonical distance oracle entered a cycle")
            seen.add(current_key)
            raw_action = module.next_preparation_macro(
                current,
                counter=counter,
            )
            action = self._convert_action(raw_action)
            if action not in enumerate_legal_actions(current):
                raise LyapunovValueError("canonical oracle emitted an illegal action")
            path.append((current, action))
            if action.kind == ACTION_HALT:
                base_distance = -1
                break
            current = apply_action(current, action)
        else:
            raise LyapunovValueError("canonical distance exceeded preparation bound")
        self.calls += counter.calls
        if base_distance is None:
            raise LyapunovValueError("distance oracle did not establish a base")

        distance = base_distance
        for state, action in reversed(path):
            distance += 1
            label = DistanceLabel(
                rows=state,
                remaining_distance=distance,
                terminal=action.kind == ACTION_HALT,
                canonical_next_action=action,
            )
            state_key = matrix_sha256(state)
            prior = self._cache.get(state_key)
            if prior is not None and prior != label:
                self.source_independence_conflicts += 1
                raise LyapunovValueError(
                    "one raw state received source-dependent distance labels"
                )
            self._cache[state_key] = label
        return self._cache[key]

    def verify_trace(self, source: Iterable[Iterable[int]]) -> int:
        """Independently certify one complete preparation trace."""

        matrix = canonical_matrix(source)
        actions: list[SuccessorAction] = []
        for _ in range(self.maximum_steps):
            label = self.label(matrix)
            actions.append(label.canonical_next_action)
            if label.terminal:
                break
            matrix = apply_action(matrix, label.canonical_next_action)
        else:
            raise LyapunovValueError("preparation trace exceeded verifier bound")
        program = compile_trace_to_primitives(canonical_matrix(source), actions)
        state = execute_program(
            canonical_matrix(source),
            program,
            register_count=DEFAULT_REGISTER_COUNT,
        )
        receipt = verify_reduction_program(canonical_matrix(source), state)
        self.strict_verifier_calls += 1
        if not receipt.passed:
            raise LyapunovValueError("preparation trace failed strict verification")
        return len(actions)


def _dataset_manifest(
    labels: Sequence[DistanceLabel],
    decisions: Sequence[DecisionState],
) -> str:
    return _digest(
        {
            "labels": [item.sha256 for item in labels],
            "decisions": [item.sha256 for item in decisions],
        }
    )


def build_potential_dataset(
    matrices: Sequence[Iterable[Iterable[int]]],
    *,
    maximum_steps: int,
    preparation_root: Path,
) -> tuple[PotentialDataset, PreparationReceipt]:
    """Build and seal state-only distance labels, then destroy transcripts."""

    if preparation_root.exists():
        raise LyapunovValueError("preparation root already exists")
    preparation_root.mkdir(parents=True)
    transcript_path = preparation_root / "canonical_distance_source.json"
    oracle = CanonicalDistanceOracle(maximum_steps)
    decisions_by_sha: dict[str, DecisionState] = {}
    successor_count = 0
    trace_state_count = 0
    try:
        for raw in matrices:
            source = canonical_matrix(raw)
            oracle.verify_trace(source)
            current = source
            for _ in range(maximum_steps):
                label = oracle.label(current)
                transitions = tuple(
                    DecisionTransition(
                        action=action,
                        successor_rows=apply_action(current, action),
                    )
                    for action in enumerate_legal_actions(current)
                )
                successor_count += len(transitions)
                for transition in transitions:
                    oracle.label(transition.successor_rows)
                decision = DecisionState(
                    rows=current,
                    canonical_next_action=label.canonical_next_action,
                    transitions=transitions,
                )
                key = matrix_sha256(current)
                prior = decisions_by_sha.get(key)
                if prior is not None and prior != decision:
                    raise LyapunovValueError("decision labels are source-dependent")
                decisions_by_sha[key] = decision
                trace_state_count += 1
                if label.terminal:
                    break
                current = apply_action(current, label.canonical_next_action)
            else:
                raise LyapunovValueError("preparation trajectory exceeded bound")

        labels = tuple(
            oracle._cache[key] for key in sorted(oracle._cache)  # noqa: SLF001
        )
        decisions = tuple(
            decisions_by_sha[key] for key in sorted(decisions_by_sha)
        )
        index = {
            matrix_sha256(label.rows): position
            for position, label in enumerate(labels)
        }
        maximum_distance = max(label.remaining_distance for label in labels)
        if maximum_distance < 1:
            raise LyapunovValueError("preparation contains no nonterminal state")
        manifest = _dataset_manifest(labels, decisions)
        source_payload = {
            "schema": PREPARATION_SCHEMA,
            "source_module_sha256": oracle.source_sha256,
            "labels": [
                {
                    "rows": label.rows,
                    "remaining_distance": label.remaining_distance,
                    "terminal": label.terminal,
                    "canonical_next_action": (
                        label.canonical_next_action.canonical_data()
                    ),
                }
                for label in labels
            ],
            "decisions": [
                {
                    "rows": decision.rows,
                    "canonical_next_action": (
                        decision.canonical_next_action.canonical_data()
                    ),
                    "transitions": [
                        {
                            "action": transition.action.canonical_data(),
                            "successor_rows": transition.successor_rows,
                        }
                        for transition in decision.transitions
                    ],
                }
                for decision in decisions
            ],
            "dataset_manifest_sha256": manifest,
        }
        transcript_path.write_bytes(_canonical_bytes(source_payload) + b"\n")
        transcript_sha = _file_sha256(transcript_path)
        transcript_bytes = transcript_path.stat().st_size
        dataset = PotentialDataset(
            labels=labels,
            decisions=decisions,
            label_index_by_matrix_sha256=index,
            maximum_distance=maximum_distance,
            manifest_sha256=manifest,
        )
        receipt = PreparationReceipt(
            schema=PREPARATION_SCHEMA,
            oracle_calls=oracle.calls,
            canonical_trace_states=trace_state_count,
            unique_labeled_states=len(labels),
            decision_states=len(decisions),
            one_step_successors_labeled=successor_count,
            source_module_path=str(oracle.source_path),
            source_module_sha256=oracle.source_sha256,
            serialized_source_sha256=transcript_sha,
            serialized_source_bytes=transcript_bytes,
            serialized_source_destroyed=True,
            serialized_source_exists_after_destroy=False,
            source_independence_conflicts=oracle.source_independence_conflicts,
            strict_preparation_verifier_calls=oracle.strict_verifier_calls,
            dataset_manifest_sha256=manifest,
        )
    finally:
        oracle.lock()
        if transcript_path.exists():
            transcript_path.unlink()
        if preparation_root.exists():
            shutil.rmtree(preparation_root)
    if transcript_path.exists() or preparation_root.exists():
        raise LyapunovValueError("preparation source destruction failed")
    return dataset, receipt


@dataclass(frozen=True, slots=True)
class LyapunovConfig:
    field_width: int = 64
    width: int = 512
    cell_hidden: int = 1024
    matrix_layers: int = 6
    state_hidden: int = 768
    coordinate_harmonics: int = 6
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for label, value in (
            ("field_width", self.field_width),
            ("width", self.width),
            ("cell_hidden", self.cell_hidden),
            ("matrix_layers", self.matrix_layers),
            ("state_hidden", self.state_hidden),
            ("coordinate_harmonics", self.coordinate_harmonics),
        ):
            _positive_int(value, label=label)
        if not isinstance(self.dropout, float) or not 0.0 <= self.dropout < 1.0:
            raise LyapunovValueError("dropout must be a float in [0, 1)")


class _MaskedGridLayer(nn.Module):
    def __init__(self, width: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(width * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, cells: Tensor, mask: Tensor) -> Tensor:
        if cells.ndim != 4 or mask.shape != cells.shape[:3]:
            raise LyapunovValueError("masked grid tensors have incompatible shapes")
        weights = mask[..., None].to(cells.dtype)
        row_denominator = weights.sum(dim=2, keepdim=True).clamp_min(1.0)
        column_denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        global_denominator = weights.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        row_mean = (cells * weights).sum(dim=2, keepdim=True) / row_denominator
        column_mean = (
            (cells * weights).sum(dim=1, keepdim=True) / column_denominator
        )
        global_mean = (
            (cells * weights).sum(dim=(1, 2), keepdim=True)
            / global_denominator
        )
        update = self.update(
            torch.cat(
                (
                    cells,
                    row_mean.expand_as(cells),
                    column_mean.expand_as(cells),
                    global_mean.expand_as(cells),
                ),
                dim=-1,
            )
        )
        return self.norm(cells + update) * weights


@dataclass(frozen=True, slots=True)
class StateScores:
    embedding: Tensor
    potential: Tensor
    terminal_logit: Tensor


class LyapunovValueController(nn.Module):
    """Shared geometry-general state value encoder and matched control head."""

    def __init__(self, config: LyapunovConfig = LyapunovConfig()) -> None:
        super().__init__()
        self.config = config
        coordinate_width = 4 * config.coordinate_harmonics + 6
        self.field_embedding = nn.Embedding(FIELD_MODULUS, config.field_width)
        self.cell_projection = nn.Sequential(
            nn.Linear(config.field_width + coordinate_width, config.cell_hidden),
            nn.GELU(),
            nn.Linear(config.cell_hidden, config.width),
            nn.LayerNorm(config.width),
        )
        self.grid_layers = nn.ModuleList(
            _MaskedGridLayer(
                config.width,
                config.cell_hidden,
                config.dropout,
            )
            for _ in range(config.matrix_layers)
        )
        self.state_projection = nn.Sequential(
            nn.Linear(config.width * 2 + 2, config.state_hidden),
            nn.GELU(),
            nn.LayerNorm(config.state_hidden),
            nn.Linear(config.state_hidden, config.state_hidden),
            nn.GELU(),
            nn.LayerNorm(config.state_hidden),
        )
        self.potential_head = nn.Sequential(
            nn.Linear(config.state_hidden, config.state_hidden),
            nn.GELU(),
            nn.Linear(config.state_hidden, 1),
        )
        self.terminal_head = nn.Sequential(
            nn.Linear(config.state_hidden, config.state_hidden // 2),
            nn.GELU(),
            nn.Linear(config.state_hidden // 2, 1),
        )
        classifier_input = config.state_hidden * 2 + ACTION_FEATURE_WIDTH
        self.action_classifier = nn.Sequential(
            nn.Linear(classifier_input, config.state_hidden),
            nn.GELU(),
            nn.LayerNorm(config.state_hidden),
            nn.Linear(config.state_hidden, 1),
        )
        if self.complete_system_parameter_count >= TOTAL_PARAMETER_BUDGET:
            raise LyapunovValueError("complete system exceeds the 200M cap")

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
            "state_projection": sum(
                parameter.numel() for parameter in self.state_projection.parameters()
            ),
            "potential_head": sum(
                parameter.numel() for parameter in self.potential_head.parameters()
            ),
            "terminal_head": sum(
                parameter.numel() for parameter in self.terminal_head.parameters()
            ),
            "action_classifier": sum(
                parameter.numel() for parameter in self.action_classifier.parameters()
            ),
        }
        groups["total"] = sum(groups.values())
        return groups

    def _coordinate_features(
        self,
        mask: Tensor,
        row_counts: Tensor,
        column_counts: Tensor,
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        batch, rows, columns = mask.shape
        device = mask.device
        row_index = torch.arange(rows, device=device, dtype=torch.float32)
        column_index = torch.arange(columns, device=device, dtype=torch.float32)
        row = (row_index[None, :, None] + 0.5) / row_counts[:, None, None]
        column = (
            (column_index[None, None, :] + 0.5)
            / column_counts[:, None, None]
        )
        row = row.expand(batch, rows, columns)
        column = column.expand(batch, rows, columns)
        row_count_feature = (
            row_counts / float(MAX_MECHANICS_ROWS)
        )[:, None, None].expand_as(row)
        column_count_feature = (
            column_counts / float(MAX_MECHANICS_COLUMNS)
        )[:, None, None].expand_as(column)
        features = [
            row,
            column,
            row_count_feature,
            column_count_feature,
            mask.to(torch.float32),
            row_count_feature * column_count_feature,
        ]
        for harmonic in range(1, self.config.coordinate_harmonics + 1):
            angle = math.pi * harmonic
            features.extend(
                (
                    torch.sin(angle * row),
                    torch.cos(angle * row),
                    torch.sin(angle * column),
                    torch.cos(angle * column),
                )
            )
        return torch.stack(features, dim=-1).to(dtype=dtype)

    def encode_states(
        self,
        values: Tensor,
        mask: Tensor,
        row_counts: Tensor,
        column_counts: Tensor,
    ) -> StateScores:
        if values.ndim != 3 or mask.shape != values.shape:
            raise LyapunovValueError("state tensors have incompatible shapes")
        batch = values.shape[0]
        if row_counts.shape != (batch,) or column_counts.shape != (batch,):
            raise LyapunovValueError("geometry tensors have incompatible shapes")
        embedded = self.field_embedding(values)
        coordinates = self._coordinate_features(
            mask,
            row_counts,
            column_counts,
            dtype=embedded.dtype,
        )
        weights = mask[..., None].to(embedded.dtype)
        cells = self.cell_projection(torch.cat((embedded, coordinates), dim=-1))
        cells = cells * weights
        for layer in self.grid_layers:
            cells = layer(cells, mask)
        denominator = weights.sum(dim=(1, 2)).clamp_min(1.0)
        mean = (cells * weights).sum(dim=(1, 2)) / denominator
        minimum = torch.finfo(cells.dtype).min
        maximum = cells.masked_fill(~mask[..., None], minimum).amax(dim=(1, 2))
        geometry = torch.stack(
            (
                row_counts / float(MAX_MECHANICS_ROWS),
                column_counts / float(MAX_MECHANICS_COLUMNS),
            ),
            dim=-1,
        ).to(cells.dtype)
        state = self.state_projection(torch.cat((mean, maximum, geometry), dim=-1))
        return StateScores(
            embedding=state,
            potential=self.potential_head(state).squeeze(-1),
            terminal_logit=self.terminal_head(state).squeeze(-1),
        )

    def classify_actions(
        self,
        current_embedding: Tensor,
        successor_embedding: Tensor,
        action_features: Tensor,
    ) -> Tensor:
        if current_embedding.shape != successor_embedding.shape:
            raise LyapunovValueError("classifier state embeddings disagree")
        if action_features.shape != (
            current_embedding.shape[0],
            ACTION_FEATURE_WIDTH,
        ):
            raise LyapunovValueError("action feature tensor has the wrong shape")
        return self.action_classifier(
            torch.cat(
                (current_embedding, successor_embedding, action_features),
                dim=-1,
            )
        ).squeeze(-1)


def tensorize_matrices(
    matrices: Sequence[Iterable[Iterable[int]]],
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    frozen = tuple(canonical_matrix(matrix) for matrix in matrices)
    if not frozen:
        raise LyapunovValueError("cannot tensorize an empty matrix collection")
    maximum_rows = max(len(matrix) for matrix in frozen)
    maximum_columns = max(len(matrix[0]) for matrix in frozen)
    values = torch.zeros(
        (len(frozen), maximum_rows, maximum_columns),
        dtype=torch.long,
        device=device,
    )
    mask = torch.zeros_like(values, dtype=torch.bool)
    row_counts = torch.empty(len(frozen), dtype=torch.float32, device=device)
    column_counts = torch.empty(len(frozen), dtype=torch.float32, device=device)
    for index, matrix in enumerate(frozen):
        rows = len(matrix)
        columns = len(matrix[0])
        values[index, :rows, :columns] = torch.tensor(
            matrix,
            dtype=torch.long,
            device=device,
        )
        mask[index, :rows, :columns] = True
        row_counts[index] = rows
        column_counts[index] = columns
    return values, mask, row_counts, column_counts


def action_features(
    action: SuccessorAction,
    *,
    row_count: int,
    column_count: int,
) -> tuple[float, ...]:
    row_denominator = float(max(row_count - 1, 1))
    column_denominator = float(max(column_count - 1, 1))
    one_hot = [0.0] * len(ACTION_TYPES)
    one_hot[ACTION_TO_INDEX[action.kind]] = 1.0
    uses_row_a = action.kind != ACTION_HALT
    uses_row_b = action.kind in (ACTION_ELIMINATE, ACTION_SWAP)
    uses_column = action.kind in (ACTION_NORMALIZE, ACTION_ELIMINATE)
    return tuple(
        one_hot
        + [
            action.row_a / row_denominator if uses_row_a else 0.0,
            action.row_b / row_denominator if uses_row_b else 0.0,
            action.column / column_denominator if uses_column else 0.0,
            float(uses_row_a),
            float(uses_row_b),
            float(uses_column),
            row_count / float(MAX_MECHANICS_ROWS),
            column_count / float(MAX_MECHANICS_COLUMNS),
            (row_count * column_count)
            / float(MAX_MECHANICS_ROWS * MAX_MECHANICS_COLUMNS),
        ]
    )


@dataclass(frozen=True, slots=True)
class ArmSpec:
    name: str
    inference_mode: str
    label_mode: str
    distance_weight: float
    terminal_weight: float
    monotonic_weight: float
    bellman_weight: float
    classification_weight: float

    def __post_init__(self) -> None:
        if self.name not in TRAINED_ARMS:
            raise LyapunovValueError("unknown trained arm")
        if self.inference_mode not in INFERENCE_MODES:
            raise LyapunovValueError("unknown inference mode")
        if self.label_mode not in LABEL_MODES:
            raise LyapunovValueError("unknown label mode")
        for value in (
            self.distance_weight,
            self.terminal_weight,
            self.monotonic_weight,
            self.bellman_weight,
            self.classification_weight,
        ):
            if not isinstance(value, float) or value < 0.0:
                raise LyapunovValueError("loss weights must be nonnegative floats")


ARM_SPECS = (
    ArmSpec(
        ARM_TREATMENT,
        INFERENCE_POTENTIAL,
        LABEL_TRUE,
        1.0,
        1.0,
        0.5,
        0.5,
        0.0,
    ),
    ArmSpec(
        ARM_CLASSIFICATION,
        INFERENCE_CLASSIFICATION,
        LABEL_TRUE,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ),
    ArmSpec(
        ARM_SHUFFLED_DISTANCE,
        INFERENCE_POTENTIAL,
        LABEL_SHUFFLED,
        1.0,
        1.0,
        0.5,
        0.5,
        0.0,
    ),
    ArmSpec(
        ARM_ZERO_STRUCTURE,
        INFERENCE_POTENTIAL,
        LABEL_TRUE,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
    ),
    ArmSpec(
        ARM_RANDOM_LABELS,
        INFERENCE_POTENTIAL,
        LABEL_RANDOM,
        1.0,
        1.0,
        0.5,
        0.5,
        0.0,
    ),
)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    optimizer_updates: int = 1600
    node_batch_size: int = 256
    decision_batch_size: int = 24
    learning_rate: float = 4e-4
    weight_decay: float = 0.01
    monotonic_margin: float = 0.5
    gradient_clip: float = 1.0
    bf16: bool = True

    def __post_init__(self) -> None:
        _positive_int(self.optimizer_updates, label="optimizer_updates")
        _positive_int(self.node_batch_size, label="node_batch_size")
        _positive_int(self.decision_batch_size, label="decision_batch_size")
        for label, value in (
            ("learning_rate", self.learning_rate),
            ("monotonic_margin", self.monotonic_margin),
            ("gradient_clip", self.gradient_clip),
        ):
            if not isinstance(value, float) or value <= 0.0:
                raise LyapunovValueError(f"{label} must be positive")
        if not isinstance(self.weight_decay, float) or self.weight_decay < 0.0:
            raise LyapunovValueError("weight_decay must be nonnegative")


@dataclass(frozen=True, slots=True)
class TrainingResources:
    optimizer_updates: int
    node_examples: int
    decision_examples: int
    transition_examples: int
    state_encoder_examples: int
    action_classifier_examples: int
    oracle_calls: int
    search_calls: int
    verifier_calls: int


@dataclass(frozen=True, slots=True)
class TrainingResult:
    final_loss: float
    mean_loss: float
    final_distance_loss: float
    final_terminal_loss: float
    final_monotonic_loss: float
    final_bellman_loss: float
    final_classification_loss: float
    resources: TrainingResources


@dataclass(slots=True)
class _TrainingTensors:
    values: Tensor
    mask: Tensor
    row_counts: Tensor
    column_counts: Tensor
    true_distances: Tensor
    true_terminal: Tensor
    decision_source_indices: tuple[int, ...]
    decision_transition_indices: tuple[tuple[int, ...], ...]
    transition_source_indices: Tensor
    transition_successor_indices: Tensor
    transition_action_features: Tensor
    transition_target: Tensor
    transition_is_halt: Tensor


def _build_training_tensors(
    dataset: PotentialDataset,
    *,
    device: torch.device,
) -> _TrainingTensors:
    matrices = tuple(label.rows for label in dataset.labels)
    values, mask, row_counts, column_counts = tensorize_matrices(
        matrices,
        device=device,
    )
    distance = torch.tensor(
        [label.remaining_distance for label in dataset.labels],
        dtype=torch.float32,
        device=device,
    )
    terminal = torch.tensor(
        [float(label.terminal) for label in dataset.labels],
        dtype=torch.float32,
        device=device,
    )
    transition_source: list[int] = []
    transition_successor: list[int] = []
    transition_features: list[tuple[float, ...]] = []
    transition_target: list[bool] = []
    transition_halt: list[bool] = []
    decision_sources: list[int] = []
    decision_transitions: list[tuple[int, ...]] = []
    for decision in dataset.decisions:
        source_index = dataset.label_index_by_matrix_sha256[
            matrix_sha256(decision.rows)
        ]
        decision_sources.append(source_index)
        current_transition_indices = []
        for transition in decision.transitions:
            current_transition_indices.append(len(transition_source))
            transition_source.append(source_index)
            transition_successor.append(
                dataset.label_index_by_matrix_sha256[
                    matrix_sha256(transition.successor_rows)
                ]
            )
            transition_features.append(
                action_features(
                    transition.action,
                    row_count=len(decision.rows),
                    column_count=len(decision.rows[0]),
                )
            )
            transition_target.append(
                transition.action == decision.canonical_next_action
            )
            transition_halt.append(transition.action.kind == ACTION_HALT)
        if sum(transition_target[index] for index in current_transition_indices) != 1:
            raise LyapunovValueError("decision does not have one classifier target")
        decision_transitions.append(tuple(current_transition_indices))
    return _TrainingTensors(
        values=values,
        mask=mask,
        row_counts=row_counts,
        column_counts=column_counts,
        true_distances=distance,
        true_terminal=terminal,
        decision_source_indices=tuple(decision_sources),
        decision_transition_indices=tuple(decision_transitions),
        transition_source_indices=torch.tensor(
            transition_source,
            dtype=torch.long,
            device=device,
        ),
        transition_successor_indices=torch.tensor(
            transition_successor,
            dtype=torch.long,
            device=device,
        ),
        transition_action_features=torch.tensor(
            transition_features,
            dtype=torch.float32,
            device=device,
        ),
        transition_target=torch.tensor(
            transition_target,
            dtype=torch.bool,
            device=device,
        ),
        transition_is_halt=torch.tensor(
            transition_halt,
            dtype=torch.bool,
            device=device,
        ),
    )


def _arm_targets(
    tensors: _TrainingTensors,
    *,
    arm: ArmSpec,
    seed: int,
    maximum_distance: int,
) -> tuple[Tensor, Tensor]:
    distances = tensors.true_distances.clone()
    terminal = tensors.true_terminal.clone()
    rng = random.Random(seed)
    if arm.label_mode == LABEL_SHUFFLED:
        indices = [
            index
            for index, value in enumerate(terminal.detach().cpu().tolist())
            if value < 0.5
        ]
        values = [float(distances[index].item()) for index in indices]
        rng.shuffle(values)
        for index, value in zip(indices, values, strict=True):
            distances[index] = value
    elif arm.label_mode == LABEL_RANDOM:
        prevalence = float(terminal.mean().item())
        for index in range(distances.numel()):
            distances[index] = rng.randrange(maximum_distance + 1)
            terminal[index] = float(rng.random() < prevalence)
    return distances / float(maximum_distance), terminal


def _autocast(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def model_state_sha256(model: LyapunovValueController) -> str:
    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _sample_indices(
    rng: random.Random,
    *,
    population: int,
    count: int,
) -> tuple[int, ...]:
    if population < 1:
        raise LyapunovValueError("cannot sample an empty population")
    if count <= population:
        return tuple(rng.sample(range(population), count))
    return tuple(rng.randrange(population) for _ in range(count))


def train_arm(
    model: LyapunovValueController,
    tensors: _TrainingTensors,
    *,
    arm: ArmSpec,
    config: TrainingConfig,
    maximum_distance: int,
    seed: int,
) -> TrainingResult:
    """Fit one matched arm without using verifier/search/oracle calls."""

    device = next(model.parameters()).device
    rng = random.Random(seed)
    target_distance, target_terminal = _arm_targets(
        tensors,
        arm=arm,
        seed=seed ^ 0x4C595041,
        maximum_distance=maximum_distance,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )
    losses: list[float] = []
    final_parts = [0.0] * 5
    node_examples = 0
    decision_examples = 0
    transition_examples = 0
    encoder_examples = 0
    classifier_examples = 0
    model.train()
    for _ in range(config.optimizer_updates):
        node_indices = _sample_indices(
            rng,
            population=tensors.values.shape[0],
            count=config.node_batch_size,
        )
        decision_indices = _sample_indices(
            rng,
            population=len(tensors.decision_source_indices),
            count=config.decision_batch_size,
        )
        transition_indices = tuple(
            transition
            for decision in decision_indices
            for transition in tensors.decision_transition_indices[decision]
        )
        node_index_tensor = torch.tensor(
            node_indices,
            dtype=torch.long,
            device=device,
        )
        transition_index_tensor = torch.tensor(
            transition_indices,
            dtype=torch.long,
            device=device,
        )
        source_indices = tensors.transition_source_indices[transition_index_tensor]
        successor_indices = tensors.transition_successor_indices[
            transition_index_tensor
        ]
        all_indices = torch.cat(
            (node_index_tensor, source_indices, successor_indices)
        )
        unique_indices, inverse = torch.unique(
            all_indices,
            sorted=True,
            return_inverse=True,
        )
        node_count = node_index_tensor.numel()
        transition_count = transition_index_tensor.numel()
        node_inverse = inverse[:node_count]
        source_inverse = inverse[node_count : node_count + transition_count]
        successor_inverse = inverse[node_count + transition_count :]
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, config.bf16):
            scores = model.encode_states(
                tensors.values[unique_indices],
                tensors.mask[unique_indices],
                tensors.row_counts[unique_indices],
                tensors.column_counts[unique_indices],
            )
            node_potential = scores.potential[node_inverse]
            node_terminal = scores.terminal_logit[node_inverse]
            distance_loss = F.mse_loss(
                node_potential.float(),
                target_distance[node_index_tensor],
            )
            terminal_loss = F.binary_cross_entropy_with_logits(
                node_terminal.float(),
                target_terminal[node_index_tensor],
            )

            source_potential = scores.potential[source_inverse].float()
            successor_potential = scores.potential[successor_inverse].float()
            source_target = target_distance[source_indices]
            successor_target = target_distance[successor_indices]
            unequal = source_target != successor_target
            direction = torch.sign(source_target - successor_target)
            ordered_gap = direction * (source_potential - successor_potential)
            if bool(unequal.any()):
                margin = config.monotonic_margin / float(maximum_distance)
                monotonic_loss = F.relu(
                    margin - ordered_gap[unequal]
                ).mean()
            else:
                monotonic_loss = source_potential.sum() * 0.0

            bellman_terms = []
            classification_terms = []
            offset = 0
            action_logits = model.classify_actions(
                scores.embedding[source_inverse],
                scores.embedding[successor_inverse],
                tensors.transition_action_features[transition_index_tensor].to(
                    scores.embedding.dtype
                ),
            ).float()
            for decision in decision_indices:
                width = len(tensors.decision_transition_indices[decision])
                local_slice = slice(offset, offset + width)
                local_halt = tensors.transition_is_halt[
                    transition_index_tensor[local_slice]
                ]
                local_nonhalt = ~local_halt
                source_value = source_potential[offset]
                if bool(local_nonhalt.any()):
                    minimum_successor = successor_potential[
                        local_slice
                    ][local_nonhalt].min()
                    bellman_target = minimum_successor + (
                        1.0 / float(maximum_distance)
                    )
                    bellman_terms.append((source_value - bellman_target).square())
                target_positions = tensors.transition_target[
                    transition_index_tensor[local_slice]
                ]
                target_position = int(
                    torch.nonzero(target_positions, as_tuple=False)[0].item()
                )
                classification_terms.append(
                    F.cross_entropy(
                        action_logits[local_slice][None, :],
                        torch.tensor(
                            [target_position],
                            dtype=torch.long,
                            device=device,
                        ),
                    )
                )
                offset += width
            bellman_loss = (
                torch.stack(bellman_terms).mean()
                if bellman_terms
                else source_potential.sum() * 0.0
            )
            classification_loss = torch.stack(classification_terms).mean()
            loss = (
                arm.distance_weight * distance_loss
                + arm.terminal_weight * terminal_loss
                + arm.monotonic_weight * monotonic_loss
                + arm.bellman_weight * bellman_loss
                + arm.classification_weight * classification_loss
            )
            # Retain every head in the graph for matched architecture audits.
            loss = loss + 0.0 * (
                scores.potential.square().mean()
                + scores.terminal_logit.square().mean()
                + action_logits.square().mean()
            )
        if not bool(torch.isfinite(loss)):
            raise LyapunovValueError("training produced a nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        final_parts = [
            float(item.detach().cpu())
            for item in (
                distance_loss,
                terminal_loss,
                monotonic_loss,
                bellman_loss,
                classification_loss,
            )
        ]
        node_examples += node_count
        decision_examples += len(decision_indices)
        transition_examples += transition_count
        encoder_examples += unique_indices.numel()
        classifier_examples += transition_count
    return TrainingResult(
        final_loss=losses[-1],
        mean_loss=sum(losses) / len(losses),
        final_distance_loss=final_parts[0],
        final_terminal_loss=final_parts[1],
        final_monotonic_loss=final_parts[2],
        final_bellman_loss=final_parts[3],
        final_classification_loss=final_parts[4],
        resources=TrainingResources(
            optimizer_updates=config.optimizer_updates,
            node_examples=node_examples,
            decision_examples=decision_examples,
            transition_examples=transition_examples,
            state_encoder_examples=encoder_examples,
            action_classifier_examples=classifier_examples,
            oracle_calls=0,
            search_calls=0,
            verifier_calls=0,
        ),
    )


@dataclass(frozen=True, slots=True)
class EvaluationBoundaryToken:
    schema: str
    dataset_manifest_sha256: str
    preparation_source_sha256: str
    preparation_source_destroyed: bool
    prepared_packet_sha256: str
    prepared_packet_destroyed: bool
    training_labels_destroyed: bool
    training_tensors_destroyed: bool
    candidate_allowed_inputs: tuple[str, ...]
    forbidden_candidate_inputs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not (
            self.preparation_source_destroyed
            and self.prepared_packet_destroyed
            and self.training_labels_destroyed
            and self.training_tensors_destroyed
        ):
            raise LyapunovValueError("evaluation boundary is not sealed")


@dataclass(slots=True)
class MutableCandidateResources:
    successor_evaluations: int = 0
    successor_matrix_cells: int = 0
    model_forward_calls: int = 0
    state_encoder_examples: int = 0
    action_classifier_examples: int = 0
    action_candidates_scored: int = 0
    hard_decisions: int = 0
    oracle_calls: int = 0
    search_calls: int = 0
    verifier_calls: int = 0

    def freeze(self) -> "CandidateResources":
        return CandidateResources(**asdict(self))


@dataclass(frozen=True, slots=True)
class CandidateResources:
    successor_evaluations: int
    successor_matrix_cells: int
    model_forward_calls: int
    state_encoder_examples: int
    action_classifier_examples: int
    action_candidates_scored: int
    hard_decisions: int
    oracle_calls: int
    search_calls: int
    verifier_calls: int


def _binding_permutation(
    actions: Sequence[SuccessorAction],
    *,
    matrix: tuple[tuple[int, ...], ...],
    seed: int,
) -> tuple[int, ...]:
    nonhalt = [
        index for index, action in enumerate(actions) if action.kind != ACTION_HALT
    ]
    permutation = list(range(len(actions)))
    if len(nonhalt) > 1:
        offset_seed = int(
            sha256(
                _canonical_bytes(
                    {
                        "seed": seed,
                        "matrix": matrix,
                        "actions": [
                            action.canonical_data() for action in actions
                        ],
                    }
                )
            ).hexdigest()[:16],
            16,
        )
        shift = 1 + offset_seed % (len(nonhalt) - 1)
        rotated = nonhalt[shift:] + nonhalt[:shift]
        for source, destination in zip(nonhalt, rotated, strict=True):
            permutation[source] = destination
    return tuple(permutation)


@dataclass(frozen=True, slots=True)
class HardDecision:
    action: SuccessorAction
    current_potential: float
    selected_successor_potential: float
    terminal_logit: float
    binding_manifest_sha256: str


def choose_hard_action(
    model: LyapunovValueController,
    rows: Iterable[Iterable[int]],
    *,
    inference_mode: str,
    binding_mode: str,
    binding_seed: int,
    boundary: EvaluationBoundaryToken,
    resources: MutableCandidateResources,
) -> HardDecision:
    """Choose one action from only raw candidate inputs."""

    if not isinstance(boundary, EvaluationBoundaryToken):
        raise LyapunovValueError("candidate lacks a sealed evaluation boundary")
    if inference_mode not in INFERENCE_MODES:
        raise LyapunovValueError("unknown inference mode")
    if binding_mode not in BINDING_MODES:
        raise LyapunovValueError("unknown binding mode")
    matrix = canonical_matrix(rows)
    actions = enumerate_legal_actions(matrix)
    true_successors = tuple(apply_action(matrix, action) for action in actions)
    resources.successor_evaluations += len(actions)
    resources.successor_matrix_cells += (
        len(actions) * len(matrix) * len(matrix[0])
    )
    if binding_mode == BINDING_SHUFFLED:
        permutation = _binding_permutation(
            actions,
            matrix=matrix,
            seed=binding_seed,
        )
        visible_successors = tuple(true_successors[index] for index in permutation)
    else:
        permutation = tuple(range(len(actions)))
        visible_successors = true_successors
    binding_manifest = _digest(
        {
            "mode": binding_mode,
            "pairs": [
                [
                    action.canonical_data(),
                    matrix_sha256(visible_successor),
                ]
                for action, visible_successor in zip(
                    actions,
                    visible_successors,
                    strict=True,
                )
            ],
        }
    )
    device = next(model.parameters()).device
    matrices = (matrix,) + visible_successors
    values, mask, rows_tensor, columns_tensor = tensorize_matrices(
        matrices,
        device=device,
    )
    with torch.no_grad():
        scores = model.encode_states(
            values,
            mask,
            rows_tensor,
            columns_tensor,
        )
        current_embedding = scores.embedding[0:1].expand(len(actions), -1)
        successor_embedding = scores.embedding[1:]
        features = torch.tensor(
            [
                action_features(
                    action,
                    row_count=len(matrix),
                    column_count=len(matrix[0]),
                )
                for action in actions
            ],
            dtype=successor_embedding.dtype,
            device=device,
        )
        classifier_logits = model.classify_actions(
            current_embedding,
            successor_embedding,
            features,
        )
    resources.model_forward_calls += 1
    resources.state_encoder_examples += len(matrices)
    resources.action_classifier_examples += len(actions)
    resources.action_candidates_scored += len(actions)
    resources.hard_decisions += 1
    current_potential = float(scores.potential[0].float().cpu())
    terminal_logit = float(scores.terminal_logit[0].float().cpu())
    successor_potentials = scores.potential[1:].float()
    if inference_mode == INFERENCE_CLASSIFICATION:
        chosen_index = int(torch.argmax(classifier_logits).item())
    elif terminal_logit >= 0.0:
        chosen_index = next(
            index
            for index, action in enumerate(actions)
            if action.kind == ACTION_HALT
        )
    else:
        nonhalt_indices = [
            index
            for index, action in enumerate(actions)
            if action.kind != ACTION_HALT
        ]
        if not nonhalt_indices:
            chosen_index = next(
                index
                for index, action in enumerate(actions)
                if action.kind == ACTION_HALT
            )
        else:
            chosen_index = min(
                nonhalt_indices,
                key=lambda index: (float(successor_potentials[index]), index),
            )
    return HardDecision(
        action=actions[chosen_index],
        current_potential=current_potential,
        selected_successor_potential=float(
            successor_potentials[chosen_index].cpu()
        ),
        terminal_logit=terminal_logit,
        binding_manifest_sha256=binding_manifest,
    )


@dataclass(frozen=True, slots=True)
class CandidateRollout:
    actions: tuple[SuccessorAction, ...]
    output_rows: tuple[tuple[int, ...], ...]
    halted: bool
    cycled: bool
    overlong: bool
    potential_descent_steps: int
    potential_nondescent_steps: int
    binding_manifest_sha256: str
    resources: CandidateResources


def candidate_rollout(
    model: LyapunovValueController,
    rows: Iterable[Iterable[int]],
    *,
    inference_mode: str,
    binding_mode: str,
    binding_seed: int,
    maximum_steps: int,
    boundary: EvaluationBoundaryToken,
) -> CandidateRollout:
    limit = _positive_int(maximum_steps, label="maximum_steps")
    matrix = canonical_matrix(rows)
    visited = {matrix_sha256(matrix)}
    actions: list[SuccessorAction] = []
    resources = MutableCandidateResources()
    descent = 0
    nondescent = 0
    bindings: list[str] = []
    model.eval()
    for _ in range(limit):
        decision = choose_hard_action(
            model,
            matrix,
            inference_mode=inference_mode,
            binding_mode=binding_mode,
            binding_seed=binding_seed,
            boundary=boundary,
            resources=resources,
        )
        actions.append(decision.action)
        bindings.append(decision.binding_manifest_sha256)
        if decision.action.kind == ACTION_HALT:
            return CandidateRollout(
                actions=tuple(actions),
                output_rows=matrix,
                halted=True,
                cycled=False,
                overlong=False,
                potential_descent_steps=descent,
                potential_nondescent_steps=nondescent,
                binding_manifest_sha256=_digest(bindings),
                resources=resources.freeze(),
            )
        if decision.selected_successor_potential < decision.current_potential:
            descent += 1
        else:
            nondescent += 1
        matrix = apply_action(matrix, decision.action)
        key = matrix_sha256(matrix)
        if key in visited:
            return CandidateRollout(
                actions=tuple(actions),
                output_rows=matrix,
                halted=False,
                cycled=True,
                overlong=False,
                potential_descent_steps=descent,
                potential_nondescent_steps=nondescent,
                binding_manifest_sha256=_digest(bindings),
                resources=resources.freeze(),
            )
        visited.add(key)
    return CandidateRollout(
        actions=tuple(actions),
        output_rows=matrix,
        halted=False,
        cycled=False,
        overlong=True,
        potential_descent_steps=descent,
        potential_nondescent_steps=nondescent,
        binding_manifest_sha256=_digest(bindings),
        resources=resources.freeze(),
    )


@dataclass(frozen=True, slots=True)
class Assessment:
    strict_canonical_certified: bool
    error: str | None
    verifier_calls: int


def assess_rollout_posthoc(
    source: Iterable[Iterable[int]],
    rollout: CandidateRollout,
) -> Assessment:
    """Run the strict endpoint verifier after candidate custody ends."""

    if not rollout.halted:
        return Assessment(False, "candidate_did_not_halt", 0)
    try:
        program = compile_trace_to_primitives(
            canonical_matrix(source),
            rollout.actions,
        )
        state = execute_program(
            canonical_matrix(source),
            program,
            register_count=DEFAULT_REGISTER_COUNT,
        )
        receipt = verify_reduction_program(canonical_matrix(source), state)
    except (ValueError, RuntimeError) as error:
        return Assessment(False, type(error).__name__, 1)
    return Assessment(bool(receipt.passed), None, 1)


def _sum_candidate_resources(
    values: Sequence[CandidateResources],
) -> CandidateResources:
    fields = CandidateResources.__dataclass_fields__
    return CandidateResources(
        **{
            name: sum(getattr(value, name) for value in values)
            for name in fields
        }
    )


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    strict_canonical_certified: int
    total: int
    cycles: int
    overlong: int
    halted_invalid: int
    potential_descent_steps: int
    potential_nondescent_steps: int
    candidate_resources: CandidateResources
    assessor_verifier_calls: int
    binding_manifest_sha256: str

    @property
    def certification_rate(self) -> float:
        return self.strict_canonical_certified / self.total if self.total else 0.0


def evaluate_model(
    model: LyapunovValueController,
    matrices: Sequence[Iterable[Iterable[int]]],
    *,
    inference_mode: str,
    binding_mode: str,
    binding_seed: int,
    maximum_steps: int,
    boundary: EvaluationBoundaryToken,
) -> EvaluationResult:
    certified = 0
    cycles = 0
    overlong = 0
    invalid = 0
    descent = 0
    nondescent = 0
    verifier_calls = 0
    resources = []
    bindings = []
    for matrix in matrices:
        rollout = candidate_rollout(
            model,
            matrix,
            inference_mode=inference_mode,
            binding_mode=binding_mode,
            binding_seed=binding_seed,
            maximum_steps=maximum_steps,
            boundary=boundary,
        )
        assessment = assess_rollout_posthoc(matrix, rollout)
        certified += int(assessment.strict_canonical_certified)
        cycles += int(rollout.cycled)
        overlong += int(rollout.overlong)
        invalid += int(rollout.halted and not assessment.strict_canonical_certified)
        descent += rollout.potential_descent_steps
        nondescent += rollout.potential_nondescent_steps
        verifier_calls += assessment.verifier_calls
        resources.append(rollout.resources)
        bindings.append(rollout.binding_manifest_sha256)
    combined = _sum_candidate_resources(resources)
    if combined.oracle_calls or combined.search_calls or combined.verifier_calls:
        raise LyapunovValueError("candidate used a forbidden inference call")
    return EvaluationResult(
        strict_canonical_certified=certified,
        total=len(matrices),
        cycles=cycles,
        overlong=overlong,
        halted_invalid=invalid,
        potential_descent_steps=descent,
        potential_nondescent_steps=nondescent,
        candidate_resources=combined,
        assessor_verifier_calls=verifier_calls,
        binding_manifest_sha256=_digest(bindings),
    )


@dataclass(frozen=True, slots=True)
class ArmReport:
    name: str
    inference_mode: str
    binding_mode: str
    label_mode: str
    loss_weights: Mapping[str, float]
    training: TrainingResult
    model_parameter_count: int
    complete_system_parameter_count: int
    model_state_sha256: str
    model_file_sha256: str
    strict_canonical_certified: int
    total: int
    certification_rate: float
    cycles: int
    overlong: int
    halted_invalid: int
    potential_descent_steps: int
    potential_nondescent_steps: int
    candidate_resources: CandidateResources
    assessor_verifier_calls: int
    binding_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    seed: int = 20260724
    train_matrices: int = 256
    evaluation_matrices: int = 128
    train_minimum_rows: int = 2
    train_maximum_rows: int = 3
    train_minimum_columns: int = 3
    train_maximum_columns: int = 4
    evaluation_minimum_rows: int = 4
    evaluation_maximum_rows: int = 4
    evaluation_minimum_columns: int = 5
    evaluation_maximum_columns: int = 6
    maximum_preparation_steps: int = 96
    maximum_rollout_steps: int = 192

    def __post_init__(self) -> None:
        for label, value in asdict(self).items():
            if label == "seed":
                _nonnegative_int(value, label=label)
            else:
                _positive_int(value, label=label)
        if self.train_minimum_rows > self.train_maximum_rows:
            raise LyapunovValueError("train row bounds are inverted")
        if self.train_minimum_columns > self.train_maximum_columns:
            raise LyapunovValueError("train column bounds are inverted")
        if self.evaluation_minimum_rows > self.evaluation_maximum_rows:
            raise LyapunovValueError("evaluation row bounds are inverted")
        if self.evaluation_minimum_columns > self.evaluation_maximum_columns:
            raise LyapunovValueError("evaluation column bounds are inverted")
        if self.evaluation_minimum_rows <= self.train_maximum_rows:
            raise LyapunovValueError("evaluation rows are not strictly larger")
        if self.evaluation_minimum_columns <= self.train_maximum_columns:
            raise LyapunovValueError("evaluation columns are not strictly larger")


@dataclass(frozen=True, slots=True)
class SealedPreparationPacket:
    schema: str
    experiment_config: ExperimentConfig
    train_matrices: tuple[tuple[tuple[int, ...], ...], ...]
    dataset: PotentialDataset
    preparation: PreparationReceipt
    payload_sha256: str
    file_sha256: str
    file_bytes: int


def _dataset_payload(dataset: PotentialDataset) -> Mapping[str, object]:
    return {
        "labels": [
            {
                "rows": label.rows,
                "remaining_distance": label.remaining_distance,
                "terminal": label.terminal,
                "canonical_next_action": (
                    label.canonical_next_action.canonical_data()
                ),
            }
            for label in dataset.labels
        ],
        "decisions": [
            {
                "rows": decision.rows,
                "canonical_next_action": (
                    decision.canonical_next_action.canonical_data()
                ),
                "transitions": [
                    {
                        "action": transition.action.canonical_data(),
                        "successor_rows": transition.successor_rows,
                    }
                    for transition in decision.transitions
                ],
            }
            for decision in dataset.decisions
        ],
        "maximum_distance": dataset.maximum_distance,
        "manifest_sha256": dataset.manifest_sha256,
    }


def _action_from_data(value: object) -> SuccessorAction:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or not isinstance(value[0], str)
    ):
        raise LyapunovValueError("packet action is malformed")
    return SuccessorAction(
        value[0],
        row_a=int(value[1]),
        row_b=int(value[2]),
        column=int(value[3]),
    )


def _matrix_from_data(value: object) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list):
        raise LyapunovValueError("packet matrix is malformed")
    return canonical_matrix(value)


def _dataset_from_payload(value: object) -> PotentialDataset:
    if not isinstance(value, dict):
        raise LyapunovValueError("packet dataset is malformed")
    raw_labels = value.get("labels")
    raw_decisions = value.get("decisions")
    if not isinstance(raw_labels, list) or not isinstance(raw_decisions, list):
        raise LyapunovValueError("packet dataset collections are malformed")
    labels = []
    for raw in raw_labels:
        if not isinstance(raw, dict):
            raise LyapunovValueError("packet label is malformed")
        labels.append(
            DistanceLabel(
                rows=_matrix_from_data(raw.get("rows")),
                remaining_distance=int(raw.get("remaining_distance")),
                terminal=bool(raw.get("terminal")),
                canonical_next_action=_action_from_data(
                    raw.get("canonical_next_action")
                ),
            )
        )
    decisions = []
    for raw in raw_decisions:
        if not isinstance(raw, dict):
            raise LyapunovValueError("packet decision is malformed")
        raw_transitions = raw.get("transitions")
        if not isinstance(raw_transitions, list):
            raise LyapunovValueError("packet transitions are malformed")
        decisions.append(
            DecisionState(
                rows=_matrix_from_data(raw.get("rows")),
                canonical_next_action=_action_from_data(
                    raw.get("canonical_next_action")
                ),
                transitions=tuple(
                    DecisionTransition(
                        action=_action_from_data(transition.get("action")),
                        successor_rows=_matrix_from_data(
                            transition.get("successor_rows")
                        ),
                    )
                    for transition in raw_transitions
                    if isinstance(transition, dict)
                ),
            )
        )
        if len(decisions[-1].transitions) != len(raw_transitions):
            raise LyapunovValueError("packet transition entry is malformed")
    frozen_labels = tuple(labels)
    frozen_decisions = tuple(decisions)
    manifest = _dataset_manifest(frozen_labels, frozen_decisions)
    if manifest != value.get("manifest_sha256"):
        raise LyapunovValueError("packet dataset manifest mismatch")
    index = {
        matrix_sha256(label.rows): position
        for position, label in enumerate(frozen_labels)
    }
    dataset = PotentialDataset(
        labels=frozen_labels,
        decisions=frozen_decisions,
        label_index_by_matrix_sha256=index,
        maximum_distance=int(value.get("maximum_distance")),
        manifest_sha256=manifest,
    )
    if max(label.remaining_distance for label in dataset.labels) != (
        dataset.maximum_distance
    ):
        raise LyapunovValueError("packet maximum distance mismatch")
    return dataset


def write_prepared_packet(
    *,
    experiment: ExperimentConfig,
    output: Path,
) -> Mapping[str, object]:
    """Run CPU-only preparation and emit one hash-bound training packet."""

    if output.exists():
        raise LyapunovValueError(f"refusing to overwrite packet {output}")
    train_matrices = generate_matrices(
        seed=experiment.seed,
        count=experiment.train_matrices,
        minimum_rows=experiment.train_minimum_rows,
        maximum_rows=experiment.train_maximum_rows,
        minimum_columns=experiment.train_minimum_columns,
        maximum_columns=experiment.train_maximum_columns,
    )
    scratch = Path(
        tempfile.mkdtemp(
            prefix="ssqac_lyapunov_prepare_",
            dir=str(output.parent),
        )
    )
    scratch.rmdir()
    dataset, preparation = build_potential_dataset(
        train_matrices,
        maximum_steps=experiment.maximum_preparation_steps,
        preparation_root=scratch,
    )
    payload = {
        "schema": PACKET_SCHEMA,
        "experiment_config": asdict(experiment),
        "train_matrices": train_matrices,
        "train_matrix_manifest_sha256": matrix_manifest(train_matrices),
        "dataset": _dataset_payload(dataset),
        "preparation": asdict(preparation),
    }
    payload_sha = _digest(payload)
    envelope = {
        "schema": PACKET_SCHEMA,
        "payload_sha256": payload_sha,
        "payload": payload,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_bytes(envelope) + b"\n")
    return {
        "schema": PACKET_SCHEMA,
        "path": str(output),
        "payload_sha256": payload_sha,
        "file_sha256": _file_sha256(output),
        "file_bytes": output.stat().st_size,
        "train_matrix_manifest_sha256": matrix_manifest(train_matrices),
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "preparation_oracle_calls": preparation.oracle_calls,
        "preparation_source_destroyed": (
            preparation.serialized_source_destroyed
        ),
        "protected_checkpoint_loaded": False,
        "pretraining_started_or_queued": False,
    }


def load_prepared_packet(path: Path) -> SealedPreparationPacket:
    """Validate and reconstruct a CPU-prepared packet without oracle calls."""

    if not path.is_file():
        raise LyapunovValueError(f"prepared packet is missing: {path}")
    raw = path.read_bytes()
    try:
        envelope = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise LyapunovValueError("prepared packet is not canonical JSON") from error
    if not isinstance(envelope, dict) or envelope.get("schema") != PACKET_SCHEMA:
        raise LyapunovValueError("prepared packet schema mismatch")
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise LyapunovValueError("prepared packet payload is malformed")
    if _digest(payload) != envelope.get("payload_sha256"):
        raise LyapunovValueError("prepared packet payload hash mismatch")
    if payload.get("schema") != PACKET_SCHEMA:
        raise LyapunovValueError("prepared payload schema mismatch")
    raw_config = payload.get("experiment_config")
    if not isinstance(raw_config, dict):
        raise LyapunovValueError("prepared experiment config is malformed")
    experiment = ExperimentConfig(**raw_config)
    raw_matrices = payload.get("train_matrices")
    if not isinstance(raw_matrices, list):
        raise LyapunovValueError("prepared train matrices are malformed")
    train_matrices = tuple(_matrix_from_data(item) for item in raw_matrices)
    if len(train_matrices) != experiment.train_matrices:
        raise LyapunovValueError("prepared train matrix count mismatch")
    if matrix_manifest(train_matrices) != payload.get(
        "train_matrix_manifest_sha256"
    ):
        raise LyapunovValueError("prepared train matrix manifest mismatch")
    dataset = _dataset_from_payload(payload.get("dataset"))
    raw_preparation = payload.get("preparation")
    if not isinstance(raw_preparation, dict):
        raise LyapunovValueError("prepared receipt is malformed")
    preparation = PreparationReceipt(**raw_preparation)
    if preparation.dataset_manifest_sha256 != dataset.manifest_sha256:
        raise LyapunovValueError("prepared receipt does not bind the dataset")
    if (
        not preparation.serialized_source_destroyed
        or preparation.serialized_source_exists_after_destroy
    ):
        raise LyapunovValueError("preparation search transcript survived sealing")
    return SealedPreparationPacket(
        schema=PACKET_SCHEMA,
        experiment_config=experiment,
        train_matrices=train_matrices,
        dataset=dataset,
        preparation=preparation,
        payload_sha256=str(envelope["payload_sha256"]),
        file_sha256=sha256(raw).hexdigest(),
        file_bytes=len(raw),
    )


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    schema: str
    architecture_schema: str
    resource_schema: str
    status: str
    claim: str
    seed: int
    device: str
    experiment_config: ExperimentConfig
    training_config: TrainingConfig
    model_config: LyapunovConfig
    parameter_count: int
    complete_system_parameter_count: int
    parameter_count_breakdown: Mapping[str, int]
    protected_checkpoint_loaded: bool
    protected_checkpoint_modified: bool
    pretraining_started_or_queued: bool
    candidate_allowed_inputs: tuple[str, ...]
    candidate_forbidden_inputs: tuple[str, ...]
    preparation: PreparationReceipt
    evaluation_boundary: EvaluationBoundaryToken
    train_matrix_manifest_sha256: str
    evaluation_matrix_manifest_sha256: str
    source_files_sha256: Mapping[str, str]
    arms: tuple[ArmReport, ...]
    controls_equal_parameter_count: bool
    controls_equal_optimizer_updates: bool
    controls_equal_node_examples: bool
    controls_equal_decision_examples: bool
    controls_equal_transition_examples: bool
    treatment_rate: float
    action_classification_rate: float
    shuffled_distance_rate: float
    zero_structure_rate: float
    shuffled_binding_rate: float
    random_label_rate: float
    treatment_minus_action_classification: float
    treatment_minus_zero_structure: float
    treatment_minus_shuffled_distance: float
    absolute_gate: float
    causal_gap_gate: float
    passed_absolute_gate: bool
    passed_causal_gap_gate: bool
    exact_h100_seconds: int | None
    notes: tuple[str, ...]

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(asdict(self)) + b"\n"


def _save_model(
    model: LyapunovValueController,
    *,
    path: Path,
    arm: str,
    seed: int,
) -> tuple[str, str]:
    if path.exists():
        raise LyapunovValueError(f"refusing to overwrite model {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    state_hash = model_state_sha256(model)
    torch.save(
        {
            "architecture_schema": ARCHITECTURE_SCHEMA,
            "arm": arm,
            "seed": seed,
            "config": asdict(model.config),
            "parameter_count": model.parameter_count,
            "complete_system_parameter_count": (
                model.complete_system_parameter_count
            ),
            "model_state_sha256": state_hash,
            "state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            },
        },
        path,
    )
    return state_hash, _file_sha256(path)


def run_bounded_experiment(
    *,
    prepared_packet_path: Path,
    model_config: LyapunovConfig,
    training_config: TrainingConfig,
    device: torch.device,
    model_dir: Path,
) -> ExperimentReport:
    """Fit from a CPU packet, destroy labels, then generate/evaluate holdout."""

    packet = load_prepared_packet(prepared_packet_path)
    experiment = packet.experiment_config
    train_matrices = packet.train_matrices
    dataset = packet.dataset
    preparation = packet.preparation
    packet_file_sha256 = packet.file_sha256
    tensors = _build_training_tensors(dataset, device=device)
    trained: dict[str, tuple[LyapunovValueController, TrainingResult, str, str]] = {}
    parameter_count: int | None = None
    complete_count: int | None = None
    parameter_breakdown: Mapping[str, int] | None = None
    for arm_index, arm in enumerate(ARM_SPECS):
        torch.manual_seed(experiment.seed + arm_index * 1009)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(experiment.seed + arm_index * 1009)
        model = LyapunovValueController(model_config).to(device)
        if parameter_count is None:
            parameter_count = model.parameter_count
            complete_count = model.complete_system_parameter_count
            parameter_breakdown = model.parameter_count_breakdown()
        elif model.parameter_count != parameter_count:
            raise LyapunovValueError("control parameter counts differ")
        training = train_arm(
            model,
            tensors,
            arm=arm,
            config=training_config,
            maximum_distance=dataset.maximum_distance,
            # Every arm sees the exact same sampled nodes and decisions.
            # Label transformations remain arm-specific through ArmSpec.
            seed=experiment.seed ^ 0x54524149,
        )
        model = model.to("cpu")
        path = model_dir / f"{arm.name}_seed{experiment.seed}.pt"
        state_hash, file_hash = _save_model(
            model,
            path=path,
            arm=arm.name,
            seed=experiment.seed,
        )
        trained[arm.name] = (model, training, state_hash, file_hash)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    dataset_manifest = dataset.manifest_sha256
    source_sha = preparation.serialized_source_sha256
    prepared_packet_path.unlink()
    if prepared_packet_path.exists():
        raise LyapunovValueError("prepared packet destruction failed")
    del tensors
    del dataset
    del packet
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    evaluation_matrices = generate_matrices(
        seed=experiment.seed ^ 0x4556414C,
        count=experiment.evaluation_matrices,
        minimum_rows=experiment.evaluation_minimum_rows,
        maximum_rows=experiment.evaluation_maximum_rows,
        minimum_columns=experiment.evaluation_minimum_columns,
        maximum_columns=experiment.evaluation_maximum_columns,
        excluded=set(train_matrices),
    )
    boundary = EvaluationBoundaryToken(
        schema=BOUNDARY_SCHEMA,
        dataset_manifest_sha256=dataset_manifest,
        preparation_source_sha256=source_sha,
        preparation_source_destroyed=True,
        prepared_packet_sha256=packet_file_sha256,
        prepared_packet_destroyed=True,
        training_labels_destroyed=True,
        training_tensors_destroyed=True,
        candidate_allowed_inputs=(
            "raw_current_field_matrix",
            "legal_action_descriptors",
            "raw_one_step_successor_matrices",
            "fixed_model_parameters",
        ),
        forbidden_candidate_inputs=(
            "canonical_trace",
            "remaining_distance_label",
            "expert_action",
            "rank",
            "structural_energy",
            "search_state",
            "oracle_output",
            "verifier_output",
            "source_matrix_identity",
        ),
    )

    reports: list[ArmReport] = []
    by_name = {arm.name: arm for arm in ARM_SPECS}
    for arm in ARM_SPECS:
        model, training, state_hash, file_hash = trained[arm.name]
        model = model.to(device)
        evaluated = evaluate_model(
            model,
            evaluation_matrices,
            inference_mode=arm.inference_mode,
            binding_mode=BINDING_RAW,
            binding_seed=experiment.seed ^ 0x42494E44,
            maximum_steps=experiment.maximum_rollout_steps,
            boundary=boundary,
        )
        reports.append(
            ArmReport(
                name=arm.name,
                inference_mode=arm.inference_mode,
                binding_mode=BINDING_RAW,
                label_mode=arm.label_mode,
                loss_weights={
                    "distance": arm.distance_weight,
                    "terminal": arm.terminal_weight,
                    "monotonic": arm.monotonic_weight,
                    "bellman": arm.bellman_weight,
                    "classification": arm.classification_weight,
                },
                training=training,
                model_parameter_count=model.parameter_count,
                complete_system_parameter_count=(
                    model.complete_system_parameter_count
                ),
                model_state_sha256=state_hash,
                model_file_sha256=file_hash,
                strict_canonical_certified=(
                    evaluated.strict_canonical_certified
                ),
                total=evaluated.total,
                certification_rate=evaluated.certification_rate,
                cycles=evaluated.cycles,
                overlong=evaluated.overlong,
                halted_invalid=evaluated.halted_invalid,
                potential_descent_steps=evaluated.potential_descent_steps,
                potential_nondescent_steps=(
                    evaluated.potential_nondescent_steps
                ),
                candidate_resources=evaluated.candidate_resources,
                assessor_verifier_calls=evaluated.assessor_verifier_calls,
                binding_manifest_sha256=(
                    evaluated.binding_manifest_sha256
                ),
            )
        )
        model = model.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    treatment_model, treatment_training, state_hash, file_hash = trained[
        ARM_TREATMENT
    ]
    treatment_model = treatment_model.to(device)
    shuffled = evaluate_model(
        treatment_model,
        evaluation_matrices,
        inference_mode=INFERENCE_POTENTIAL,
        binding_mode=BINDING_SHUFFLED,
        binding_seed=experiment.seed ^ 0x53485546,
        maximum_steps=experiment.maximum_rollout_steps,
        boundary=boundary,
    )
    treatment_spec = by_name[ARM_TREATMENT]
    reports.append(
        ArmReport(
            name=ARM_SHUFFLED_BINDINGS,
            inference_mode=INFERENCE_POTENTIAL,
            binding_mode=BINDING_SHUFFLED,
            label_mode=LABEL_TRUE,
            loss_weights={
                "distance": treatment_spec.distance_weight,
                "terminal": treatment_spec.terminal_weight,
                "monotonic": treatment_spec.monotonic_weight,
                "bellman": treatment_spec.bellman_weight,
                "classification": treatment_spec.classification_weight,
            },
            training=treatment_training,
            model_parameter_count=treatment_model.parameter_count,
            complete_system_parameter_count=(
                treatment_model.complete_system_parameter_count
            ),
            model_state_sha256=state_hash,
            model_file_sha256=file_hash,
            strict_canonical_certified=shuffled.strict_canonical_certified,
            total=shuffled.total,
            certification_rate=shuffled.certification_rate,
            cycles=shuffled.cycles,
            overlong=shuffled.overlong,
            halted_invalid=shuffled.halted_invalid,
            potential_descent_steps=shuffled.potential_descent_steps,
            potential_nondescent_steps=shuffled.potential_nondescent_steps,
            candidate_resources=shuffled.candidate_resources,
            assessor_verifier_calls=shuffled.assessor_verifier_calls,
            binding_manifest_sha256=shuffled.binding_manifest_sha256,
        )
    )
    report_by_name = {report.name: report for report in reports}
    treatment_rate = report_by_name[ARM_TREATMENT].certification_rate
    classification_rate = report_by_name[ARM_CLASSIFICATION].certification_rate
    shuffled_distance_rate = report_by_name[
        ARM_SHUFFLED_DISTANCE
    ].certification_rate
    zero_rate = report_by_name[ARM_ZERO_STRUCTURE].certification_rate
    shuffled_binding_rate = report_by_name[
        ARM_SHUFFLED_BINDINGS
    ].certification_rate
    random_rate = report_by_name[ARM_RANDOM_LABELS].certification_rate
    absolute_gate = 0.80
    causal_gap_gate = 0.10
    causal_gap = min(
        treatment_rate - classification_rate,
        treatment_rate - zero_rate,
        treatment_rate - shuffled_distance_rate,
        treatment_rate - shuffled_binding_rate,
    )
    passed_absolute = treatment_rate >= absolute_gate
    passed_causal = causal_gap >= causal_gap_gate
    if passed_absolute and passed_causal:
        claim = CLAIM_MATERIAL
    elif treatment_rate > max(
        classification_rate,
        zero_rate,
        shuffled_distance_rate,
        shuffled_binding_rate,
    ):
        claim = CLAIM_SUGGESTIVE
    else:
        claim = CLAIM_NO_GO
    parameter_counts = {report.model_parameter_count for report in reports}
    updates = {
        report.training.resources.optimizer_updates
        for report in reports
        if report.name != ARM_SHUFFLED_BINDINGS
    }
    node_examples = {
        report.training.resources.node_examples
        for report in reports
        if report.name != ARM_SHUFFLED_BINDINGS
    }
    decision_examples = {
        report.training.resources.decision_examples
        for report in reports
        if report.name != ARM_SHUFFLED_BINDINGS
    }
    transition_examples = {
        report.training.resources.transition_examples
        for report in reports
        if report.name != ARM_SHUFFLED_BINDINGS
    }
    source_path = Path(__file__)
    test_path = source_path.with_name("test_ssqac_lyapunov_value_controller.py")
    launcher_path = source_path.parent / "jobs" / "ssqac_lyapunov_value.sbatch"
    source_hashes = {"controller": _file_sha256(source_path)}
    if test_path.is_file():
        source_hashes["tests"] = _file_sha256(test_path)
    if launcher_path.is_file():
        source_hashes["launcher"] = _file_sha256(launcher_path)
    if parameter_count is None or complete_count is None or parameter_breakdown is None:
        raise LyapunovValueError("model accounting was not initialized")
    return ExperimentReport(
        schema=EXPERIMENT_SCHEMA,
        architecture_schema=ARCHITECTURE_SCHEMA,
        resource_schema=RESOURCE_SCHEMA,
        status=STATUS,
        claim=claim,
        seed=experiment.seed,
        device=str(device),
        experiment_config=experiment,
        training_config=training_config,
        model_config=model_config,
        parameter_count=parameter_count,
        complete_system_parameter_count=complete_count,
        parameter_count_breakdown=parameter_breakdown,
        protected_checkpoint_loaded=False,
        protected_checkpoint_modified=False,
        pretraining_started_or_queued=False,
        candidate_allowed_inputs=boundary.candidate_allowed_inputs,
        candidate_forbidden_inputs=boundary.forbidden_candidate_inputs,
        preparation=preparation,
        evaluation_boundary=boundary,
        train_matrix_manifest_sha256=matrix_manifest(train_matrices),
        evaluation_matrix_manifest_sha256=matrix_manifest(evaluation_matrices),
        source_files_sha256=source_hashes,
        arms=tuple(reports),
        controls_equal_parameter_count=len(parameter_counts) == 1,
        controls_equal_optimizer_updates=len(updates) == 1,
        controls_equal_node_examples=len(node_examples) == 1,
        controls_equal_decision_examples=len(decision_examples) == 1,
        controls_equal_transition_examples=len(transition_examples) == 1,
        treatment_rate=treatment_rate,
        action_classification_rate=classification_rate,
        shuffled_distance_rate=shuffled_distance_rate,
        zero_structure_rate=zero_rate,
        shuffled_binding_rate=shuffled_binding_rate,
        random_label_rate=random_rate,
        treatment_minus_action_classification=(
            treatment_rate - classification_rate
        ),
        treatment_minus_zero_structure=treatment_rate - zero_rate,
        treatment_minus_shuffled_distance=(
            treatment_rate - shuffled_distance_rate
        ),
        absolute_gate=absolute_gate,
        causal_gap_gate=causal_gap_gate,
        passed_absolute_gate=passed_absolute,
        passed_causal_gap_gate=passed_causal,
        exact_h100_seconds=None,
        notes=(
            "The treatment receives no expert action loss.",
            "Canonical remaining distance is a state-only preparation label.",
            "The action classifier shares the exact treatment architecture.",
            "Shuffled binding is a paired candidate-time treatment ablation.",
            "All strict verification is posthoc assessor computation.",
            "A single-seed pass cannot establish replication.",
        ),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepared-packet", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--train-matrices", type=int, default=256)
    parser.add_argument("--evaluation-matrices", type=int, default=128)
    parser.add_argument("--train-minimum-rows", type=int, default=2)
    parser.add_argument("--train-maximum-rows", type=int, default=3)
    parser.add_argument("--train-minimum-columns", type=int, default=3)
    parser.add_argument("--train-maximum-columns", type=int, default=4)
    parser.add_argument("--evaluation-minimum-rows", type=int, default=4)
    parser.add_argument("--evaluation-maximum-rows", type=int, default=4)
    parser.add_argument("--evaluation-minimum-columns", type=int, default=5)
    parser.add_argument("--evaluation-maximum-columns", type=int, default=6)
    parser.add_argument("--maximum-preparation-steps", type=int, default=96)
    parser.add_argument("--maximum-rollout-steps", type=int, default=192)
    parser.add_argument("--optimizer-updates", type=int, default=1600)
    parser.add_argument("--node-batch-size", type=int, default=256)
    parser.add_argument("--decision-batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--monotonic-margin", type=float, default=0.5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--field-width", type=int, default=64)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--cell-hidden", type=int, default=1024)
    parser.add_argument("--matrix-layers", type=int, default=6)
    parser.add_argument("--state-hidden", type=int, default=768)
    parser.add_argument("--coordinate-harmonics", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-bf16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    experiment = ExperimentConfig(
            seed=args.seed,
            train_matrices=args.train_matrices,
            evaluation_matrices=args.evaluation_matrices,
            train_minimum_rows=args.train_minimum_rows,
            train_maximum_rows=args.train_maximum_rows,
            train_minimum_columns=args.train_minimum_columns,
            train_maximum_columns=args.train_maximum_columns,
            evaluation_minimum_rows=args.evaluation_minimum_rows,
            evaluation_maximum_rows=args.evaluation_maximum_rows,
            evaluation_minimum_columns=args.evaluation_minimum_columns,
            evaluation_maximum_columns=args.evaluation_maximum_columns,
            maximum_preparation_steps=args.maximum_preparation_steps,
            maximum_rollout_steps=args.maximum_rollout_steps,
        )
    if args.prepare_only:
        if args.model_dir is not None or args.output is not None:
            raise LyapunovValueError(
                "prepare-only does not accept model-dir or output"
            )
        receipt = write_prepared_packet(
            experiment=experiment,
            output=args.prepared_packet,
        )
        print((_canonical_bytes(receipt) + b"\n").decode("ascii"), end="")
        return
    if args.model_dir is None or args.output is None:
        raise LyapunovValueError("fit mode requires model-dir and output")
    if args.output.exists():
        raise LyapunovValueError(f"refusing to overwrite report {args.output}")
    if args.model_dir.exists():
        raise LyapunovValueError(
            f"refusing to overwrite model directory {args.model_dir}"
        )
    args.model_dir.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise LyapunovValueError("CUDA was requested but is unavailable")
    report = run_bounded_experiment(
        prepared_packet_path=args.prepared_packet,
        model_config=LyapunovConfig(
            field_width=args.field_width,
            width=args.width,
            cell_hidden=args.cell_hidden,
            matrix_layers=args.matrix_layers,
            state_hidden=args.state_hidden,
            coordinate_harmonics=args.coordinate_harmonics,
        ),
        training_config=TrainingConfig(
            optimizer_updates=args.optimizer_updates,
            node_batch_size=args.node_batch_size,
            decision_batch_size=args.decision_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            monotonic_margin=args.monotonic_margin,
            gradient_clip=args.gradient_clip,
            bf16=not args.no_bf16,
        ),
        device=device,
        model_dir=args.model_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(report.canonical_bytes())
    print(report.canonical_bytes().decode("ascii"), end="")


if __name__ == "__main__":
    main()
