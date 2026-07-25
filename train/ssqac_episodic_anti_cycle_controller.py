#!/usr/bin/env python3
"""Full-trajectory episodic anti-cycle controller for SSQAC.

The candidate stores learned encodings of previously visited raw matrix states
in a bounded model-owned memory. Each legal successor is compared with that
memory before an action is emitted. Training unrolls complete expert
trajectories so both the recurrent state and episodic memory see the same
temporal regime used at autonomous inference.

Oracle traces are consumed only by the separate ``prepare`` command. The
``run`` command loads a sealed JSON artifact and candidate rollout calls no
oracle, search routine, or verifier. Strict verification is posthoc.
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
import time
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

import ssqac_proof_carrying_controller as proof
from ssqac_successor_value_controller import (
    ACTION_HALT,
    ACTION_TO_INDEX,
    PROTECTED_FLAGSHIP_PARAMETERS,
    TOTAL_PARAMETER_BUDGET,
    CandidateRollout,
    ResourceCounts,
    SuccessorAction,
    apply_action,
    assess_rollout_posthoc,
    canonical_matrix,
    enumerate_legal_actions,
    generate_matrices,
    matrix_manifest,
    matrix_sha256,
)


PREPARATION_SCHEMA = "ssqac_episodic_anti_cycle_preparation_v1"
REPORT_SCHEMA = "ssqac_episodic_anti_cycle_experiment_v1"
STATUS = "isolated_full_trajectory_episodic_memory_falsifier_not_reasoning"

MODE_REAL = "episodic_memory"
MODE_ZERO = "memory_zeroed"
MODE_SHUFFLED = "memory_feature_shuffled"
MODE_CLASSIFIER = "full_trajectory_classifier"
MODE_RANDOM = "randomized_labels"
MODE_BARRIER = "semantic_cycle_barrier"
MODE_BARRIER_SHUFFLED = "semantic_cycle_barrier_feature_shuffled"
MODE_EXACT_BARRIER = "exact_discrete_cycle_barrier"
MODE_EXACT_BARRIER_SHUFFLED = "exact_discrete_cycle_barrier_feature_shuffled"
MODES = (
    MODE_REAL,
    MODE_ZERO,
    MODE_SHUFFLED,
    MODE_CLASSIFIER,
    MODE_RANDOM,
    MODE_BARRIER,
    MODE_BARRIER_SHUFFLED,
    MODE_EXACT_BARRIER,
    MODE_EXACT_BARRIER_SHUFFLED,
)


class EpisodicMemoryError(ValueError):
    """The episodic-memory experiment failed closed."""


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
        raise EpisodicMemoryError("value is not canonical ASCII JSON") from error


def _digest(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EpisodicMemoryError(f"{label} must be a positive integer")
    return value


def _action_data(action: SuccessorAction) -> list[object]:
    return action.canonical_data()


def _action_from_data(value: Sequence[object]) -> SuccessorAction:
    if len(value) != 4:
        raise EpisodicMemoryError("action record has the wrong width")
    return SuccessorAction(
        str(value[0]),
        row_a=int(value[1]),
        row_b=int(value[2]),
        column=int(value[3]),
    )


@dataclass(frozen=True, slots=True)
class ExpertTrajectory:
    states: tuple[tuple[tuple[int, ...], ...], ...]
    actions: tuple[SuccessorAction, ...]

    def __post_init__(self) -> None:
        if not self.states or len(self.states) != len(self.actions):
            raise EpisodicMemoryError("trajectory state/action lengths differ")
        for index, (state, action) in enumerate(zip(self.states, self.actions)):
            if action not in enumerate_legal_actions(state):
                raise EpisodicMemoryError("trajectory contains an illegal action")
            if index + 1 < len(self.states):
                if action.kind == ACTION_HALT:
                    raise EpisodicMemoryError("trajectory halts before its final state")
                if apply_action(state, action) != self.states[index + 1]:
                    raise EpisodicMemoryError("trajectory transition is inconsistent")
        if self.actions[-1].kind != ACTION_HALT:
            raise EpisodicMemoryError("trajectory does not terminate with HALT")

    def canonical_data(self) -> Mapping[str, object]:
        return {
            "states": [
                [list(row) for row in matrix] for matrix in self.states
            ],
            "actions": [_action_data(action) for action in self.actions],
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
    train_matrix_manifest_sha256: str
    evaluation_matrix_manifest_sha256: str
    trajectory_manifest_sha256: str
    trajectories: tuple[ExpertTrajectory, ...]
    evaluation_matrices: tuple[tuple[tuple[int, ...], ...], ...]
    oracle_calls: int
    train_maximum_rows: int
    train_maximum_columns: int
    evaluation_minimum_rows: int
    evaluation_minimum_columns: int
    evaluation_maximum_rows: int
    evaluation_maximum_columns: int
    maximum_preparation_steps: int

    def canonical_data(self) -> Mapping[str, object]:
        return {
            "schema": self.schema,
            "seed": self.seed,
            "source_sha256": self.source_sha256,
            "oracle_source_sha256": self.oracle_source_sha256,
            "train_matrix_manifest_sha256": self.train_matrix_manifest_sha256,
            "evaluation_matrix_manifest_sha256": (
                self.evaluation_matrix_manifest_sha256
            ),
            "trajectory_manifest_sha256": self.trajectory_manifest_sha256,
            "trajectories": [
                trajectory.canonical_data() for trajectory in self.trajectories
            ],
            "evaluation_matrices": [
                [list(row) for row in matrix]
                for matrix in self.evaluation_matrices
            ],
            "oracle_calls": self.oracle_calls,
            "train_maximum_rows": self.train_maximum_rows,
            "train_maximum_columns": self.train_maximum_columns,
            "evaluation_minimum_rows": self.evaluation_minimum_rows,
            "evaluation_minimum_columns": self.evaluation_minimum_columns,
            "evaluation_maximum_rows": self.evaluation_maximum_rows,
            "evaluation_maximum_columns": self.evaluation_maximum_columns,
            "maximum_preparation_steps": self.maximum_preparation_steps,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.canonical_data()) + b"\n"


def _trajectory_manifest(trajectories: Sequence[ExpertTrajectory]) -> str:
    return sha256(
        (
            "\n".join(trajectory.sha256 for trajectory in trajectories) + "\n"
        ).encode("ascii")
    ).hexdigest()


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
    """Create full expert trajectories and seal them before fitting."""

    import ssqac_soft_value_iteration_controller as oracle

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
    counter = oracle.PreparationOracleCounter()
    trajectories = []
    for source in train:
        states = []
        actions = []
        matrix = source
        for _ in range(maximum_preparation_steps):
            raw = oracle.next_preparation_macro(matrix, counter=counter)
            action = SuccessorAction(
                raw.kind,
                row_a=raw.row_a,
                row_b=raw.row_b,
                column=raw.column,
            )
            states.append(matrix)
            actions.append(action)
            if action.kind == ACTION_HALT:
                break
            matrix = apply_action(matrix, action)
        else:
            raise EpisodicMemoryError("expert trajectory exceeded preparation bound")
        trajectories.append(ExpertTrajectory(tuple(states), tuple(actions)))
    oracle_path = Path(inspect.getsourcefile(oracle) or "")
    return PreparationArtifact(
        schema=PREPARATION_SCHEMA,
        seed=seed,
        source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        oracle_source_sha256=sha256(oracle_path.read_bytes()).hexdigest(),
        train_matrix_manifest_sha256=matrix_manifest(train),
        evaluation_matrix_manifest_sha256=matrix_manifest(evaluation),
        trajectory_manifest_sha256=_trajectory_manifest(trajectories),
        trajectories=tuple(trajectories),
        evaluation_matrices=evaluation,
        oracle_calls=counter.calls,
        train_maximum_rows=train_maximum_rows,
        train_maximum_columns=train_maximum_columns,
        evaluation_minimum_rows=evaluation_minimum_rows,
        evaluation_minimum_columns=evaluation_minimum_columns,
        evaluation_maximum_rows=evaluation_maximum_rows,
        evaluation_maximum_columns=evaluation_maximum_columns,
        maximum_preparation_steps=maximum_preparation_steps,
    )


def _load_preparation(path: Path) -> PreparationArtifact:
    data = json.loads(path.read_text(encoding="ascii"))
    trajectories = tuple(
        ExpertTrajectory(
            states=tuple(
                canonical_matrix(matrix) for matrix in item["states"]
            ),
            actions=tuple(_action_from_data(action) for action in item["actions"]),
        )
        for item in data["trajectories"]
    )
    artifact = PreparationArtifact(
        schema=str(data["schema"]),
        seed=int(data["seed"]),
        source_sha256=str(data["source_sha256"]),
        oracle_source_sha256=str(data["oracle_source_sha256"]),
        train_matrix_manifest_sha256=str(data["train_matrix_manifest_sha256"]),
        evaluation_matrix_manifest_sha256=str(
            data["evaluation_matrix_manifest_sha256"]
        ),
        trajectory_manifest_sha256=str(data["trajectory_manifest_sha256"]),
        trajectories=trajectories,
        evaluation_matrices=tuple(
            canonical_matrix(matrix) for matrix in data["evaluation_matrices"]
        ),
        oracle_calls=int(data["oracle_calls"]),
        train_maximum_rows=int(data["train_maximum_rows"]),
        train_maximum_columns=int(data["train_maximum_columns"]),
        evaluation_minimum_rows=int(data["evaluation_minimum_rows"]),
        evaluation_minimum_columns=int(data["evaluation_minimum_columns"]),
        evaluation_maximum_rows=int(data["evaluation_maximum_rows"]),
        evaluation_maximum_columns=int(data["evaluation_maximum_columns"]),
        maximum_preparation_steps=int(data["maximum_preparation_steps"]),
    )
    if artifact.schema != PREPARATION_SCHEMA:
        raise EpisodicMemoryError("preparation schema mismatch")
    if artifact.trajectory_manifest_sha256 != _trajectory_manifest(trajectories):
        raise EpisodicMemoryError("trajectory manifest mismatch")
    train = tuple(trajectory.states[0] for trajectory in trajectories)
    if artifact.train_matrix_manifest_sha256 != matrix_manifest(train):
        raise EpisodicMemoryError("train matrix manifest mismatch")
    if artifact.evaluation_matrix_manifest_sha256 != matrix_manifest(
        artifact.evaluation_matrices
    ):
        raise EpisodicMemoryError("evaluation matrix manifest mismatch")
    if set(train) & set(artifact.evaluation_matrices):
        raise EpisodicMemoryError("train and evaluation matrices overlap")
    return artifact


@dataclass(frozen=True, slots=True)
class EpisodicConfig:
    base: proof.ControllerConfig = proof.ControllerConfig()
    state_width: int = 256
    state_hidden: int = 384
    state_layers: int = 3
    memory_slots: int = 32
    barrier_temperature: float = 0.02
    barrier_penalty: float = 8.0

    def __post_init__(self) -> None:
        for label, value in (
            ("state_width", self.state_width),
            ("state_hidden", self.state_hidden),
            ("state_layers", self.state_layers),
            ("memory_slots", self.memory_slots),
        ):
            _positive_int(value, label=label)
        if self.barrier_temperature <= 0.0:
            raise EpisodicMemoryError("barrier_temperature must be positive")
        if self.barrier_penalty <= 0.0:
            raise EpisodicMemoryError("barrier_penalty must be positive")


@dataclass(frozen=True, slots=True)
class EpisodicState:
    keys: tuple[Tensor, ...] = ()
    recurrent: Tensor | None = None
    raw_states: tuple[tuple[tuple[int, ...], ...], ...] = ()


@dataclass(frozen=True, slots=True)
class EpisodicScores:
    actions: tuple[SuccessorAction, ...]
    successors: tuple[tuple[tuple[int, ...], ...], ...]
    logits: Tensor
    revisit_logits: Tensor
    current_key: Tensor
    action_hidden: Tensor
    maximum_similarity: Tensor
    cycle_evidence: Tensor
    exact_cycle_evidence: Tensor
    current_rows: tuple[tuple[int, ...], ...]


class EpisodicAntiCycleController(nn.Module):
    """Raw-state controller with bounded learned associative memory."""

    def __init__(self, config: EpisodicConfig = EpisodicConfig()) -> None:
        super().__init__()
        self.config = config
        self.base = proof.ProofCarryingController(config.base)
        coordinate_width = 4 * config.base.coordinate_harmonics + 2
        self.state_field = nn.Embedding(
            proof.FIELD_MODULUS,
            config.base.field_width,
        )
        self.state_projection = nn.Sequential(
            nn.Linear(
                config.base.field_width + coordinate_width,
                config.state_hidden,
            ),
            nn.GELU(),
            nn.Linear(config.state_hidden, config.state_width),
            nn.LayerNorm(config.state_width),
        )
        self.state_layers = nn.ModuleList(
            proof.EquivariantGridLayer(
                config.state_width,
                config.state_hidden,
                config.base.dropout,
            )
            for _ in range(config.state_layers)
        )
        self.state_pool = nn.Sequential(
            nn.Linear(2 * config.state_width, config.state_hidden),
            nn.GELU(),
            nn.Linear(config.state_hidden, config.state_width),
            nn.LayerNorm(config.state_width),
        )
        fusion_width = (
            config.base.contract_hidden + 4 * config.state_width + 1
        )
        self.memory_delta = nn.Sequential(
            nn.Linear(fusion_width, config.state_hidden),
            nn.GELU(),
            nn.Linear(config.state_hidden, 1),
        )
        self.revisit_head = nn.Sequential(
            nn.Linear(3 * config.state_width + 1, config.state_hidden),
            nn.GELU(),
            nn.Linear(config.state_hidden, 1),
        )
        if self.complete_system_parameters >= TOTAL_PARAMETER_BUDGET:
            raise EpisodicMemoryError("complete system exceeds 200M parameters")

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def complete_system_parameters(self) -> int:
        return PROTECTED_FLAGSHIP_PARAMETERS + self.parameter_count

    def _coordinates(
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
        for harmonic in range(1, self.config.base.coordinate_harmonics + 1):
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

    def encode_states(self, values: Tensor) -> Tensor:
        """Encode one or more raw matrices into normalized memory keys."""

        squeeze = values.ndim == 2
        if squeeze:
            values = values[None]
        if values.ndim != 3:
            raise EpisodicMemoryError("state values need [batch, rows, columns]")
        batch, rows, columns = values.shape
        fields = self.state_field(values)
        coordinates = self._coordinates(
            rows,
            columns,
            device=values.device,
            dtype=fields.dtype,
        )[None].expand(batch, -1, -1, -1)
        cells = self.state_projection(torch.cat((fields, coordinates), dim=-1))
        for layer in self.state_layers:
            cells = layer(cells)
        pooled = self.state_pool(
            torch.cat(
                (cells.mean(dim=(1, 2)), cells.amax(dim=(1, 2))),
                dim=-1,
            )
        )
        keys = F.normalize(pooled.float(), dim=-1).to(pooled.dtype)
        return keys[0] if squeeze else keys

    def score(
        self,
        rows: Iterable[Iterable[int]],
        memory: EpisodicState,
        *,
        mode: str,
        actions: Sequence[SuccessorAction] | None = None,
    ) -> EpisodicScores:
        if mode not in MODES:
            raise EpisodicMemoryError(f"unknown memory mode {mode!r}")
        matrix = canonical_matrix(rows)
        rendered = (
            enumerate_legal_actions(matrix) if actions is None else tuple(actions)
        )
        if set(rendered) != set(enumerate_legal_actions(matrix)):
            raise EpisodicMemoryError("rendered actions differ from legal actions")
        successors = tuple(apply_action(matrix, action) for action in rendered)
        reference = next(self.parameters())
        device = reference.device
        current_tensor = torch.tensor(matrix, dtype=torch.long, device=device)
        successor_tensor = torch.tensor(
            successors,
            dtype=torch.long,
            device=device,
        )
        base = self.base(
            current_tensor,
            successor_tensor,
            torch.tensor(
                [ACTION_TO_INDEX[action.kind] for action in rendered],
                dtype=torch.long,
                device=device,
            ),
            torch.tensor(
                [action.row_a for action in rendered],
                dtype=torch.long,
                device=device,
            ),
            torch.tensor(
                [action.row_b for action in rendered],
                dtype=torch.long,
                device=device,
            ),
            torch.tensor(
                [action.column for action in rendered],
                dtype=torch.long,
                device=device,
            ),
            memory.recurrent,
        )
        current_key = self.encode_states(current_tensor)
        successor_keys = self.encode_states(successor_tensor)
        use_memory = mode in (
            MODE_REAL,
            MODE_SHUFFLED,
            MODE_BARRIER,
            MODE_BARRIER_SHUFFLED,
            MODE_EXACT_BARRIER,
            MODE_EXACT_BARRIER_SHUFFLED,
        ) and bool(memory.keys)
        if use_memory:
            keys = torch.stack(memory.keys[-self.config.memory_slots :])
            if mode in (MODE_SHUFFLED, MODE_BARRIER_SHUFFLED):
                keys = torch.roll(keys, shifts=1, dims=-1)
            similarity = successor_keys.float() @ keys.float().T
            maximum_similarity, indices = similarity.max(dim=1)
            retrieved = keys[indices]
            gate = torch.ones(
                len(rendered),
                1,
                dtype=successor_keys.dtype,
                device=device,
            )
        else:
            maximum_similarity = torch.zeros(
                len(rendered),
                dtype=torch.float32,
                device=device,
            )
            retrieved = torch.zeros_like(successor_keys)
            gate = torch.zeros(
                len(rendered),
                1,
                dtype=successor_keys.dtype,
                device=device,
            )
        current = current_key[None].expand_as(successor_keys)
        similarity_feature = maximum_similarity[:, None].to(successor_keys.dtype)
        fusion = torch.cat(
            (
                base.action_hidden,
                current,
                successor_keys,
                retrieved,
                successor_keys - retrieved,
                similarity_feature,
            ),
            dim=-1,
        )
        delta = self.memory_delta(fusion).squeeze(-1) * gate.squeeze(-1)
        revisit = self.revisit_head(
            torch.cat(
                (
                    current,
                    successor_keys,
                    retrieved,
                    similarity_feature,
                ),
                dim=-1,
            )
        ).squeeze(-1)
        cycle_evidence = torch.exp(
            -(
                (1.0 - maximum_similarity)
                .clamp_min(0.0)
                / self.config.barrier_temperature
            )
        ) * gate.squeeze(-1).float()
        if mode in (MODE_BARRIER, MODE_BARRIER_SHUFFLED):
            logits = (
                base.preference_logits.float()
                - self.config.barrier_penalty * cycle_evidence
            ).to(base.preference_logits.dtype)
        else:
            logits = base.preference_logits + delta
        exact_cycle_evidence = torch.zeros(
            len(rendered),
            dtype=torch.float32,
            device=device,
        )
        if (
            mode in (MODE_EXACT_BARRIER, MODE_EXACT_BARRIER_SHUFFLED)
            and memory.raw_states
        ):
            raw_memory = torch.tensor(
                memory.raw_states[-self.config.memory_slots :],
                dtype=torch.long,
                device=device,
            )
            if mode == MODE_EXACT_BARRIER_SHUFFLED:
                raw_memory = torch.roll(raw_memory, shifts=(1, 1), dims=(-2, -1))
            exact_cycle_evidence = (
                successor_tensor[:, None]
                .eq(raw_memory[None])
                .all(dim=(-1, -2))
                .any(dim=1)
                .float()
            )
            logits = (
                base.preference_logits.float()
                - self.config.barrier_penalty * exact_cycle_evidence
            ).to(base.preference_logits.dtype)
        return EpisodicScores(
            actions=rendered,
            successors=successors,
            logits=logits,
            revisit_logits=revisit,
            current_key=current_key,
            action_hidden=base.action_hidden,
            maximum_similarity=maximum_similarity,
            cycle_evidence=cycle_evidence,
            exact_cycle_evidence=exact_cycle_evidence,
            current_rows=matrix,
        )

    def advance(
        self,
        memory: EpisodicState,
        scores: EpisodicScores,
        selected_index: int,
    ) -> EpisodicState:
        keys = (*memory.keys, scores.current_key)
        if len(keys) > self.config.memory_slots:
            keys = keys[-self.config.memory_slots :]
        return EpisodicState(
            keys=keys,
            recurrent=scores.action_hidden[selected_index],
            raw_states=(
                *memory.raw_states,
                scores.current_rows,
            )[-self.config.memory_slots :],
        )


def _target_index(
    actions: Sequence[SuccessorAction],
    target: SuccessorAction,
) -> int:
    try:
        return tuple(actions).index(target)
    except ValueError as error:
        raise EpisodicMemoryError("target action is absent") from error


def _random_target_index(
    trajectory: ExpertTrajectory,
    step: int,
    actions: Sequence[SuccessorAction],
    *,
    seed: int,
) -> int:
    expert = _target_index(actions, trajectory.actions[step])
    alternatives = [index for index in range(len(actions)) if index != expert]
    if not alternatives:
        return expert
    digest = sha256(
        f"{seed}:{trajectory.sha256}:{step}".encode("ascii")
    ).digest()
    return alternatives[int.from_bytes(digest[:8], "big") % len(alternatives)]


@dataclass(frozen=True, slots=True)
class TrainingReceipt:
    optimizer_updates: int
    trajectory_presentations: int
    state_presentations: int
    action_candidates: int
    batch_schedule_sha256: str
    mean_loss: float
    final_loss: float
    wall_seconds: float
    peak_cuda_memory_bytes: int


def _autocast(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def train_full_trajectories(
    model: EpisodicAntiCycleController,
    trajectories: Sequence[ExpertTrajectory],
    *,
    mode: str,
    optimizer_updates: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    amp_bfloat16: bool,
) -> TrainingReceipt:
    """Train recurrent state and episodic memory on complete trajectories."""

    if mode not in (
        MODE_REAL,
        MODE_EXACT_BARRIER,
        MODE_CLASSIFIER,
        MODE_RANDOM,
    ):
        raise EpisodicMemoryError("unsupported training mode")
    updates = _positive_int(optimizer_updates, label="optimizer_updates")
    batch = _positive_int(batch_size, label="batch_size")
    if not trajectories:
        raise EpisodicMemoryError("training trajectories are empty")
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        fused=device.type == "cuda",
    )
    rng = random.Random(seed)
    order = list(range(len(trajectories)))
    cursor = len(order)
    losses = []
    schedule_digest = sha256()
    trajectory_presentations = state_presentations = candidates = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        batch_losses = []
        for _ in range(batch):
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            trajectory = trajectories[order[cursor]]
            cursor += 1
            schedule_digest.update(trajectory.sha256.encode("ascii"))
            schedule_digest.update(b"\n")
            memory = EpisodicState()
            visited: list[tuple[tuple[int, ...], ...]] = []
            step_losses = []
            for step, rows in enumerate(trajectory.states):
                if mode == MODE_REAL:
                    score_mode = MODE_REAL
                elif mode == MODE_EXACT_BARRIER:
                    score_mode = MODE_EXACT_BARRIER
                else:
                    score_mode = MODE_ZERO
                with _autocast(device, amp_bfloat16):
                    scores = model.score(rows, memory, mode=score_mode)
                    if mode == MODE_RANDOM:
                        target_index = _random_target_index(
                            trajectory,
                            step,
                            scores.actions,
                            seed=seed,
                        )
                    else:
                        target_index = _target_index(
                            scores.actions,
                            trajectory.actions[step],
                        )
                    action_loss = F.cross_entropy(
                        scores.logits[None],
                        torch.tensor([target_index], device=device),
                    )
                    if mode == MODE_REAL and visited:
                        revisit_targets = torch.tensor(
                            [
                                float(successor in visited)
                                for successor in scores.successors
                            ],
                            dtype=scores.revisit_logits.dtype,
                            device=device,
                        )
                        revisit_loss = F.binary_cross_entropy_with_logits(
                            scores.revisit_logits,
                            revisit_targets,
                        )
                        step_loss = action_loss + 0.25 * revisit_loss
                    else:
                        step_loss = action_loss
                step_losses.append(step_loss)
                memory = model.advance(memory, scores, target_index)
                visited.append(rows)
                state_presentations += 1
                candidates += len(scores.actions)
            batch_losses.append(torch.stack(step_losses).mean())
            trajectory_presentations += 1
        loss = torch.stack(batch_losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    peak = (
        torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0
    )
    return TrainingReceipt(
        optimizer_updates=updates,
        trajectory_presentations=trajectory_presentations,
        state_presentations=state_presentations,
        action_candidates=candidates,
        batch_schedule_sha256=schedule_digest.hexdigest(),
        mean_loss=sum(losses) / len(losses),
        final_loss=losses[-1],
        wall_seconds=time.perf_counter() - started,
        peak_cuda_memory_bytes=peak,
    )


@dataclass(frozen=True, slots=True)
class EpisodicRollout:
    halted: bool
    invalid: bool
    overlong: bool
    actions: tuple[SuccessorAction, ...]
    output_rows: tuple[tuple[int, ...], ...]
    model_decisions: int
    cycles: int


@torch.no_grad()
def autonomous_rollout(
    model: EpisodicAntiCycleController,
    rows: Iterable[Iterable[int]],
    *,
    mode: str,
    maximum_steps: int,
    renderer_seed: int | None = None,
) -> EpisodicRollout:
    """Run with raw matrices, model state, and the primitive VM only."""

    matrix = canonical_matrix(rows)
    memory = EpisodicState()
    emitted = []
    seen = set()
    cycles = 0
    model.eval()
    for step in range(maximum_steps):
        actions = list(enumerate_legal_actions(matrix))
        if renderer_seed is not None:
            digest = sha256(
                f"{renderer_seed}:{step}:{matrix_sha256(matrix)}".encode("ascii")
            ).digest()
            random.Random(int.from_bytes(digest[:8], "big")).shuffle(actions)
        scores = model.score(matrix, memory, mode=mode, actions=actions)
        selected_index = int(scores.logits.argmax())
        selected = scores.actions[selected_index]
        memory = model.advance(memory, scores, selected_index)
        emitted.append(selected)
        state_hash = matrix_sha256(matrix)
        cycles += int(state_hash in seen)
        seen.add(state_hash)
        if selected.kind == ACTION_HALT:
            return EpisodicRollout(
                True,
                False,
                False,
                tuple(emitted),
                matrix,
                len(emitted),
                cycles,
            )
        try:
            matrix = apply_action(matrix, selected)
        except ValueError:
            return EpisodicRollout(
                False,
                True,
                False,
                tuple(emitted),
                matrix,
                len(emitted),
                cycles,
            )
    return EpisodicRollout(
        False,
        False,
        True,
        tuple(emitted),
        matrix,
        len(emitted),
        cycles,
    )


@dataclass(frozen=True, slots=True)
class EvaluationReceipt:
    strict_certified: int
    total: int
    invalid: int
    overlong: int
    halted: int
    cycles: int
    model_decisions: int
    candidate_oracle_calls: int = 0
    candidate_search_calls: int = 0
    candidate_verifier_calls: int = 0

    @property
    def rate(self) -> float:
        return self.strict_certified / self.total if self.total else 0.0


def evaluate(
    model: EpisodicAntiCycleController,
    matrices: Sequence[tuple[tuple[int, ...], ...]],
    *,
    mode: str,
    maximum_steps: int,
    renderer_seed: int | None = None,
) -> EvaluationReceipt:
    certified = invalid = overlong = halted = cycles = decisions = 0
    for matrix in matrices:
        rollout = autonomous_rollout(
            model,
            matrix,
            mode=mode,
            maximum_steps=maximum_steps,
            renderer_seed=renderer_seed,
        )
        compatible = CandidateRollout(
            halted=rollout.halted,
            invalid=rollout.invalid,
            overlong=rollout.overlong,
            actions=rollout.actions,
            output_rows=rollout.output_rows,
            resources=ResourceCounts(
                schema="episodic_posthoc_adapter",
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
        cycles += rollout.cycles
        decisions += rollout.model_decisions
    return EvaluationReceipt(
        strict_certified=certified,
        total=len(matrices),
        invalid=invalid,
        overlong=overlong,
        halted=halted,
        cycles=cycles,
        model_decisions=decisions,
    )


def _state_sha256(model: nn.Module) -> str:
    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_experiment(args: argparse.Namespace) -> Mapping[str, object]:
    preparation_path = Path(args.preparation)
    artifact = _load_preparation(preparation_path)
    device = torch.device(args.device)
    base_config = proof.ControllerConfig(
        field_width=args.field_width,
        width=args.width,
        cell_hidden=args.cell_hidden,
        matrix_layers=args.matrix_layers,
        contract_hidden=args.contract_hidden,
        coordinate_harmonics=args.coordinate_harmonics,
    )
    config = EpisodicConfig(
        base=base_config,
        state_width=args.state_width,
        state_hidden=args.state_hidden,
        state_layers=args.state_layers,
        memory_slots=args.memory_slots,
    )
    evaluation_matrices = artifact.evaluation_matrices
    fresh_evaluation = args.evaluation_seed is not None
    if fresh_evaluation:
        excluded = {
            trajectory.states[0] for trajectory in artifact.trajectories
        } | set(artifact.evaluation_matrices)
        evaluation_matrices = generate_matrices(
            seed=args.evaluation_seed,
            count=args.evaluation_matrices,
            minimum_rows=artifact.evaluation_minimum_rows,
            maximum_rows=artifact.evaluation_maximum_rows,
            minimum_columns=artifact.evaluation_minimum_columns,
            maximum_columns=artifact.evaluation_maximum_columns,
            excluded=excluded,
        )
    torch.manual_seed(args.seed)
    template = EpisodicAntiCycleController(config)
    initial = {name: tensor.clone() for name, tensor in template.state_dict().items()}
    models = {}
    training = {}
    for mode in (
        MODE_REAL,
        MODE_EXACT_BARRIER,
        MODE_CLASSIFIER,
        MODE_RANDOM,
    ):
        torch.manual_seed(args.seed)
        model = EpisodicAntiCycleController(config).to(device)
        model.load_state_dict(initial)
        training[mode] = train_full_trajectories(
            model,
            artifact.trajectories,
            mode=mode,
            optimizer_updates=args.optimizer_updates,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            amp_bfloat16=args.amp_bfloat16,
        )
        models[mode] = model
    evaluations = {
        MODE_REAL: evaluate(
            models[MODE_REAL],
            evaluation_matrices,
            mode=MODE_REAL,
            maximum_steps=args.maximum_rollout_steps,
        ),
        MODE_ZERO: evaluate(
            models[MODE_REAL],
            evaluation_matrices,
            mode=MODE_ZERO,
            maximum_steps=args.maximum_rollout_steps,
        ),
        MODE_SHUFFLED: evaluate(
            models[MODE_REAL],
            evaluation_matrices,
            mode=MODE_SHUFFLED,
            maximum_steps=args.maximum_rollout_steps,
        ),
        MODE_CLASSIFIER: evaluate(
            models[MODE_CLASSIFIER],
            evaluation_matrices,
            mode=MODE_ZERO,
            maximum_steps=args.maximum_rollout_steps,
        ),
        MODE_RANDOM: evaluate(
            models[MODE_RANDOM],
            evaluation_matrices,
            mode=MODE_ZERO,
            maximum_steps=args.maximum_rollout_steps,
        ),
        "episodic_memory_shuffled_renderer": evaluate(
            models[MODE_REAL],
            evaluation_matrices,
            mode=MODE_REAL,
            maximum_steps=args.maximum_rollout_steps,
            renderer_seed=args.renderer_seed,
        ),
        "exact_trained_exact_barrier": evaluate(
            models[MODE_EXACT_BARRIER],
            evaluation_matrices,
            mode=MODE_EXACT_BARRIER,
            maximum_steps=args.maximum_rollout_steps,
        ),
        "exact_trained_feature_shuffled": evaluate(
            models[MODE_EXACT_BARRIER],
            evaluation_matrices,
            mode=MODE_EXACT_BARRIER_SHUFFLED,
            maximum_steps=args.maximum_rollout_steps,
        ),
        "exact_trained_barrier_off": evaluate(
            models[MODE_EXACT_BARRIER],
            evaluation_matrices,
            mode=MODE_ZERO,
            maximum_steps=args.maximum_rollout_steps,
        ),
    }
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=False)
    model_hashes = {}
    for mode, model in models.items():
        path = model_dir / f"{mode}.pt"
        torch.save(model.state_dict(), path)
        model_hashes[mode] = {
            "state_sha256": _state_sha256(model),
            "file_sha256": sha256(path.read_bytes()).hexdigest(),
        }
    result = {
        "schema": REPORT_SCHEMA,
        "status": STATUS,
        "seed": args.seed,
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "preparation_sha256": sha256(preparation_path.read_bytes()).hexdigest(),
        "preparation": {
            "trajectory_manifest_sha256": artifact.trajectory_manifest_sha256,
            "train_matrix_manifest_sha256": (
                artifact.train_matrix_manifest_sha256
            ),
            "evaluation_matrix_manifest_sha256": (
                matrix_manifest(evaluation_matrices)
            ),
            "fresh_evaluation_seed": args.evaluation_seed,
            "fresh_evaluation_board": fresh_evaluation,
            "trajectories": len(artifact.trajectories),
            "states": sum(
                len(trajectory.states) for trajectory in artifact.trajectories
            ),
            "oracle_calls": artifact.oracle_calls,
        },
        "config": {
            "controller": asdict(config),
            "optimizer_updates": args.optimizer_updates,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "maximum_rollout_steps": args.maximum_rollout_steps,
        },
        "controller_parameters": template.parameter_count,
        "complete_system_parameters": template.complete_system_parameters,
        "identical_initial_weights": True,
        "matched_parameter_budget": True,
        "matched_update_budget": True,
        "candidate_runtime": (
            "raw_matrix_plus_model_owned_recurrence_and_associative_memory"
        ),
        "candidate_oracle_calls": 0,
        "candidate_search_calls": 0,
        "candidate_verifier_calls": 0,
        "training": {
            mode: asdict(receipt) for mode, receipt in training.items()
        },
        "evaluations": {
            mode: {**asdict(receipt), "rate": receipt.rate}
            for mode, receipt in evaluations.items()
        },
        "model_hashes": model_hashes,
        "reasoning_claim_authorized": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_bytes(result) + b"\n")
    return result


def rescore_semantic_barrier(args: argparse.Namespace) -> Mapping[str, object]:
    """Evaluate a frozen treatment model on a fresh unseen board."""

    artifact = _load_preparation(Path(args.preparation))
    excluded = {
        trajectory.states[0] for trajectory in artifact.trajectories
    } | set(artifact.evaluation_matrices)
    evaluation = generate_matrices(
        seed=args.evaluation_seed,
        count=args.evaluation_matrices,
        minimum_rows=args.evaluation_minimum_rows,
        maximum_rows=args.evaluation_maximum_rows,
        minimum_columns=args.evaluation_minimum_columns,
        maximum_columns=args.evaluation_maximum_columns,
        excluded=excluded,
    )
    config = EpisodicConfig(
        base=proof.ControllerConfig(
            field_width=args.field_width,
            width=args.width,
            cell_hidden=args.cell_hidden,
            matrix_layers=args.matrix_layers,
            contract_hidden=args.contract_hidden,
            coordinate_harmonics=args.coordinate_harmonics,
        ),
        state_width=args.state_width,
        state_hidden=args.state_hidden,
        state_layers=args.state_layers,
        memory_slots=args.memory_slots,
        barrier_temperature=args.barrier_temperature,
        barrier_penalty=args.barrier_penalty,
    )
    model = EpisodicAntiCycleController(config).to(torch.device(args.device))
    state = torch.load(
        args.model,
        map_location=args.device,
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    evaluations = {
        mode: evaluate(
            model,
            evaluation,
            mode=mode,
            maximum_steps=args.maximum_rollout_steps,
        )
        for mode in (
            MODE_BARRIER,
            MODE_BARRIER_SHUFFLED,
            MODE_EXACT_BARRIER,
            MODE_EXACT_BARRIER_SHUFFLED,
            MODE_REAL,
            MODE_ZERO,
        )
    }
    result = {
        "schema": "ssqac_episodic_semantic_barrier_rescore_v1",
        "status": "fresh_board_semantic_barrier_falsifier_not_reasoning",
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_file_sha256": sha256(Path(args.model).read_bytes()).hexdigest(),
        "evaluation_seed": args.evaluation_seed,
        "evaluation_matrices": len(evaluation),
        "evaluation_matrix_manifest_sha256": matrix_manifest(evaluation),
        "training_board_excluded": True,
        "prior_evaluation_board_excluded": True,
        "config": asdict(config),
        "candidate_oracle_calls": 0,
        "candidate_search_calls": 0,
        "candidate_verifier_calls": 0,
        "evaluations": {
            mode: {**asdict(receipt), "rate": receipt.rate}
            for mode, receipt in evaluations.items()
        },
        "reasoning_claim_authorized": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_bytes(result) + b"\n")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--seed", type=int, default=20260724)
    prepare.add_argument("--train-matrices", type=int, default=256)
    prepare.add_argument("--evaluation-matrices", type=int, default=256)
    prepare.add_argument("--train-maximum-rows", type=int, default=4)
    prepare.add_argument("--train-maximum-columns", type=int, default=6)
    prepare.add_argument("--evaluation-minimum-rows", type=int, default=5)
    prepare.add_argument("--evaluation-minimum-columns", type=int, default=7)
    prepare.add_argument("--evaluation-maximum-rows", type=int, default=5)
    prepare.add_argument("--evaluation-maximum-columns", type=int, default=8)
    prepare.add_argument("--maximum-preparation-steps", type=int, default=64)
    prepare.add_argument("--output", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--preparation", required=True)
    run.add_argument("--seed", type=int, default=20260724)
    run.add_argument("--optimizer-updates", type=int, default=1500)
    run.add_argument("--batch-size", type=int, default=2)
    run.add_argument("--learning-rate", type=float, default=5e-4)
    run.add_argument("--maximum-rollout-steps", type=int, default=192)
    run.add_argument("--renderer-seed", type=int, default=99173)
    run.add_argument("--evaluation-seed", type=int)
    run.add_argument("--evaluation-matrices", type=int, default=512)
    run.add_argument("--field-width", type=int, default=64)
    run.add_argument("--width", type=int, default=384)
    run.add_argument("--cell-hidden", type=int, default=512)
    run.add_argument("--matrix-layers", type=int, default=4)
    run.add_argument("--contract-hidden", type=int, default=512)
    run.add_argument("--coordinate-harmonics", type=int, default=4)
    run.add_argument("--state-width", type=int, default=256)
    run.add_argument("--state-hidden", type=int, default=384)
    run.add_argument("--state-layers", type=int, default=3)
    run.add_argument("--memory-slots", type=int, default=32)
    run.add_argument("--device", default="cuda")
    run.add_argument(
        "--amp-bfloat16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    run.add_argument("--output", required=True)
    run.add_argument("--model-dir", required=True)
    rescore = subparsers.add_parser("rescore")
    rescore.add_argument("--preparation", required=True)
    rescore.add_argument("--model", required=True)
    rescore.add_argument("--evaluation-seed", type=int, required=True)
    rescore.add_argument("--evaluation-matrices", type=int, default=512)
    rescore.add_argument("--evaluation-minimum-rows", type=int, default=5)
    rescore.add_argument("--evaluation-maximum-rows", type=int, default=5)
    rescore.add_argument("--evaluation-minimum-columns", type=int, default=7)
    rescore.add_argument("--evaluation-maximum-columns", type=int, default=8)
    rescore.add_argument("--maximum-rollout-steps", type=int, default=192)
    rescore.add_argument("--field-width", type=int, default=64)
    rescore.add_argument("--width", type=int, default=384)
    rescore.add_argument("--cell-hidden", type=int, default=512)
    rescore.add_argument("--matrix-layers", type=int, default=4)
    rescore.add_argument("--contract-hidden", type=int, default=512)
    rescore.add_argument("--coordinate-harmonics", type=int, default=4)
    rescore.add_argument("--state-width", type=int, default=256)
    rescore.add_argument("--state-hidden", type=int, default=384)
    rescore.add_argument("--state-layers", type=int, default=3)
    rescore.add_argument("--memory-slots", type=int, default=32)
    rescore.add_argument("--barrier-temperature", type=float, default=0.02)
    rescore.add_argument("--barrier-penalty", type=float, default=8.0)
    rescore.add_argument("--device", default="cuda")
    rescore.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
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
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(artifact.canonical_bytes())
        print(
            json.dumps(
                {
                    "output": str(output),
                    "sha256": sha256(output.read_bytes()).hexdigest(),
                    "trajectories": len(artifact.trajectories),
                    "states": sum(
                        len(trajectory.states)
                        for trajectory in artifact.trajectories
                    ),
                },
                sort_keys=True,
            )
        )
        return
    if args.command == "rescore":
        result = rescore_semantic_barrier(args)
        print(
            json.dumps(
                {
                    "output": args.output,
                    "evaluations": result["evaluations"],
                },
                sort_keys=True,
            )
        )
        return
    result = run_experiment(args)
    print(
        json.dumps(
            {
                "output": args.output,
                "evaluations": result["evaluations"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
