#!/usr/bin/env python3
"""Reactive DAgger falsifier for the primitive SSQAC algebra controller.

This is an isolated mechanics experiment, not a reasoning result.  A
preparation-only oracle labels expert and controller-reached states.  The
candidate runtime remains a step-free, memoryless neural controller plus the
primitive VM.  Final larger-geometry rollout is performed without invoking
the oracle.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Iterable, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from episode_functor_algebra_machine import (
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
    AlgebraMachineState,
    execute_program,
    verify_reduction_program,
)
from episode_functor_neural_algebra_controller import (
    PREVIOUS_START,
    ControllerConfig,
    ControllerLogits,
    NeuralAlgebraController,
    NeuralAlgebraControllerError,
    harden_controller_instruction,
)
from pipeline.ssqac_controller_trace_pilot import (
    TraceExample,
    generate_examples,
    next_reference_repair_instruction,
)


PILOT_SCHEMA = "ssqac_reactive_dagger_pilot_v1"
ROUND_SCHEMA = "ssqac_reactive_dagger_round_v1"
STATUS = "reactive_dagger_mechanics_falsifier_only_not_reasoning"
RUNTIME_BOUNDARY = "step_free_neural_controller_plus_primitive_vm_only"
ORACLE_BOUNDARY = (
    "preparation_and_on_policy_labeling_only_never_final_rollout"
)


@dataclass(frozen=True, slots=True)
class ReactiveStateLabel:
    """One deduplicated observable controller state and corrective label."""

    rows: tuple[tuple[int, ...], ...]
    registers: tuple[int, ...]
    previous_instruction: AlgebraInstruction | None
    target_instruction: AlgebraInstruction
    pre_error: bool = False

    def observation_data(self) -> list[object]:
        previous = (
            ["START", 0, 0, 0]
            if self.previous_instruction is None
            else self.previous_instruction.canonical_data()
        )
        return [
            [list(row) for row in self.rows],
            list(self.registers),
            previous,
        ]

    @property
    def observation_sha256(self) -> str:
        return sha256(
            json.dumps(
                self.observation_data(),
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()

    @property
    def labeled_sha256(self) -> str:
        return sha256(
            json.dumps(
                [
                    self.observation_data(),
                    self.target_instruction.canonical_data(),
                    self.pre_error,
                ],
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class MatrixCase:
    """Matrix-only candidate input with all expert traces deleted."""

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
class DAggerRoundReport:
    schema: str
    round_index: int
    rollout_matrices: int
    reached_valid_states: int
    unique_reached_states: int
    newly_added_states: int
    duplicate_states: int
    pre_error_states: int
    certified: int
    invalid: int
    overlong: int
    aggregate_states: int
    rollout_matrix_manifest_sha256: str
    reached_state_manifest_sha256: str
    aggregate_state_manifest_sha256: str
    optimizer_updates: int


@dataclass(frozen=True, slots=True)
class ReactiveDAggerReport:
    schema: str
    status: str
    candidate_runtime: str
    preparation_oracle_boundary: str
    final_rollout_oracle_calls: int
    seed: int
    expert_train_matrices: int
    expert_audit_matrices: int
    collection_rounds: int
    collection_matrices_per_round: int
    evaluation_matrices: int
    fit_maximum_rows: int
    fit_maximum_columns: int
    evaluation_minimum_rows: int
    evaluation_minimum_columns: int
    evaluation_maximum_rows: int
    evaluation_maximum_columns: int
    maximum_rollout_steps: int
    controller_parameters: int
    initial_expert_states: int
    final_aggregate_states: int
    initial_optimizer_updates: int
    dagger_optimizer_updates: int
    total_optimizer_updates: int
    initial_expert_instruction_accuracy: float
    final_expert_instruction_accuracy: float
    final_closed_loop_certified: int
    final_closed_loop_total: int
    final_invalid_programs: int
    final_overlong_programs: int
    expert_train_matrix_manifest_sha256: str
    expert_audit_matrix_manifest_sha256: str
    collection_matrix_manifest_sha256: str
    evaluation_matrix_manifest_sha256: str
    initial_state_manifest_sha256: str
    final_state_manifest_sha256: str
    initial_model_state_sha256: str
    final_model_state_sha256: str
    rounds: tuple[DAggerRoundReport, ...]

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
class RolloutCounts:
    certified: int
    invalid: int
    overlong: int
    oracle_calls: int = 0

    @property
    def total(self) -> int:
        return self.certified + self.invalid + self.overlong


@dataclass(frozen=True, slots=True)
class CollectionResult:
    states: tuple[ReactiveStateLabel, ...]
    reached_valid_states: int
    unique_reached_states: int
    duplicate_states: int
    pre_error_states: int
    counts: RolloutCounts


def _instruction_data(
    instruction: AlgebraInstruction | None,
) -> tuple[int, int, int, int]:
    if instruction is None:
        return PREVIOUS_START, 0, 0, 0
    return (
        OPCODES.index(instruction.opcode),
        instruction.a,
        instruction.b,
        instruction.c,
    )


def _matrix_manifest(
    examples: Iterable[TraceExample | MatrixCase],
) -> str:
    return sha256(
        (
            "\n".join(example.matrix_sha256 for example in examples) + "\n"
        ).encode("ascii")
    ).hexdigest()


def delete_expert_traces(
    examples: Iterable[TraceExample],
) -> tuple[MatrixCase, ...]:
    """Cross the candidate boundary by retaining matrices and nothing else."""

    return tuple(MatrixCase(matrix=example.matrix) for example in examples)


def _state_manifest(states: Iterable[ReactiveStateLabel]) -> str:
    return sha256(
        (
            "\n".join(
                state.labeled_sha256
                for state in sorted(
                    states,
                    key=lambda item: item.observation_sha256,
                )
            )
            + "\n"
        ).encode("ascii")
    ).hexdigest()


def _model_state_sha256(model: NeuralAlgebraController) -> str:
    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def flatten_expert_states(
    examples: Iterable[TraceExample],
) -> tuple[ReactiveStateLabel, ...]:
    """Flatten expert traces into a step-free reactive state dataset."""

    result: list[ReactiveStateLabel] = []
    for example in examples:
        if len(example.snapshots) != len(example.program):
            raise ValueError("expert trace snapshots and labels differ")
        previous: AlgebraInstruction | None = None
        for snapshot, target in zip(
            example.snapshots,
            example.program,
            strict=True,
        ):
            result.append(
                ReactiveStateLabel(
                    rows=snapshot.rows,
                    registers=snapshot.registers,
                    previous_instruction=previous,
                    target_instruction=target,
                )
            )
            previous = target
    return deduplicate_states(result)


def deduplicate_states(
    states: Iterable[ReactiveStateLabel],
) -> tuple[ReactiveStateLabel, ...]:
    """Deduplicate by candidate-visible state and previous instruction."""

    unique: dict[str, ReactiveStateLabel] = {}
    for state in states:
        key = state.observation_sha256
        prior = unique.get(key)
        if prior is None:
            unique[key] = state
            continue
        if prior.observation_data() != state.observation_data():
            raise RuntimeError("reactive observation digest collision")
        if prior.target_instruction != state.target_instruction:
            raise RuntimeError("preparation oracle labels conflict")
        if state.pre_error and not prior.pre_error:
            unique[key] = replace(prior, pre_error=True)
    return tuple(unique[key] for key in sorted(unique))


def merge_state_aggregates(
    aggregate: Iterable[ReactiveStateLabel],
    additions: Iterable[ReactiveStateLabel],
) -> tuple[tuple[ReactiveStateLabel, ...], int]:
    frozen_aggregate = tuple(aggregate)
    frozen_additions = tuple(additions)
    before = {
        state.observation_sha256
        for state in frozen_aggregate
    }
    merged = deduplicate_states(
        (*frozen_aggregate, *frozen_additions)
    )
    after = {state.observation_sha256 for state in merged}
    return merged, len(after - before)


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
        if len(result) >= count:
            break
        batch = generate_examples(
            seed=seed + 104_729 * attempt,
            count=max(8, count - len(result)),
            maximum_rows=maximum_rows,
            maximum_columns=maximum_columns,
            register_count=register_count,
            minimum_rows=minimum_rows,
            minimum_columns=minimum_columns,
        )
        for example in batch:
            if example.matrix in excluded:
                continue
            excluded.add(example.matrix)
            result.append(example)
            if len(result) == count:
                break
    if len(result) != count:
        raise RuntimeError("could not generate a disjoint matrix split")
    return tuple(result)


def _batch_inputs(
    states: Sequence[ReactiveStateLabel],
    config: ControllerConfig,
    device: torch.device,
) -> dict[str, Tensor]:
    batch = len(states)
    rows = torch.zeros(
        batch,
        config.maximum_rows,
        config.maximum_columns,
        dtype=torch.long,
        device=device,
    )
    registers = torch.zeros(
        batch,
        config.register_count,
        dtype=torch.long,
        device=device,
    )
    row_mask = torch.zeros(
        batch,
        config.maximum_rows,
        dtype=torch.bool,
        device=device,
    )
    column_mask = torch.zeros(
        batch,
        config.maximum_columns,
        dtype=torch.bool,
        device=device,
    )
    previous = []
    for index, state in enumerate(states):
        row_count = len(state.rows)
        column_count = len(state.rows[0])
        if row_count > config.maximum_rows or column_count > config.maximum_columns:
            raise ValueError("reactive state leaves controller geometry")
        if len(state.registers) != config.register_count:
            raise ValueError("reactive state register count differs")
        rows[index, :row_count, :column_count] = torch.tensor(
            state.rows,
            dtype=torch.long,
            device=device,
        )
        registers[index] = torch.tensor(
            state.registers,
            dtype=torch.long,
            device=device,
        )
        row_mask[index, :row_count] = True
        column_mask[index, :column_count] = True
        previous.append(_instruction_data(state.previous_instruction))
    previous_tensor = torch.tensor(previous, dtype=torch.long, device=device)
    return {
        "column_mask": column_mask,
        "previous_a": previous_tensor[:, 1],
        "previous_b": previous_tensor[:, 2],
        "previous_c": previous_tensor[:, 3],
        "previous_opcode": previous_tensor[:, 0],
        "registers": registers,
        "row_mask": row_mask,
        "rows": rows,
        # Deliberately constant: the candidate is reactive and step-free.
        "step": torch.zeros(batch, dtype=torch.long, device=device),
    }


def _weighted_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    weights: Tensor,
) -> Tensor:
    losses = F.cross_entropy(logits, targets, reduction="none")
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def _instruction_loss(
    logits: ControllerLogits,
    states: Sequence[ReactiveStateLabel],
    *,
    pre_error_weight: float,
) -> Tensor:
    device = logits.opcode.device
    targets = [state.target_instruction for state in states]
    weights = torch.tensor(
        [
            pre_error_weight if state.pre_error else 1.0
            for state in states
        ],
        dtype=torch.float32,
        device=device,
    )
    opcode_targets = torch.tensor(
        [OPCODES.index(target.opcode) for target in targets],
        dtype=torch.long,
        device=device,
    )
    losses = [
        _weighted_cross_entropy(logits.opcode, opcode_targets, weights)
    ]

    def add(
        head: Tensor,
        selected: list[int],
        values: list[int],
    ) -> None:
        if not selected:
            return
        indices = torch.tensor(selected, dtype=torch.long, device=device)
        target = torch.tensor(values, dtype=torch.long, device=device)
        losses.append(
            _weighted_cross_entropy(
                head[indices],
                target,
                weights[indices],
            )
        )

    row_a = [
        index
        for index, target in enumerate(targets)
        if target.opcode in (OP_LOAD, OP_SCALE, OP_AXPY, OP_SWAP)
    ]
    add(logits.row_a, row_a, [targets[index].a for index in row_a])
    row_b = [
        index
        for index, target in enumerate(targets)
        if target.opcode in (OP_AXPY, OP_SWAP)
    ]
    add(logits.row_b, row_b, [targets[index].b for index in row_b])
    columns = [
        index
        for index, target in enumerate(targets)
        if target.opcode == OP_LOAD
    ]
    add(logits.column, columns, [targets[index].b for index in columns])
    register_a = [
        index
        for index, target in enumerate(targets)
        if target.opcode in (OP_INV, OP_NEG, OP_SCALE, OP_AXPY, OP_LOAD)
    ]
    register_a_values = []
    for index in register_a:
        target = targets[index]
        if target.opcode in (OP_INV, OP_NEG):
            register_a_values.append(target.a)
        elif target.opcode == OP_SCALE:
            register_a_values.append(target.b)
        else:
            register_a_values.append(target.c)
    add(logits.register_a, register_a, register_a_values)
    register_b = [
        index
        for index, target in enumerate(targets)
        if target.opcode in (OP_INV, OP_NEG)
    ]
    add(
        logits.register_b,
        register_b,
        [targets[index].b for index in register_b],
    )
    return torch.stack(losses).mean()


def _single_logits(logits: ControllerLogits, index: int) -> ControllerLogits:
    return ControllerLogits(
        **{
            name: value[index : index + 1]
            for name, value in logits.as_mapping().items()
        }
    )


def _predict_batch(
    controller: NeuralAlgebraController,
    states: Sequence[ReactiveStateLabel],
    *,
    device: torch.device,
) -> tuple[AlgebraInstruction, ...]:
    inputs = _batch_inputs(states, controller.config, device)
    hidden = controller.initial_hidden(len(states), device=device)
    logits, _ = controller(hidden=hidden, **inputs)
    return tuple(
        harden_controller_instruction(
            _single_logits(logits, index),
            minimum_margin=0.0,
        ).instruction
        for index in range(len(states))
    )


def train_reactive_policy(
    controller: NeuralAlgebraController,
    states: Sequence[ReactiveStateLabel],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    pre_error_weight: float,
    device: torch.device,
    shuffle_seed: int,
) -> int:
    if not states:
        raise ValueError("reactive training dataset is empty")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    optimizer = torch.optim.AdamW(
        controller.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )
    generator = random.Random(shuffle_seed)
    updates = 0
    controller.train()
    for _ in range(epochs):
        order = list(range(len(states)))
        generator.shuffle(order)
        for offset in range(0, len(order), batch_size):
            batch = [
                states[index]
                for index in order[offset : offset + batch_size]
            ]
            inputs = _batch_inputs(batch, controller.config, device)
            hidden = controller.initial_hidden(len(batch), device=device)
            logits, _ = controller(hidden=hidden, **inputs)
            loss = _instruction_loss(
                logits,
                batch,
                pre_error_weight=pre_error_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
            optimizer.step()
            updates += 1
    return updates


@torch.no_grad()
def expert_instruction_accuracy(
    controller: NeuralAlgebraController,
    states: Sequence[ReactiveStateLabel],
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    controller.eval()
    correct = 0
    total = 0
    for offset in range(0, len(states), batch_size):
        batch = states[offset : offset + batch_size]
        predictions = _predict_batch(controller, batch, device=device)
        correct += sum(
            prediction == state.target_instruction
            for prediction, state in zip(predictions, batch, strict=True)
        )
        total += len(batch)
    return correct / total if total else 0.0


def _state_label(
    snapshot: AlgebraMachineState,
    previous: AlgebraInstruction | None,
    target: AlgebraInstruction,
    *,
    pre_error: bool,
) -> ReactiveStateLabel:
    return ReactiveStateLabel(
        rows=snapshot.rows,
        registers=snapshot.registers,
        previous_instruction=previous,
        target_instruction=target,
        pre_error=pre_error,
    )


def _predict_instruction(
    controller: NeuralAlgebraController,
    snapshot: AlgebraMachineState,
    previous: AlgebraInstruction | None,
    *,
    device: torch.device,
) -> AlgebraInstruction:
    state = _state_label(
        snapshot,
        previous,
        AlgebraInstruction(OP_HALT),
        pre_error=False,
    )
    return _predict_batch(controller, (state,), device=device)[0]


@torch.no_grad()
def collect_on_policy_states(
    controller: NeuralAlgebraController,
    examples: Sequence[MatrixCase],
    *,
    device: torch.device,
    maximum_rollout_steps: int,
) -> CollectionResult:
    """Collect oracle corrections only at controller-reached valid states."""

    controller.eval()
    reached: list[ReactiveStateLabel] = []
    certified = 0
    invalid = 0
    overlong = 0
    for example in examples:
        emitted: list[AlgebraInstruction] = []
        previous: AlgebraInstruction | None = None
        for _ in range(maximum_rollout_steps):
            snapshot = execute_program(
                example.matrix,
                emitted,
                register_count=controller.config.register_count,
            )
            target = next_reference_repair_instruction(snapshot, previous)
            try:
                instruction = _predict_instruction(
                    controller,
                    snapshot,
                    previous,
                    device=device,
                )
            except NeuralAlgebraControllerError:
                reached.append(
                    _state_label(
                        snapshot,
                        previous,
                        target,
                        pre_error=True,
                    )
                )
                invalid += 1
                break
            pre_error = False
            if instruction.opcode == OP_HALT:
                candidate = (*emitted, instruction)
                try:
                    final_state = execute_program(
                        example.matrix,
                        candidate,
                        register_count=controller.config.register_count,
                    )
                    verify_reduction_program(example.matrix, final_state)
                except AlgebraMachineError:
                    invalid += 1
                    pre_error = True
                else:
                    certified += 1
                reached.append(
                    _state_label(
                        snapshot,
                        previous,
                        target,
                        pre_error=pre_error,
                    )
                )
                break
            try:
                execute_program(
                    example.matrix,
                    (*emitted, instruction),
                    register_count=controller.config.register_count,
                )
            except AlgebraMachineError:
                invalid += 1
                pre_error = True
            reached.append(
                _state_label(
                    snapshot,
                    previous,
                    target,
                    pre_error=pre_error,
                )
            )
            if pre_error:
                break
            emitted.append(instruction)
            previous = instruction
        else:
            overlong += 1
    unique = deduplicate_states(reached)
    return CollectionResult(
        states=unique,
        reached_valid_states=len(reached),
        unique_reached_states=len(unique),
        duplicate_states=len(reached) - len(unique),
        pre_error_states=sum(state.pre_error for state in unique),
        counts=RolloutCounts(
            certified=certified,
            invalid=invalid,
            overlong=overlong,
            oracle_calls=len(reached),
        ),
    )


@torch.no_grad()
def final_oracle_free_rollout(
    controller: NeuralAlgebraController,
    examples: Sequence[MatrixCase],
    *,
    device: torch.device,
    maximum_rollout_steps: int,
) -> RolloutCounts:
    """Evaluate controller+VM with no preparation-oracle call path."""

    controller.eval()
    certified = 0
    invalid = 0
    overlong = 0
    for example in examples:
        emitted: list[AlgebraInstruction] = []
        previous: AlgebraInstruction | None = None
        for _ in range(maximum_rollout_steps):
            try:
                snapshot = execute_program(
                    example.matrix,
                    emitted,
                    register_count=controller.config.register_count,
                )
                instruction = _predict_instruction(
                    controller,
                    snapshot,
                    previous,
                    device=device,
                )
            except (AlgebraMachineError, NeuralAlgebraControllerError):
                invalid += 1
                break
            emitted.append(instruction)
            previous = instruction
            if instruction.opcode == OP_HALT:
                try:
                    final_state = execute_program(
                        example.matrix,
                        emitted,
                        register_count=controller.config.register_count,
                    )
                    verify_reduction_program(example.matrix, final_state)
                except AlgebraMachineError:
                    invalid += 1
                else:
                    certified += 1
                break
        else:
            overlong += 1
    return RolloutCounts(
        certified=certified,
        invalid=invalid,
        overlong=overlong,
        oracle_calls=0,
    )


def _validate_geometry(args: argparse.Namespace) -> None:
    if not 2 <= args.fit_maximum_rows < args.evaluation_minimum_rows:
        raise ValueError(
            "evaluation rows must be strictly larger than fit/collection rows"
        )
    if not 2 <= args.fit_maximum_columns < args.evaluation_minimum_columns:
        raise ValueError(
            "evaluation columns must be strictly larger than fit/collection columns"
        )
    if not args.evaluation_minimum_rows <= args.maximum_rows:
        raise ValueError("evaluation row minimum leaves controller geometry")
    if not args.evaluation_minimum_columns <= args.maximum_columns:
        raise ValueError("evaluation column minimum leaves controller geometry")
    if args.fit_maximum_columns < args.fit_maximum_rows:
        raise ValueError("fit columns must admit every fit row geometry")
    if args.maximum_columns < args.maximum_rows:
        raise ValueError("controller columns must admit every row geometry")
    if args.dagger_rounds < 1 or args.collection_examples < 1:
        raise ValueError("DAgger collection bounds must be positive")
    if args.maximum_rollout_steps < 1:
        raise ValueError("maximum rollout steps must be positive")
    if args.pre_error_weight <= 0.0:
        raise ValueError("pre-error weight must be positive")


def run_pilot(args: argparse.Namespace) -> ReactiveDAggerReport:
    _validate_geometry(args)
    torch.manual_seed(args.seed)
    config = ControllerConfig(
        maximum_rows=args.maximum_rows,
        maximum_columns=args.maximum_columns,
        register_count=args.registers,
        width=args.width,
        layers=args.layers,
        heads=args.heads,
        feedforward=args.feedforward,
        maximum_steps=1,
    )
    excluded: set[tuple[tuple[int, ...], ...]] = set()
    expert_train = _generate_disjoint_examples(
        seed=args.seed,
        count=args.expert_train_examples,
        maximum_rows=args.fit_maximum_rows,
        maximum_columns=args.fit_maximum_columns,
        register_count=config.register_count,
        minimum_rows=2,
        minimum_columns=2,
        excluded=excluded,
    )
    expert_audit = _generate_disjoint_examples(
        seed=args.seed + 1,
        count=args.expert_audit_examples,
        maximum_rows=args.fit_maximum_rows,
        maximum_columns=args.fit_maximum_columns,
        register_count=config.register_count,
        minimum_rows=2,
        minimum_columns=2,
        excluded=excluded,
    )
    collection_splits = tuple(
        _generate_disjoint_examples(
            seed=args.seed + 10 + round_index,
            count=args.collection_examples,
            maximum_rows=args.fit_maximum_rows,
            maximum_columns=args.fit_maximum_columns,
            register_count=config.register_count,
            minimum_rows=2,
            minimum_columns=2,
            excluded=excluded,
        )
        for round_index in range(args.dagger_rounds)
    )
    evaluation = _generate_disjoint_examples(
        seed=args.seed + 10_000,
        count=args.evaluation_examples,
        maximum_rows=config.maximum_rows,
        maximum_columns=config.maximum_columns,
        register_count=config.register_count,
        minimum_rows=args.evaluation_minimum_rows,
        minimum_columns=args.evaluation_minimum_columns,
        excluded=excluded,
    )
    collection_cases = tuple(
        delete_expert_traces(split)
        for split in collection_splits
    )
    evaluation_cases = delete_expert_traces(evaluation)

    expert_states = flatten_expert_states(expert_train)
    audit_states = flatten_expert_states(expert_audit)
    device = torch.device(args.device)
    controller = NeuralAlgebraController(config).to(device)
    initial_updates = train_reactive_policy(
        controller,
        expert_states,
        epochs=args.initial_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        pre_error_weight=args.pre_error_weight,
        device=device,
        shuffle_seed=args.seed + 20_000,
    )
    initial_accuracy = expert_instruction_accuracy(
        controller,
        audit_states,
        batch_size=args.batch_size,
        device=device,
    )
    initial_model_hash = _model_state_sha256(controller)
    aggregate = expert_states
    rounds: list[DAggerRoundReport] = []
    dagger_updates = 0
    for round_index, (split, cases) in enumerate(
        zip(collection_splits, collection_cases, strict=True),
        start=1,
    ):
        collection = collect_on_policy_states(
            controller,
            cases,
            device=device,
            maximum_rollout_steps=args.maximum_rollout_steps,
        )
        aggregate, newly_added = merge_state_aggregates(
            aggregate,
            collection.states,
        )
        round_updates = train_reactive_policy(
            controller,
            aggregate,
            epochs=args.dagger_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            pre_error_weight=args.pre_error_weight,
            device=device,
            shuffle_seed=args.seed + 30_000 + round_index,
        )
        dagger_updates += round_updates
        rounds.append(
            DAggerRoundReport(
                schema=ROUND_SCHEMA,
                round_index=round_index,
                rollout_matrices=len(split),
                reached_valid_states=collection.reached_valid_states,
                unique_reached_states=collection.unique_reached_states,
                newly_added_states=newly_added,
                duplicate_states=collection.duplicate_states,
                pre_error_states=collection.pre_error_states,
                certified=collection.counts.certified,
                invalid=collection.counts.invalid,
                overlong=collection.counts.overlong,
                aggregate_states=len(aggregate),
                rollout_matrix_manifest_sha256=_matrix_manifest(split),
                reached_state_manifest_sha256=_state_manifest(
                    collection.states
                ),
                aggregate_state_manifest_sha256=_state_manifest(aggregate),
                optimizer_updates=round_updates,
            )
        )

    final_accuracy = expert_instruction_accuracy(
        controller,
        audit_states,
        batch_size=args.batch_size,
        device=device,
    )
    final_counts = final_oracle_free_rollout(
        controller,
        evaluation_cases,
        device=device,
        maximum_rollout_steps=args.maximum_rollout_steps,
    )
    if final_counts.oracle_calls != 0:
        raise RuntimeError("final rollout crossed the preparation-oracle boundary")
    collection_examples = tuple(
        example
        for split in collection_splits
        for example in split
    )
    return ReactiveDAggerReport(
        schema=PILOT_SCHEMA,
        status=STATUS,
        candidate_runtime=RUNTIME_BOUNDARY,
        preparation_oracle_boundary=ORACLE_BOUNDARY,
        final_rollout_oracle_calls=final_counts.oracle_calls,
        seed=args.seed,
        expert_train_matrices=len(expert_train),
        expert_audit_matrices=len(expert_audit),
        collection_rounds=args.dagger_rounds,
        collection_matrices_per_round=args.collection_examples,
        evaluation_matrices=len(evaluation),
        fit_maximum_rows=args.fit_maximum_rows,
        fit_maximum_columns=args.fit_maximum_columns,
        evaluation_minimum_rows=args.evaluation_minimum_rows,
        evaluation_minimum_columns=args.evaluation_minimum_columns,
        evaluation_maximum_rows=config.maximum_rows,
        evaluation_maximum_columns=config.maximum_columns,
        maximum_rollout_steps=args.maximum_rollout_steps,
        controller_parameters=controller.parameter_count,
        initial_expert_states=len(expert_states),
        final_aggregate_states=len(aggregate),
        initial_optimizer_updates=initial_updates,
        dagger_optimizer_updates=dagger_updates,
        total_optimizer_updates=initial_updates + dagger_updates,
        initial_expert_instruction_accuracy=initial_accuracy,
        final_expert_instruction_accuracy=final_accuracy,
        final_closed_loop_certified=final_counts.certified,
        final_closed_loop_total=final_counts.total,
        final_invalid_programs=final_counts.invalid,
        final_overlong_programs=final_counts.overlong,
        expert_train_matrix_manifest_sha256=_matrix_manifest(expert_train),
        expert_audit_matrix_manifest_sha256=_matrix_manifest(expert_audit),
        collection_matrix_manifest_sha256=_matrix_manifest(
            collection_examples
        ),
        evaluation_matrix_manifest_sha256=_matrix_manifest(evaluation),
        initial_state_manifest_sha256=_state_manifest(expert_states),
        final_state_manifest_sha256=_state_manifest(aggregate),
        initial_model_state_sha256=initial_model_hash,
        final_model_state_sha256=_model_state_sha256(controller),
        rounds=tuple(rounds),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--expert-train-examples", type=int, default=256)
    parser.add_argument("--expert-audit-examples", type=int, default=64)
    parser.add_argument("--collection-examples", type=int, default=128)
    parser.add_argument("--dagger-rounds", type=int, default=4)
    parser.add_argument("--evaluation-examples", type=int, default=64)
    parser.add_argument("--fit-maximum-rows", type=int, default=3)
    parser.add_argument("--fit-maximum-columns", type=int, default=4)
    parser.add_argument("--evaluation-minimum-rows", type=int, default=4)
    parser.add_argument("--evaluation-minimum-columns", type=int, default=5)
    parser.add_argument("--maximum-rows", type=int, default=5)
    parser.add_argument("--maximum-columns", type=int, default=7)
    parser.add_argument("--registers", type=int, default=4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--feedforward", type=int, default=384)
    parser.add_argument("--initial-epochs", type=int, default=12)
    parser.add_argument("--dagger-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--pre-error-weight", type=float, default=2.0)
    parser.add_argument("--maximum-rollout-steps", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
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
