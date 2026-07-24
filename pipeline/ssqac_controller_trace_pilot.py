#!/usr/bin/env python3
"""Scoreless learned-controller trace pilot for primitive F_257 row reduction.

This is a preparation-only mechanics experiment. The reference scheduler is
used solely to generate training traces and is never part of the candidate
runtime. Closed-loop evaluation supplies the trained controller only the
current primitive-VM state and accepts a case only when the independent RREF
verifier certifies the emitted program.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Iterable, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from train.episode_functor_algebra_machine import (
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
    AlgebraMachineState,
    execute_program,
    verify_reduction_program,
)
from train.episode_functor_neural_algebra_controller import (
    PREVIOUS_START,
    ControllerConfig,
    ControllerLogits,
    NeuralAlgebraController,
    NeuralAlgebraControllerError,
    harden_controller_instruction,
)


PILOT_SCHEMA = "ssqac_controller_trace_pilot_v1"


@dataclass(frozen=True, slots=True)
class TraceExample:
    matrix: tuple[tuple[int, ...], ...]
    program: tuple[AlgebraInstruction, ...]
    snapshots: tuple[AlgebraMachineState, ...]

    @property
    def matrix_sha256(self) -> str:
        return sha256(
            json.dumps(
                self.matrix,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class PilotReport:
    schema: str
    status: str
    seed: int
    train_examples: int
    evaluation_examples: int
    train_maximum_rows: int
    train_maximum_columns: int
    evaluation_minimum_rows: int
    evaluation_minimum_columns: int
    evaluation_maximum_rows: int
    evaluation_maximum_columns: int
    controller_parameters: int
    optimizer_updates: int
    teacher_forced_instruction_accuracy: float
    closed_loop_certified: int
    closed_loop_total: int
    invalid_programs: int
    overlong_programs: int
    train_matrix_manifest_sha256: str
    evaluation_matrix_manifest_sha256: str
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


def _matrix_digest(matrices: Iterable[TraceExample]) -> str:
    payload = "\n".join(example.matrix_sha256 for example in matrices).encode(
        "ascii"
    )
    return sha256(payload).hexdigest()


def compile_reference_program(
    matrix: Iterable[Iterable[int]],
    *,
    register_count: int = 4,
    maximum_cycles: int = 4096,
) -> tuple[AlgebraInstruction, ...]:
    """Generate a self-stabilizing preparation-only RREF trace."""

    frozen = tuple(
        tuple(int(value) % FIELD_MODULUS for value in row)
        for row in matrix
    )
    if not frozen or not frozen[0] or any(
        len(row) != len(frozen[0]) for row in frozen
    ):
        raise ValueError("reference matrix must be nonempty and rectangular")
    program: list[AlgebraInstruction] = []
    previous: AlgebraInstruction | None = None
    for _ in range(maximum_cycles):
        state = execute_program(
            frozen,
            program,
            register_count=register_count,
        )
        instruction = next_reference_repair_instruction(
            state,
            previous,
        )
        program.append(instruction)
        previous = instruction
        if instruction.opcode == OP_HALT:
            return tuple(program)
    raise RuntimeError("reference repair policy exceeded its cycle bound")


def next_reference_repair_instruction(
    state: AlgebraMachineState,
    previous: AlgebraInstruction | None,
) -> AlgebraInstruction:
    """Return one state-local repair primitive.

    This preparation oracle has no persistent cursor. It identifies the
    already settled prefix directly from the current rows, then either repairs
    the next structural violation or completes a pending scalar macro whose
    operands are verified against the current registers.
    """

    rows = state.rows
    row_count = len(rows)
    column_count = len(rows[0])
    settled = 0
    previous_column = -1
    while settled < row_count:
        row = rows[settled]
        nonzero = tuple(index for index, value in enumerate(row) if value)
        if not nonzero:
            if any(any(value for value in later) for later in rows[settled + 1 :]):
                break
            settled += 1
            continue
        column = nonzero[0]
        if column <= previous_column or row[column] != 1:
            break
        if any(
            other != settled and rows[other][column]
            for other in range(row_count)
        ):
            break
        previous_column = column
        settled += 1
    if settled == row_count:
        return AlgebraInstruction(OP_HALT)

    pivot_column = next(
        (
            column
            for column in range(previous_column + 1, column_count)
            if any(rows[row][column] for row in range(settled, row_count))
        ),
        None,
    )
    if pivot_column is None:
        raise RuntimeError("unsettled matrix has no admissible pivot column")
    source = next(
        row
        for row in range(settled, row_count)
        if rows[row][pivot_column]
    )
    if source != settled:
        return AlgebraInstruction(OP_SWAP, settled, source)

    pivot_value = rows[settled][pivot_column]
    if pivot_value != 1:
        if (
            previous is not None
            and previous.opcode == OP_LOAD
            and (previous.a, previous.b, previous.c)
            == (settled, pivot_column, 0)
            and state.registers[0] == pivot_value
        ):
            return AlgebraInstruction(OP_INV, 0, 1)
        if (
            previous is not None
            and previous.opcode == OP_INV
            and (previous.a, previous.b) == (0, 1)
            and state.registers[1] * pivot_value % FIELD_MODULUS == 1
        ):
            return AlgebraInstruction(OP_SCALE, settled, 1)
        return AlgebraInstruction(OP_LOAD, settled, pivot_column, 0)

    destination = next(
        (
            row
            for row in range(row_count)
            if row != settled and rows[row][pivot_column]
        ),
        None,
    )
    if destination is None:
        raise RuntimeError("settled-prefix detector drifted")
    factor = rows[destination][pivot_column]
    if (
        previous is not None
        and previous.opcode == OP_LOAD
        and (previous.a, previous.b, previous.c)
        == (destination, pivot_column, 0)
        and state.registers[0] == factor
    ):
        return AlgebraInstruction(OP_NEG, 0, 2)
    if (
        previous is not None
        and previous.opcode == OP_NEG
        and (previous.a, previous.b) == (0, 2)
        and state.registers[2] == (-factor) % FIELD_MODULUS
    ):
        return AlgebraInstruction(OP_AXPY, destination, settled, 2)
    return AlgebraInstruction(OP_LOAD, destination, pivot_column, 0)


def build_trace_example(
    matrix: Iterable[Iterable[int]],
    *,
    register_count: int,
) -> TraceExample:
    frozen = tuple(tuple(int(value) for value in row) for row in matrix)
    program = compile_reference_program(
        frozen,
        register_count=register_count,
    )
    snapshots = tuple(
        execute_program(
            frozen,
            program[:index],
            register_count=register_count,
        )
        for index in range(len(program))
    )
    final_state = execute_program(
        frozen,
        program,
        register_count=register_count,
    )
    verify_reduction_program(frozen, final_state)
    return TraceExample(
        matrix=frozen,
        program=program,
        snapshots=snapshots,
    )


def generate_examples(
    *,
    seed: int,
    count: int,
    maximum_rows: int,
    maximum_columns: int,
    register_count: int,
    minimum_rows: int = 2,
    minimum_columns: int = 2,
) -> tuple[TraceExample, ...]:
    rng = random.Random(seed)
    if not 2 <= minimum_rows <= maximum_rows:
        raise ValueError("minimum rows leave the generation bounds")
    if not 2 <= minimum_columns <= maximum_columns:
        raise ValueError("minimum columns leave the generation bounds")
    examples = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    while len(examples) < count:
        row_count = rng.randint(minimum_rows, maximum_rows)
        column_count = rng.randint(
            max(row_count, minimum_columns),
            maximum_columns,
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
        examples.append(
            build_trace_example(
                matrix,
                register_count=register_count,
            )
        )
    return tuple(examples)


def _previous_instruction(
    example: TraceExample,
    step: int,
) -> tuple[int, int, int, int]:
    if step == 0:
        return PREVIOUS_START, 0, 0, 0
    instruction = example.program[step - 1]
    return (
        OPCODES.index(instruction.opcode),
        instruction.a,
        instruction.b,
        instruction.c,
    )


def _batch_state(
    examples: Sequence[TraceExample],
    step: int,
    config: ControllerConfig,
    device: torch.device,
) -> tuple[dict[str, Tensor], Tensor]:
    batch = len(examples)
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
    active = torch.tensor(
        [step < len(example.program) for example in examples],
        dtype=torch.bool,
        device=device,
    )
    previous = []
    for index, example in enumerate(examples):
        trace_step = min(step, len(example.program) - 1)
        snapshot = example.snapshots[trace_step]
        row_count = len(snapshot.rows)
        column_count = len(snapshot.rows[0])
        rows[index, :row_count, :column_count] = torch.tensor(
            snapshot.rows,
            dtype=torch.long,
            device=device,
        )
        registers[index] = torch.tensor(
            snapshot.registers,
            dtype=torch.long,
            device=device,
        )
        row_mask[index, :row_count] = True
        column_mask[index, :column_count] = True
        previous.append(_previous_instruction(example, trace_step))
    previous_tensor = torch.tensor(previous, dtype=torch.long, device=device)
    inputs = {
        "column_mask": column_mask,
        "previous_a": previous_tensor[:, 1],
        "previous_b": previous_tensor[:, 2],
        "previous_c": previous_tensor[:, 3],
        "previous_opcode": previous_tensor[:, 0],
        "registers": registers,
        "row_mask": row_mask,
        "rows": rows,
        "step": torch.full(
            (batch,),
            step,
            dtype=torch.long,
            device=device,
        ),
    }
    return inputs, active


def _instruction_loss(
    logits: ControllerLogits,
    examples: Sequence[TraceExample],
    step: int,
    active: Tensor,
) -> tuple[Tensor, int, int]:
    device = logits.opcode.device
    indices = active.nonzero(as_tuple=False).flatten()
    targets = [examples[index].program[step] for index in indices.tolist()]
    opcode_targets = torch.tensor(
        [OPCODES.index(target.opcode) for target in targets],
        dtype=torch.long,
        device=device,
    )
    loss = F.cross_entropy(logits.opcode[indices], opcode_targets)
    correct = int(
        (logits.opcode[indices].argmax(dim=-1) == opcode_targets).sum().item()
    )
    decisions = len(targets)

    def add(head: Tensor, values: list[int], selected: list[int]) -> None:
        nonlocal loss, correct, decisions
        if not selected:
            return
        selected_tensor = torch.tensor(selected, dtype=torch.long, device=device)
        target_tensor = torch.tensor(values, dtype=torch.long, device=device)
        chosen = indices[selected_tensor]
        loss = loss + F.cross_entropy(head[chosen], target_tensor)
        correct += int((head[chosen].argmax(dim=-1) == target_tensor).sum().item())
        decisions += len(selected)

    row_a_selected = [
        index
        for index, target in enumerate(targets)
        if target.opcode in (OP_LOAD, OP_SCALE, OP_AXPY, OP_SWAP)
    ]
    add(
        logits.row_a,
        [targets[index].a for index in row_a_selected],
        row_a_selected,
    )
    row_b_selected = [
        index
        for index, target in enumerate(targets)
        if target.opcode in (OP_AXPY, OP_SWAP)
    ]
    add(
        logits.row_b,
        [targets[index].b for index in row_b_selected],
        row_b_selected,
    )
    column_selected = [
        index for index, target in enumerate(targets) if target.opcode == OP_LOAD
    ]
    add(
        logits.column,
        [targets[index].b for index in column_selected],
        column_selected,
    )
    register_a_selected = [
        index
        for index, target in enumerate(targets)
        if target.opcode in (OP_INV, OP_NEG, OP_SCALE, OP_AXPY)
    ]
    register_a_values = [
        (
            targets[index].a
            if targets[index].opcode in (OP_INV, OP_NEG)
            else (
                targets[index].b
                if targets[index].opcode == OP_SCALE
                else targets[index].c
            )
        )
        for index in register_a_selected
    ]
    add(
        logits.register_a,
        register_a_values,
        register_a_selected,
    )
    register_b_selected = [
        index
        for index, target in enumerate(targets)
        if target.opcode in (OP_INV, OP_NEG)
    ]
    add(
        logits.register_b,
        [targets[index].b for index in register_b_selected],
        register_b_selected,
    )
    load_register_selected = [
        index for index, target in enumerate(targets) if target.opcode == OP_LOAD
    ]
    add(
        logits.register_a,
        [targets[index].c for index in load_register_selected],
        load_register_selected,
    )
    return loss, correct, decisions


def train_controller(
    controller: NeuralAlgebraController,
    examples: Sequence[TraceExample],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> int:
    optimizer = torch.optim.AdamW(
        controller.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )
    generator = random.Random(20260724)
    updates = 0
    controller.train()
    for _ in range(epochs):
        order = list(range(len(examples)))
        generator.shuffle(order)
        for offset in range(0, len(order), batch_size):
            batch = [examples[index] for index in order[offset : offset + batch_size]]
            hidden = controller.initial_hidden(len(batch), device=device)
            maximum_steps = max(len(example.program) for example in batch)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for step in range(maximum_steps):
                inputs, active = _batch_state(
                    batch,
                    step,
                    controller.config,
                    device,
                )
                logits, next_hidden = controller(hidden=hidden, **inputs)
                hidden = torch.where(active[:, None], next_hidden, hidden)
                if active.any():
                    loss, _, _ = _instruction_loss(
                        logits,
                        batch,
                        step,
                        active,
                    )
                    losses.append(loss)
            torch.stack(losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
            optimizer.step()
            updates += 1
    return updates


@torch.no_grad()
def teacher_forced_accuracy(
    controller: NeuralAlgebraController,
    examples: Sequence[TraceExample],
    *,
    device: torch.device,
) -> float:
    controller.eval()
    correct = 0
    total = 0
    hidden = controller.initial_hidden(len(examples), device=device)
    maximum_steps = max(len(example.program) for example in examples)
    for step in range(maximum_steps):
        inputs, active = _batch_state(
            examples,
            step,
            controller.config,
            device,
        )
        logits, next_hidden = controller(hidden=hidden, **inputs)
        hidden = torch.where(active[:, None], next_hidden, hidden)
        if active.any():
            _, step_correct, step_total = _instruction_loss(
                logits,
                examples,
                step,
                active,
            )
            correct += step_correct
            total += step_total
    return correct / total if total else 0.0


@torch.no_grad()
def closed_loop_evaluate(
    controller: NeuralAlgebraController,
    examples: Sequence[TraceExample],
    *,
    device: torch.device,
    cycle_multiplier: int = 2,
) -> tuple[int, int, int]:
    controller.eval()
    certified = 0
    invalid = 0
    overlong = 0
    for example in examples:
        hidden = controller.initial_hidden(1, device=device)
        emitted: list[AlgebraInstruction] = []
        limit = cycle_multiplier * len(example.program)
        previous = (PREVIOUS_START, 0, 0, 0)
        for step in range(limit):
            try:
                snapshot = execute_program(
                    example.matrix,
                    emitted,
                    register_count=controller.config.register_count,
                )
            except AlgebraMachineError:
                invalid += 1
                break
            rows = torch.zeros(
                1,
                controller.config.maximum_rows,
                controller.config.maximum_columns,
                dtype=torch.long,
                device=device,
            )
            row_count = len(snapshot.rows)
            column_count = len(snapshot.rows[0])
            rows[0, :row_count, :column_count] = torch.tensor(
                snapshot.rows,
                dtype=torch.long,
                device=device,
            )
            row_mask = torch.zeros(
                1,
                controller.config.maximum_rows,
                dtype=torch.bool,
                device=device,
            )
            row_mask[0, :row_count] = True
            column_mask = torch.zeros(
                1,
                controller.config.maximum_columns,
                dtype=torch.bool,
                device=device,
            )
            column_mask[0, :column_count] = True
            logits, hidden = controller(
                rows=rows,
                registers=torch.tensor(
                    snapshot.registers,
                    dtype=torch.long,
                    device=device,
                )[None, :],
                row_mask=row_mask,
                column_mask=column_mask,
                previous_opcode=torch.tensor([previous[0]], device=device),
                previous_a=torch.tensor([previous[1]], device=device),
                previous_b=torch.tensor([previous[2]], device=device),
                previous_c=torch.tensor([previous[3]], device=device),
                step=torch.tensor([step], device=device),
                hidden=hidden,
            )
            try:
                instruction = harden_controller_instruction(
                    logits,
                    minimum_margin=0.0,
                ).instruction
            except NeuralAlgebraControllerError:
                invalid += 1
                break
            emitted.append(instruction)
            previous = (
                OPCODES.index(instruction.opcode),
                instruction.a,
                instruction.b,
                instruction.c,
            )
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
    return certified, invalid, overlong


def _model_state_sha256(model: NeuralAlgebraController) -> str:
    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_pilot(args: argparse.Namespace) -> PilotReport:
    torch.manual_seed(args.seed)
    config = ControllerConfig(
        maximum_rows=args.maximum_rows,
        maximum_columns=args.maximum_columns,
        register_count=args.registers,
        width=args.width,
        layers=args.layers,
        heads=args.heads,
        feedforward=args.feedforward,
        maximum_steps=args.maximum_steps,
    )
    if not 2 <= args.train_maximum_rows <= config.maximum_rows:
        raise ValueError("training row geometry leaves controller bounds")
    if not 2 <= args.train_maximum_columns <= config.maximum_columns:
        raise ValueError("training column geometry leaves controller bounds")
    train = generate_examples(
        seed=args.seed,
        count=args.train_examples,
        maximum_rows=args.train_maximum_rows,
        maximum_columns=args.train_maximum_columns,
        register_count=config.register_count,
    )
    evaluation = generate_examples(
        seed=args.seed + 1,
        count=args.evaluation_examples,
        maximum_rows=config.maximum_rows,
        maximum_columns=config.maximum_columns,
        register_count=config.register_count,
        minimum_rows=args.evaluation_minimum_rows,
        minimum_columns=args.evaluation_minimum_columns,
    )
    if {example.matrix for example in train} & {
        example.matrix for example in evaluation
    }:
        raise RuntimeError("train/evaluation matrix overlap")
    device = torch.device(args.device)
    controller = NeuralAlgebraController(config).to(device)
    updates = train_controller(
        controller,
        train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
    )
    accuracy = teacher_forced_accuracy(
        controller,
        evaluation,
        device=device,
    )
    certified, invalid, overlong = closed_loop_evaluate(
        controller,
        evaluation,
        device=device,
    )
    return PilotReport(
        schema=PILOT_SCHEMA,
        status="scoreless_mechanics_only_not_reasoning",
        seed=args.seed,
        train_examples=len(train),
        evaluation_examples=len(evaluation),
        train_maximum_rows=args.train_maximum_rows,
        train_maximum_columns=args.train_maximum_columns,
        evaluation_minimum_rows=args.evaluation_minimum_rows,
        evaluation_minimum_columns=args.evaluation_minimum_columns,
        evaluation_maximum_rows=config.maximum_rows,
        evaluation_maximum_columns=config.maximum_columns,
        controller_parameters=controller.parameter_count,
        optimizer_updates=updates,
        teacher_forced_instruction_accuracy=accuracy,
        closed_loop_certified=certified,
        closed_loop_total=len(evaluation),
        invalid_programs=invalid,
        overlong_programs=overlong,
        train_matrix_manifest_sha256=_matrix_digest(train),
        evaluation_matrix_manifest_sha256=_matrix_digest(evaluation),
        model_state_sha256=_model_state_sha256(controller),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--train-examples", type=int, default=128)
    parser.add_argument("--evaluation-examples", type=int, default=32)
    parser.add_argument("--maximum-rows", type=int, default=4)
    parser.add_argument("--maximum-columns", type=int, default=6)
    parser.add_argument("--train-maximum-rows", type=int)
    parser.add_argument("--train-maximum-columns", type=int)
    parser.add_argument("--evaluation-minimum-rows", type=int, default=2)
    parser.add_argument("--evaluation-minimum-columns", type=int, default=2)
    parser.add_argument("--registers", type=int, default=4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--feedforward", type=int, default=128)
    parser.add_argument("--maximum-steps", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.train_maximum_rows is None:
        args.train_maximum_rows = args.maximum_rows
    if args.train_maximum_columns is None:
        args.train_maximum_columns = args.maximum_columns
    report = run_pilot(args)
    payload = report.canonical_bytes()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
    print(payload.decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
