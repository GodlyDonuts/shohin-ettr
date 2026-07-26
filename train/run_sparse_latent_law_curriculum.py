"""Train and evaluate the shared sparse latent-law compiler.

Exact parsing is preparation-only. Candidate evaluation scans raw bytes with
the restricted lexical interface, predicts a complete transition packet, then
executes the late query from that packet after source bytes are discarded.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

from source_deleted_sparse_latent_law_board import (  # noqa: E402
    FAMILIES,
    CandidateEpisode,
    GeneratedEpisode,
    build_frozen_board,
    compile_source,
    generate_episode,
)
from sparse_latent_law_compiler import (  # noqa: E402
    MAX_ACTIONS,
    MAX_CARDINALITY,
    FactorizedSparseLatentLawCompiler,
    MicrocodedSparseLatentLawCompiler,
    SparseLatentLawCompiler,
    SparseLawCompilerError,
    SparseSourceBatch,
    collate_sparse_sources,
    execute_sparse_query,
    scan_sparse_query,
    scan_sparse_source,
    seal_sparse_machine,
)


def _json_sha256(value: object, domain: bytes) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return sha256(domain + b"\0" + payload).hexdigest()


def _auxiliary_training_rows(
    *,
    seed: int,
    count: int,
) -> list[GeneratedEpisode]:
    topologies = ((8, 2), (8, 3), (16, 2), (16, 3))
    rows: list[GeneratedEpisode] = []
    for index in range(count):
        family = FAMILIES[index % len(FAMILIES)]
        renderer = (index // len(FAMILIES)) % 5
        cardinality, action_count = topologies[
            (index // (len(FAMILIES) * 5)) % len(topologies)
        ]
        rows.append(
            generate_episode(
                seed=seed + 10_000_000 + index,
                split="train",
                family=family,
                renderer=renderer,
                cell="fit",
                cardinality=cardinality,
                action_count=action_count,
            )
        )
    return rows


def _counterfactual_training_rows(
    rows: list[GeneratedEpisode],
) -> tuple[list[GeneratedEpisode], list[bool]]:
    counterfactual: list[GeneratedEpisode] = []
    directions: list[bool] = []
    for row in rows:
        machine = compile_source(row.candidate.source)
        templates = (
            (
                "from {source}, {target} is reached using {action}",
                True,
            ),
            (
                "using {action}, from {source} reaches {target}",
                True,
            ),
            (
                "using {action}, {target} is reached from {source}",
                False,
            ),
            (
                "{target}, using {action}, is reached from {source}",
                False,
            ),
        )
        for template, first_is_source in templates:
            records = [
                template.format(
                    source=source,
                    target=machine.transition[action][source],
                    action=machine.action_keys[action],
                )
                for action in range(len(machine.action_keys))
                for source in machine.visible_inputs[action]
            ]
            counterfactual.append(
                replace(
                    row,
                    candidate=CandidateEpisode(
                        source="\n".join(
                            [
                                f"domain-size={machine.cardinality}",
                                *records,
                            ]
                        ),
                        query=row.candidate.query,
                    ),
                )
            )
            directions.append(first_is_source)
    return counterfactual, directions


def _gray(value: int) -> int:
    return value ^ (value >> 1)


def _inverse_gray(value: int) -> int:
    result = value
    while value:
        value >>= 1
        result ^= value
    return result


def _program_parameters(
    *,
    family: str,
    transition: tuple[int, ...],
) -> tuple[int, int, int, int, int]:
    cardinality = len(transition)
    if family in {"affine_modular", "gray_conjugate_affine"}:
        family_index = 0 if family == "affine_modular" else 2
        for multiplier in range(1, cardinality):
            if math.gcd(multiplier, cardinality) != 1:
                continue
            for offset in range(cardinality):
                candidate = (
                    tuple(
                        (multiplier * state + offset) % cardinality
                        for state in range(cardinality)
                    )
                    if family_index == 0
                    else tuple(
                        _inverse_gray(
                            (
                                multiplier * _gray(state)
                                + offset
                            )
                            % cardinality
                        )
                        for state in range(cardinality)
                    )
                )
                if candidate == transition:
                    return family_index, multiplier, offset, 0, 0
    elif family == "bitwise_rotate_xor":
        width = int(math.log2(cardinality))
        mask_all = cardinality - 1
        for shift in range(width):
            for xor_mask in range(1, cardinality):
                candidate = tuple(
                    (
                        ((state ^ xor_mask) << shift)
                        | ((state ^ xor_mask) >> (width - shift))
                    )
                    & mask_all
                    for state in range(cardinality)
                )
                if candidate == transition:
                    return 1, 0, 0, shift, xor_mask
    raise ValueError("preparation microcode is not identifiable")


def _prepare(
    rows: list[GeneratedEpisode],
    *,
    device: torch.device,
    direction_overrides: list[bool] | None = None,
) -> tuple[
    SparseSourceBatch,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    if (
        direction_overrides is not None
        and len(direction_overrides) != len(rows)
    ):
        raise ValueError("direction override count differs")
    scanned = [
        scan_sparse_source(row.candidate.source.encode("ascii"))
        for row in rows
    ]
    batch = collate_sparse_sources(scanned, device=device)
    transition_target = torch.full(
        (len(rows), MAX_ACTIONS, MAX_CARDINALITY),
        -100,
        dtype=torch.long,
        device=device,
    )
    direction_target = torch.full(
        batch.record_valid.shape,
        -100.0,
        dtype=torch.float32,
        device=device,
    )
    program_target = {
        name: torch.full(
            (len(rows), MAX_ACTIONS),
            -100,
            dtype=torch.long,
            device=device,
        )
        for name in (
            "family",
            "multiplier",
            "offset",
            "shift",
            "mask",
        )
    }
    for row_index, (row, source) in enumerate(zip(rows, scanned, strict=True)):
        exact = compile_source(row.candidate.source)
        if tuple(key.decode("ascii") for key in source.action_keys) != (
            exact.action_keys
        ):
            raise ValueError("candidate/exact action order differs")
        for action, transition in enumerate(exact.transition):
            transition_target[
                row_index,
                action,
                : exact.cardinality,
            ] = torch.tensor(
                transition,
                dtype=torch.long,
                device=device,
            )
            values = _program_parameters(
                family=row.supervisor.family,
                transition=transition,
            )
            for name, value in zip(
                program_target,
                values,
                strict=True,
            ):
                program_target[name][row_index, action] = value
        count = len(source.records)
        first_is_source = (
            direction_overrides[row_index]
            if direction_overrides is not None
            else row.supervisor.renderer in {0, 1, 2, 3}
        )
        direction_target[row_index, :count] = float(first_is_source)
    return (
        batch,
        transition_target,
        direction_target,
        program_target,
    )


def _slice_batch(
    batch: SparseSourceBatch,
    indices: torch.Tensor,
) -> SparseSourceBatch:
    host_indices = indices.detach().cpu().tolist()
    return SparseSourceBatch(
        unit_ids=batch.unit_ids[indices],
        unit_valid=batch.unit_valid[indices],
        record_valid=batch.record_valid[indices],
        action_positions=batch.action_positions[indices],
        number_positions=batch.number_positions[indices],
        number_values=batch.number_values[indices],
        record_action_indices=batch.record_action_indices[indices],
        action_valid=batch.action_valid[indices],
        cardinalities=batch.cardinalities[indices],
        action_keys=tuple(batch.action_keys[index] for index in host_indices),
        source_sha256=tuple(
            batch.source_sha256[index] for index in host_indices
        ),
    )


def _train(
    *,
    model: SparseLatentLawCompiler,
    batch: SparseSourceBatch,
    transition_target: torch.Tensor,
    direction_target: torch.Tensor,
    program_target: dict[str, torch.Tensor],
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[float, float]:
    def masked_cross_entropy(
        logits: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(valid.any()):
            return logits.sum() * 0.0
        return F.cross_entropy(logits[valid], target[valid])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    generator = torch.Generator(device=batch.unit_ids.device)
    generator.manual_seed(seed)
    first_loss = 0.0
    final_loss = 0.0
    for step in range(steps):
        indices = torch.randint(
            len(batch.action_keys),
            (batch_size,),
            generator=generator,
            device=batch.unit_ids.device,
        )
        mini = _slice_batch(batch, indices)
        output = model(mini)
        target = transition_target[indices]
        transition_loss = F.cross_entropy(
            output.transition_logits.reshape(
                -1,
                MAX_CARDINALITY,
            ),
            target.reshape(-1),
            ignore_index=-100,
        )
        direction = direction_target[indices]
        direction_valid = direction.ne(-100)
        direction_loss = F.binary_cross_entropy_with_logits(
            output.direction_logits[direction_valid],
            direction[direction_valid],
        )
        probabilities = output.transition_logits.softmax(dim=-1)
        column_sums = probabilities.sum(dim=-2)
        column_valid = (
            mini.action_valid[..., None]
            & (
                torch.arange(
                    MAX_CARDINALITY,
                    device=batch.unit_ids.device,
                )[None, None]
                < mini.cardinalities[:, None, None]
            )
        )
        permutation_loss = (
            column_sums[column_valid] - 1.0
        ).square().mean()
        loss = (
            transition_loss
            + 0.25 * direction_loss
            + 0.10 * permutation_loss
        )
        if output.microcode is not None:
            family_target = program_target["family"][indices]
            family_valid = family_target.ne(-100)
            family_loss = masked_cross_entropy(
                output.microcode.family_logits,
                family_target,
                family_valid,
            )
            affine_valid = family_valid & family_target.ne(1)
            bitwise_valid = family_valid & family_target.eq(1)
            parameter_loss = (
                masked_cross_entropy(
                    output.microcode.multiplier_logits,
                    program_target["multiplier"][indices],
                    affine_valid,
                )
                + masked_cross_entropy(
                    output.microcode.offset_logits,
                    program_target["offset"][indices],
                    affine_valid,
                )
                + masked_cross_entropy(
                    output.microcode.shift_logits,
                    program_target["shift"][indices],
                    bitwise_valid,
                )
                + masked_cross_entropy(
                    output.microcode.mask_logits,
                    program_target["mask"][indices],
                    bitwise_valid,
                )
            )
            loss = loss + 0.50 * family_loss + 0.25 * parameter_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        value = float(loss.detach())
        if step == 0:
            first_loss = value
        final_loss = value
    return first_loss, final_loss


@torch.no_grad()
def _evaluate(
    *,
    model: SparseLatentLawCompiler,
    rows: list[GeneratedEpisode],
    batch: SparseSourceBatch,
    transition_target: torch.Tensor,
    direction_target: torch.Tensor,
    program_target: dict[str, torch.Tensor],
    control: str,
) -> dict[str, object]:
    if control == "treatment":
        output = model(batch)
    elif control == "direction_negated":
        output = model(batch, direction_sign=-1.0)
    elif control == "observation_targets_shifted":
        output = model(batch, observation_target_shift=1)
    elif control == "observations_zeroed":
        output = model(batch, observations_zeroed=True)
    else:
        raise ValueError("sparse evaluation control differs")
    direction_valid = direction_target.ne(-100)
    direction_exact = int(
        output.direction_logits.gt(0)[direction_valid]
        .eq(direction_target[direction_valid].bool())
        .sum()
    )
    direction_total = int(direction_valid.sum())
    state_predictions = output.transition_logits.argmax(dim=-1)
    state_valid = transition_target.ne(-100)
    state_exact = int(
        state_predictions[state_valid]
        .eq(transition_target[state_valid])
        .sum()
    )
    state_total = int(state_valid.sum())
    microcode_correct = 0
    microcode_total = 0
    if output.microcode is not None:
        family_target = program_target["family"]
        family_valid = family_target.ne(-100)
        family_prediction = output.microcode.family_logits.argmax(-1)
        multiplier_prediction = (
            output.microcode.multiplier_logits.argmax(-1)
        )
        offset_prediction = output.microcode.offset_logits.argmax(-1)
        shift_prediction = output.microcode.shift_logits.argmax(-1)
        mask_prediction = output.microcode.mask_logits.argmax(-1)
        affine = family_valid & family_target.ne(1)
        bitwise = family_valid & family_target.eq(1)
        program_correct = family_prediction.eq(family_target)
        program_correct &= (
            (~affine)
            | (
                multiplier_prediction.eq(program_target["multiplier"])
                & offset_prediction.eq(program_target["offset"])
            )
        )
        program_correct &= (
            (~bitwise)
            | (
                shift_prediction.eq(program_target["shift"])
                & mask_prediction.eq(program_target["mask"])
            )
        )
        microcode_correct = int(program_correct[family_valid].sum())
        microcode_total = int(family_valid.sum())
    exact = 0
    invalid = 0
    map_exact = 0
    source_deletion_passes = 0
    cell_exact: Counter[str] = Counter()
    cell_total: Counter[str] = Counter()
    family_exact: Counter[str] = Counter()
    family_total: Counter[str] = Counter()
    for index, row in enumerate(rows):
        cardinality = row.supervisor.cardinality
        action_count = row.supervisor.action_count
        map_correct = bool(
            state_predictions[
                index,
                :action_count,
                :cardinality,
            ].eq(
                transition_target[
                    index,
                    :action_count,
                    :cardinality,
                ]
            ).all()
        )
        map_exact += int(map_correct)
        cell_total[row.supervisor.cell] += 1
        family_total[row.supervisor.family] += 1
        try:
            machine = seal_sparse_machine(batch, output, row=index)
            wire = machine.deployed_wire()
            source_bytes = row.candidate.source.encode("ascii")
            source_absent = source_bytes not in wire
            del source_bytes
            query = scan_sparse_query(
                row.candidate.query.encode("ascii")
            )
            answer = execute_sparse_query(machine, query)
            correct = answer == row.supervisor.answer
            source_deletion_passes += int(source_absent)
        except SparseLawCompilerError:
            invalid += 1
            correct = False
        exact += int(correct)
        cell_exact[row.supervisor.cell] += int(correct)
        family_exact[row.supervisor.family] += int(correct)
    return {
        "cell_exact": {
            cell: {
                "correct": cell_exact[cell],
                "total": cell_total[cell],
            }
            for cell in sorted(cell_total)
        },
        "control": control,
        "direction_accuracy": direction_exact / direction_total,
        "direction_correct": direction_exact,
        "direction_total": direction_total,
        "exact": exact,
        "family_exact": {
            family: {
                "correct": family_exact[family],
                "total": family_total[family],
            }
            for family in sorted(family_total)
        },
        "invalid": invalid,
        "map_exact": map_exact,
        "microcode_accuracy": (
            microcode_correct / microcode_total
            if microcode_total
            else None
        ),
        "microcode_correct": microcode_correct,
        "microcode_total": microcode_total,
        "source_deletion_passes": source_deletion_passes,
        "state_accuracy": state_exact / state_total,
        "state_correct": state_exact,
        "state_total": state_total,
        "total": len(rows),
    }


def run_experiment(
    *,
    seed: int,
    steps: int,
    width: int,
    layers: int,
    heads: int,
    learning_rate: float,
    batch_size: int,
    auxiliary_rows: int,
    architecture: str,
    device: torch.device,
    model_output: Path | None = None,
) -> dict[str, object]:
    random.seed(seed)
    torch.manual_seed(seed)
    board = build_frozen_board(
        seed=seed,
        train_per_renderer=4,
        development_per_cell=4,
    )
    frozen_train = [
        row for row in board if row.supervisor.split == "train"
    ]
    development = [
        row for row in board if row.supervisor.split == "development"
    ]
    original_training = [
        *frozen_train,
        *_auxiliary_training_rows(seed=seed, count=auxiliary_rows),
    ]
    counterfactual, counterfactual_direction = (
        _counterfactual_training_rows(frozen_train)
    )
    training = [*original_training, *counterfactual]
    training_direction = [
        row.supervisor.renderer in {0, 1, 2, 3}
        for row in original_training
    ] + counterfactual_direction
    (
        train_batch,
        train_transition,
        train_direction,
        train_program,
    ) = _prepare(
        training,
        device=device,
        direction_overrides=training_direction,
    )
    (
        dev_batch,
        dev_transition,
        dev_direction,
        dev_program,
    ) = _prepare(
        development,
        device=device,
    )
    if architecture == "attention":
        model = SparseLatentLawCompiler(
            width=width,
            layers=layers,
            heads=heads,
        ).to(device)
    elif architecture == "factorized":
        model = FactorizedSparseLatentLawCompiler(
            width=width,
            layers=layers,
            heads=heads,
        ).to(device)
    elif architecture == "microcoded":
        model = MicrocodedSparseLatentLawCompiler(
            width=width,
            layers=layers,
            heads=heads,
        ).to(device)
    else:
        raise ValueError("sparse architecture differs")
    losses = _train(
        model=model,
        batch=train_batch,
        transition_target=train_transition,
        direction_target=train_direction,
        program_target=train_program,
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    model_receipt: dict[str, object] | None = None
    if model_output is not None:
        model_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "architecture": architecture,
                "heads": heads,
                "layers": layers,
                "seed": seed,
                "state_dict": {
                    name: tensor.detach().cpu()
                    for name, tensor in model.state_dict().items()
                },
                "width": width,
            },
            model_output,
        )
        model_receipt = {
            "path": str(model_output),
            "sha256": sha256(model_output.read_bytes()).hexdigest(),
        }
    development_report = {
        control: _evaluate(
            model=model,
            rows=development,
            batch=dev_batch,
            transition_target=dev_transition,
            direction_target=dev_direction,
            program_target=dev_program,
            control=control,
        )
        for control in (
            "treatment",
            "direction_negated",
            "observation_targets_shifted",
            "observations_zeroed",
        )
    }
    train_action_laws = {
        digest
        for row in training
        for digest in row.supervisor.action_law_sha256
    }
    development_action_laws = {
        digest
        for row in development
        for digest in row.supervisor.action_law_sha256
    }
    return {
        "board_manifest_sha256": _json_sha256(
            [
                {
                    "candidate": asdict(row.candidate),
                    "supervisor": asdict(row.supervisor),
                }
                for row in board
            ],
            b"SPARSE-LATENT-LAW-QUALIFICATION-BOARD-V1",
        ),
        "candidate_imports_exact_compiler": False,
        "candidate_architecture": architecture,
        "candidate_time_oracle_calls": 0,
        "candidate_time_search_calls": 0,
        "candidate_time_verifier_calls": 0,
        "development": development_report,
        "development_action_laws": len(development_action_laws),
        "device": str(device),
        "equal_budget": {
            "control_additional_updates": 0,
            "models_trained": 1,
            "optimizer_updates": steps,
            "same_weights_controls": True,
        },
        "model_receipt": model_receipt,
        "parameter_receipt": asdict(model.parameter_receipt()),
        "preparation_exact_parser_calls": len(training) + len(development),
        "seed": seed,
        "status": "sparse_latent_law_curriculum",
        "train": _evaluate(
            model=model,
            rows=training,
            batch=train_batch,
            transition_target=train_transition,
            direction_target=train_direction,
            program_target=train_program,
            control="treatment",
        ),
        "train_action_laws": len(train_action_laws),
        "train_development_action_law_overlap": len(
            train_action_laws & development_action_laws
        ),
        "training_loss": {
            "initial": losses[0],
            "final": losses[1],
        },
        "training_rows": {
            "auxiliary": auxiliary_rows,
            "counterfactual": len(counterfactual),
            "frozen": len(frozen_train),
            "total": len(training),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--auxiliary-rows", type=int, default=3_000)
    parser.add_argument(
        "--architecture",
        choices=("attention", "factorized", "microcoded"),
        default="microcoded",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-output", type=Path)
    args = parser.parse_args()
    report = run_experiment(
        seed=args.seed,
        steps=args.steps,
        width=args.width,
        layers=args.layers,
        heads=args.heads,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        auxiliary_rows=args.auxiliary_rows,
        architecture=args.architecture,
        device=torch.device(args.device),
        model_output=args.model_output,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="ascii")
    print(payload, end="")


if __name__ == "__main__":
    main()
