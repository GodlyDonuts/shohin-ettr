"""Train and audit the episode-local generator constraint compiler."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import random
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

from source_deleted_episodic_generator_law_board import (  # noqa: E402
    TRAIN_FAMILIES,
    CandidateEpisode,
    GeneratedEpisode,
    build_frozen_board,
    compile_source,
    generate_episode,
)
from episodic_generator_constraint_compiler import (  # noqa: E402
    EpisodicGeneratorConstraintCompiler,
    execute_episodic_generator_query,
    scan_episodic_generator_source,
    seal_episodic_generator_packet,
)
from sparse_latent_law_compiler import (  # noqa: E402
    MAX_ACTIONS,
    MAX_CARDINALITY,
    SparseLawCompilerError,
    SparseSourceBatch,
    collate_sparse_sources,
    scan_sparse_query,
)


def _counterfactual_rows(
    rows: list[GeneratedEpisode],
) -> tuple[list[GeneratedEpisode], list[bool]]:
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
    result: list[GeneratedEpisode] = []
    directions: list[bool] = []
    for row in rows:
        scanned = scan_episodic_generator_source(
            row.candidate.source.encode("ascii")
        )
        machine = compile_source(row.candidate.source)
        all_keys = tuple(
            key.decode("ascii") for key in scanned.action_keys
        )
        support_keys = tuple(
            key for key in all_keys if key not in machine.target_keys
        )
        maps = {
            **dict(
                zip(
                    support_keys,
                    row.supervisor.support_transition,
                    strict=True,
                )
            ),
            **dict(
                zip(
                    machine.target_keys,
                    row.supervisor.target_transition,
                    strict=True,
                )
            ),
        }
        visible = {
            **{
                key: tuple(range(row.supervisor.cardinality))
                for key in support_keys
            },
            **dict(
                zip(
                    machine.target_keys,
                    row.supervisor.target_visible_inputs,
                    strict=True,
                )
            ),
        }
        for template, first_is_source in templates:
            records = [
                template.format(
                    source=source,
                    target=maps[action][source],
                    action=action,
                )
                for action in all_keys
                for source in visible[action]
            ]
            result.append(
                replace(
                    row,
                    candidate=CandidateEpisode(
                        source="\n".join(
                            [
                                (
                                    "domain-size="
                                    f"{row.supervisor.cardinality}"
                                ),
                                *records,
                            ]
                        ),
                        query=row.candidate.query,
                    ),
                )
            )
            directions.append(first_is_source)
    return result, directions


def _auxiliary_rows(
    *,
    seed: int,
    count: int,
) -> list[GeneratedEpisode]:
    rows: list[GeneratedEpisode] = []
    for index in range(count):
        rows.append(
            generate_episode(
                seed=seed + 10_000_000 + index,
                split="train",
                family=TRAIN_FAMILIES[index % len(TRAIN_FAMILIES)],
                renderer=(
                    index // len(TRAIN_FAMILIES)
                ) % 5,
                cell="fit",
                cardinality=8,
            )
        )
    return rows


def _prepare(
    rows: list[GeneratedEpisode],
    *,
    device: torch.device,
    direction_overrides: list[bool] | None = None,
) -> tuple[SparseSourceBatch, torch.Tensor, torch.Tensor]:
    if (
        direction_overrides is not None
        and len(direction_overrides) != len(rows)
    ):
        raise ValueError("episodic direction count differs")
    scanned = [
        scan_episodic_generator_source(
            row.candidate.source.encode("ascii")
        )
        for row in rows
    ]
    batch = collate_sparse_sources(scanned, device=device)
    transition = torch.full(
        (len(rows), MAX_ACTIONS, MAX_CARDINALITY),
        -100,
        dtype=torch.long,
        device=device,
    )
    direction = torch.full(
        batch.record_valid.shape,
        -100.0,
        dtype=torch.float32,
        device=device,
    )
    for row_index, (row, source) in enumerate(
        zip(rows, scanned, strict=True)
    ):
        keys = tuple(
            key.decode("ascii") for key in source.action_keys
        )
        counts = Counter(
            record.action_key.decode("ascii")
            for record in source.records
        )
        support_keys = tuple(
            key
            for key in keys
            if counts[key] == row.supervisor.cardinality
        )
        target_keys = tuple(
            key
            for key in keys
            if counts[key] < row.supervisor.cardinality
        )
        if len(support_keys) != 2 or len(target_keys) != 2:
            raise ValueError("episodic action roles differ")
        map_by_key = {
            **dict(
                zip(
                    support_keys,
                    row.supervisor.support_transition,
                    strict=True,
                )
            ),
            **dict(
                zip(
                    target_keys,
                    row.supervisor.target_transition,
                    strict=True,
                )
            ),
        }
        for action, key in enumerate(keys):
            transition[
                row_index,
                action,
                : row.supervisor.cardinality,
            ] = torch.tensor(
                map_by_key[key],
                dtype=torch.long,
                device=device,
            )
        first_is_source = (
            direction_overrides[row_index]
            if direction_overrides is not None
            else row.supervisor.renderer in {0, 1, 2, 3}
        )
        direction[
            row_index,
            : len(source.records),
        ] = float(first_is_source)
    return batch, transition, direction


def _slice_batch(
    batch: SparseSourceBatch,
    indices: torch.Tensor,
) -> SparseSourceBatch:
    host = indices.detach().cpu().tolist()
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
        action_keys=tuple(batch.action_keys[index] for index in host),
        source_sha256=tuple(
            batch.source_sha256[index] for index in host
        ),
    )


def _permute_records(batch: SparseSourceBatch) -> SparseSourceBatch:
    order = torch.arange(
        batch.unit_ids.shape[1] - 1,
        -1,
        -1,
        device=batch.unit_ids.device,
    )
    return SparseSourceBatch(
        unit_ids=batch.unit_ids[:, order],
        unit_valid=batch.unit_valid[:, order],
        record_valid=batch.record_valid[:, order],
        action_positions=batch.action_positions[:, order],
        number_positions=batch.number_positions[:, order],
        number_values=batch.number_values[:, order],
        record_action_indices=batch.record_action_indices[:, order],
        action_valid=batch.action_valid,
        cardinalities=batch.cardinalities,
        action_keys=batch.action_keys,
        source_sha256=batch.source_sha256,
    )


def _delete_target_witness(batch: SparseSourceBatch) -> SparseSourceBatch:
    record_valid = batch.record_valid.clone()
    for row in range(record_valid.shape[0]):
        counts = torch.zeros(
            MAX_ACTIONS,
            dtype=torch.long,
            device=record_valid.device,
        )
        counts.scatter_add_(
            0,
            batch.record_action_indices[row],
            record_valid[row].long(),
        )
        targets = torch.nonzero(
            (counts > 0) & (counts < batch.cardinalities[row]),
            as_tuple=False,
        ).flatten()
        for action in targets.tolist():
            positions = torch.nonzero(
                record_valid[row]
                & batch.record_action_indices[row].eq(action),
                as_tuple=False,
            ).flatten()
            record_valid[row, positions[0]] = False
    return SparseSourceBatch(
        unit_ids=batch.unit_ids,
        unit_valid=batch.unit_valid,
        record_valid=record_valid,
        action_positions=batch.action_positions,
        number_positions=batch.number_positions,
        number_values=batch.number_values,
        record_action_indices=batch.record_action_indices,
        action_valid=batch.action_valid,
        cardinalities=batch.cardinalities,
        action_keys=batch.action_keys,
        source_sha256=batch.source_sha256,
    )


def _train(
    *,
    model: EpisodicGeneratorConstraintCompiler,
    batch: SparseSourceBatch,
    transition_target: torch.Tensor,
    direction_target: torch.Tensor,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[float, float]:
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
        transition_loss = F.cross_entropy(
            output.transition_logits.reshape(
                -1,
                MAX_CARDINALITY,
            ),
            transition_target[indices].reshape(-1),
            ignore_index=-100,
        )
        direction = direction_target[indices]
        valid = direction.ne(-100)
        direction_loss = F.binary_cross_entropy_with_logits(
            output.direction_logits[valid],
            direction[valid],
        )
        loss = transition_loss + 0.5 * direction_loss
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
    model: EpisodicGeneratorConstraintCompiler,
    rows: list[GeneratedEpisode],
    batch: SparseSourceBatch,
    transition_target: torch.Tensor,
    direction_target: torch.Tensor,
    control: str,
) -> dict[str, object]:
    evaluated_batch = batch
    if control == "treatment":
        output = model(batch)
    elif control == "direction_negated":
        output = model(batch, direction_sign=-1.0)
    elif control == "observations_shifted":
        output = model(batch, observation_target_shift=1)
    elif control == "observations_zeroed":
        output = model(batch, observations_zeroed=True)
    elif control == "record_order_reversed":
        evaluated_batch = _permute_records(batch)
        output = model(evaluated_batch)
    elif control == "support_order_reversed":
        output = model(batch, support_order_reversed=True)
    elif control == "support_semantics_deranged":
        output = model(batch, support_semantics_deranged=True)
    elif control == "target_witness_deleted":
        evaluated_batch = _delete_target_witness(batch)
        output = model(evaluated_batch)
    else:
        raise ValueError("episodic control differs")
    prediction = output.transition_logits.argmax(dim=-1)
    state_valid = transition_target.ne(-100)
    state_exact = int(
        prediction[state_valid].eq(
            transition_target[state_valid]
        ).sum()
    )
    direction_valid = direction_target.ne(-100)
    direction_exact = int(
        output.direction_logits.gt(0)[direction_valid].eq(
            direction_target[direction_valid].bool()
        ).sum()
    )
    exact = 0
    target_maps = 0
    invalid = 0
    source_deletion = 0
    support_deletion = 0
    cell_exact: Counter[str] = Counter()
    cell_total: Counter[str] = Counter()
    family_exact: Counter[str] = Counter()
    family_total: Counter[str] = Counter()
    for index, row in enumerate(rows):
        counts = evaluated_batch.record_valid[index].new_zeros(
            MAX_ACTIONS,
            dtype=torch.long,
        )
        counts.scatter_add_(
            0,
            evaluated_batch.record_action_indices[index],
            evaluated_batch.record_valid[index].long(),
        )
        target_indices = torch.nonzero(
            (counts > 0)
            & (counts < row.supervisor.cardinality),
            as_tuple=False,
        ).flatten()
        target_maps += int(
            target_indices.numel() == 2
            and all(
                prediction[
                    index,
                    action,
                    : row.supervisor.cardinality,
                ].eq(
                    transition_target[
                        index,
                        action,
                        : row.supervisor.cardinality,
                    ]
                ).all()
                for action in target_indices.tolist()
            )
        )
        cell_total[row.supervisor.cell] += 1
        family_total[row.supervisor.family] += 1
        try:
            packet = seal_episodic_generator_packet(
                evaluated_batch,
                output,
                row=index,
            )
            wire = packet.deployed_wire()
            source_bytes = row.candidate.source.encode("ascii")
            support_keys = set(
                evaluated_batch.action_keys[index]
            ) - set(packet.target_keys)
            source_deletion += int(source_bytes not in wire)
            support_deletion += int(
                all(key not in wire for key in support_keys)
            )
            query = scan_sparse_query(
                row.candidate.query.encode("ascii")
            )
            correct = (
                execute_episodic_generator_query(packet, query)
                == row.supervisor.answer
            )
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
        "direction_accuracy": (
            direction_exact / int(direction_valid.sum())
        ),
        "exact": exact,
        "family_exact": {
            family: {
                "correct": family_exact[family],
                "total": family_total[family],
            }
            for family in sorted(family_total)
        },
        "invalid": invalid,
        "source_deletion_passes": source_deletion,
        "state_accuracy": state_exact / int(state_valid.sum()),
        "support_deletion_passes": support_deletion,
        "target_map_exact": target_maps,
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
    device: torch.device,
    model_output: Path | None,
) -> dict[str, object]:
    random.seed(seed)
    torch.manual_seed(seed)
    board = list(build_frozen_board(seed=seed))
    frozen_train = [
        row for row in board if row.supervisor.split == "train"
    ]
    development = [
        row
        for row in board
        if row.supervisor.split == "development"
    ]
    auxiliary = _auxiliary_rows(
        seed=seed,
        count=auxiliary_rows,
    )
    originals = [*frozen_train, *auxiliary]
    counterfactual, counterfactual_directions = (
        _counterfactual_rows(frozen_train)
    )
    training = [*originals, *counterfactual]
    training_directions = [
        row.supervisor.renderer in {0, 1, 2, 3}
        for row in originals
    ] + counterfactual_directions
    train_batch, train_transition, train_direction = _prepare(
        training,
        device=device,
        direction_overrides=training_directions,
    )
    dev_batch, dev_transition, dev_direction = _prepare(
        development,
        device=device,
    )
    model = EpisodicGeneratorConstraintCompiler(
        width=width,
        layers=layers,
        heads=heads,
    ).to(device)
    losses = _train(
        model=model,
        batch=train_batch,
        transition_target=train_transition,
        direction_target=train_direction,
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
                "heads": heads,
                "layers": layers,
                "seed": seed,
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
                "width": width,
            },
            model_output,
        )
        model_receipt = {
            "path": str(model_output),
            "sha256": sha256(
                model_output.read_bytes()
            ).hexdigest(),
        }
    controls = (
        "treatment",
        "direction_negated",
        "observations_shifted",
        "observations_zeroed",
        "record_order_reversed",
        "support_order_reversed",
        "support_semantics_deranged",
        "target_witness_deleted",
    )
    train_target_laws = {
        digest
        for row in training
        for digest in row.supervisor.target_law_sha256
    }
    development_target_laws = {
        digest
        for row in development
        for digest in row.supervisor.target_law_sha256
    }
    return {
        "candidate_time_oracle_calls": 0,
        "candidate_time_search_calls": 0,
        "candidate_time_verifier_calls": 0,
        "closure_receipt": asdict(model.closure_receipt),
        "development": {
            control: _evaluate(
                model=model,
                rows=development,
                batch=dev_batch,
                transition_target=dev_transition,
                direction_target=dev_direction,
                control=control,
            )
            for control in controls
        },
        "device": str(device),
        "equal_budget": {
            "control_additional_updates": 0,
            "models_trained": 1,
            "optimizer_updates": steps,
            "same_weights_controls": True,
        },
        "model_receipt": model_receipt,
        "parameter_receipt": asdict(model.parameter_receipt()),
        "preparation_exact_parser_calls": (
            len(training) + len(development)
        ),
        "seed": seed,
        "status": "episodic_generator_constraint",
        "target_law_overlap": len(
            train_target_laws & development_target_laws
        ),
        "train": _evaluate(
            model=model,
            rows=training,
            batch=train_batch,
            transition_target=train_transition,
            direction_target=train_direction,
            control="treatment",
        ),
        "training_loss": {
            "final": losses[1],
            "initial": losses[0],
        },
        "training_rows": {
            "auxiliary": len(auxiliary),
            "counterfactual": len(counterfactual),
            "frozen": len(frozen_train),
            "total": len(training),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--auxiliary-rows", type=int, default=300)
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
