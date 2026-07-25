"""Learned semantic typing for the source-deleted variable-topology board.

Incidence typing is used only when state and action frequencies differ. In
collision geometries the shared byte compiler must infer all three semantic
roles. Counterfactual preparation supplies a target/source/action renderer
under symbols absent from the held-out renderer. Matched controls swap either
direction or state/action type labels while preserving bytes and compute.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from dataclasses import asdict, dataclass
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

from source_deleted_variable_topology_board import (  # noqa: E402
    CandidateEpisode,
    FAMILIES,
    GeneratedEpisode,
    build_frozen_board,
    compile_source,
    decode_query,
)
from multifamily_raw_machine_compiler import (  # noqa: E402
    MAX_QUERY_OCCURRENCES,
    MAX_RECORDS,
    QUERY_ACTION,
    QUERY_START,
    ROLE_ACTION,
    ROLE_SOURCE,
    ROLE_TARGET,
    CompilerOutput,
    QueryOutput,
    SharedRawMachineCompiler,
    collate_queries,
    collate_sources,
    execute_query,
    scan_query,
    scan_source,
    seal_machine,
)


def _json_sha256(value: object, domain: bytes) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return sha256(domain + b"\0" + payload).hexdigest()


@dataclass(frozen=True, slots=True)
class LabeledEpisode:
    row: GeneratedEpisode
    source_roles: tuple[int, int, int]
    query_actions_first: bool
    counterfactual: bool


def _source_roles(renderer: int) -> tuple[int, int, int]:
    if renderer in {0, 3}:
        return (ROLE_SOURCE, ROLE_ACTION, ROLE_TARGET)
    if renderer in {1, 2}:
        return (ROLE_ACTION, ROLE_SOURCE, ROLE_TARGET)
    if renderer == 4:
        return (ROLE_TARGET, ROLE_ACTION, ROLE_SOURCE)
    if renderer == 5:
        return (ROLE_TARGET, ROLE_SOURCE, ROLE_ACTION)
    raise ValueError("renderer leaves role contract")


def _base_example(row: GeneratedEpisode) -> LabeledEpisode:
    renderer = row.supervisor.renderer
    return LabeledEpisode(
        row=row,
        source_roles=_source_roles(renderer),
        query_actions_first=renderer in {4, 5},
        counterfactual=False,
    )


def _counterfactual_examples(row: GeneratedEpisode) -> list[LabeledEpisode]:
    machine = compile_source(row.candidate.source)
    start, actions = decode_query(machine, row.candidate.query)
    word = ",".join(machine.action_keys[action] for action in actions)
    record_templates = (
        (
            "from {source}; reaches {target}; using {action}",
            (ROLE_SOURCE, ROLE_TARGET, ROLE_ACTION),
        ),
        (
            "using {action}; reaches {target}; from {source}",
            (ROLE_ACTION, ROLE_TARGET, ROLE_SOURCE),
        ),
        (
            "from {source}; using {action}; reaches {target}",
            (ROLE_SOURCE, ROLE_ACTION, ROLE_TARGET),
        ),
        (
            "{target} reached; using {action}; from {source}",
            (ROLE_TARGET, ROLE_ACTION, ROLE_SOURCE),
        ),
    )
    query_templates = (
        "transform-sequence::{word} departure::{start}",
        "method-list={word}; prior={start}",
        "{word} <=sequence-from= {start}",
        "operations {word} source {start}",
    )
    examples: list[LabeledEpisode] = []
    for style, ((record_template, source_roles), query_template) in enumerate(
        zip(record_templates, query_templates, strict=True)
    ):
        records = [
            record_template.format(
                target=machine.state_keys[
                    machine.transition[action][state]
                ],
                source=machine.state_keys[state],
                action=machine.action_keys[action],
            )
            for action in range(len(machine.action_keys))
            for state in range(len(machine.state_keys))
        ]
        rng = random.Random(
            int.from_bytes(
                sha256(
                    (
                        "VARIABLE-TOPOLOGY-COUNTERFACTUAL-V3|"
                        f"{row.supervisor.episode_seed}|"
                        f"{row.supervisor.law_sha256}|{style}"
                    ).encode("ascii")
                ).digest()[:8],
                "big",
            )
        )
        rng.shuffle(records)
        augmented = GeneratedEpisode(
            candidate=CandidateEpisode(
                source="\n".join(records),
                query=query_template.format(
                    word=word,
                    start=machine.state_keys[start],
                ),
            ),
            supervisor=row.supervisor,
        )
        examples.append(
            LabeledEpisode(
                row=augmented,
                source_roles=source_roles,
                query_actions_first=True,
                counterfactual=True,
            )
        )
    return examples


def _collate(
    examples: list[LabeledEpisode],
    *,
    device: torch.device,
    control: str = "treatment",
):
    if control not in {"treatment", "direction_shuffled", "type_shuffled"}:
        raise ValueError("control leaves frozen contract")
    source = collate_sources(
        tuple(
            scan_source(example.row.candidate.source.encode("ascii"))
            for example in examples
        ),
        device=device,
    )
    query = collate_queries(
        tuple(
            scan_query(example.row.candidate.query.encode("ascii"))
            for example in examples
        ),
        device=device,
    )
    source_labels = torch.full(
        (len(examples), MAX_RECORDS, 3),
        -100,
        dtype=torch.long,
        device=device,
    )
    query_labels = torch.full(
        (len(examples), MAX_QUERY_OCCURRENCES),
        -100,
        dtype=torch.long,
        device=device,
    )
    for index, example in enumerate(examples):
        roles = example.source_roles
        if example.counterfactual and control == "direction_shuffled":
            roles = tuple(
                ROLE_TARGET
                if role == ROLE_SOURCE
                else ROLE_SOURCE
                if role == ROLE_TARGET
                else role
                for role in roles
            )
        elif example.counterfactual and control == "type_shuffled":
            roles = tuple(
                ROLE_ACTION
                if role == ROLE_SOURCE
                else ROLE_SOURCE
                if role == ROLE_ACTION
                else role
                for role in roles
            )
        record_count = int(source.record_valid[index].sum())
        source_labels[index, :record_count] = torch.tensor(
            roles,
            dtype=torch.long,
            device=device,
        )
        occurrence_count = int(query.occurrence_valid[index].sum())
        if example.query_actions_first:
            labels = (QUERY_ACTION,) * (occurrence_count - 1) + (QUERY_START,)
        else:
            labels = (QUERY_START,) + (QUERY_ACTION,) * (occurrence_count - 1)
        query_labels[index, :occurrence_count] = torch.tensor(
            labels,
            dtype=torch.long,
            device=device,
        )
    return source, query, source_labels, query_labels


def _role_loss(
    source_logits: torch.Tensor,
    query_logits: torch.Tensor,
    source_labels: torch.Tensor,
    query_labels: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(
        source_logits.flatten(0, 2),
        source_labels.flatten(),
        ignore_index=-100,
    ) + F.cross_entropy(
        query_logits.flatten(0, 1),
        query_labels.flatten(),
        ignore_index=-100,
    )


def _train(
    *,
    model: SharedRawMachineCompiler,
    tensors,
    steps: int,
    learning_rate: float,
) -> tuple[float, float]:
    source, query, source_labels, query_labels = tensors
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )
    initial = 0.0
    final = 0.0
    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        source_output = model.compile_source(source)
        query_output = model.parse_query(query)
        loss = _role_loss(
            source_output.source_role_logits,
            query_output.query_role_logits,
            source_labels,
            query_labels,
        )
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        if step == 0:
            initial = value
        final = value
    model.eval()
    return initial, final


@torch.no_grad()
def _evaluate(
    model: SharedRawMachineCompiler,
    examples: list[LabeledEpisode],
    *,
    device: torch.device,
    source_ablation: str = "none",
    query_ablation: str = "none",
) -> dict[str, object]:
    if source_ablation not in {"none", "direction_swap", "type_swap"}:
        raise ValueError("source ablation leaves frozen contract")
    if query_ablation not in {"none", "role_swap"}:
        raise ValueError("query ablation leaves frozen contract")
    source, query, source_labels, query_labels = _collate(
        examples,
        device=device,
    )
    source_output = model.compile_source(source)
    query_output = model.parse_query(query)
    source_logits = source_output.source_role_logits.clone()
    query_logits = query_output.query_role_logits.clone()
    if source_ablation == "direction_swap":
        original = source_logits.clone()
        source_logits[..., ROLE_SOURCE] = original[..., ROLE_TARGET]
        source_logits[..., ROLE_TARGET] = original[..., ROLE_SOURCE]
    elif source_ablation == "type_swap":
        original = source_logits.clone()
        source_logits[..., ROLE_SOURCE] = original[..., ROLE_ACTION]
        source_logits[..., ROLE_ACTION] = original[..., ROLE_SOURCE]
    if query_ablation == "role_swap":
        query_logits = query_logits.flip(-1)
    source_valid = source_labels.ne(-100)
    query_valid = query_labels.ne(-100)
    source_predictions = source_logits.argmax(-1)
    query_predictions = query_logits.argmax(-1)
    source_correct = int(
        source_predictions[source_valid].eq(source_labels[source_valid]).sum()
    )
    query_correct = int(
        query_predictions[query_valid].eq(query_labels[query_valid]).sum()
    )
    exact = 0
    invalid = 0
    by_cell: Counter[str] = Counter()
    cell_total: Counter[str] = Counter()
    collision_exact = 0
    collision_total = 0
    for index, example in enumerate(examples):
        cell = example.row.supervisor.cell
        cell_total[cell] += 1
        collision = example.row.supervisor.incidence_collision
        collision_total += int(collision)
        try:
            machine = seal_machine(
                source,
                CompilerOutput(source_logits),
                row=index,
                learned_global_key_classes=True,
            )
            machine = type(machine).from_deployed_wire(
                machine.deployed_wire()
            )
            answer = execute_query(
                machine,
                query,
                QueryOutput(query_logits),
                row=index,
                structural_key_classes=False,
            ).decode("ascii")
        except ValueError:
            invalid += 1
            continue
        if answer == example.row.supervisor.answer:
            exact += 1
            by_cell[cell] += 1
            collision_exact += int(collision)
    return {
        "cell_exact": {
            cell: {"correct": by_cell[cell], "total": total}
            for cell, total in sorted(cell_total.items())
        },
        "collision_exact": collision_exact,
        "collision_total": collision_total,
        "exact": exact,
        "invalid": invalid,
        "query_role_accuracy": query_correct / int(query_valid.sum()),
        "query_ablation": query_ablation,
        "sealed_wire_roundtrip": True,
        "source_ablation": source_ablation,
        "source_role_accuracy": source_correct / int(source_valid.sum()),
        "total": len(examples),
    }


def run_experiment(
    *,
    seed: int,
    steps: int,
    width: int,
    layers: int,
    learning_rate: float,
    device: torch.device,
    held_out_family: str | None,
    train_per_renderer: int = 4,
    development_per_cell: int = 4,
) -> dict[str, object]:
    if held_out_family is not None and held_out_family not in FAMILIES:
        raise ValueError("held-out family leaves frozen board")
    random.seed(seed)
    torch.manual_seed(seed)
    board = build_frozen_board(
        seed=seed,
        train_per_renderer=train_per_renderer,
        development_per_cell=development_per_cell,
    )
    base_rows = [
        row
        for row in board
        if row.supervisor.split == "train"
        and row.supervisor.family != held_out_family
    ]
    development_rows = [
        row
        for row in board
        if row.supervisor.split == "development"
        and (
            held_out_family is None
            or row.supervisor.family == held_out_family
        )
    ]
    base_examples = [_base_example(row) for row in base_rows]
    counterfactual_examples = [
        example
        for row in base_rows
        for example in _counterfactual_examples(row)
    ]
    train_examples = [*base_examples, *counterfactual_examples]
    development_examples = [_base_example(row) for row in development_rows]
    tensors = {
        arm: _collate(train_examples, device=device, control=arm)
        for arm in ("treatment", "direction_shuffled", "type_shuffled")
    }
    template = SharedRawMachineCompiler(width=width, layers=layers).to(device)
    models = {
        arm: copy.deepcopy(template)
        for arm in tensors
    }
    losses = {
        arm: _train(
            model=models[arm],
            tensors=tensors[arm],
            steps=steps,
            learning_rate=learning_rate,
        )
        for arm in tensors
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
            b"VARIABLE-TOPOLOGY-QUALIFICATION-BOARD-V1",
        ),
        "candidate_time_oracle_calls": 0,
        "candidate_time_search_calls": 0,
        "candidate_time_verifier_calls": 0,
        "candidate_uses_learned_global_partition_on_all_rows": True,
        "candidate_uses_query_role_logits": True,
        "candidate_source_bytes_absent_from_deployed_wire": True,
        "development": {
            **{
                arm: _evaluate(
                    model,
                    development_examples,
                    device=device,
                )
                for arm, model in models.items()
            },
            "same_weights_direction_swapped": _evaluate(
                models["treatment"],
                development_examples,
                device=device,
                source_ablation="direction_swap",
            ),
            "same_weights_query_roles_swapped": _evaluate(
                models["treatment"],
                development_examples,
                device=device,
                query_ablation="role_swap",
            ),
            "same_weights_type_swapped": _evaluate(
                models["treatment"],
                development_examples,
                device=device,
                source_ablation="type_swap",
            ),
        },
        "device": str(device),
        "equal_budget": {
            "base_rows": len(base_examples),
            "counterfactual_rows": len(counterfactual_examples),
            "initialization_identical": True,
            "optimizer_updates_per_arm": steps,
            "parameters_per_arm": template.parameter_count(),
        },
        "held_out_family": held_out_family,
        "parameter_receipt": asdict(template.parameter_receipt()),
        "preparation_exact_parser_calls": len(base_examples),
        "preparation_query_parser_calls": len(base_examples),
        "preparation_source_parser_calls": len(base_examples),
        "seed": seed,
        "status": "variable_topology_semantic_type_curriculum",
        "train": {
            arm: _evaluate(model, train_examples, device=device)
            for arm, model in models.items()
        },
        "training_loss": {
            arm: {"initial": pair[0], "final": pair[1]}
            for arm, pair in losses.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--held-out-family", choices=FAMILIES)
    parser.add_argument("--train-per-renderer", type=int, default=4)
    parser.add_argument("--development-per-cell", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_experiment(
        seed=args.seed,
        steps=args.steps,
        width=args.width,
        layers=args.layers,
        learning_rate=args.learning_rate,
        device=torch.device(args.device),
        held_out_family=args.held_out_family,
        train_per_renderer=args.train_per_renderer,
        development_per_cell=args.development_per_cell,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="ascii")
    print(payload, end="")


if __name__ == "__main__":
    main()
