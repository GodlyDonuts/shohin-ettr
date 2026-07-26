"""Counterfactual renderer curriculum for anonymous machine compilation.

Preparation renders each fitting law once more with target-first direction
under symbols that do not occur in the held-out renderer. The treatment gets
the correct source/target labels. A matched control gets those direction
labels swapped while preserving bytes, parameters, updates, initialization,
and query labels. Candidate evaluation retains source deletion and receives
no exact parser, oracle, search, or verifier.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import random
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

from source_deleted_multifamily_machine_board import (  # noqa: E402
    CandidateEpisode,
    FAMILIES,
    GeneratedEpisode,
    SupervisorEpisode,
    build_frozen_board,
    compile_source,
    decode_late_query,
)
from multifamily_raw_machine_compiler import (  # noqa: E402
    ROLE_SOURCE,
    ROLE_TARGET,
    SharedRawMachineCompiler,
)
from run_multifamily_raw_machine_smoke import (  # noqa: E402
    _collate,
    _evaluate,
    _role_loss,
)


def _target_first_augmentation(row: GeneratedEpisode) -> GeneratedEpisode:
    machine = compile_source(row.candidate.source)
    records = [
        (
            f"{machine.state_keys[machine.transition[action][state]]}"
            f" <-{{{machine.action_keys[action]}}}- "
            f"({machine.state_keys[state]})"
        )
        for action in range(len(machine.action_keys))
        for state in range(len(machine.state_keys))
    ]
    rng = random.Random(
        int.from_bytes(
            sha256(
                (
                    "MFM-TARGET-FIRST-V1|"
                    f"{row.supervisor.episode_seed}|"
                    f"{row.supervisor.law_sha256}"
                ).encode("ascii")
            ).digest()[:8],
            "big",
        )
    )
    rng.shuffle(records)
    source = "\n".join(records)
    start, actions = decode_late_query(machine, row.candidate.query)
    query = (
        "then "
        + ",".join(machine.action_keys[action] for action in actions)
        + " beginning-from "
        + machine.state_keys[start]
    )
    return GeneratedEpisode(
        candidate=CandidateEpisode(source=source, query=query),
        supervisor=SupervisorEpisode(
            family=row.supervisor.family,
            split="train",
            cell="fit",
            renderer=3,
            law_sha256=row.supervisor.law_sha256,
            source_sha256=sha256(source.encode("ascii")).hexdigest(),
            query_sha256=sha256(query.encode("ascii")).hexdigest(),
            answer=row.supervisor.answer,
            composition_length=row.supervisor.composition_length,
            episode_seed=row.supervisor.episode_seed,
        ),
    )


def _swap_augmented_direction_labels(
    labels: torch.Tensor,
    *,
    augmented_start: int,
) -> torch.Tensor:
    output = labels.clone()
    region = output[augmented_start:]
    source = region.eq(ROLE_SOURCE)
    target = region.eq(ROLE_TARGET)
    region[source] = ROLE_TARGET
    region[target] = ROLE_SOURCE
    return output


def _train(
    *,
    model: SharedRawMachineCompiler,
    source,
    query,
    source_labels: torch.Tensor,
    query_labels: torch.Tensor,
    steps: int,
    learning_rate: float,
) -> list[float]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )
    losses: list[float] = []
    model.train()
    for _step in range(steps):
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
        losses.append(float(loss.detach()))
    model.eval()
    return losses


def run_renderer_curriculum_smoke(
    *,
    seed: int,
    steps: int,
    width: int,
    layers: int,
    learning_rate: float,
    device: torch.device,
    held_out_family: str | None = None,
) -> dict[str, object]:
    if held_out_family is not None and held_out_family not in FAMILIES:
        raise ValueError("held-out family leaves the frozen board")
    random.seed(seed)
    torch.manual_seed(seed)
    board = build_frozen_board(
        seed=20260725,
        train_per_renderer=4,
        development_per_cell=2,
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
    augmented_rows = [_target_first_augmentation(row) for row in base_rows]
    train_rows = [*base_rows, *augmented_rows]
    source, query, source_labels, query_labels = _collate(
        train_rows,
        device=device,
    )
    control_labels = _swap_augmented_direction_labels(
        source_labels,
        augmented_start=len(base_rows),
    )
    template = SharedRawMachineCompiler(width=width, layers=layers).to(device)
    treatment = copy.deepcopy(template)
    control = copy.deepcopy(template)
    treatment_losses = _train(
        model=treatment,
        source=source,
        query=query,
        source_labels=source_labels,
        query_labels=query_labels,
        steps=steps,
        learning_rate=learning_rate,
    )
    control_losses = _train(
        model=control,
        source=source,
        query=query,
        source_labels=control_labels,
        query_labels=query_labels,
        steps=steps,
        learning_rate=learning_rate,
    )
    return {
        "candidate_time_oracle_calls": 0,
        "candidate_time_search_calls": 0,
        "candidate_time_verifier_calls": 0,
        "development": {
            "direction_shuffled_control": _evaluate(
                control,
                development_rows,
                device=device,
                structural_key_classes=True,
            ),
            "learned_roles_without_structural_typing": _evaluate(
                treatment,
                development_rows,
                device=device,
            ),
            "renderer_curriculum_treatment": _evaluate(
                treatment,
                development_rows,
                device=device,
                structural_key_classes=True,
            ),
        },
        "device": str(device),
        "equal_budget": {
            "base_rows": len(base_rows),
            "counterfactual_rows": len(augmented_rows),
            "initialization_identical": True,
            "optimizer_updates_per_arm": steps,
            "parameters_per_arm": treatment.parameter_count(),
        },
        "parameter_receipt": asdict(treatment.parameter_receipt()),
        "preparation_exact_parser_calls": len(base_rows),
        "seed": seed,
        "status": "target_first_renderer_curriculum_smoke",
        "held_out_family": held_out_family,
        "train": {
            "direction_shuffled_control": _evaluate(
                control,
                train_rows,
                device=device,
                structural_key_classes=True,
            ),
            "renderer_curriculum_treatment": _evaluate(
                treatment,
                train_rows,
                device=device,
                structural_key_classes=True,
            ),
        },
        "training_loss": {
            "control_final": control_losses[-1],
            "control_initial": control_losses[0],
            "treatment_final": treatment_losses[-1],
            "treatment_initial": treatment_losses[0],
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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_renderer_curriculum_smoke(
        seed=args.seed,
        steps=args.steps,
        width=args.width,
        layers=args.layers,
        learning_rate=args.learning_rate,
        device=torch.device(args.device),
        held_out_family=args.held_out_family,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="ascii")
    print(payload, end="")


if __name__ == "__main__":
    main()
