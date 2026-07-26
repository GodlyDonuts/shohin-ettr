"""Renderer-neutral structural type inference for the multi-family compiler.

The treatment uses only the source equality/incidence graph to identify
anonymous action and state key classes. It leaves source-versus-target
direction to the learned compiler and identifies the late-query start by its
membership in the sealed state set. The exact board parser is never imported
by candidate inference.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

from source_deleted_multifamily_machine_board import build_frozen_board  # noqa: E402
from multifamily_raw_machine_compiler import SharedRawMachineCompiler  # noqa: E402
from run_multifamily_raw_machine_smoke import (  # noqa: E402
    _collate,
    _evaluate,
    _role_loss,
)


def run_structural_type_smoke(
    *,
    seed: int,
    steps: int,
    width: int,
    layers: int,
    learning_rate: float,
    device: torch.device,
) -> dict[str, object]:
    random.seed(seed)
    torch.manual_seed(seed)
    board = build_frozen_board(
        seed=20260725,
        train_per_renderer=4,
        development_per_cell=2,
    )
    train_rows = [row for row in board if row.supervisor.split == "train"]
    development_rows = [
        row for row in board if row.supervisor.split == "development"
    ]
    source, query, source_labels, query_labels = _collate(
        train_rows,
        device=device,
    )
    model = SharedRawMachineCompiler(width=width, layers=layers).to(device)
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
    return {
        "candidate_time_oracle_calls": 0,
        "candidate_time_search_calls": 0,
        "candidate_time_verifier_calls": 0,
        "development": {
            "learned_roles": _evaluate(
                model,
                development_rows,
                device=device,
            ),
            "structural_key_classes": _evaluate(
                model,
                development_rows,
                device=device,
                structural_key_classes=True,
            ),
            "structural_key_shuffle": _evaluate(
                model,
                development_rows,
                device=device,
                structural_key_classes=True,
                structural_key_shuffle=True,
            ),
        },
        "device": str(device),
        "final_loss": losses[-1],
        "initial_loss": losses[0],
        "parameter_receipt": asdict(model.parameter_receipt()),
        "seed": seed,
        "status": "renderer_neutral_structural_type_inference_smoke",
        "steps": steps,
        "train": {
            "learned_roles": _evaluate(
                model,
                train_rows,
                device=device,
            ),
            "structural_key_classes": _evaluate(
                model,
                train_rows,
                device=device,
                structural_key_classes=True,
            ),
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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_structural_type_smoke(
        seed=args.seed,
        steps=args.steps,
        width=args.width,
        layers=args.layers,
        learning_rate=args.learning_rate,
        device=torch.device(args.device),
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="ascii")
    print(payload, end="")


if __name__ == "__main__":
    main()
