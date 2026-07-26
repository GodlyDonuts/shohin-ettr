"""Bounded learned smoke for the multi-family raw-machine compiler.

This trains only semantic role parsing. The exact board compiler is never
called by the candidate. The run is a mechanics/optimization diagnostic and
does not authorize an H100 qualification or a reasoning claim.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
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

from source_deleted_multifamily_machine_board import (  # noqa: E402
    GeneratedEpisode,
    build_frozen_board,
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


def _source_role_order(renderer: int) -> tuple[int, int, int]:
    if renderer == 0:
        return (ROLE_SOURCE, ROLE_ACTION, ROLE_TARGET)
    if renderer in {1, 2}:
        return (ROLE_ACTION, ROLE_SOURCE, ROLE_TARGET)
    if renderer == 3:
        return (ROLE_TARGET, ROLE_ACTION, ROLE_SOURCE)
    raise ValueError("renderer leaves source role contract")


def _collate(
    rows: list[GeneratedEpisode],
    *,
    device: torch.device,
):
    source = collate_sources(
        tuple(
            scan_source(row.candidate.source.encode("ascii"))
            for row in rows
        ),
        device=device,
    )
    query = collate_queries(
        tuple(
            scan_query(row.candidate.query.encode("ascii"))
            for row in rows
        ),
        device=device,
    )
    source_labels = torch.full(
        (len(rows), MAX_RECORDS, 3),
        -100,
        dtype=torch.long,
        device=device,
    )
    query_labels = torch.full(
        (len(rows), MAX_QUERY_OCCURRENCES),
        -100,
        dtype=torch.long,
        device=device,
    )
    for index, row in enumerate(rows):
        records = int(source.record_valid[index].sum())
        source_labels[index, :records] = torch.tensor(
            _source_role_order(row.supervisor.renderer),
            dtype=torch.long,
            device=device,
        )
        occurrences = int(query.occurrence_valid[index].sum())
        if row.supervisor.renderer == 3:
            labels = (QUERY_ACTION,) * (occurrences - 1) + (QUERY_START,)
        else:
            labels = (QUERY_START,) + (QUERY_ACTION,) * (occurrences - 1)
        query_labels[index, :occurrences] = torch.tensor(
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


@torch.no_grad()
def _evaluate(
    model: SharedRawMachineCompiler,
    rows: list[GeneratedEpisode],
    *,
    device: torch.device,
    source_external: torch.Tensor | None = None,
    query_external: torch.Tensor | None = None,
    structural_key_classes: bool = False,
    structural_key_shuffle: bool = False,
) -> dict[str, object]:
    source, query, source_labels, query_labels = _collate(rows, device=device)
    source_output = model.compile_source(
        source,
        external_unit_features=source_external,
    )
    query_output = model.parse_query(
        query,
        external_unit_features=query_external,
    )
    source_predictions = source_output.source_role_logits.argmax(-1)
    query_predictions = query_output.query_role_logits.argmax(-1)
    source_valid = source_labels.ne(-100)
    query_valid = query_labels.ne(-100)
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
    for index, row in enumerate(rows):
        cell_total[row.supervisor.cell] += 1
        try:
            machine = seal_machine(
                source,
                CompilerOutput(
                    source_role_logits=source_output.source_role_logits
                ),
                row=index,
                structural_key_classes=structural_key_classes,
                structural_key_shuffle=structural_key_shuffle,
            )
            answer = execute_query(
                machine,
                query,
                QueryOutput(
                    query_role_logits=query_output.query_role_logits
                ),
                row=index,
                structural_key_classes=structural_key_classes,
            ).decode("ascii")
        except ValueError:
            invalid += 1
            continue
        if answer == row.supervisor.answer:
            exact += 1
            by_cell[row.supervisor.cell] += 1
    return {
        "cell_exact": {
            cell: {
                "correct": by_cell[cell],
                "total": total,
            }
            for cell, total in sorted(cell_total.items())
        },
        "exact": exact,
        "invalid": invalid,
        "query_role_accuracy": query_correct / int(query_valid.sum()),
        "source_role_accuracy": source_correct / int(source_valid.sum()),
        "total": len(rows),
    }


def run_smoke(
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
        "development": _evaluate(
            model,
            development_rows,
            device=device,
        ),
        "device": str(device),
        "final_loss": losses[-1],
        "initial_loss": losses[0],
        "parameter_receipt": asdict(model.parameter_receipt()),
        "seed": seed,
        "status": "standalone_mechanics_not_connected_to_shohin_trunk",
        "steps": steps,
        "train": _evaluate(model, train_rows, device=device),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_smoke(
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
