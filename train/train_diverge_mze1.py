#!/usr/bin/env python3
"""Train and gate the bounded DIVERGE-MZE1 finite-field executor."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping

import torch

from diverge_mze1_runtime import (
    CHECKPOINT_SCHEMA,
    MZE1Config,
    OPERATIONS,
    PRIME,
    PresentedZ97Executor,
    module_state_sha256,
    sha256_path,
)


REPORT_SCHEMA = "shohin-diverge-mze1-training-report-v1"
SEED = 2026080741
UPDATES = 256
BATCH_SIZE = 4096
LEARNING_RATE = 0.2
HELD_DEPTHS = (4, 8, 16, 32)
HELD_PROGRAMS = 2_000


def oracle_transition(operation: int, state: tuple[int, int]) -> tuple[int, int]:
    """Independent assessor used only to create outcome supervision."""

    x, y = state
    matrices = (
        ((1, 1), (0, 1)),
        ((1, 0), (1, 1)),
        ((1, -1), (0, 1)),
        ((1, 0), (-1, 1)),
        ((2, 1), (0, 1)),
        ((1, 0), (1, 2)),
        ((0, 1), (1, 0)),
        ((-1, 1), (0, 1)),
    )
    rows = matrices[operation]
    return tuple((row[0] * x + row[1] * y) % PRIME for row in rows)  # type: ignore[return-value]


def _targets(
    operations: torch.Tensor, states: torch.Tensor, shift: int
) -> torch.Tensor:
    values = [
        oracle_transition(
            (int(operation) + shift) % OPERATIONS, (int(state[0]), int(state[1]))
        )
        for operation, state in zip(operations.tolist(), states.tolist(), strict=True)
    ]
    return torch.tensor(values, dtype=torch.long)


def _train(
    model: PresentedZ97Executor,
    *,
    shift: int,
    generator: torch.Generator,
) -> tuple[list[dict[str, float]], float]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.0
    )
    history = []
    started = time.perf_counter()
    for update in range(1, UPDATES + 1):
        operations = torch.randint(
            OPERATIONS, (BATCH_SIZE,), generator=generator, dtype=torch.long
        )
        states = torch.randint(
            PRIME, (BATCH_SIZE, 2), generator=generator, dtype=torch.long
        )
        targets = _targets(operations, states, shift)
        loss = model.outcome_nll(operations, states, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0))
        if not torch.isfinite(loss) or not torch.isfinite(torch.tensor(gradient_norm)):
            raise RuntimeError("MZE1 training became nonfinite")
        optimizer.step()
        if update in (1, 2, 4, 8, 16, 32, 64, 128, UPDATES):
            history.append(
                {
                    "update": float(update),
                    "loss": float(loss.detach()),
                    "gradient_norm": gradient_norm,
                }
            )
    return history, time.perf_counter() - started


def _exhaustive(model: PresentedZ97Executor) -> dict[str, Any]:
    exact = 0
    total = OPERATIONS * PRIME * PRIME
    by_operation = []
    for operation in range(OPERATIONS):
        operation_exact = 0
        for x in range(PRIME):
            for y in range(PRIME):
                operation_exact += model.transition(
                    operation, (x, y)
                ) == oracle_transition(operation, (x, y))
        exact += operation_exact
        by_operation.append(
            {
                "operation": operation,
                "exact": operation_exact,
                "total": PRIME * PRIME,
                "rate": operation_exact / (PRIME * PRIME),
            }
        )
    return {
        "exact": exact,
        "total": total,
        "rate": exact / total,
        "by_operation": by_operation,
    }


def _depths(model: PresentedZ97Executor) -> dict[str, Any]:
    result = {}
    for depth in HELD_DEPTHS:
        rng = random.Random(f"mze1-depth:{SEED}:{depth}")
        exact = 0
        for _ in range(HELD_PROGRAMS):
            learned = (rng.randrange(PRIME), rng.randrange(PRIME))
            oracle = learned
            operations = tuple(rng.randrange(OPERATIONS) for _ in range(depth))
            for operation in operations:
                learned = model.transition(operation, learned)
                oracle = oracle_transition(operation, oracle)
            exact += learned == oracle
        result[str(depth)] = {
            "exact": exact,
            "total": HELD_PROGRAMS,
            "rate": exact / HELD_PROGRAMS,
        }
    return result


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise RuntimeError("MZE1 temporary checkpoint already exists")
    torch.save(dict(payload), temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.checkpoint.exists() or args.report.exists():
        raise SystemExit("refusing existing MZE1 output")

    torch.manual_seed(SEED)
    treatment = PresentedZ97Executor()
    shuffled = PresentedZ97Executor()
    shuffled.load_state_dict(treatment.state_dict(), strict=True)
    treatment_history, treatment_seconds = _train(
        treatment,
        shift=0,
        generator=torch.Generator().manual_seed(SEED + 1),
    )
    shuffled_history, shuffled_seconds = _train(
        shuffled,
        shift=1,
        generator=torch.Generator().manual_seed(SEED + 1),
    )

    treatment_exhaustive = _exhaustive(treatment)
    shuffled_exhaustive = _exhaustive(shuffled)
    depths = _depths(treatment)
    runtime_path = Path(__file__).with_name("diverge_mze1_runtime.py")
    runtime_source = runtime_path.read_text(encoding="utf-8")
    runtime_audit = {
        "runtime": str(runtime_path),
        "runtime_sha256": hashlib.sha256(runtime_source.encode()).hexdigest(),
        "imports_exact_pl1_operation": "diverge_pl1_data" in runtime_source,
        "contains_oracle_transition": "oracle_transition" in runtime_source,
    }
    conditions = {
        "treatment_exhaustive_one_step_exact": treatment_exhaustive["rate"] == 1.0,
        "treatment_all_held_depths_exact": all(
            value["rate"] == 1.0 for value in depths.values()
        ),
        "shuffled_control_at_most_5_percent": shuffled_exhaustive["rate"] <= 0.05,
        "runtime_has_no_exact_operation_import": not runtime_audit[
            "imports_exact_pl1_operation"
        ],
        "runtime_has_no_oracle_function": not runtime_audit[
            "contains_oracle_transition"
        ],
        "matched_parameter_and_update_budget": sum(
            parameter.numel() for parameter in treatment.parameters()
        )
        == sum(parameter.numel() for parameter in shuffled.parameters())
        and len(treatment_history) == len(shuffled_history),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "pass" if all(conditions.values()) else "fail",
        "source_commit": args.source_commit,
        "seed": SEED,
        "training": {
            "updates": UPDATES,
            "batch_size": BATCH_SIZE,
            "charged_transitions_per_arm": UPDATES * BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "treatment_seconds": treatment_seconds,
            "shuffled_seconds": shuffled_seconds,
            "treatment_history": treatment_history,
            "shuffled_history": shuffled_history,
        },
        "treatment": {
            "model": treatment.record(),
            "exhaustive_one_step": treatment_exhaustive,
            "held_free_running_depths": depths,
        },
        "shuffled_control": {
            "model": shuffled.record(),
            "target_operation_shift": 1,
            "exhaustive_against_true": shuffled_exhaustive,
        },
        "runtime_source_audit": runtime_audit,
        "gate": {"conditions": conditions, "passed": all(conditions.values())},
    }
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "source_commit": args.source_commit,
        "config": asdict(MZE1Config()),
        "treatment_state_dict": treatment.state_dict(),
        "treatment_state_sha256": module_state_sha256(treatment),
        "shuffled_state_dict": shuffled.state_dict(),
        "shuffled_state_sha256": module_state_sha256(shuffled),
        "gate_passed": all(conditions.values()),
    }
    _atomic_torch(args.checkpoint, checkpoint)
    report["checkpoint"] = str(args.checkpoint)
    report["checkpoint_sha256"] = sha256_path(args.checkpoint)
    _atomic_json(args.report, report)
    os.chmod(args.checkpoint, 0o444)
    os.chmod(args.report, 0o444)
    print(
        json.dumps(
            {
                "status": report["status"],
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": report["checkpoint_sha256"],
                "report": str(args.report),
                "report_sha256": sha256_path(args.report),
                "treatment_rate": treatment_exhaustive["rate"],
                "shuffled_rate": shuffled_exhaustive["rate"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
