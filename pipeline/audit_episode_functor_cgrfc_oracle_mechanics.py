#!/usr/bin/env python3
"""Deterministic CPU oracle-mechanics audit for CGRFC.

This audit asks only whether the tied revision cell can learn to repair
bounded categorical faults when every local section and its incidence are
oracle supplied. It does not exercise raw-source compilation, language
grounding, late-query parsing, or unrestricted reasoning.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train"
if str(TRAIN) not in sys.path:
    sys.path.insert(0, str(TRAIN))

from episode_functor_conflict_reentrant_revision import (  # noqa: E402
    ConflictGatedReentrantRevision,
    ConflictRevisionBatch,
    MACHINE_CATEGORIES,
    MACHINE_ROWS,
)
from episode_functor_constrained_transport import (  # noqa: E402
    PRIMARY_ACTIONS,
    PRIMARY_ANSWERS,
    PRIMARY_OBSERVERS,
    PRIMARY_STATES,
)


class CGRFCOracleAuditError(ValueError):
    """The fixed oracle mechanics audit failed closed."""


@dataclass(frozen=True, slots=True)
class ExactMetrics:
    exact_machines: int
    total_machines: int
    transition_cells: int
    transition_total: int
    observer_cells: int
    observer_total: int
    final_energy: float

    @property
    def exact_rate(self) -> float:
        return self.exact_machines / self.total_machines


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("ascii")


def _oracle_board(
    count: int,
    *,
    seed: int,
    record_width: int,
    fault_margin: float,
) -> tuple[ConflictRevisionBatch, torch.Tensor, torch.Tensor]:
    if (
        count < 1
        or record_width < 32
        or not math.isfinite(fault_margin)
        or not 0.0 < fault_margin < 1.0
    ):
        raise CGRFCOracleAuditError("oracle board geometry differs")
    generator = torch.Generator().manual_seed(seed)
    transitions = torch.stack(
        tuple(
            torch.stack(
                tuple(
                    torch.randperm(
                        PRIMARY_STATES,
                        generator=generator,
                    )
                    for _ in range(PRIMARY_ACTIONS)
                )
            )
            for _ in range(count)
        )
    )
    observers = torch.randint(
        PRIMARY_ANSWERS,
        (count, PRIMARY_OBSERVERS, PRIMARY_STATES),
        generator=generator,
    )
    transition_logits = (
        F.one_hot(transitions, PRIMARY_STATES).float() * 4.0 - 2.0
    )
    observer_logits = (
        F.one_hot(observers, PRIMARY_ANSWERS).float() * 4.0 - 2.0
    )
    for row in range(count):
        for action in range(PRIMARY_ACTIONS):
            state = int(
                torch.randint(
                    PRIMARY_STATES,
                    (1,),
                    generator=generator,
                )
            )
            target = int(transitions[row, action, state])
            wrong = (
                target
                + 1
                + int(
                    torch.randint(
                        PRIMARY_STATES - 1,
                        (1,),
                        generator=generator,
                    )
                )
            ) % PRIMARY_STATES
            transition_logits[row, action, state].fill_(-2.0)
            transition_logits[row, action, state, wrong] = fault_margin / 2
            transition_logits[row, action, state, target] = -fault_margin / 2
        for observer in range(PRIMARY_OBSERVERS):
            state = int(
                torch.randint(
                    PRIMARY_STATES,
                    (1,),
                    generator=generator,
                )
            )
            target = int(observers[row, observer, state])
            wrong = (
                target
                + 1
                + int(
                    torch.randint(
                        PRIMARY_ANSWERS - 1,
                        (1,),
                        generator=generator,
                    )
                )
            ) % PRIMARY_ANSWERS
            observer_logits[row, observer, state].fill_(-2.0)
            observer_logits[row, observer, state, wrong] = fault_margin / 2
            observer_logits[row, observer, state, target] = -fault_margin / 2

    oracle = torch.cat(
        (
            (
                F.one_hot(transitions, PRIMARY_STATES).float()
                * 12.0
                - 6.0
            ).reshape(
                count,
                PRIMARY_ACTIONS * PRIMARY_STATES,
                PRIMARY_STATES,
            ),
            F.pad(
                (
                    F.one_hot(observers, PRIMARY_ANSWERS).float()
                    * 12.0
                    - 6.0
                ).reshape(
                    count,
                    PRIMARY_OBSERVERS * PRIMARY_STATES,
                    PRIMARY_ANSWERS,
                ),
                (0, PRIMARY_STATES - PRIMARY_ANSWERS),
                value=-20.0,
            ),
        ),
        dim=1,
    )
    claims = torch.zeros(
        (
            count,
            MACHINE_ROWS,
            MACHINE_ROWS,
            MACHINE_CATEGORIES,
        )
    )
    index = torch.arange(MACHINE_ROWS)
    claims[:, index, index] = oracle
    incidence = torch.eye(MACHINE_ROWS)[None].expand(
        count,
        -1,
        -1,
    ).contiguous()
    record_features = torch.zeros(
        (count, MACHINE_ROWS, record_width)
    )
    record_features[:, : PRIMARY_ACTIONS * PRIMARY_STATES, 0] = 1.0
    record_features[
        :,
        PRIMARY_ACTIONS * PRIMARY_STATES :,
        1,
    ] = 1.0
    valid = torch.ones(
        (count, MACHINE_ROWS),
        dtype=torch.bool,
    )
    return (
        ConflictRevisionBatch(
            transition_logits=transition_logits,
            observer_logits=observer_logits,
            claim_logits=claims,
            closure_claim_logits=claims.clone(),
            claim_incidence=incidence,
            closure_incidence=incidence.clone(),
            record_features=record_features,
            record_valid=valid,
        ),
        transitions,
        observers,
    )


@torch.no_grad()
def _metrics(
    module: ConflictGatedReentrantRevision,
    board: ConflictRevisionBatch,
    transitions: torch.Tensor,
    observers: torch.Tensor,
    *,
    routing_mode: str,
) -> ExactMetrics:
    result = module(board, routing_mode=routing_mode)
    predicted_transition = result.projection.machine.action_next[
        :,
        :PRIMARY_ACTIONS,
        :PRIMARY_STATES,
        :PRIMARY_STATES,
    ].argmax(-1)
    predicted_observer = result.projection.machine.observer_answer[
        :,
        :PRIMARY_OBSERVERS,
        :PRIMARY_STATES,
        :PRIMARY_ANSWERS,
    ].argmax(-1)
    transition_exact = predicted_transition.eq(transitions)
    observer_exact = predicted_observer.eq(observers)
    machine_exact = transition_exact.all(
        (1, 2)
    ) & observer_exact.all((1, 2))
    return ExactMetrics(
        exact_machines=int(machine_exact.sum()),
        total_machines=int(machine_exact.numel()),
        transition_cells=int(transition_exact.sum()),
        transition_total=int(transition_exact.numel()),
        observer_cells=int(observer_exact.sum()),
        observer_total=int(observer_exact.numel()),
        final_energy=float(
            result.contradiction_energy[:, -1].mean()
        ),
    )


def run_audit(
    *,
    train_count: int = 16,
    development_count: int = 64,
    record_width: int = 32,
    controller_width: int = 128,
    cycles: int = 4,
    updates: int = 101,
    learning_rate: float = 0.002,
    fault_margin: float = 0.5,
) -> dict[str, object]:
    if updates < 1 or learning_rate <= 0.0:
        raise CGRFCOracleAuditError("oracle optimization contract differs")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(7)
    train, train_transition, train_observer = _oracle_board(
        train_count,
        seed=11,
        record_width=record_width,
        fault_margin=fault_margin,
    )
    development, dev_transition, dev_observer = _oracle_board(
        development_count,
        seed=19,
        record_width=record_width,
        fault_margin=fault_margin,
    )
    module = ConflictGatedReentrantRevision(
        record_width=record_width,
        controller_width=controller_width,
        cycles=cycles,
        max_step=1.0,
    )
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=learning_rate,
    )
    trace: list[dict[str, object]] = []
    selected = {0, 10, 25, 50, updates - 1}
    first_gradient_complete = False
    for update in range(updates):
        optimizer.zero_grad(set_to_none=True)
        result = module(train)
        transition_logits = result.projection.machine.action_next[
            :,
            :PRIMARY_ACTIONS,
            :PRIMARY_STATES,
            :PRIMARY_STATES,
        ]
        observer_logits = result.projection.machine.observer_answer[
            :,
            :PRIMARY_OBSERVERS,
            :PRIMARY_STATES,
            :PRIMARY_ANSWERS,
        ]
        loss = F.cross_entropy(
            transition_logits.reshape(-1, PRIMARY_STATES),
            train_transition.reshape(-1),
        ) + F.cross_entropy(
            observer_logits.reshape(-1, PRIMARY_ANSWERS),
            train_observer.reshape(-1),
        )
        if not bool(torch.isfinite(loss)):
            raise CGRFCOracleAuditError("oracle loss is nonfinite")
        loss.backward()
        gradients = tuple(
            parameter.grad
            for parameter in module.parameters()
            if parameter.requires_grad
        )
        if (
            not gradients
            or any(value is None for value in gradients)
            or any(
                not bool(torch.isfinite(value).all())
                for value in gradients
                if value is not None
            )
        ):
            raise CGRFCOracleAuditError(
                "oracle gradient coverage differs"
            )
        if update == 0:
            first_gradient_complete = True
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        optimizer.step()
        if update in selected:
            trace.append(
                {
                    "development": asdict(
                        _metrics(
                            module,
                            development,
                            dev_transition,
                            dev_observer,
                            routing_mode="causal",
                        )
                    ),
                    "loss": float(loss.detach()),
                    "update": update,
                }
            )

    modes = {
        mode: _metrics(
            module,
            development,
            dev_transition,
            dev_observer,
            routing_mode=mode,
        )
        for mode in (
            "causal",
            "deranged",
            "open-loop",
            "sign-scrambled",
        )
    }
    causal = modes["causal"].exact_rate
    strongest_control = max(
        metrics.exact_rate
        for mode, metrics in modes.items()
        if mode != "causal"
    )
    decision = (
        "oracle_mechanics_pass"
        if (
            first_gradient_complete
            and causal >= 0.99
            and causal - strongest_control >= 0.20
            and modes["deranged"].exact_rate <= 0.10
            and modes["open-loop"].exact_rate <= 0.10
        )
        else "oracle_mechanics_no_go"
    )
    report: dict[str, object] = {
        "claim_boundary": (
            "oracle local sections only; no raw-source or reasoning claim"
        ),
        "configuration": {
            "controller_width": controller_width,
            "cycles": cycles,
            "development_count": development_count,
            "fault_margin": fault_margin,
            "learning_rate": learning_rate,
            "record_width": record_width,
            "train_count": train_count,
            "updates": updates,
        },
        "decision": decision,
        "first_gradient_complete": first_gradient_complete,
        "modes": {
            mode: asdict(metrics)
            | {"exact_rate": metrics.exact_rate}
            for mode, metrics in modes.items()
        },
        "parameter_count": module.parameter_count(),
        "strongest_control_advantage": causal - strongest_control,
        "trace": trace,
    }
    report["payload_sha256"] = sha256(_canonical(report)).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise CGRFCOracleAuditError("oracle audit output already exists")
    report = run_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical(report)
    args.output.write_bytes(payload)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
