"""Frozen optimizer and learning-rate contract for ETTR continuation.

This module constructs optimizers but never launches a training loop.  Base
and architecture parameters receive explicit, disjoint groups so a later
authorized run can prove exactly which tensors participated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from endogenous_typed_theory_reactor import (
    EndogenousTypedTheoryReactorGPT,
    TheoryReactorError,
)
from muon import Muon


@dataclass(frozen=True, slots=True)
class ETTROptimizerConfig:
    train_base: bool = True
    base_lr_muon: float = 0.005
    base_lr_adam: float = 0.001
    architecture_lr_muon: float = 0.005
    architecture_lr_adam: float = 0.001
    adam_betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    warmup_updates: int = 2_000
    total_updates: int = 300_000
    decay_fraction: float = 0.2
    final_lr_fraction: float = 0.1

    def validate(self) -> None:
        rates = (
            self.base_lr_muon,
            self.base_lr_adam,
            self.architecture_lr_muon,
            self.architecture_lr_adam,
        )
        if any(not 0 < value < 1 for value in rates):
            raise TheoryReactorError(
                "ETTR learning rates must lie strictly between zero and one"
            )
        if (
            len(self.adam_betas) != 2
            or not 0 <= self.adam_betas[0] < 1
            or not 0 <= self.adam_betas[1] < 1
            or self.weight_decay < 0
            or self.warmup_updates < 0
            or self.total_updates <= self.warmup_updates
            or not 0 < self.decay_fraction <= 1
            or not 0 < self.final_lr_fraction <= 1
        ):
            raise TheoryReactorError("ETTR optimizer schedule configuration differs")


@dataclass(frozen=True, slots=True)
class ETTROptimizerReceipt:
    base_muon_parameters: int
    base_adam_parameters: int
    architecture_muon_parameters: int
    architecture_adam_parameters: int
    unique_trainable_parameters: int
    complete_system_parameters: int
    train_base: bool


class ETTROptimizerBundle:
    """Two optimizer families with exact disjoint parameter ownership."""

    def __init__(
        self,
        model: EndogenousTypedTheoryReactorGPT,
        config: ETTROptimizerConfig,
    ):
        config.validate()
        self.config = config
        self.next_update = 0
        base_ids = {id(parameter) for parameter in model.base.parameters()}
        if not config.train_base:
            model.freeze_base()

        groups: dict[str, list[torch.nn.Parameter]] = {
            "base_muon": [],
            "base_adam": [],
            "architecture_muon": [],
            "architecture_adam": [],
        }
        seen: set[int] = set()
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            identity = id(parameter)
            if identity in seen:
                raise TheoryReactorError("optimizer parameter ownership is duplicated")
            seen.add(identity)
            owner = "base" if identity in base_ids else "architecture"
            family = (
                "muon"
                if (parameter.ndim == 2 and "tok" not in name and "head" not in name)
                else "adam"
            )
            groups[f"{owner}_{family}"].append(parameter)

        architecture_ids = {
            id(parameter)
            for module in (
                model.compiler,
                model.reactor,
                model.query_reader,
            )
            for parameter in module.parameters()
        }
        if not architecture_ids.issubset(seen):
            raise TheoryReactorError("an ETTR architecture parameter is not trainable")
        expected_trainable = architecture_ids | (
            base_ids if config.train_base else set()
        )
        if seen != expected_trainable:
            raise TheoryReactorError("optimizer trainable parameter set differs")
        self.failed_update = False

        muon_groups = [
            {
                "params": groups["base_muon"],
                "lr": config.base_lr_muon,
                "initial_lr": config.base_lr_muon,
                "ettr_group": "base_muon",
            },
            {
                "params": groups["architecture_muon"],
                "lr": config.architecture_lr_muon,
                "initial_lr": config.architecture_lr_muon,
                "ettr_group": "architecture_muon",
            },
        ]
        muon_groups = [group for group in muon_groups if group["params"]]
        adam_groups = [
            {
                "params": groups["base_adam"],
                "lr": config.base_lr_adam,
                "initial_lr": config.base_lr_adam,
                "ettr_group": "base_adam",
            },
            {
                "params": groups["architecture_adam"],
                "lr": config.architecture_lr_adam,
                "initial_lr": config.architecture_lr_adam,
                "ettr_group": "architecture_adam",
            },
        ]
        adam_groups = [group for group in adam_groups if group["params"]]
        self.muon = Muon(muon_groups, lr=config.base_lr_muon) if muon_groups else None
        self.adam = torch.optim.AdamW(
            adam_groups,
            lr=config.base_lr_adam,
            betas=config.adam_betas,
            weight_decay=config.weight_decay,
        )
        counts = {
            name: sum(parameter.numel() for parameter in parameters)
            for name, parameters in groups.items()
        }
        trainable = sum(counts.values())
        if trainable != sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ):
            raise TheoryReactorError("optimizer parameter receipt does not reconcile")
        complete = model.parameter_receipt().complete_system_parameters
        self.receipt = ETTROptimizerReceipt(
            base_muon_parameters=counts["base_muon"],
            base_adam_parameters=counts["base_adam"],
            architecture_muon_parameters=counts["architecture_muon"],
            architecture_adam_parameters=counts["architecture_adam"],
            unique_trainable_parameters=trainable,
            complete_system_parameters=complete,
            train_base=config.train_base,
        )

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    def assert_bound_to(self, model: EndogenousTypedTheoryReactorGPT) -> None:
        model_parameters = tuple(
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        optimizer_parameters = tuple(
            parameter
            for optimizer in (self.muon, self.adam)
            if optimizer is not None
            for group in optimizer.param_groups
            for parameter in group["params"]
        )
        model_ids = {id(parameter) for parameter in model_parameters}
        optimizer_ids = {
            id(parameter) for parameter in optimizer_parameters
        }
        if (
            len(model_ids) != len(model_parameters)
            or len(optimizer_ids) != len(optimizer_parameters)
            or model_ids != optimizer_ids
        ):
            raise TheoryReactorError(
                "ETTR optimizer is not bound to the supplied model parameters"
            )

    def assert_healthy(self) -> None:
        if self.failed_update:
            raise TheoryReactorError(
                "ETTR optimizer is fail-stop; restore the last verified "
                "checkpoint"
            )

    def mark_failed_update(self) -> None:
        self.failed_update = True

    def step(self) -> None:
        self.assert_healthy()
        if self.next_update >= self.config.total_updates:
            raise TheoryReactorError(
                "ETTR optimizer cannot step beyond the frozen horizon"
            )
        if self.muon is not None:
            self.muon.step()
        self.adam.step()
        self.next_update += 1

    def apply_schedule(self, update: int | None = None) -> float:
        self.assert_healthy()
        if update is None:
            update = self.next_update
        if not 0 <= update <= self.config.total_updates:
            raise TheoryReactorError(
                "ETTR scheduler update is outside the frozen horizon"
            )
        scale = _schedule_scale(update, self.config)
        if self.muon is not None:
            for group in self.muon.param_groups:
                group["lr"] = self._base_lr(group["ettr_group"]) * scale
        for group in self.adam.param_groups:
            group["lr"] = self._base_lr(group["ettr_group"]) * scale
        return scale

    def state_dict(self) -> dict[str, Any]:
        self.assert_healthy()
        return {
            "schema": "shohin-ettr-optimizer-v1",
            "config": asdict(self.config),
            "receipt": asdict(self.receipt),
            "next_update": self.next_update,
            "muon": None if self.muon is None else self.muon.state_dict(),
            "adam": self.adam.state_dict(),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.assert_healthy()
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != "shohin-ettr-optimizer-v1"
            or payload.get("config") != asdict(self.config)
            or payload.get("receipt") != asdict(self.receipt)
        ):
            raise TheoryReactorError("ETTR optimizer checkpoint contract differs")
        next_update = payload.get("next_update")
        if (
            not isinstance(next_update, int)
            or not 0 <= next_update <= self.config.total_updates
        ):
            raise TheoryReactorError("ETTR optimizer update cursor differs")
        muon_state = payload.get("muon")
        if (self.muon is None) != (muon_state is None):
            raise TheoryReactorError("ETTR Muon checkpoint presence differs")
        if self.muon is not None:
            self.muon.load_state_dict(muon_state)
        self.adam.load_state_dict(payload["adam"])
        self.next_update = next_update
        self.apply_schedule()

    def _base_lr(self, name: str) -> float:
        return {
            "base_muon": self.config.base_lr_muon,
            "base_adam": self.config.base_lr_adam,
            "architecture_muon": self.config.architecture_lr_muon,
            "architecture_adam": self.config.architecture_lr_adam,
        }[name]


def _schedule_scale(
    update: int,
    config: ETTROptimizerConfig,
) -> float:
    if update < config.warmup_updates:
        return update / max(1, config.warmup_updates)
    decay_start = config.total_updates * (1.0 - config.decay_fraction)
    if update < decay_start:
        return 1.0
    progress = (update - decay_start) / max(
        1.0,
        config.total_updates - decay_start,
    )
    return 1.0 + (config.final_lr_fraction - 1.0) * progress


__all__ = [
    "ETTROptimizerBundle",
    "ETTROptimizerConfig",
    "ETTROptimizerReceipt",
]
