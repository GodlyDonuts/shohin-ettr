"""Bounded ETTR optimizer-step core.

This is a library component, not a trainer or launcher.  It receives already
frozen continuation batches and has no shard, filesystem, checkpoint, Slurm,
or network access.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from endogenous_typed_theory_reactor import (
    EndogenousTypedTheoryReactorGPT,
    TheoryReactorError,
)
from ettr_data_contract import ETTRContinuationBatch
from ettr_episode import CausalETTREpisodeRunner
from ettr_objectives import (
    ETTRCompositeLoss,
    ETTRCompositeObjective,
    ETTRObjectiveConfig,
    ETTRObjectiveWeights,
)
from ettr_optimization import ETTROptimizerBundle


@dataclass(frozen=True, slots=True)
class ETTRTrainStepConfig:
    gradient_accumulation_steps: int = 1
    gradient_clip: float = 1.0
    hard_transactions: bool = False
    autocast_dtype: torch.dtype = torch.bfloat16

    def validate(self) -> None:
        if (
            not isinstance(self.gradient_accumulation_steps, int)
            or self.gradient_accumulation_steps <= 0
            or not 0 < self.gradient_clip <= 100
            or not isinstance(self.hard_transactions, bool)
            or self.autocast_dtype not in (torch.bfloat16, torch.float16)
        ):
            raise TheoryReactorError("ETTR train-step configuration differs")


@dataclass(frozen=True, slots=True)
class ETTRUpdateReceipt:
    optimizer_step: int
    learning_rate_scale: float
    total_loss: torch.Tensor
    token_lm_loss: torch.Tensor
    packet_loss: torch.Tensor
    transaction_loss: torch.Tensor
    equivariance_loss: torch.Tensor
    commit_halt_loss: torch.Tensor
    sparsity_loss: torch.Tensor
    anti_bypass_loss: torch.Tensor
    gradient_norm: torch.Tensor
    supervised_token_count: torch.Tensor


class ETTRTrainStep(nn.Module):
    """One exact optimizer update over one accumulation window."""

    def __init__(
        self,
        model: EndogenousTypedTheoryReactorGPT,
        optimizer: ETTROptimizerBundle,
        objective_config: ETTRObjectiveConfig,
        *,
        objective_weights: ETTRObjectiveWeights | None = None,
        step_config: ETTRTrainStepConfig | None = None,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.objective_config = objective_config
        self.step_config = ETTRTrainStepConfig() if step_config is None else step_config
        self.step_config.validate()
        if (
            optimizer.receipt.complete_system_parameters
            != model.parameter_receipt().complete_system_parameters
        ):
            raise TheoryReactorError(
                "ETTR optimizer is not bound to the supplied model"
            )
        self.runner = CausalETTREpisodeRunner(model)
        self.objective = ETTRCompositeObjective(
            objective_config,
            weights=objective_weights,
        )

    def forward_loss(
        self,
        batch: ETTRContinuationBatch,
    ) -> ETTRCompositeLoss:
        batch.validate(
            self.model.config,
            self.objective_config,
        )
        steps = batch.transaction_targets.opcode.shape[1]
        output = self.runner(
            batch.episodes,
            reactor_steps=steps,
            hard=self.step_config.hard_transactions,
            validate_batch=False,
        )
        return self.objective(batch.objective_batch(output))

    def update(
        self,
        batches: Sequence[ETTRContinuationBatch],
    ) -> ETTRUpdateReceipt:
        if len(batches) != self.step_config.gradient_accumulation_steps:
            raise TheoryReactorError("ETTR accumulation window differs")
        scale = self.optimizer.apply_schedule()
        self.optimizer.zero_grad(set_to_none=True)
        fields = {
            "total": [],
            "token_lm": [],
            "packet": [],
            "transaction": [],
            "equivariance": [],
            "commit_halt": [],
            "sparsity": [],
            "anti_bypass": [],
        }
        token_counts: list[torch.Tensor] = []
        for batch in batches:
            device_type = batch.episodes.world.tokens.device.type
            with torch.autocast(
                device_type=device_type,
                dtype=self.step_config.autocast_dtype,
                enabled=device_type in {"cuda", "cpu"},
            ):
                loss = self.forward_loss(batch)
                scaled = loss.total / self.step_config.gradient_accumulation_steps
            scaled.backward()
            for name in fields:
                fields[name].append(getattr(loss, name).detach())
            token_counts.append(loss.receipt.lm_target_tokens.detach())
        trainable = tuple(
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable,
            self.step_config.gradient_clip,
        )
        torch._assert_async(
            torch.isfinite(gradient_norm),
            "ETTR gradient norm is nonfinite",
        )
        self.optimizer.step()

        def mean(name: str) -> torch.Tensor:
            return torch.stack(fields[name]).mean()

        return ETTRUpdateReceipt(
            optimizer_step=self.optimizer.next_update,
            learning_rate_scale=scale,
            total_loss=mean("total"),
            token_lm_loss=mean("token_lm"),
            packet_loss=mean("packet"),
            transaction_loss=mean("transaction"),
            equivariance_loss=mean("equivariance"),
            commit_halt_loss=mean("commit_halt"),
            sparsity_loss=mean("sparsity"),
            anti_bypass_loss=mean("anti_bypass"),
            gradient_norm=gradient_norm.detach(),
            supervised_token_count=torch.stack(token_counts).sum(),
        )


__all__ = [
    "ETTRTrainStep",
    "ETTRTrainStepConfig",
    "ETTRUpdateReceipt",
]
