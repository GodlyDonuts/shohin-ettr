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
from ettr_data_contract import (
    ETTRContinuationBatch,
    ETTRContinuationManifest,
    ETTRPacketSufficiencyIndex,
)
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
    hard_transactions: bool = True
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
    world_intervention_loss: torch.Tensor
    command_intervention_loss: torch.Tensor
    world_query_binding_loss: torch.Tensor
    command_query_binding_loss: torch.Tensor
    transaction_loss: torch.Tensor
    equivariance_loss: torch.Tensor
    commit_halt_loss: torch.Tensor
    sparsity_loss: torch.Tensor
    anti_bypass_loss: torch.Tensor
    gradient_norm: torch.Tensor
    supervised_token_count: torch.Tensor
    supervised_world_query_pairs: torch.Tensor
    supervised_command_query_pairs: torch.Tensor
    world_query_margin_satisfied: torch.Tensor
    command_query_margin_satisfied: torch.Tensor


class ETTRTrainStep(nn.Module):
    """One exact optimizer update over one accumulation window."""

    def __init__(
        self,
        model: EndogenousTypedTheoryReactorGPT,
        optimizer: ETTROptimizerBundle,
        objective_config: ETTRObjectiveConfig,
        *,
        manifest: ETTRContinuationManifest,
        packet_sufficiency: ETTRPacketSufficiencyIndex,
        manifest_sha256: str,
        objective_weights: ETTRObjectiveWeights | None = None,
        step_config: ETTRTrainStepConfig | None = None,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.objective_config = objective_config
        self.step_config = ETTRTrainStepConfig() if step_config is None else step_config
        self.step_config.validate()
        if len(manifest_sha256) != 64:
            raise TheoryReactorError("ETTR trainer snapshot receipt differs")
        try:
            bytes.fromhex(manifest_sha256)
        except ValueError as error:
            raise TheoryReactorError(
                "ETTR trainer snapshot receipt differs"
            ) from error
        self.manifest_sha256 = manifest_sha256
        if not isinstance(manifest, ETTRContinuationManifest):
            raise TheoryReactorError("ETTR trainer manifest type differs")
        if not isinstance(packet_sufficiency, ETTRPacketSufficiencyIndex):
            raise TheoryReactorError(
                "ETTR trainer packet sufficiency index type differs"
            )
        manifest.validate()
        if manifest.sha256() != manifest_sha256:
            raise TheoryReactorError("ETTR trainer manifest hash differs")
        if packet_sufficiency.receipt != manifest.packet_sufficiency_receipt():
            raise TheoryReactorError(
                "ETTR trainer packet sufficiency receipt differs"
            )
        if (
            packet_sufficiency.train_batches
            != manifest.packet_sufficiency_train_batches
            or packet_sufficiency.validation_batches
            != manifest.packet_sufficiency_validation_batches
            or packet_sufficiency.train_rows != manifest.train_rows
            or packet_sufficiency.validation_rows != manifest.validation_rows
            or packet_sufficiency.train_contexts
            != manifest.packet_sufficiency_train_contexts
            or packet_sufficiency.validation_contexts
            != manifest.packet_sufficiency_validation_contexts
            or packet_sufficiency.train_payload_sha256
            != manifest.train_payload_sha256
            or packet_sufficiency.validation_payload_sha256
            != manifest.validation_payload_sha256
        ):
            raise TheoryReactorError(
                "ETTR trainer packet sufficiency split differs"
            )
        self.manifest = manifest
        self.dataset_sha256 = manifest.dataset_sha256
        self.packet_sufficiency = packet_sufficiency
        self._poisoned = False
        optimizer.assert_bound_to(model)
        optimizer.assert_healthy()
        self.runner = CausalETTREpisodeRunner(model)
        self.objective = ETTRCompositeObjective(
            objective_config,
            weights=objective_weights,
        )

    def forward_loss(
        self,
        batch: ETTRContinuationBatch,
    ) -> ETTRCompositeLoss:
        if self._poisoned:
            raise TheoryReactorError(
                "ETTR train step is fail-stop; restart from the last "
                "verified checkpoint"
            )
        self.optimizer.assert_healthy()
        if (
            batch.manifest_sha256 != self.manifest_sha256
            or batch.dataset_sha256 != self.dataset_sha256
        ):
            raise TheoryReactorError("ETTR batch snapshot differs from the trainer")
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
            compute_losses=False,
        )
        (
            world_packet,
            world_command,
            world_target,
            command_packet,
            command_command,
            command_target,
        ) = batch.causal_rectangles.intervention_indices()
        interventions = self.runner.intervene(
            batch.episodes,
            output.initial_state,
            reactor_steps=steps,
            world_packet_index=world_packet,
            world_command_index=world_command,
            world_query_index=world_target,
            command_packet_index=command_packet,
            command_command_index=command_command,
            command_query_index=command_target,
            hard=self.step_config.hard_transactions,
        )
        return self.objective(
            batch.objective_batch(output, interventions)
        )

    def update(
        self,
        batches: Sequence[ETTRContinuationBatch],
    ) -> ETTRUpdateReceipt:
        if self._poisoned:
            raise TheoryReactorError(
                "ETTR train step is fail-stop; restart from the last "
                "verified checkpoint"
            )
        self.optimizer.assert_healthy()
        if len(batches) != self.step_config.gradient_accumulation_steps:
            raise TheoryReactorError("ETTR accumulation window differs")
        for batch in batches:
            batch.validate(
                self.model.config,
                self.objective_config,
            )
        self.packet_sufficiency.verify_train(tuple(batches))
        self.optimizer.zero_grad(set_to_none=True)
        fields = {
            "total": [],
            "token_lm": [],
            "packet": [],
            "world_intervention": [],
            "command_intervention": [],
            "world_query_binding": [],
            "command_query_binding": [],
            "transaction": [],
            "equivariance": [],
            "commit_halt": [],
            "sparsity": [],
            "anti_bypass": [],
        }
        token_counts: list[torch.Tensor] = []
        world_query_pairs: list[torch.Tensor] = []
        command_query_pairs: list[torch.Tensor] = []
        world_query_margin: list[torch.Tensor] = []
        command_query_margin: list[torch.Tensor] = []
        try:
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
                world_query_pairs.append(
                    loss.receipt.supervised_world_query_pairs.detach()
                )
                command_query_pairs.append(
                    loss.receipt.supervised_command_query_pairs.detach()
                )
                world_query_margin.append(
                    loss.receipt.world_query_margin_satisfied.detach()
                )
                command_query_margin.append(
                    loss.receipt.command_query_margin_satisfied.detach()
                )
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True)
            raise
        try:
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
            scale = self.optimizer.apply_schedule()
            self.optimizer.step()
        except BaseException as error:
            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer.mark_failed_update()
            self._poisoned = True
            raise TheoryReactorError(
                "ETTR optimizer update failed after backward; restart from "
                "the last verified checkpoint"
            ) from error

        def mean(name: str) -> torch.Tensor:
            return torch.stack(fields[name]).mean()

        return ETTRUpdateReceipt(
            optimizer_step=self.optimizer.next_update,
            learning_rate_scale=scale,
            total_loss=mean("total"),
            token_lm_loss=mean("token_lm"),
            packet_loss=mean("packet"),
            world_intervention_loss=mean("world_intervention"),
            command_intervention_loss=mean("command_intervention"),
            world_query_binding_loss=mean("world_query_binding"),
            command_query_binding_loss=mean("command_query_binding"),
            transaction_loss=mean("transaction"),
            equivariance_loss=mean("equivariance"),
            commit_halt_loss=mean("commit_halt"),
            sparsity_loss=mean("sparsity"),
            anti_bypass_loss=mean("anti_bypass"),
            gradient_norm=gradient_norm.detach(),
            supervised_token_count=torch.stack(token_counts).sum(),
            supervised_world_query_pairs=torch.stack(world_query_pairs).sum(),
            supervised_command_query_pairs=torch.stack(command_query_pairs).sum(),
            world_query_margin_satisfied=torch.stack(world_query_margin).sum(),
            command_query_margin_satisfied=torch.stack(command_query_margin).sum(),
        )


__all__ = [
    "ETTRTrainStep",
    "ETTRTrainStepConfig",
    "ETTRUpdateReceipt",
]
