"""Bounded ETTR optimizer-step core.

This is a library component, not a trainer or launcher.  It receives already
frozen continuation batches and has no shard, filesystem, checkpoint, Slurm,
or network access.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from endogenous_typed_theory_reactor import (
    EndogenousTypedTheoryReactorGPT,
    TRANSACTION_COUNT,
    TheoryReactorError,
    TransactionPolicy,
    TypedTheoryState,
)
from ettr_data_contract import (
    ETTRContinuationBatch,
    ETTRContinuationManifest,
    ETTRPacketSufficiencyVerifier,
)
from ettr_episode import CausalETTREpisodeRunner
from ettr_objectives import (
    ETTRCompositeLoss,
    ETTRCompositeObjective,
    ETTRObjectiveConfig,
    ETTRObjectiveWeights,
    ETTRPacketTargets,
    ETTRTransactionPredictions,
    ETTRTransactionTargets,
    _transaction_prediction_loss,
)
from ettr_optimization import ETTROptimizerBundle


_LOSS_FIELDS = (
    "total",
    "token_lm",
    "packet",
    "world_intervention",
    "command_intervention",
    "world_query_binding",
    "command_query_binding",
    "transaction",
    "equivariance",
    "commit_halt",
    "sparsity",
    "anti_bypass",
)
_RECEIPT_FIELDS = (
    "lm_target_tokens",
    "supervised_world_query_pairs",
    "supervised_command_query_pairs",
    "world_query_contrast_pairs",
    "command_query_contrast_pairs",
    "world_query_invariance_pairs",
    "command_query_invariance_pairs",
    "world_query_margin_satisfied",
    "command_query_margin_satisfied",
)
_COMPILE_BACKENDS = {"eager", "inductor"}
_COMPILE_MODES = {
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
}
_GRADIENT_CLIP_MODES = {"global", "owner", "component"}


@dataclass(frozen=True, slots=True)
class ETTRTrainStepConfig:
    gradient_accumulation_steps: int = 1
    gradient_clip: float = 1.0
    gradient_clip_mode: str = "global"
    hard_transactions: bool = True
    autocast_dtype: torch.dtype = torch.bfloat16
    compile_backend: str | None = None
    compile_mode: str | None = None
    teacher_forced_transaction_weight: float = 0.0

    def validate(self) -> None:
        if (
            not isinstance(self.gradient_accumulation_steps, int)
            or self.gradient_accumulation_steps <= 0
            or not 0 < self.gradient_clip <= 100
            or self.gradient_clip_mode not in _GRADIENT_CLIP_MODES
            or not isinstance(self.hard_transactions, bool)
            or self.autocast_dtype not in (torch.bfloat16, torch.float16)
            or (
                self.compile_backend is not None
                and self.compile_backend not in _COMPILE_BACKENDS
            )
            or (
                self.compile_backend is None
                and self.compile_mode is not None
            )
            or (
                self.compile_mode is not None
                and self.compile_mode not in _COMPILE_MODES
            )
            or not isinstance(
                self.teacher_forced_transaction_weight,
                float,
            )
            or not math.isfinite(
                self.teacher_forced_transaction_weight
            )
            or not 0.0
            <= self.teacher_forced_transaction_weight
            <= 1_000.0
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
    base_gradient_norm: torch.Tensor
    architecture_gradient_norm: torch.Tensor
    compiler_gradient_norm: torch.Tensor
    reactor_gradient_norm: torch.Tensor
    query_reader_gradient_norm: torch.Tensor
    supervised_token_count: torch.Tensor
    supervised_world_query_pairs: torch.Tensor
    supervised_command_query_pairs: torch.Tensor
    world_query_contrast_pairs: torch.Tensor
    command_query_contrast_pairs: torch.Tensor
    world_query_invariance_pairs: torch.Tensor
    command_query_invariance_pairs: torch.Tensor
    world_query_margin_satisfied: torch.Tensor
    command_query_margin_satisfied: torch.Tensor


def _packet_targets_to_state(
    targets: ETTRPacketTargets,
    model: EndogenousTypedTheoryReactorGPT,
) -> TypedTheoryState:
    """Materialize the exact initial state for training-only imitation."""

    dtype = next(model.reactor.parameters()).dtype
    active = targets.active.to(dtype)
    return TypedTheoryState(
        value_probabilities=(
            F.one_hot(
                targets.value_code,
                model.config.num_value_codes,
            ).to(dtype)
            * active.unsqueeze(-1)
        ),
        type_probabilities=(
            F.one_hot(
                targets.type_index,
                model.config.num_types,
            ).to(dtype)
            * active.unsqueeze(-1)
        ),
        relations=targets.relations.to(dtype),
        active=active,
        root=targets.root.to(dtype),
        committed=targets.committed.to(dtype),
        halted=targets.halted.to(dtype),
        step=0,
    )


def _target_policy(
    targets: ETTRTransactionTargets,
    model: EndogenousTypedTheoryReactorGPT,
    step: int,
    *,
    dtype: torch.dtype,
) -> TransactionPolicy:
    """Return the offline transaction used only to advance imitation state."""

    values = {
        "opcode": F.one_hot(
            targets.opcode[:, step],
            TRANSACTION_COUNT,
        ).to(dtype),
        "source": F.one_hot(
            targets.source[:, step],
            model.config.num_slots,
        ).to(dtype),
        "target": F.one_hot(
            targets.target[:, step],
            model.config.num_slots,
        ).to(dtype),
        "relation": F.one_hot(
            targets.relation[:, step],
            model.config.num_relations,
        ).to(dtype),
        "type_index": F.one_hot(
            targets.type_index[:, step],
            model.config.num_types,
        ).to(dtype),
        "value_code": F.one_hot(
            targets.value_code[:, step],
            model.config.num_value_codes,
        ).to(dtype),
    }
    return TransactionPolicy(
        **values,
        opcode_probabilities=values["opcode"],
        source_probabilities=values["source"],
        target_probabilities=values["target"],
        relation_probabilities=values["relation"],
        type_probabilities=values["type_index"],
        value_probabilities=values["value_code"],
    )


def _teacher_forced_transaction_predictions(
    model: EndogenousTypedTheoryReactorGPT,
    batch: ETTRContinuationBatch,
) -> ETTRTransactionPredictions:
    """Predict each action from exact prior state without changing inference."""

    targets = batch.transaction_targets
    state = _packet_targets_to_state(batch.packet_targets, model)
    command_hidden = model._encode_to_stage(
        batch.episodes.command.tokens,
        pos=0,
    )
    policies: list[TransactionPolicy] = []
    states: list[TypedTheoryState] = []
    for step in range(targets.opcode.shape[1]):
        policies.append(
            model.reactor.policy(
                state,
                hard=False,
                command_hidden=command_hidden,
                command_attention_mask=(
                    batch.episodes.command.attention_mask
                ),
                validate=False,
            )
        )
        state = model.reactor.apply(
            state,
            _target_policy(
                targets,
                model,
                step,
                dtype=state.active.dtype,
            ),
            hard=True,
            validate=False,
        )
        states.append(state)
    return ETTRTransactionPredictions(
        opcode=torch.stack(
            [policy.opcode_probabilities for policy in policies],
            dim=1,
        ),
        source=torch.stack(
            [policy.source_probabilities for policy in policies],
            dim=1,
        ),
        target=torch.stack(
            [policy.target_probabilities for policy in policies],
            dim=1,
        ),
        relation=torch.stack(
            [policy.relation_probabilities for policy in policies],
            dim=1,
        ),
        type_index=torch.stack(
            [policy.type_probabilities for policy in policies],
            dim=1,
        ),
        value_code=torch.stack(
            [policy.value_probabilities for policy in policies],
            dim=1,
        ),
        active=torch.stack([state.active for state in states], dim=1),
        committed=torch.stack(
            [state.committed for state in states],
            dim=1,
        ),
        halted=torch.stack(
            [state.halted for state in states],
            dim=1,
        ),
    )


class ETTRCompositeTrainingSubject(nn.Module):
    """Compile-safe factual, intervention, and objective tensor path."""

    def __init__(
        self,
        model: EndogenousTypedTheoryReactorGPT,
        objective_config: ETTRObjectiveConfig,
        objective_weights: ETTRObjectiveWeights | None,
        *,
        hard_transactions: bool,
        teacher_forced_transaction_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.runner = CausalETTREpisodeRunner(model)
        self.objective = ETTRCompositeObjective(
            objective_config,
            weights=objective_weights,
        )
        self.hard_transactions = hard_transactions
        self.teacher_forced_transaction_weight = (
            teacher_forced_transaction_weight
        )

    def objective_loss(
        self,
        batch: ETTRContinuationBatch,
    ) -> ETTRCompositeLoss:
        steps = batch.transaction_targets.opcode.shape[1]
        output = self.runner(
            batch.episodes,
            reactor_steps=steps,
            hard=self.hard_transactions,
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
            hard=self.hard_transactions,
        )
        loss = self.objective(
            batch.objective_batch(output, interventions)
        )
        if self.teacher_forced_transaction_weight == 0.0:
            return loss
        teacher_predictions = _teacher_forced_transaction_predictions(
            self.runner.model,
            batch,
        )
        teacher_transaction, _ = _transaction_prediction_loss(
            teacher_predictions,
            batch.transaction_targets,
            self.objective.config,
        )
        weighted_teacher = (
            self.teacher_forced_transaction_weight
            * teacher_transaction
        )
        return replace(
            loss,
            total=(
                loss.total
                + self.objective.weights.transaction
                * weighted_teacher
            ),
            transaction=loss.transaction + weighted_teacher,
        )

    def forward(
        self,
        batch: ETTRContinuationBatch,
    ) -> tuple[torch.Tensor, ...]:
        loss = self.objective_loss(batch)
        return tuple(getattr(loss, name) for name in _LOSS_FIELDS) + tuple(
            getattr(loss.receipt, name) for name in _RECEIPT_FIELDS
        )


class ETTRTrainStep(nn.Module):
    """One exact optimizer update over one accumulation window."""

    def __init__(
        self,
        model: EndogenousTypedTheoryReactorGPT,
        optimizer: ETTROptimizerBundle,
        objective_config: ETTRObjectiveConfig,
        *,
        manifest: ETTRContinuationManifest,
        packet_sufficiency: ETTRPacketSufficiencyVerifier,
        manifest_sha256: str,
        objective_weights: ETTRObjectiveWeights | None = None,
        step_config: ETTRTrainStepConfig | None = None,
        gradient_synchronizer: (
            Callable[[Sequence[nn.Parameter]], None] | None
        ) = None,
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
        if not isinstance(packet_sufficiency, ETTRPacketSufficiencyVerifier):
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
        if gradient_synchronizer is not None and not callable(
            gradient_synchronizer
        ):
            raise TheoryReactorError(
                "ETTR gradient synchronizer is not callable"
            )
        self.gradient_synchronizer = gradient_synchronizer
        self._poisoned = False
        optimizer.assert_bound_to(model)
        optimizer.assert_healthy()
        subject = ETTRCompositeTrainingSubject(
            model,
            objective_config,
            objective_weights,
            hard_transactions=self.step_config.hard_transactions,
            teacher_forced_transaction_weight=(
                self.step_config.teacher_forced_transaction_weight
            ),
        )
        object.__setattr__(self, "eager_subject", subject)
        if self.step_config.compile_backend is None:
            training_subject = subject
        else:
            compile_options: dict[str, str] = {
                "backend": self.step_config.compile_backend,
            }
            if self.step_config.compile_mode is not None:
                compile_options["mode"] = self.step_config.compile_mode
            training_subject = torch.compile(
                subject,
                **compile_options,
            )
        object.__setattr__(self, "training_subject", training_subject)

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
        return self.eager_subject.objective_loss(batch)

    def _training_forward(
        self,
        batch: ETTRContinuationBatch,
    ) -> tuple[
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
    ]:
        if (
            self.step_config.compile_backend is not None
            and batch.episodes.world.tokens.device.type == "cuda"
        ):
            torch.compiler.cudagraph_mark_step_begin()
        values = self.training_subject(batch)
        expected = len(_LOSS_FIELDS) + len(_RECEIPT_FIELDS)
        if not isinstance(values, tuple) or len(values) != expected:
            raise TheoryReactorError(
                "ETTR compiled objective output differs"
            )
        return (
            values[: len(_LOSS_FIELDS)],
            values[len(_LOSS_FIELDS) :],
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
            if (
                batch.manifest_sha256 != self.manifest_sha256
                or batch.dataset_sha256 != self.dataset_sha256
            ):
                raise TheoryReactorError(
                    "ETTR batch snapshot differs from the trainer"
                )
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
        world_query_contrast: list[torch.Tensor] = []
        command_query_contrast: list[torch.Tensor] = []
        world_query_invariance: list[torch.Tensor] = []
        command_query_invariance: list[torch.Tensor] = []
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
                    losses, counts = self._training_forward(batch)
                    scaled = (
                        losses[0]
                        / self.step_config.gradient_accumulation_steps
                    )
                scaled.backward()
                for name, value in zip(
                    _LOSS_FIELDS,
                    losses,
                    strict=True,
                ):
                    fields[name].append(value.detach())
                count_values = dict(
                    zip(_RECEIPT_FIELDS, counts, strict=True)
                )
                token_counts.append(
                    count_values["lm_target_tokens"].detach()
                )
                world_query_pairs.append(
                    count_values[
                        "supervised_world_query_pairs"
                    ].detach()
                )
                command_query_pairs.append(
                    count_values[
                        "supervised_command_query_pairs"
                    ].detach()
                )
                world_query_contrast.append(
                    count_values["world_query_contrast_pairs"].detach()
                )
                command_query_contrast.append(
                    count_values[
                        "command_query_contrast_pairs"
                    ].detach()
                )
                world_query_invariance.append(
                    count_values[
                        "world_query_invariance_pairs"
                    ].detach()
                )
                command_query_invariance.append(
                    count_values[
                        "command_query_invariance_pairs"
                    ].detach()
                )
                world_query_margin.append(
                    count_values[
                        "world_query_margin_satisfied"
                    ].detach()
                )
                command_query_margin.append(
                    count_values[
                        "command_query_margin_satisfied"
                    ].detach()
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
            if self.gradient_synchronizer is not None:
                self.gradient_synchronizer(trainable)
            base_parameter_ids = {
                id(parameter) for parameter in self.model.base.parameters()
            }
            base_trainable = tuple(
                parameter
                for parameter in trainable
                if id(parameter) in base_parameter_ids
            )
            architecture_trainable = tuple(
                parameter
                for parameter in trainable
                if id(parameter) not in base_parameter_ids
            )
            component_trainable = tuple(
                tuple(
                    parameter
                    for parameter in component.parameters()
                    if parameter.requires_grad
                )
                for component in (
                    self.model.compiler,
                    self.model.reactor,
                    self.model.query_reader,
                )
            )
            component_parameter_ids = tuple(
                {id(parameter) for parameter in parameters}
                for parameters in component_trainable
            )
            if (
                any(
                    left & right
                    for index, left in enumerate(component_parameter_ids)
                    for right in component_parameter_ids[index + 1 :]
                )
                or set().union(*component_parameter_ids)
                != {id(parameter) for parameter in architecture_trainable}
            ):
                raise TheoryReactorError(
                    "ETTR architecture gradient ownership differs"
                )
            (
                compiler_trainable,
                reactor_trainable,
                query_reader_trainable,
            ) = component_trainable
            if self.step_config.gradient_clip_mode == "component":
                base_gradient_norm = self._clip_gradient_owner(
                    base_trainable
                )
                compiler_gradient_norm = self._clip_gradient_owner(
                    compiler_trainable
                )
                reactor_gradient_norm = self._clip_gradient_owner(
                    reactor_trainable
                )
                query_reader_gradient_norm = self._clip_gradient_owner(
                    query_reader_trainable
                )
                architecture_gradient_norm = torch.linalg.vector_norm(
                    torch.stack(
                        (
                            compiler_gradient_norm.float(),
                            reactor_gradient_norm.float(),
                            query_reader_gradient_norm.float(),
                        )
                    )
                )
                gradient_norm = torch.linalg.vector_norm(
                    torch.stack(
                        (
                            base_gradient_norm.float(),
                            compiler_gradient_norm.float(),
                            reactor_gradient_norm.float(),
                            query_reader_gradient_norm.float(),
                        )
                    )
                )
            elif self.step_config.gradient_clip_mode == "owner":
                compiler_gradient_norm = self._gradient_norm(
                    compiler_trainable
                )
                reactor_gradient_norm = self._gradient_norm(
                    reactor_trainable
                )
                query_reader_gradient_norm = self._gradient_norm(
                    query_reader_trainable
                )
                base_gradient_norm = self._clip_gradient_owner(
                    base_trainable
                )
                architecture_gradient_norm = self._clip_gradient_owner(
                    architecture_trainable
                )
                gradient_norm = torch.linalg.vector_norm(
                    torch.stack(
                        (
                            base_gradient_norm.float(),
                            architecture_gradient_norm.float(),
                        )
                    )
                )
            else:
                base_gradient_norm = self._gradient_norm(base_trainable)
                architecture_gradient_norm = self._gradient_norm(
                    architecture_trainable
                )
                compiler_gradient_norm = self._gradient_norm(
                    compiler_trainable
                )
                reactor_gradient_norm = self._gradient_norm(
                    reactor_trainable
                )
                query_reader_gradient_norm = self._gradient_norm(
                    query_reader_trainable
                )
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    trainable,
                    self.step_config.gradient_clip,
                )
            torch._assert_async(
                torch.stack(
                    (
                        gradient_norm.float(),
                        base_gradient_norm.float(),
                        architecture_gradient_norm.float(),
                        compiler_gradient_norm.float(),
                        reactor_gradient_norm.float(),
                        query_reader_gradient_norm.float(),
                    )
                )
                .isfinite()
                .all(),
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
            base_gradient_norm=base_gradient_norm.detach(),
            architecture_gradient_norm=(
                architecture_gradient_norm.detach()
            ),
            compiler_gradient_norm=compiler_gradient_norm.detach(),
            reactor_gradient_norm=reactor_gradient_norm.detach(),
            query_reader_gradient_norm=query_reader_gradient_norm.detach(),
            supervised_token_count=torch.stack(token_counts).sum(),
            supervised_world_query_pairs=torch.stack(world_query_pairs).sum(),
            supervised_command_query_pairs=torch.stack(command_query_pairs).sum(),
            world_query_contrast_pairs=torch.stack(
                world_query_contrast
            ).sum(),
            command_query_contrast_pairs=torch.stack(
                command_query_contrast
            ).sum(),
            world_query_invariance_pairs=torch.stack(
                world_query_invariance
            ).sum(),
            command_query_invariance_pairs=torch.stack(
                command_query_invariance
            ).sum(),
            world_query_margin_satisfied=torch.stack(world_query_margin).sum(),
            command_query_margin_satisfied=torch.stack(command_query_margin).sum(),
        )

    def _clip_gradient_owner(
        self,
        parameters: tuple[nn.Parameter, ...],
    ) -> torch.Tensor:
        if not parameters:
            return torch.zeros(
                (),
                device=next(self.model.parameters()).device,
            )
        return torch.nn.utils.clip_grad_norm_(
            parameters,
            self.step_config.gradient_clip,
        )

    def _gradient_norm(
        self,
        parameters: tuple[nn.Parameter, ...],
    ) -> torch.Tensor:
        gradients = tuple(
            parameter.grad.detach().float()
            for parameter in parameters
            if parameter.grad is not None
        )
        if not gradients:
            if not parameters:
                return torch.zeros(
                    (),
                    device=next(self.model.parameters()).device,
                )
            return torch.zeros((), device=parameters[0].device)
        return torch.linalg.vector_norm(
            torch.stack(
                tuple(
                    torch.linalg.vector_norm(gradient)
                    for gradient in gradients
                )
            )
        )


__all__ = [
    "ETTRCompositeTrainingSubject",
    "ETTRTrainStep",
    "ETTRTrainStepConfig",
    "ETTRUpdateReceipt",
]
