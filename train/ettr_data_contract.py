"""Frozen continuation-data and runner-to-objective contract for ETTR.

Candidate-visible tensors contain token segments and generic categorical
supervision only.  No family identifier, semantic executor, parser product,
host callback, answer verifier, or continuous source payload is admitted.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import re

import torch

from endogenous_typed_theory_reactor import (
    TheoryReactorConfig,
    TheoryReactorError,
)
from ettr_episode import ETTREpisodeBatch, ETTREpisodeOutput
from ettr_objectives import (
    ETTRObjectiveBatch,
    ETTRObjectiveConfig,
    ETTRPacketTargets,
    ETTRTokenTargets,
    ETTRTransactionPredictions,
    ETTRTransactionTargets,
    ETTRVariantAlignment,
)


ETTR_CONTINUATION_SCHEMA = "shohin-ettr-continuation-data-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ETTRContinuationBatch:
    """One architecture batch plus its generic offline supervision."""

    episodes: ETTREpisodeBatch
    packet_targets: ETTRPacketTargets
    transaction_targets: ETTRTransactionTargets
    initial_committed: torch.Tensor
    initial_halted: torch.Tensor
    equivariance: ETTRVariantAlignment | None

    def validate(
        self,
        reactor_config: TheoryReactorConfig,
        objective_config: ETTRObjectiveConfig,
    ) -> None:
        self.episodes.validate()
        batch = self.episodes.world.tokens.shape[0]
        steps = self.transaction_targets.opcode.shape[1]
        if (
            objective_config.num_slots != reactor_config.num_slots
            or objective_config.num_types != reactor_config.num_types
            or objective_config.num_relations != reactor_config.num_relations
            or objective_config.num_value_codes != reactor_config.num_value_codes
            or objective_config.relation_edge_budget != reactor_config.max_edges
            or steps > reactor_config.max_steps
            or self.packet_targets.active.shape != (batch, reactor_config.num_slots)
            or self.transaction_targets.opcode.shape[0] != batch
            or self.initial_committed.shape != (batch,)
            or self.initial_halted.shape != (batch,)
            or self.initial_committed.dtype != torch.bool
            or self.initial_halted.dtype != torch.bool
        ):
            raise TheoryReactorError("ETTR continuation/objective geometry differs")
        devices = {
            self.episodes.world.tokens.device,
            self.packet_targets.active.device,
            self.transaction_targets.opcode.device,
            self.initial_committed.device,
            self.initial_halted.device,
        }
        if self.equivariance is not None:
            devices.add(self.equivariance.left_index.device)
        if len(devices) != 1:
            raise TheoryReactorError("ETTR continuation tensors must share one device")
        declared = {
            field.name
            for value in (
                self,
                self.packet_targets,
                self.transaction_targets,
            )
            for field in fields(value)
        }
        if any("family" in name or "ontology" in name for name in declared):
            raise TheoryReactorError(
                "ETTR continuation contract exposes a family label"
            )

    def objective_batch(
        self,
        output: ETTREpisodeOutput,
    ) -> ETTRObjectiveBatch:
        """Join reset segment logits without creating boundary targets."""

        if not isinstance(output, ETTREpisodeOutput):
            raise TheoryReactorError("ETTR episode output type differs")
        segments = (
            self.episodes.world,
            self.episodes.command,
            self.episodes.query,
        )
        logits = torch.cat(
            (
                output.world_logits,
                output.command_logits,
                output.query_logits,
            ),
            dim=1,
        )
        token_ids = torch.cat(
            tuple(segment.tokens for segment in segments),
            dim=1,
        )
        mask = torch.cat(
            tuple(segment.attention_mask for segment in segments),
            dim=1,
        ).bool()
        reset_mask = torch.zeros_like(mask)
        cursor = 0
        for segment in segments:
            reset_mask[:, cursor] = True
            cursor += segment.tokens.shape[1]
        return ETTRObjectiveBatch(
            token_logits=logits,
            token_targets=ETTRTokenTargets(
                token_ids=token_ids,
                mask=mask,
                reset_mask=reset_mask,
            ),
            packet_prediction=output.initial_state,
            packet_targets=self.packet_targets,
            transactions=(ETTRTransactionPredictions.from_reactor_trace(output.trace)),
            transaction_targets=self.transaction_targets,
            initial_committed=self.initial_committed,
            initial_halted=self.initial_halted,
            equivariance=self.equivariance,
        )


@dataclass(frozen=True, slots=True)
class ETTRContinuationManifest:
    schema: str
    protected_checkpoint_sha256: str
    tokenizer_sha256: str
    qualification_payload_sha256: str
    hybrid_payload_sha256: str
    train_rows: int
    validation_rows: int
    train_payload_sha256: str
    validation_payload_sha256: str
    source_deleted: bool
    immutable_snapshot: bool
    live_writer_input: bool
    family_label_fields: tuple[str, ...]

    def validate(self) -> None:
        if self.schema != ETTR_CONTINUATION_SCHEMA:
            raise TheoryReactorError("ETTR continuation manifest schema differs")
        for name in (
            "protected_checkpoint_sha256",
            "tokenizer_sha256",
            "qualification_payload_sha256",
            "hybrid_payload_sha256",
            "train_payload_sha256",
            "validation_payload_sha256",
        ):
            if not _SHA256.fullmatch(getattr(self, name)):
                raise TheoryReactorError(f"ETTR continuation manifest {name} differs")
        if (
            self.train_rows <= 0
            or self.validation_rows <= 0
            or not self.source_deleted
            or not self.immutable_snapshot
            or self.live_writer_input
            or self.family_label_fields
            or self.train_payload_sha256 == self.validation_payload_sha256
        ):
            raise TheoryReactorError("ETTR continuation data custody differs")


__all__ = [
    "ETTR_CONTINUATION_SCHEMA",
    "ETTRContinuationBatch",
    "ETTRContinuationManifest",
]
