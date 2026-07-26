"""Causal episode lifecycle for the endogenous typed-theory reactor.

An episode has three independently causal token streams:

``WORLD -> sealed typed state -> COMMAND transactions -> QUERY answer``.

Transformer context is reset at every arrow.  The only information crossing
from WORLD into COMMAND or QUERY is ``TypedTheoryState``.  This module is an
architecture/training interface; it does not parse source text, execute an
ontology, or construct labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import torch
import torch.nn as nn

from endogenous_typed_theory_reactor import (
    EndogenousTypedTheoryReactorGPT,
    ReactorTrace,
    TheoryReactorError,
    TypedTheoryState,
)
from model import _supervised_lm_loss


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ETTREpisodeSegment:
    """One right-padded, independently causal token segment."""

    tokens: torch.Tensor
    targets: torch.Tensor
    attention_mask: torch.Tensor

    @classmethod
    def from_tokens(
        cls,
        tokens: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> "ETTREpisodeSegment":
        if tokens.ndim != 2 or tokens.dtype != torch.long:
            raise TheoryReactorError("episode tokens must be a rank-two long tensor")
        if tokens.shape[1] < 2:
            raise TheoryReactorError(
                "episode segments need at least two token positions"
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(tokens, dtype=torch.bool)
        else:
            if attention_mask.shape != tokens.shape:
                raise TheoryReactorError("episode attention mask geometry differs")
            if attention_mask.device != tokens.device:
                raise TheoryReactorError(
                    "episode tokens and attention mask must share one device"
                )
            if attention_mask.dtype != torch.bool:
                _require_tensor(
                    ((attention_mask == 0) | (attention_mask == 1)).all(),
                    "episode attention mask must be binary",
                )
            attention_mask = attention_mask.to(
                device=tokens.device,
                dtype=torch.bool,
            )
        targets = torch.full_like(tokens, -1)
        causal_pairs = attention_mask[:, :-1] & attention_mask[:, 1:]
        targets[:, :-1] = torch.where(
            causal_pairs,
            tokens[:, 1:],
            torch.full_like(tokens[:, 1:], -1),
        )
        segment = cls(
            tokens=tokens,
            targets=targets,
            attention_mask=attention_mask,
        )
        segment.validate()
        return segment

    def validate(self) -> None:
        if (
            self.tokens.ndim != 2
            or self.tokens.dtype != torch.long
            or self.targets.shape != self.tokens.shape
            or self.targets.dtype != torch.long
            or self.attention_mask.shape != self.tokens.shape
        ):
            raise TheoryReactorError("episode segment geometry differs")
        if self.tokens.shape[1] < 2:
            raise TheoryReactorError(
                "episode segments need at least two token positions"
            )
        mask = self.attention_mask
        if mask.dtype != torch.bool:
            _require_tensor(
                ((mask == 0) | (mask == 1)).all(),
                "episode attention mask must be binary",
            )
            mask = mask.bool()
        _require_tensor(
            ~(mask[:, 1:].to(torch.int8) > mask[:, :-1].to(torch.int8)).any(),
            "episode attention mask must be right padded",
        )
        _require_tensor(
            ~(self.targets[~mask] != -1).any(),
            "padded episode targets must be ignored",
        )
        _require_tensor(
            self.targets.ne(-1).any(dim=1).all(),
            "an episode segment row has no supervised token",
        )
        expected_targets = torch.full_like(self.tokens, -1)
        causal_pairs = mask[:, :-1] & mask[:, 1:]
        expected_targets[:, :-1] = torch.where(
            causal_pairs,
            self.tokens[:, 1:],
            torch.full_like(self.tokens[:, 1:], -1),
        )
        _require_tensor(
            (self.targets == expected_targets).all(),
            "episode targets must equal the causal token shift",
        )
        if (
            self.tokens.device != self.targets.device
            or self.tokens.device != self.attention_mask.device
        ):
            raise TheoryReactorError("episode segment tensors must share one device")

    @property
    def supervised_tokens(self) -> torch.Tensor:
        return self.targets.ne(-1).sum()


@dataclass(frozen=True, slots=True)
class ETTREpisodeBatch:
    """A batch whose rows all begin a fresh source-deleted episode."""

    episode_ids: tuple[str, ...]
    reset_mask: torch.Tensor
    world: ETTREpisodeSegment
    command: ETTREpisodeSegment
    query: ETTREpisodeSegment

    def validate(self) -> None:
        self.world.validate()
        self.command.validate()
        self.query.validate()
        batch = self.world.tokens.shape[0]
        if (
            self.command.tokens.shape[0] != batch
            or self.query.tokens.shape[0] != batch
            or len(self.episode_ids) != batch
            or len(set(self.episode_ids)) != batch
            or any(_SHA256.fullmatch(value) is None for value in self.episode_ids)
        ):
            raise TheoryReactorError("episode batch identity or batch geometry differs")
        if (
            self.reset_mask.shape != (batch,)
            or self.reset_mask.device != self.world.tokens.device
        ):
            raise TheoryReactorError("episode reset mask differs")
        reset = self.reset_mask
        if reset.dtype != torch.bool:
            _require_tensor(
                ((reset == 0) | (reset == 1)).all(),
                "episode reset mask must be binary",
            )
            reset = reset.bool()
        _require_tensor(
            reset.all(),
            "every ETTR batch row must explicitly reset",
        )
        devices = {
            self.world.tokens.device,
            self.command.tokens.device,
            self.query.tokens.device,
        }
        if len(devices) != 1:
            raise TheoryReactorError("episode segments must share one device")


@dataclass(frozen=True, slots=True)
class ETTREpisodeLosses:
    world: torch.Tensor
    command: torch.Tensor
    query: torch.Tensor
    token_lm: torch.Tensor
    supervised_token_count: torch.Tensor


@dataclass(frozen=True, slots=True)
class ETTREpisodeOutput:
    world_logits: torch.Tensor
    command_logits: torch.Tensor
    query_logits: torch.Tensor
    initial_state: TypedTheoryState
    terminal_state: TypedTheoryState
    trace: ReactorTrace
    losses: ETTREpisodeLosses | None


class CausalETTREpisodeRunner(nn.Module):
    """Run complete episodes without any cross-segment transformer cache."""

    def __init__(self, model: EndogenousTypedTheoryReactorGPT):
        super().__init__()
        self.model = model

    def forward(
        self,
        batch: ETTREpisodeBatch,
        *,
        reactor_steps: int,
        hard: bool = False,
        validate_batch: bool = True,
        compute_losses: bool = True,
    ) -> ETTREpisodeOutput:
        if validate_batch:
            batch.validate()
        world_hidden, world_logits = self._segment_forward(batch.world.tokens)
        initial_state = self.model.compiler(
            world_hidden,
            attention_mask=batch.world.attention_mask,
            hard=hard,
        )

        command_hidden, command_logits = self._segment_forward(batch.command.tokens)
        terminal_state, trace = self.model.reactor(
            initial_state,
            steps=reactor_steps,
            hard=hard,
            command_hidden=command_hidden,
            command_attention_mask=batch.command.attention_mask,
        )

        query_hidden = self.model._encode_to_stage(
            batch.query.tokens,
            pos=0,
        )
        query_hidden = query_hidden + self.model.query_reader(
            query_hidden,
            terminal_state,
            attention_mask=batch.query.attention_mask,
        )
        query_hidden = self.model._decode_from_stage(
            query_hidden,
            pos=0,
        )
        query_logits = self.model.base.head(self.model.base.norm(query_hidden))

        losses = (
            self._losses(
                batch,
                world_logits,
                command_logits,
                query_logits,
            )
            if compute_losses
            else None
        )
        return ETTREpisodeOutput(
            world_logits=world_logits,
            command_logits=command_logits,
            query_logits=query_logits,
            initial_state=initial_state,
            terminal_state=terminal_state,
            trace=trace,
            losses=losses,
        )

    def _segment_forward(
        self,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return stage residual and logits from one reset causal stream."""

        hidden = self.model._encode_to_stage(tokens, pos=0)
        stage_hidden = hidden
        hidden = self.model._decode_from_stage(hidden, pos=0)
        logits = self.model.base.head(self.model.base.norm(hidden))
        return stage_hidden, logits

    def _losses(
        self,
        batch: ETTREpisodeBatch,
        world_logits: torch.Tensor,
        command_logits: torch.Tensor,
        query_logits: torch.Tensor,
    ) -> ETTREpisodeLosses:
        triples = (
            (world_logits, batch.world.targets),
            (command_logits, batch.command.targets),
            (query_logits, batch.query.targets),
        )
        segment_losses = tuple(
            _supervised_lm_loss(
                logits,
                targets,
                self.model.base.cfg.zloss,
            )
            for logits, targets in triples
        )
        counts = tuple(targets.ne(-1).sum() for _, targets in triples)
        total_count = sum(counts)
        token_lm = sum(
            loss * count.to(loss.dtype)
            for loss, count in zip(
                segment_losses,
                counts,
                strict=True,
            )
        ) / total_count.to(segment_losses[0].dtype)
        return ETTREpisodeLosses(
            world=segment_losses[0],
            command=segment_losses[1],
            query=segment_losses[2],
            token_lm=token_lm,
            supervised_token_count=total_count.detach(),
        )


def _require_tensor(
    condition: torch.Tensor,
    message: str,
) -> None:
    if condition.ndim:
        raise TheoryReactorError("internal episode assertion must be scalar")
    if condition.device.type == "cuda" or torch.compiler.is_compiling():
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise TheoryReactorError(message)


__all__ = [
    "CausalETTREpisodeRunner",
    "ETTREpisodeBatch",
    "ETTREpisodeLosses",
    "ETTREpisodeOutput",
    "ETTREpisodeSegment",
]
