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
    query_read_index: torch.Tensor
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
        query_tokens = self.query.tokens.shape[1]
        if (
            self.query_read_index.shape != (batch,)
            or self.query_read_index.dtype != torch.long
            or self.query_read_index.device != self.query.tokens.device
        ):
            raise TheoryReactorError("episode query read index geometry differs")
        _require_tensor(
            (
                (self.query_read_index >= 0)
                & (self.query_read_index < query_tokens - 1)
            ).all(),
            "episode query read index leaves the causal query range",
        )
        read_mask = self.query.attention_mask.gather(
            1,
            self.query_read_index[:, None],
        ).squeeze(1)
        target_mask = self.query.attention_mask.gather(
            1,
            (self.query_read_index + 1)[:, None],
        ).squeeze(1)
        _require_tensor(
            (read_mask & target_mask).all(),
            "episode query read and next-token target must both be valid",
        )
        devices = {
            self.world.tokens.device,
            self.command.tokens.device,
            self.query.tokens.device,
            self.query_read_index.device,
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


@dataclass(frozen=True, slots=True)
class ETTRInterventionOutput:
    """Free-running terminal states under orthogonal causal swaps."""

    world_terminal_state: TypedTheoryState
    world_trace: ReactorTrace
    world_query_logits: torch.Tensor
    command_terminal_state: TypedTheoryState
    command_trace: ReactorTrace
    command_query_logits: torch.Tensor


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

    def intervene(
        self,
        batch: ETTREpisodeBatch,
        initial_state: TypedTheoryState,
        *,
        reactor_steps: int,
        world_packet_index: torch.Tensor,
        world_command_index: torch.Tensor,
        world_query_index: torch.Tensor,
        command_packet_index: torch.Tensor,
        command_command_index: torch.Tensor,
        command_query_index: torch.Tensor,
        hard: bool = False,
    ) -> ETTRInterventionOutput:
        """Execute state and command swaps inside the learned reactor.

        Packet swaps keep each row's command fixed. Command swaps keep each
        row's compiled packet fixed. The two arms are concatenated into one
        reactor call and never expose a parser, executor, or answer oracle.
        """

        batch.validate()
        batch_size = batch.world.tokens.shape[0]
        intervention_size = world_packet_index.shape[0]
        if intervention_size < 1:
            raise TheoryReactorError("intervention batch is empty")
        for name, index in (
            ("world_packet_index", world_packet_index),
            ("world_command_index", world_command_index),
            ("world_query_index", world_query_index),
            ("command_packet_index", command_packet_index),
            ("command_command_index", command_command_index),
            ("command_query_index", command_query_index),
        ):
            if (
                index.shape != (intervention_size,)
                or index.dtype != torch.long
                or index.device != batch.world.tokens.device
            ):
                raise TheoryReactorError(f"{name} geometry differs")
            _require_tensor(
                ((index >= 0) & (index < batch_size)).all(),
                f"{name} leaves the batch",
            )
        command_hidden = self.model._encode_to_stage(
            batch.command.tokens,
            pos=0,
        )
        world_state = _index_state(initial_state, world_packet_index)
        command_state = _index_state(initial_state, command_packet_index)
        combined_state = _cat_states((world_state, command_state))
        combined_command = torch.cat(
            (
                command_hidden.index_select(0, world_command_index),
                command_hidden.index_select(0, command_command_index),
            ),
            dim=0,
        )
        combined_mask = torch.cat(
            (
                batch.command.attention_mask.index_select(
                    0,
                    world_command_index,
                ),
                batch.command.attention_mask.index_select(
                    0,
                    command_command_index,
                ),
            ),
            dim=0,
        )
        terminal, trace = self.model.reactor(
            combined_state,
            steps=reactor_steps,
            hard=hard,
            command_hidden=combined_command,
            command_attention_mask=combined_mask,
        )
        combined_query_index = torch.cat(
            (world_query_index, command_query_index),
            dim=0,
        )
        query_logits, _ = self.model.answer_query(
            terminal,
            batch.query.tokens.index_select(0, combined_query_index),
            targets=None,
            attention_mask=batch.query.attention_mask.index_select(
                0,
                combined_query_index,
            ),
        )
        query_read_index = batch.query_read_index.index_select(
            0,
            combined_query_index,
        )
        gathered_query_logits = query_logits.gather(
            1,
            query_read_index[:, None, None].expand(
                -1,
                1,
                query_logits.shape[-1],
            ),
        ).squeeze(1)
        return ETTRInterventionOutput(
            world_terminal_state=_slice_state(
                terminal,
                0,
                intervention_size,
            ),
            world_trace=_slice_trace(
                trace,
                0,
                intervention_size,
            ),
            world_query_logits=gathered_query_logits[:intervention_size],
            command_terminal_state=_slice_state(
                terminal,
                intervention_size,
                2 * intervention_size,
            ),
            command_trace=_slice_trace(
                trace,
                intervention_size,
                2 * intervention_size,
            ),
            command_query_logits=gathered_query_logits[intervention_size:],
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


def _index_state(
    state: TypedTheoryState,
    index: torch.Tensor,
) -> TypedTheoryState:
    return TypedTheoryState(
        value_probabilities=state.value_probabilities.index_select(0, index),
        type_probabilities=state.type_probabilities.index_select(0, index),
        relations=state.relations.index_select(0, index),
        active=state.active.index_select(0, index),
        root=state.root.index_select(0, index),
        committed=state.committed.index_select(0, index),
        halted=state.halted.index_select(0, index),
        step=state.step,
    )


def _cat_states(
    states: tuple[TypedTheoryState, ...],
) -> TypedTheoryState:
    if not states or any(state.step != states[0].step for state in states):
        raise TheoryReactorError("intervention state steps differ")
    return TypedTheoryState(
        value_probabilities=torch.cat(
            tuple(state.value_probabilities for state in states),
            dim=0,
        ),
        type_probabilities=torch.cat(
            tuple(state.type_probabilities for state in states),
            dim=0,
        ),
        relations=torch.cat(tuple(state.relations for state in states), dim=0),
        active=torch.cat(tuple(state.active for state in states), dim=0),
        root=torch.cat(tuple(state.root for state in states), dim=0),
        committed=torch.cat(tuple(state.committed for state in states), dim=0),
        halted=torch.cat(tuple(state.halted for state in states), dim=0),
        step=states[0].step,
    )


def _slice_state(
    state: TypedTheoryState,
    start: int,
    stop: int,
) -> TypedTheoryState:
    return TypedTheoryState(
        value_probabilities=state.value_probabilities[start:stop],
        type_probabilities=state.type_probabilities[start:stop],
        relations=state.relations[start:stop],
        active=state.active[start:stop],
        root=state.root[start:stop],
        committed=state.committed[start:stop],
        halted=state.halted[start:stop],
        step=state.step,
    )


def _slice_trace(
    trace: ReactorTrace,
    start: int,
    stop: int,
) -> ReactorTrace:
    return ReactorTrace(
        opcode=trace.opcode[start:stop],
        source=trace.source[start:stop],
        target=trace.target[start:stop],
        relation=trace.relation[start:stop],
        type_index=trace.type_index[start:stop],
        value_code=trace.value_code[start:stop],
        applied_opcode=trace.applied_opcode[start:stop],
        applied_source=trace.applied_source[start:stop],
        applied_target=trace.applied_target[start:stop],
        applied_relation=trace.applied_relation[start:stop],
        applied_type_index=trace.applied_type_index[start:stop],
        applied_value_code=trace.applied_value_code[start:stop],
        active=trace.active[start:stop],
        committed=trace.committed[start:stop],
        halted=trace.halted[start:stop],
    )


__all__ = [
    "CausalETTREpisodeRunner",
    "ETTREpisodeBatch",
    "ETTRInterventionOutput",
    "ETTREpisodeLosses",
    "ETTREpisodeOutput",
    "ETTREpisodeSegment",
]
