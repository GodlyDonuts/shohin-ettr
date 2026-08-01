"""Native source-deleted answer head for ETTR terminal states.

The generic ETTR reader injects a residual into the pretrained language-model
decoder and asks the full vocabulary head to rediscover four protocol answers.
This module keeps query/state interpretation learned, but makes the final
causal interface explicit: ANSWER states choose false/true through a dedicated
binary motor while ABSTAIN and REJECT are grounded by the terminal disposition.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from endogenous_typed_theory_reactor import (
    SourceDeletedQueryReader,
    TheoryReactorConfig,
    TheoryReactorError,
    TypedTheoryState,
    _disposition_probabilities,
)


def answer_token_ids_from_tokenizer(
    tokenizer_path: Path,
) -> tuple[int, int, int, int]:
    """Derive and validate the immutable ``R=0..3`` next-token codebook."""

    from tokenizers import Tokenizer

    if not isinstance(tokenizer_path, Path) or not tokenizer_path.is_file():
        raise TheoryReactorError("causal answer tokenizer differs")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    prefix = tokenizer.encode("\nR=", add_special_tokens=False).ids
    result: list[int] = []
    for code in range(4):
        boundary = tokenizer.encode(
            f"\nR={code}",
            add_special_tokens=False,
        ).ids
        if (
            len(boundary) != len(prefix) + 1
            or boundary[: len(prefix)] != prefix
        ):
            raise TheoryReactorError(
                "causal answer boundary is not one token"
            )
        result.append(boundary[-1])
    answer_token_ids = tuple(result)
    if len(set(answer_token_ids)) != 4:
        raise TheoryReactorError("causal answer tokens are not unique")
    return answer_token_ids  # type: ignore[return-value]


class NativeCausalDispositionReader(nn.Module):
    """Map a source-deleted typed state and late query to four legal answers."""

    def __init__(
        self,
        config: TheoryReactorConfig,
        *,
        vocab_size: int,
        answer_token_ids: tuple[int, int, int, int],
        truth_motor_hidden: int = 0,
    ) -> None:
        super().__init__()
        config.validate()
        if (
            not isinstance(vocab_size, int)
            or vocab_size <= 4
            or not isinstance(answer_token_ids, tuple)
            or len(answer_token_ids) != 4
            or any(
                not isinstance(value, int) or not 0 <= value < vocab_size
                for value in answer_token_ids
            )
            or len(set(answer_token_ids)) != 4
            or not isinstance(truth_motor_hidden, int)
            or truth_motor_hidden < 0
        ):
            raise TheoryReactorError("causal answer-token codebook differs")
        self.config = config
        self.vocab_size = vocab_size
        self.reader = SourceDeletedQueryReader(config)
        self.readout_norm = nn.LayerNorm(config.d_model)
        self.truth_motor_hidden = truth_motor_hidden
        if truth_motor_hidden == 0:
            self.truth_motor = nn.Linear(config.d_model, 2)
        else:
            self.truth_motor = nn.Sequential(
                nn.Linear(config.d_model, truth_motor_hidden),
                nn.GELU(),
                nn.Linear(truth_motor_hidden, 2),
            )
        self.register_buffer(
            "answer_token_ids",
            torch.tensor(answer_token_ids, dtype=torch.long),
            persistent=True,
        )

    def load_reader_state(
        self,
        reader: SourceDeletedQueryReader,
    ) -> None:
        """Warm-start only the exact generic reader submodule."""

        if not isinstance(reader, SourceDeletedQueryReader):
            raise TheoryReactorError("causal reader warm start differs")
        self.reader.load_state_dict(reader.state_dict(), strict=True)

    def class_logits(
        self,
        query_hidden: torch.Tensor,
        state: TypedTheoryState,
        *,
        trace=None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ordered false/true/abstain/reject logits."""

        if (
            query_hidden.ndim != 3
            or query_hidden.shape[0] != state.active.shape[0]
            or query_hidden.shape[-1] != self.config.d_model
        ):
            raise TheoryReactorError("causal query-hidden geometry differs")
        read = self.reader(
            query_hidden,
            state,
            trace=trace,
            attention_mask=attention_mask,
        )
        truth = F.log_softmax(
            self.truth_motor(self.readout_norm(query_hidden + read)),
            dim=-1,
        )
        disposition = _disposition_probabilities(state).to(truth.dtype)
        if disposition.shape != (query_hidden.shape[0], 4):
            raise TheoryReactorError("terminal disposition geometry differs")
        tiny = torch.finfo(truth.dtype).tiny
        answer_log_gate = disposition[:, 1].clamp_min(tiny).log()
        abstain = disposition[:, 2].clamp_min(tiny).log()
        reject = disposition[:, 3].clamp_min(tiny).log()
        class_logits = torch.cat(
            (
                truth + answer_log_gate[:, None, None],
                abstain[:, None, None].expand(-1, query_hidden.shape[1], 1),
                reject[:, None, None].expand(-1, query_hidden.shape[1], 1),
            ),
            dim=-1,
        )
        if not bool(torch.isfinite(class_logits).all()):
            raise TheoryReactorError("causal class logits are nonfinite")
        return class_logits

    def forward(
        self,
        query_hidden: torch.Tensor,
        state: TypedTheoryState,
        *,
        trace=None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Scatter four legal class logits into the immutable vocabulary."""

        class_logits = self.class_logits(
            query_hidden,
            state,
            trace=trace,
            attention_mask=attention_mask,
        )
        floor = -min(1.0e4, math.sqrt(torch.finfo(class_logits.dtype).max))
        logits = torch.full(
            (*class_logits.shape[:-1], self.vocab_size),
            floor,
            dtype=class_logits.dtype,
            device=class_logits.device,
        )
        token_ids = self.answer_token_ids.to(class_logits.device)
        logits.scatter_(
            -1,
            token_ids.view(1, 1, 4).expand(*class_logits.shape[:-1], 4),
            class_logits,
        )
        return logits


__all__ = [
    "NativeCausalDispositionReader",
    "answer_token_ids_from_tokenizer",
]
