"""Target-free document masking for fixed-width token-native transports.

ETTR transports place one complete syntax tree first and deterministic cover
codewords afterward.  The ordinary attention mask deliberately covers the
whole transport, so semantic compilers otherwise treat cover as source.  This
router recovers only the tree boundary from the public token-native grammar.
It does not decode ontology symbols, inspect targets, or execute a program.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


MAX_NATIVE_ARITY = 32
MAX_HEAD = 15
CALL_STRIDE = MAX_NATIVE_ARITY + 1
CALL_END = (MAX_HEAD + 1) * CALL_STRIDE
FRAME_A = CALL_END
FRAME_B = FRAME_A + 1
FRAME_END = FRAME_B + 1
FRAME_FILL = FRAME_END + 1
REIFY_BASE = FRAME_FILL + 1
REIFY_END = REIFY_BASE + MAX_NATIVE_ARITY + 1
INTEGER_BASE = REIFY_END
ROOT_CODES = (
    14 * CALL_STRIDE + 3,
    15 * CALL_STRIDE + 2,
)


class TokenNativeSyntaxRouterError(ValueError):
    """A token sequence cannot be routed under the public grammar."""


class TokenNativeDocumentMask(nn.Module):
    """Recover the exact leading AST span and delete transport cover."""

    def __init__(
        self,
        codebook_token_ids: Sequence[int],
        *,
        vocab_size: int,
    ) -> None:
        super().__init__()
        ids = tuple(int(value) for value in codebook_token_ids)
        if (
            not ids
            or len(set(ids)) != len(ids)
            or not isinstance(vocab_size, int)
            or vocab_size < 1
            or min(ids) < 0
            or max(ids) >= vocab_size
            or len(ids) <= REIFY_END
        ):
            raise TokenNativeSyntaxRouterError(
                "token-native codebook geometry differs"
            )
        inverse = torch.full((vocab_size,), -1, dtype=torch.long)
        inverse[torch.tensor(ids, dtype=torch.long)] = torch.arange(
            len(ids),
            dtype=torch.long,
        )
        self.register_buffer("inverse_codebook", inverse)

    def forward(
        self,
        tokens: torch.Tensor,
        transport_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            tokens.ndim != 2
            or tokens.dtype != torch.long
            or transport_mask.shape != tokens.shape
            or transport_mask.dtype != torch.bool
            or tokens.shape[1] < 3
            or tokens.device != transport_mask.device
        ):
            raise TokenNativeSyntaxRouterError(
                "token-native transport geometry differs"
            )
        codes = self.inverse_codebook[tokens]
        torch._assert_async(
            codes.ge(0).all(),
            "token-native transport leaves the bound codebook",
        )
        preamble = (
            codes[:, 0].eq(FRAME_A) | codes[:, 0].eq(FRAME_B)
        ) & (
            codes[:, 1].eq(FRAME_A) | codes[:, 1].eq(FRAME_B)
        )
        torch._assert_async(
            preamble.all(),
            "token-native transport preamble differs",
        )

        body = codes[:, 2:]
        call = body.lt(CALL_END)
        arity = body.remainder(CALL_STRIDE)
        reified = body.ge(REIFY_BASE) & body.lt(REIFY_END)
        reified_arity = body - REIFY_BASE

        prefix_effect = torch.full_like(body, -1)
        prefix_effect = torch.where(call, arity - 1, prefix_effect)
        prefix_effect = torch.where(reified, reified_arity, prefix_effect)
        remaining = 1 + prefix_effect.cumsum(dim=1)
        prefix_complete = remaining.eq(0)

        postfix_effect = torch.ones_like(body)
        postfix_effect = torch.where(call, 1 - arity, postfix_effect)
        postfix_effect = torch.where(reified, -reified_arity, postfix_effect)
        stack = postfix_effect.cumsum(dim=1)
        root = torch.zeros_like(body, dtype=torch.bool)
        for root_code in ROOT_CODES:
            root |= body.eq(root_code)
        postfix_complete = stack.eq(1) & root

        is_prefix = codes[:, 0].eq(FRAME_A)
        completion = torch.where(
            is_prefix[:, None],
            prefix_complete,
            postfix_complete,
        )
        has_completion = completion.any(dim=1)
        torch._assert_async(
            has_completion.all(),
            "token-native syntax tree does not terminate",
        )
        terminal_index = completion.to(torch.int64).argmax(dim=1) + 2
        document_mask = (
            torch.arange(tokens.shape[1], device=tokens.device)[None, :]
            <= terminal_index[:, None]
        )
        routed = transport_mask & document_mask
        torch._assert_async(
            routed.sum(dim=1).eq(terminal_index + 1).all(),
            "token-native source mask truncates the syntax tree",
        )
        return routed


class TokenNativeOccurrenceEncoder(nn.Module):
    """Expose grammar roles and bind repeated local identifiers explicitly.

    Token-native symbol names are deliberately opaque and vary between
    documents.  Their usable invariant is equality between occurrences, not
    the accidental tokenizer embedding assigned to an ordinal.  This encoder
    factorizes public call heads/arities and integer atoms, then broadcasts a
    shared representation across equal identifier occurrences before a
    second contextual pass.  It performs no semantic execution.
    """

    _CATEGORY_COUNT = 5
    _HEAD_COUNT = MAX_HEAD + 2

    def __init__(
        self,
        codebook_token_ids: Sequence[int],
        *,
        vocab_size: int,
        width: int,
        num_heads: int,
        maximum_positions: int = 96,
        maximum_identifier_codes: int = 96,
    ) -> None:
        super().__init__()
        ids = tuple(int(value) for value in codebook_token_ids)
        if (
            not ids
            or len(set(ids)) != len(ids)
            or not isinstance(vocab_size, int)
            or vocab_size < 1
            or min(ids) < 0
            or max(ids) >= vocab_size
            or len(ids) <= INTEGER_BASE + maximum_identifier_codes
            or not isinstance(width, int)
            or width < 64
            or not isinstance(num_heads, int)
            or num_heads < 1
            or width % num_heads
            or not isinstance(maximum_positions, int)
            or maximum_positions < 3
            or not isinstance(maximum_identifier_codes, int)
            or not 1 <= maximum_identifier_codes < maximum_positions + 1
        ):
            raise TokenNativeSyntaxRouterError(
                "token-native occurrence geometry differs"
            )
        inverse = torch.full((vocab_size,), -1, dtype=torch.long)
        inverse[torch.tensor(ids, dtype=torch.long)] = torch.arange(
            len(ids),
            dtype=torch.long,
        )
        self.register_buffer("inverse_codebook", inverse)
        self.codebook_size = len(ids)
        self.width = width
        self.maximum_positions = maximum_positions
        self.maximum_identifier_codes = maximum_identifier_codes

        self.category_embedding = nn.Embedding(self._CATEGORY_COUNT, width)
        self.head_embedding = nn.Embedding(self._HEAD_COUNT, width)
        self.arity_embedding = nn.Embedding(MAX_NATIVE_ARITY + 2, width)
        self.integer_embedding = nn.Embedding(
            self.codebook_size - INTEGER_BASE,
            width,
        )
        self.renderer_embedding = nn.Embedding(4, width)
        self.position_embedding = nn.Embedding(maximum_positions, width)
        self.input_norm = nn.LayerNorm(width)
        self.pre_binding = self._layer(width, num_heads)
        self.binding_projection = nn.Linear(width, width, bias=False)
        self.post_binding = self._layer(width, num_heads)
        self.output_norm = nn.LayerNorm(width)

    @staticmethod
    def _layer(width: int, num_heads: int) -> nn.TransformerEncoderLayer:
        return nn.TransformerEncoderLayer(
            d_model=width,
            nhead=num_heads,
            dim_feedforward=4 * width,
            batch_first=True,
            norm_first=True,
            activation="gelu",
            dropout=0.0,
        )

    def forward(
        self,
        memory: torch.Tensor,
        tokens: torch.Tensor,
        document_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            memory.ndim != 3
            or memory.shape[:2] != tokens.shape
            or memory.shape[-1] != self.width
            or tokens.ndim != 2
            or tokens.dtype != torch.long
            or document_mask.shape != tokens.shape
            or document_mask.dtype != torch.bool
            or tokens.shape[1] > self.maximum_positions
            or memory.device != tokens.device
            or tokens.device != document_mask.device
        ):
            raise TokenNativeSyntaxRouterError(
                "token-native occurrence input differs"
            )
        codes = self.inverse_codebook[tokens]
        torch._assert_async(
            codes.ge(0).all(),
            "token-native occurrence leaves the bound codebook",
        )
        is_call = codes.lt(CALL_END)
        is_reified = codes.ge(REIFY_BASE) & codes.lt(REIFY_END)
        identifier_floor = self.codebook_size - self.maximum_identifier_codes
        is_identifier = codes.ge(identifier_floor) & document_mask
        is_integer = (
            codes.ge(INTEGER_BASE) & ~is_identifier & document_mask
        )
        is_frame = document_mask & ~(is_call | is_reified | is_integer | is_identifier)
        category = torch.zeros_like(codes)
        category = torch.where(is_call, torch.ones_like(category), category)
        category = torch.where(
            is_reified,
            torch.full_like(category, 2),
            category,
        )
        category = torch.where(
            is_integer,
            torch.full_like(category, 3),
            category,
        )
        category = torch.where(
            is_identifier,
            torch.full_like(category, 4),
            category,
        )
        torch._assert_async(
            (is_frame | is_call | is_reified | is_integer | is_identifier)[
                document_mask
            ].all(),
            "token-native occurrence category is incomplete",
        )

        head = torch.zeros_like(codes)
        head = torch.where(is_call, codes.div(CALL_STRIDE, rounding_mode="floor"), head)
        head = torch.where(
            is_reified,
            torch.full_like(head, MAX_HEAD + 1),
            head,
        )
        arity = torch.zeros_like(codes)
        arity = torch.where(is_call, codes.remainder(CALL_STRIDE), arity)
        arity = torch.where(is_reified, codes - REIFY_BASE + 1, arity)
        integer = (codes - INTEGER_BASE).clamp(
            min=0,
            max=self.codebook_size - INTEGER_BASE - 1,
        )
        renderer = (
            codes[:, 0].eq(FRAME_B).to(torch.long) * 2
            + codes[:, 1].eq(FRAME_B).to(torch.long)
        )
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        structural = (
            self.category_embedding(category)
            + self.head_embedding(head)
            + self.arity_embedding(arity)
            + self.renderer_embedding(renderer)[:, None, :]
            + self.position_embedding(positions)[None, :, :]
        )
        structural = structural + (
            self.integer_embedding(integer) * is_integer.unsqueeze(-1)
        )
        padding = ~document_mask
        hidden = self.pre_binding(
            self.input_norm(memory + structural.to(memory.dtype)),
            src_key_padding_mask=padding,
        )

        equality = (
            codes[:, :, None].eq(codes[:, None, :])
            & is_identifier[:, :, None]
            & is_identifier[:, None, :]
        )
        counts = equality.sum(dim=-1, keepdim=True).clamp_min(1)
        shared = torch.bmm(
            equality.to(hidden.dtype),
            hidden,
        ) / counts.to(hidden.dtype)
        hidden = hidden + self.binding_projection(shared) * is_identifier.unsqueeze(-1)
        hidden = self.post_binding(hidden, src_key_padding_mask=padding)
        return self.output_norm(hidden)


__all__ = [
    "TokenNativeDocumentMask",
    "TokenNativeOccurrenceEncoder",
    "TokenNativeSyntaxRouterError",
]
