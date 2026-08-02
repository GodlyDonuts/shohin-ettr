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
FRAME_FILL = FRAME_B + 1
REIFY_BASE = FRAME_FILL + 1
REIFY_END = REIFY_BASE + MAX_NATIVE_ARITY + 1
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


__all__ = [
    "TokenNativeDocumentMask",
    "TokenNativeSyntaxRouterError",
]
