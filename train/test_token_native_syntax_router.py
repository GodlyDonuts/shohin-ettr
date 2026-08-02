from __future__ import annotations

import pytest
import torch

from token_native_syntax_router import (
    TokenNativeDocumentMask,
    TokenNativeSyntaxRouterError,
)


def _codec():
    from ettr_il_v2_token_native_surface import (
        DEFAULT_TOKENIZER_PATH,
        TokenNativeSurfaceCodec,
    )

    return TokenNativeSurfaceCodec(DEFAULT_TOKENIZER_PATH)


def test_router_removes_cover_for_every_renderer() -> None:
    from ettr_il_v2_surface import SurfaceRenderer, call, integer

    codec = _codec()
    ast = call(
        15,
        integer(1),
        call(
            13,
            call(4, integer(0), integer(1), integer(0)),
            call(4, integer(1), integer(4), integer(1)),
        ),
    )
    transports = [
        codec.pack(codec.serialize(ast, renderer), width=96)
        for renderer in SurfaceRenderer
    ]
    tokens = torch.tensor(
        [transport.token_ids for transport in transports],
        dtype=torch.long,
    )
    router = TokenNativeDocumentMask(
        codec.codebook.token_ids,
        vocab_size=codec.tokenizer.get_vocab_size(),
    )
    routed = router(tokens, torch.ones_like(tokens, dtype=torch.bool))
    expected = torch.zeros_like(routed)
    expected[:, :13] = True
    assert torch.equal(routed, expected)


def test_router_accepts_legacy_head_14_root() -> None:
    from ettr_il_v2_surface import SurfaceRenderer, call, integer

    codec = _codec()
    ast = call(14, integer(2), call(1, integer(3)), call(13, integer(4)))
    transports = [
        codec.pack(codec.serialize(ast, renderer), width=32)
        for renderer in SurfaceRenderer
    ]
    tokens = torch.tensor(
        [transport.token_ids for transport in transports],
        dtype=torch.long,
    )
    router = TokenNativeDocumentMask(
        codec.codebook.token_ids,
        vocab_size=codec.tokenizer.get_vocab_size(),
    )
    routed = router(tokens, torch.ones_like(tokens, dtype=torch.bool))
    assert routed.sum(dim=1).tolist() == [8, 8, 8, 8]


def test_router_rejects_non_codebook_source() -> None:
    codec = _codec()
    router = TokenNativeDocumentMask(
        codec.codebook.token_ids,
        vocab_size=codec.tokenizer.get_vocab_size(),
    )
    unknown = next(
        token
        for token in range(codec.tokenizer.get_vocab_size())
        if token not in set(codec.codebook.token_ids)
    )
    tokens = torch.full((1, 8), unknown, dtype=torch.long)
    with pytest.raises(RuntimeError, match="bound codebook"):
        router(tokens, torch.ones_like(tokens, dtype=torch.bool))


def test_router_rejects_wrong_geometry() -> None:
    codec = _codec()
    router = TokenNativeDocumentMask(
        codec.codebook.token_ids,
        vocab_size=codec.tokenizer.get_vocab_size(),
    )
    with pytest.raises(TokenNativeSyntaxRouterError, match="geometry"):
        router(
            torch.ones(2, 3, dtype=torch.float32),
            torch.ones(2, 3, dtype=torch.bool),
        )
