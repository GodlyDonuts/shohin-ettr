from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest
from tokenizers import Tokenizer

from ettr_il_v2_candidate_search import (
    find_first_depth1_checkerboard,
    semantic_core_id,
)
from ettr_il_v2_custody import derive_public_split_key, prf
from ettr_il_v2_semantics import Ontology
from ettr_il_v2_surface import (
    MAX_INTEGER,
    SurfaceCall,
    SurfaceInteger,
    SurfaceRenderer,
    SurfaceSymbol,
)
from ettr_il_v2_surface_adapter import (
    SurfaceAdapterContext,
    SurfaceStage,
    build_base_surface_bundle,
)
from ettr_il_v2_token_native_surface import (
    CODEWORD_BYTES,
    DEFAULT_TOKENIZER_PATH,
    DEFAULT_TOKENIZER_SHA256,
    TokenNativeBoundError,
    TokenNativeDocument,
    TokenNativeDocumentError,
    TokenNativeSurfaceCodec,
    TokenNativeSymbolContext,
    TokenNativeTransport,
    canonical_symbol_table,
    count_surface_nodes,
    encode_token_native_surface,
    parse_token_native_surface,
)


S0 = SurfaceSymbol("x0123456789abcdef")
S1 = SurfaceSymbol("xfedcba9876543210")
SAMPLE = SurfaceCall(
    3,
    (
        SurfaceInteger(7),
        S1,
        SurfaceCall(2, (S0, SurfaceInteger(11))),
        S1,
    ),
)
REIFIED = SurfaceCall(
    2,
    (
        SurfaceCall(12, (S0, SurfaceInteger(0), S1)),
        SurfaceCall(
            12,
            (
                S0,
                SurfaceInteger(1),
                SurfaceCall(4, (S1, SurfaceInteger(7))),
            ),
        ),
    ),
)

FULL_UNIVERSE_NODE_BOUNDS = {
    Ontology.HORN: {"WORLD": 138, "COMMAND": 87},
    Ontology.REWRITE: {"WORLD": 114, "COMMAND": 81},
    Ontology.RESOURCE: {"WORLD": 125, "COMMAND": 60},
}
STAGE_TOKEN_BUDGETS = {
    SurfaceStage.WORLD: 192,
    SurfaceStage.COMMAND: 96,
    SurfaceStage.QUERY: 48,
}


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_file(str(DEFAULT_TOKENIZER_PATH))


@pytest.fixture(scope="module")
def codec() -> TokenNativeSurfaceCodec:
    return TokenNativeSurfaceCodec(DEFAULT_TOKENIZER_PATH)


def _prf_callback(key: bytes):
    def callback(label: str, context: bytes) -> bytes:
        return prf(key, label, context)

    return callback


def _all_bundle_documents(bundle: object):
    assert hasattr(bundle, "world_variants")
    assert hasattr(bundle, "command_variants")
    assert hasattr(bundle, "query_prefixes")
    for pair in bundle.world_variants:
        for document in pair:
            yield SurfaceStage.WORLD, document.ast
    for pair in bundle.command_variants:
        for document in pair:
            yield SurfaceStage.COMMAND, document.ast
    for pair in bundle.query_prefixes:
        for prefix in pair:
            yield SurfaceStage.QUERY, prefix.document.ast


def test_public_codebook_is_deterministic_atomic_and_exact(
    codec: TokenNativeSurfaceCodec,
    tokenizer: Tokenizer,
) -> None:
    object_codec = TokenNativeSurfaceCodec(tokenizer)
    assert codec.tokenizer_sha256 == DEFAULT_TOKENIZER_SHA256
    assert object_codec.tokenizer_sha256 == codec.tokenizer_sha256
    assert object_codec.codebook == codec.codebook
    assert len(codec.codebook.atoms) >= 2048
    assert len(codec.codebook.atoms) == len(codec.codebook.token_ids)
    assert all(
        atom.isascii()
        and atom.startswith(" ")
        and len(atom.encode("ascii")) == CODEWORD_BYTES
        for atom in codec.codebook.atoms
    )

    joined = "".join(codec.codebook.atoms)
    assert tokenizer.encode(
        joined,
        add_special_tokens=False,
    ).ids == list(codec.codebook.token_ids)


@pytest.mark.parametrize("renderer", tuple(SurfaceRenderer))
def test_exactly_one_token_per_node_and_lossless_roundtrip(
    codec: TokenNativeSurfaceCodec,
    tokenizer: Tokenizer,
    renderer: SurfaceRenderer,
) -> None:
    document = encode_token_native_surface(
        SAMPLE,
        renderer,
        tokenizer,
        symbol_context=TokenNativeSymbolContext(
            canonical_symbol_table(SAMPLE),
            context="sample-document",
        ),
    )
    assert document.payload.isascii()
    assert not document.payload.endswith(b"\n")
    assert document.token_ids == codec.token_ids(document.payload)
    assert len(document.token_ids) == count_surface_nodes(SAMPLE) + 2
    assert len(document.token_ids) == 9
    assert codec.deserialize(document) == SAMPLE
    assert parse_token_native_surface(
        document.payload,
        renderer,
        tokenizer,
        symbol_context=document.symbol_context,
    ) == SAMPLE
    measurement = codec.measure(SAMPLE, renderer)
    assert measurement.node_count == 7
    assert measurement.token_count == 9
    assert measurement.ast_token_count == 7
    assert measurement.tokens_per_node == 1.0


def test_four_renderer_ids_are_byte_distinct_reversible_variants(
    codec: TokenNativeSurfaceCodec,
) -> None:
    documents = tuple(codec.serialize(SAMPLE, renderer) for renderer in SurfaceRenderer)
    assert len({document.payload for document in documents}) == 4
    assert len({document.token_ids for document in documents}) == 4
    assert all(codec.deserialize(document) == SAMPLE for document in documents)
    assert len({document.symbol_context for document in documents}) == 1
    for source in documents:
        for wrong_renderer in SurfaceRenderer:
            if wrong_renderer is source.renderer:
                continue
            with pytest.raises(
                TokenNativeDocumentError,
                match="grammar preamble",
            ):
                codec.parse(
                    source.payload,
                    wrong_renderer,
                    symbols=source.symbols,
                )


@pytest.mark.parametrize("renderer", tuple(SurfaceRenderer))
def test_reified_incidence_fusion_is_lossless_and_smaller(
    codec: TokenNativeSurfaceCodec,
    renderer: SurfaceRenderer,
) -> None:
    document = codec.serialize(REIFIED, renderer)
    assert codec.deserialize(document) == REIFIED
    assert len(document.token_ids) < count_surface_nodes(REIFIED) + 2
    assert codec.measure(REIFIED, renderer).tokens_per_node < 1.0


def test_per_document_symbol_context_preserves_opaque_values_exactly(
    codec: TokenNativeSurfaceCodec,
) -> None:
    context = TokenNativeSymbolContext(
        (S0, S1),
        context="rectangle/fold-0/world-1",
    )
    document = codec.serialize(
        SAMPLE,
        SurfaceRenderer.RECORD_INFIX,
        symbol_context=context,
    )
    assert document.symbol_map == (
        (0, "x0123456789abcdef"),
        (1, "xfedcba9876543210"),
    )
    assert codec.deserialize(document) == SAMPLE
    assert tuple(
        symbol.value for symbol in document.symbol_context.symbols
    ) == (S0.value, S1.value)

    with pytest.raises(TokenNativeDocumentError, match="exact canonical"):
        codec.render(
            SAMPLE,
            SurfaceRenderer.RECORD_INFIX,
            symbol_context=TokenNativeSymbolContext(
                (
                    S0,
                    SurfaceSymbol("x1111111111111111"),
                    S1,
                )
            ),
        )


@pytest.mark.parametrize(
    ("ontology", "stage", "node_count"),
    tuple(
        (ontology, stage, node_count)
        for ontology, bounds in FULL_UNIVERSE_NODE_BOUNDS.items()
        for stage, node_count in bounds.items()
    ),
)
def test_proven_full_universe_node_bounds_fit_stage_widths_exactly(
    codec: TokenNativeSurfaceCodec,
    ontology: Ontology,
    stage: str,
    node_count: int,
) -> None:
    del ontology
    node: SurfaceCall | SurfaceInteger = SurfaceInteger(0)
    for _ in range(node_count - 1):
        node = SurfaceCall(0, (node,))
    document = codec.serialize(node, SurfaceRenderer.CANONICAL_JSON)
    assert len(document.token_ids) == node_count + 2
    budget = 192 if stage == "WORLD" else 96
    assert len(document.token_ids) <= budget


def test_all_current_candidates_fit_with_existing_answer_prefix(
    codec: TokenNativeSurfaceCodec,
    tokenizer: Tokenizer,
) -> None:
    split_key = derive_public_split_key(0, "development")
    callback = _prf_callback(split_key)
    observed_maxima = {
        SurfaceStage.WORLD: 0,
        SurfaceStage.COMMAND: 0,
        SurfaceStage.QUERY: 0,
    }
    observed_examples = 0

    for ontology in Ontology:
        candidate = find_first_depth1_checkerboard(ontology)
        core_id = semantic_core_id(candidate)
        for source_renderer in SurfaceRenderer:
            bundle = build_base_surface_bundle(
                candidate.worlds,
                candidate.commands,
                (candidate.queries.slot_0, candidate.queries.slot_1),
                context=SurfaceAdapterContext(
                    fold=0,
                    split="development",
                    semantic_core_id=core_id,
                    semantic_rectangle_id=f"token-native-{ontology.value}",
                    renderer=source_renderer,
                    prf=callback,
                ),
            )
            for stage, ast in _all_bundle_documents(bundle):
                for token_renderer in SurfaceRenderer:
                    document = codec.serialize(ast, token_renderer)
                    assert codec.deserialize(document) == ast
                    assert len(document.token_ids) == count_surface_nodes(ast) + 2
                    if stage is SurfaceStage.QUERY:
                        model_input = document.payload + b"\nR="
                        model_input_ids = tokenizer.encode(
                            model_input.decode("ascii"),
                            add_special_tokens=False,
                        ).ids
                        assert model_input_ids[: len(document.token_ids)] == list(
                            document.token_ids
                        )
                        width = len(model_input_ids)
                    else:
                        width = len(document.token_ids)
                    observed_maxima[stage] = max(observed_maxima[stage], width)
                    assert width <= STAGE_TOKEN_BUDGETS[stage]
                    observed_examples += 1

    assert observed_examples == 3 * 4 * 12 * 4
    assert observed_maxima == {
        SurfaceStage.WORLD: 125,
        SurfaceStage.COMMAND: 73,
        SurfaceStage.QUERY: 11,
    }


@pytest.mark.parametrize(
    "width",
    (48, 96, 192),
)
def test_fixed_transport_equalizes_tokens_bytes_and_roundtrips(
    codec: TokenNativeSurfaceCodec,
    width: int,
) -> None:
    document = codec.serialize(SAMPLE, SurfaceRenderer.REVERSE_POSTFIX)
    transport = codec.pack(document, width=width)
    assert isinstance(transport, TokenNativeTransport)
    assert len(transport.token_ids) == width
    assert len(transport.payload) == width * CODEWORD_BYTES
    assert codec.unpack(transport) == SAMPLE
    cover = transport.token_ids[len(document.token_ids) :]
    assert cover
    assert len(set(cover)) > 1


def test_transport_cover_tampering_fails_closed(
    codec: TokenNativeSurfaceCodec,
) -> None:
    document = codec.serialize(SAMPLE, SurfaceRenderer.CANONICAL_JSON)
    transport = codec.pack(document, width=48)
    replacement = codec.codebook.atoms[0].encode("ascii")
    if transport.payload[-CODEWORD_BYTES:] == replacement:
        replacement = codec.codebook.atoms[1].encode("ascii")
    tampered = replace(
        transport,
        payload=transport.payload[:-CODEWORD_BYTES] + replacement,
    )
    with pytest.raises(
        TokenNativeDocumentError,
        match="token IDs differ|cover differs",
    ):
        codec.unpack(tampered)


@pytest.mark.parametrize(
    "payload_transform",
    (
        lambda payload, codec: payload[1:],
        lambda payload, codec: b" " + payload,
        lambda payload, codec: payload + b" ",
        lambda payload, codec: payload + b"\n",
        lambda payload, codec: payload + codec.codebook.atoms[0].encode("ascii"),
        lambda payload, codec: payload.replace(b" ", b"  ", 1),
        lambda payload, codec: b" \xff",
        lambda payload, codec: b" NotInTheShohinAtomicCodebook",
    ),
)
def test_malformed_and_noncanonical_payloads_reject(
    codec: TokenNativeSurfaceCodec,
    payload_transform: object,
) -> None:
    assert callable(payload_transform)
    document = codec.serialize(SAMPLE, SurfaceRenderer.PREFIX_SEXPR)
    malformed = payload_transform(document.payload, codec)
    with pytest.raises(
        (TokenNativeDocumentError, TokenNativeBoundError),
    ):
        codec.parse(
            malformed,
            document.renderer,
            symbols=document.symbols,
        )


def test_invalid_sidecars_tokens_types_and_out_of_range_values_reject(
    codec: TokenNativeSurfaceCodec,
) -> None:
    with pytest.raises(TokenNativeDocumentError, match="unique and sorted"):
        TokenNativeSymbolContext((S1, S0))
    with pytest.raises(TokenNativeDocumentError, match="strict ASCII"):
        TokenNativeSymbolContext((S0,), context="not-ascii-\N{SNOWMAN}")
    with pytest.raises(TokenNativeBoundError, match="one-token direct range"):
        codec.serialize(
            SurfaceInteger(MAX_INTEGER),
            SurfaceRenderer.CANONICAL_JSON,
        )
    with pytest.raises(TypeError, match="immutable bytes"):
        codec.parse(  # type: ignore[arg-type]
            "not-bytes",
            SurfaceRenderer.CANONICAL_JSON,
            symbols=(),
        )

    document = codec.serialize(SAMPLE, SurfaceRenderer.CANONICAL_JSON)
    tampered_ids = replace(
        document,
        token_ids=(
            document.token_ids[0] ^ 1,
            *document.token_ids[1:],
        ),
    )
    assert isinstance(tampered_ids, TokenNativeDocument)
    with pytest.raises(TokenNativeDocumentError, match="token IDs"):
        codec.deserialize(tampered_ids)


def test_tokenizer_artifact_identity_is_public_and_fixed() -> None:
    assert (
        hashlib.sha256(DEFAULT_TOKENIZER_PATH.read_bytes()).hexdigest()
        == DEFAULT_TOKENIZER_SHA256
    )
