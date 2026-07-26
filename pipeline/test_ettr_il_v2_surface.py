from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json

import pytest

from ettr_il_v2_surface import (
    MAX_CHILDREN,
    OpaqueNameContext,
    SurfaceCall,
    SurfaceCodecError,
    SurfaceInteger,
    SurfaceRenderer,
    SurfaceSchemaError,
    SurfaceSymbol,
    assign_opaque_symbols,
    ast_from_json_value,
    ast_to_json_value,
    call,
    canonical_json_bytes,
    integer,
    parse_canonical_json,
    parse_infix,
    parse_postfix,
    parse_prefix,
    parse_surface,
    render_infix,
    render_postfix,
    render_prefix,
    render_surface,
    semantic_canonicalize,
    symbol,
)


S0 = symbol("x0000000000000001")
S1 = symbol("x0000000000000002")
S2 = symbol("x0000000000000003")
SAMPLE = call(
    14,
    integer(2),
    call(1, S0, integer(0)),
    call(4, S1, integer(7)),
)


GOLDEN_JSON = (
    b'{"a":[{"i":2},{"a":[{"s":"x0000000000000001"},{"i":0}],"h":1},'
    b'{"a":[{"s":"x0000000000000002"},{"i":7}],"h":4}],"h":14}\n'
)
GOLDEN_PREFIX = (
    b"(14 #2 (1 @x0000000000000001 #0) "
    b"(4 @x0000000000000002 #7))\n"
)
GOLDEN_INFIX = (
    b"V2\x1e"
    b"N0=14%3%1%2%5\x1e"
    b"I1=2\x1e"
    b"N2=1%2%3%4\x1e"
    b"S3=x0000000000000001\x1e"
    b"I4=0\x1e"
    b"N5=4%2%6%7\x1e"
    b"S6=x0000000000000002\x1e"
    b"I7=7\x1e"
    b"R=0\x1e"
)
GOLDEN_POSTFIX = (
    b"#7 $x0000000000000002 ^4/2 #0 $x0000000000000001 "
    b"^1/2 #2 ^14/3 !\n"
)


def test_ast_is_typed_deeply_immutable_and_schema_bounded() -> None:
    node = SurfaceCall(0, (SurfaceInteger(1), SurfaceSymbol(S0.value)))
    with pytest.raises(FrozenInstanceError):
        node.head = 1  # type: ignore[misc]
    with pytest.raises(SurfaceSchemaError, match="immutable tuple"):
        SurfaceCall(0, [integer(1)])  # type: ignore[arg-type]
    with pytest.raises(SurfaceSchemaError, match="exact int"):
        SurfaceInteger(True)  # type: ignore[arg-type]
    with pytest.raises(SurfaceSchemaError, match="head"):
        SurfaceCall(16, ())
    with pytest.raises(SurfaceSchemaError, match="pattern"):
        SurfaceSymbol("xABCDEF0000000000")
    with pytest.raises(SurfaceSchemaError, match="at most"):
        SurfaceCall(0, tuple(integer(0) for _ in range(MAX_CHILDREN + 1)))


def test_plain_json_conversion_validates_exact_schema_without_dependency() -> None:
    value = ast_to_json_value(SAMPLE)
    assert ast_from_json_value(value) == SAMPLE
    assert value == json.loads(GOLDEN_JSON)
    for invalid in (
        {"i": 0, "extra": 1},
        {"s": "x0000000000000001", "i": 0},
        {"h": 0},
        {"a": (), "h": 0},
        {"a": [], "h": False},
        [],
    ):
        with pytest.raises(SurfaceSchemaError):
            ast_from_json_value(invalid)


@pytest.mark.parametrize(
    ("renderer", "expected"),
    [
        (SurfaceRenderer.CANONICAL_JSON, GOLDEN_JSON),
        (SurfaceRenderer.PREFIX_SEXPR, GOLDEN_PREFIX),
        (SurfaceRenderer.RECORD_INFIX, GOLDEN_INFIX),
        (SurfaceRenderer.REVERSE_POSTFIX, GOLDEN_POSTFIX),
    ],
)
def test_four_golden_renderers_and_dispatch_roundtrip(
    renderer: SurfaceRenderer,
    expected: bytes,
) -> None:
    assert render_surface(SAMPLE, renderer) == expected
    assert parse_surface(expected, renderer) == SAMPLE
    assert render_surface(parse_surface(expected, renderer), renderer) == expected


def test_same_ast_has_four_distinct_exact_byte_documents() -> None:
    rendered = tuple(render_surface(SAMPLE, renderer) for renderer in SurfaceRenderer)
    assert len(set(rendered)) == 4
    assert rendered[0].endswith(b"\n")
    assert rendered[1].endswith(b"\n")
    assert rendered[2].endswith(b"\x1e") and b"\n" not in rendered[2]
    assert rendered[3].endswith(b"!\n")


def test_infix_numbers_reused_immutable_objects_by_tree_occurrence() -> None:
    reused_leaf = symbol("x000000000000000a")
    tree = call(0, reused_leaf, call(1, reused_leaf), reused_leaf)
    payload = render_infix(tree)
    assert payload == (
        b"V2\x1e"
        b"N0=0%3%1%2%4\x1e"
        b"S1=x000000000000000a\x1e"
        b"N2=1%1%3\x1e"
        b"S3=x000000000000000a\x1e"
        b"S4=x000000000000000a\x1e"
        b"R=0\x1e"
    )
    assert parse_infix(payload) == tree


@pytest.mark.parametrize(
    "node",
    [
        integer(0),
        integer(2_147_483_647),
        S0,
        call(0),
        call(15, *(integer(index) for index in range(MAX_CHILDREN))),
        call(
            14,
            call(3, S0, integer(0), call(0, integer(4))),
            call(2, call(0, S1, integer(2)), call(0, S2, integer(1))),
        ),
    ],
)
def test_all_codecs_roundtrip_schema_boundaries(node: object) -> None:
    assert isinstance(node, (SurfaceInteger, SurfaceSymbol, SurfaceCall))
    for renderer in SurfaceRenderer:
        payload = render_surface(node, renderer)
        assert parse_surface(payload, renderer) == node


@pytest.mark.parametrize(
    "payload",
    [
        GOLDEN_JSON[:-1],
        b" " + GOLDEN_JSON,
        GOLDEN_JSON + b"\n",
        b'{"i":0,"i":0}\n',
        b'{"i":0.0}\n',
        b'{"i":0e0}\n',
        b'{"i":NaN}\n',
        b'{"i":0}\r\n',
        b'{"s":"x000000000000000A"}\n',
        b'{"a":[],"h":0,"z":null}\n',
        b"\xff\n",
    ],
)
def test_json_rejects_malformed_and_noncanonical_bytes(payload: bytes) -> None:
    with pytest.raises((SurfaceCodecError, SurfaceSchemaError)):
        parse_canonical_json(payload)


@pytest.mark.parametrize(
    "payload",
    [
        GOLDEN_PREFIX[:-1],
        GOLDEN_PREFIX + b"\n",
        b"(14  #2)\n",
        b"(14\t#2)\n",
        b"(14 #02)\n",
        b"(014 #2)\n",
        b"(16 #2)\n",
        b"(14 @x000000000000000A)\n",
        b"(14 #2) \n",
        b"(14 #2\n",
        b"#0 #1\n",
        b"\n",
        b"\xff\n",
    ],
)
def test_prefix_rejects_malformed_and_noncanonical_bytes(payload: bytes) -> None:
    with pytest.raises((SurfaceCodecError, SurfaceSchemaError)):
        parse_prefix(payload)


@pytest.mark.parametrize(
    "payload",
    [
        GOLDEN_INFIX[:-1],
        GOLDEN_INFIX + b"\n",
        GOLDEN_INFIX.replace(b"N0=", b"N00=", 1),
        GOLDEN_INFIX.replace(b"I1=2", b"I01=2", 1),
        GOLDEN_INFIX.replace(b"I1=2", b"I1=02", 1),
        GOLDEN_INFIX.replace(b"N0=14%3", b"N0=14%2", 1),
        GOLDEN_INFIX.replace(b"%1%2%5", b"%2%2%5", 1),
        GOLDEN_INFIX.replace(b"%1%2%5", b"%1%2%99", 1),
        GOLDEN_INFIX.replace(b"N0=14%3%1%2%5", b"N0=14%3%2%1%5", 1),
        GOLDEN_INFIX.replace(b"R=0", b"R=1", 1),
        GOLDEN_INFIX.replace(b"\x1eI1=2", b"\x1e\x1eI1=2", 1),
        b"V2\x1eR=0\x1e",
        b"V2\x1eI0=0\x1eI1=1\x1eR=0\x1e",
        b"\xff\x1e",
    ],
)
def test_infix_rejects_malformed_noncanonical_or_nontree_bytes(
    payload: bytes,
) -> None:
    with pytest.raises((SurfaceCodecError, SurfaceSchemaError)):
        parse_infix(payload)


@pytest.mark.parametrize(
    "payload",
    [
        GOLDEN_POSTFIX[:-1],
        GOLDEN_POSTFIX + b"\n",
        GOLDEN_POSTFIX.replace(b"#7 ", b"#07 ", 1),
        GOLDEN_POSTFIX.replace(b"^4/2", b"^04/2", 1),
        GOLDEN_POSTFIX.replace(b"^4/2", b"^4/02", 1),
        GOLDEN_POSTFIX.replace(b"^4/2", b"^4/3", 1),
        GOLDEN_POSTFIX.replace(b" !\n", b"  !\n", 1),
        GOLDEN_POSTFIX.replace(b" !\n", b"\t!\n", 1),
        GOLDEN_POSTFIX.replace(b" !\n", b" #1 !\n", 1),
        GOLDEN_POSTFIX.replace(b" !\n", b" ?\n", 1),
        b"^0/0 ^0/0 !\n",
        b"$x000000000000000A !\n",
        b"\xff !\n",
    ],
)
def test_postfix_rejects_malformed_and_noncanonical_bytes(payload: bytes) -> None:
    with pytest.raises((SurfaceCodecError, SurfaceSchemaError)):
        parse_postfix(payload)


def test_parser_preserves_surface_order_and_semantic_canonicalizer_sorts_later() -> None:
    unsorted = call(
        14,
        call(1, integer(9), integer(1), integer(4)),
        call(2, call(0, integer(8), S1), call(0, integer(2), S0)),
        call(13, integer(3), integer(1)),
    )
    for renderer in SurfaceRenderer:
        assert parse_surface(render_surface(unsorted, renderer), renderer) == unsorted

    normalized = semantic_canonicalize(unsorted)
    assert normalized == call(
        14,
        call(1, integer(1), integer(4), integer(9)),
        call(2, call(0, integer(2), S0), call(0, integer(8), S1)),
        call(13, integer(3), integer(1)),
    )
    # Ordered command operations remain untouched.
    assert isinstance(normalized, SurfaceCall)
    assert normalized.children[2] == call(13, integer(3), integer(1))
    assert semantic_canonicalize(normalized) == normalized


def test_semantic_canonicalization_resolves_aliases_before_sorting() -> None:
    alias = symbol("x0000000000000009")
    canonical = symbol("x0000000000000001")
    document = call(
        14,
        call(1, call(11, alias, canonical), call(4, alias), call(4, canonical)),
        call(1, alias, canonical),
    )
    normalized = semantic_canonicalize(document)
    expected_application = call(4, canonical)
    assert normalized == call(
        14,
        call(
            1,
            call(11, canonical, canonical),
            expected_application,
            expected_application,
        ),
        call(1, canonical, canonical),
    )

    explicit = semantic_canonicalize(
        call(1, S2, S1, S0),
        aliases={S2.value: S0.value, S1.value: S0.value},
    )
    assert explicit == call(1, S0, S0, S0)


def test_opaque_symbol_assignment_uses_exact_context_and_collision_retry() -> None:
    observed: list[tuple[str, bytes]] = []

    def prf(label: str, context: bytes) -> bytes:
        observed.append((label, context))
        value = json.loads(context)
        # Force ordinal 1 to collide at counter zero, then resolve at counter one.
        if value["symbol_ordinal"] == 1 and value["counter"] == 0:
            return hashlib.sha256(
                b'forced-same-as-ordinal-zero'
            ).digest()
        if value["symbol_ordinal"] == 0:
            return hashlib.sha256(
                b'forced-same-as-ordinal-zero'
            ).digest()
        return hashlib.sha256(label.encode("ascii") + b"\0" + context).digest()

    context = OpaqueNameContext(
        cell_salt="shared-query",
        fold=2,
        presentation="alias_split",
        semantic_core_id="a" * 64,
        split="confirmation",
    )
    names = assign_opaque_symbols(3, prf=prf, context=context)
    assert len(names) == len({item.value for item in names}) == 3
    assert all(item.value.startswith("x") and len(item.value) == 17 for item in names)
    assert [json.loads(payload)["counter"] for _, payload in observed] == [0, 0, 1, 0]
    assert all(label == "opaque-name" for label, _ in observed)
    assert all(payload.endswith(b"\n") for _, payload in observed)
    assert assign_opaque_symbols(0, prf=prf, context=context) == ()


def test_opaque_assignment_is_deterministic_for_a_passed_prf() -> None:
    context = OpaqueNameContext(
        cell_salt="world-0",
        fold=0,
        presentation="base",
        semantic_core_id="core-7",
        split="train",
    )

    def prf(label: str, payload: bytes) -> bytes:
        return hashlib.sha256(label.encode("ascii") + b"\0" + payload).digest()

    assert assign_opaque_symbols(8, prf=prf, context=context) == (
        assign_opaque_symbols(8, prf=prf, context=context)
    )


def test_dispatch_and_input_types_fail_closed() -> None:
    with pytest.raises(TypeError):
        parse_surface(bytearray(GOLDEN_JSON), 0)  # type: ignore[arg-type]
    with pytest.raises(SurfaceCodecError, match="renderer"):
        render_surface(SAMPLE, 4)
    with pytest.raises(SurfaceCodecError, match="Boolean"):
        parse_surface(GOLDEN_JSON, True)  # type: ignore[arg-type]
    with pytest.raises(SurfaceSchemaError, match="exactly 32"):
        assign_opaque_symbols(
            1,
            prf=lambda _label, _context: b"short",
            context=OpaqueNameContext("world-0", 0, "base", "core", "train"),
        )
    with pytest.raises(SurfaceSchemaError, match="five v2 salts"):
        OpaqueNameContext("other", 0, "base", "core", "train")
    with pytest.raises(SurfaceSchemaError, match="exact int"):
        OpaqueNameContext("world-0", 3, "base", "core", "train")


def test_direct_codec_functions_match_dispatch() -> None:
    direct_renderers = (
        canonical_json_bytes,
        render_prefix,
        render_infix,
        render_postfix,
    )
    direct_parsers = (
        parse_canonical_json,
        parse_prefix,
        parse_infix,
        parse_postfix,
    )
    for renderer, render, parse in zip(
        SurfaceRenderer,
        direct_renderers,
        direct_parsers,
        strict=True,
    ):
        payload = render(SAMPLE)
        assert payload == render_surface(SAMPLE, renderer)
        assert parse(payload) == parse_surface(payload, renderer) == SAMPLE
