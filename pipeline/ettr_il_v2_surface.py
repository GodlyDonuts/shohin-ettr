"""Strict surface AST and byte codecs for R12-ETTR-IL-v2.

The parsers in this module preserve surface order.  Semantic normalization is
an explicit, separate operation so a parser can never hide a noncanonical or
presentation-changing input.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
import json
import re
from typing import Any, TypeAlias


MAX_INTEGER = 2_147_483_647
MAX_HEAD = 15
MAX_CHILDREN = 256
_SYMBOL_RE = re.compile(r"x[0-9a-f]{16}\Z")
_DECIMAL_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_ASCII_TEXT_RE = re.compile(r"[\x20-\x7e]*\Z")
_RS = b"\x1e"
_CELL_SALTS = frozenset(
    {"world-0", "world-1", "command-0", "command-1", "shared-query"}
)
_PRESENTATIONS = frozenset(
    {
        "base",
        "alpha_reorder",
        "alias_split",
        "relation_reification",
        "type_twin",
        "execution_semantics_twin",
    }
)
_SPLITS = frozenset({"train", "development", "confirmation"})


class SurfaceError(ValueError):
    """Base class for rejected surface values and byte documents."""


class SurfaceSchemaError(SurfaceError):
    """The value is not an instance of the frozen surface AST schema."""


class SurfaceCodecError(SurfaceError):
    """The byte document is malformed or noncanonical."""


@dataclass(frozen=True, slots=True)
class SurfaceInteger:
    """A nonnegative, schema-bounded integer leaf."""

    value: int

    def __post_init__(self) -> None:
        if type(self.value) is not int or not 0 <= self.value <= MAX_INTEGER:
            raise SurfaceSchemaError(
                f"integer must be an exact int in [0,{MAX_INTEGER}]"
            )


@dataclass(frozen=True, slots=True)
class SurfaceSymbol:
    """An opaque 64-bit hexadecimal surface name."""

    value: str

    def __post_init__(self) -> None:
        if type(self.value) is not str or _SYMBOL_RE.fullmatch(self.value) is None:
            raise SurfaceSchemaError(
                "symbol must match the exact pattern x[0-9a-f]{16}"
            )


@dataclass(frozen=True, slots=True)
class SurfaceCall:
    """A head code and an immutable ordered tuple of child nodes."""

    head: int
    children: tuple[SurfaceNode, ...]

    def __post_init__(self) -> None:
        if type(self.head) is not int or not 0 <= self.head <= MAX_HEAD:
            raise SurfaceSchemaError(f"head must be an exact int in [0,{MAX_HEAD}]")
        if type(self.children) is not tuple:
            raise SurfaceSchemaError("call children must be an immutable tuple")
        if len(self.children) > MAX_CHILDREN:
            raise SurfaceSchemaError(
                f"call may contain at most {MAX_CHILDREN} children"
            )
        for index, child in enumerate(self.children):
            if not isinstance(child, (SurfaceInteger, SurfaceSymbol, SurfaceCall)):
                raise SurfaceSchemaError(
                    f"child {index} is not a typed surface AST node"
                )


SurfaceNode: TypeAlias = SurfaceInteger | SurfaceSymbol | SurfaceCall
PRFCallback: TypeAlias = Callable[[str, bytes], bytes]

# Descriptive aliases make downstream annotations readable without weakening
# the single immutable representation.
IntegerNode = SurfaceInteger
SymbolNode = SurfaceSymbol
CallNode = SurfaceCall


class SurfaceRenderer(IntEnum):
    CANONICAL_JSON = 0
    PREFIX_SEXPR = 1
    RECORD_INFIX = 2
    REVERSE_POSTFIX = 3


@dataclass(frozen=True, slots=True)
class OpaqueNameContext:
    """Public context fields in the v2 opaque-name PRF preimage."""

    cell_salt: str
    fold: int
    presentation: str
    semantic_core_id: str
    split: str

    def __post_init__(self) -> None:
        for label, value in (
            ("cell_salt", self.cell_salt),
            ("presentation", self.presentation),
            ("semantic_core_id", self.semantic_core_id),
            ("split", self.split),
        ):
            if (
                type(value) is not str
                or not value
                or _ASCII_TEXT_RE.fullmatch(value) is None
            ):
                raise SurfaceSchemaError(
                    f"{label} must be nonempty printable ASCII text"
                )
        if self.cell_salt not in _CELL_SALTS:
            raise SurfaceSchemaError("cell_salt is not one of the five v2 salts")
        if self.presentation not in _PRESENTATIONS:
            raise SurfaceSchemaError("presentation is not admitted by v2")
        if self.split not in _SPLITS:
            raise SurfaceSchemaError("split is not admitted by v2")
        if type(self.fold) is not int or self.fold not in {0, 1, 2}:
            raise SurfaceSchemaError("fold must be the exact int 0, 1, or 2")


def integer(value: int) -> SurfaceInteger:
    return SurfaceInteger(value)


def symbol(value: str) -> SurfaceSymbol:
    return SurfaceSymbol(value)


def call(head: int, *children: SurfaceNode) -> SurfaceCall:
    return SurfaceCall(head, tuple(children))


def validate_surface_ast(node: SurfaceNode) -> None:
    """Validate the complete recursive schema without jsonschema."""

    if isinstance(node, (SurfaceInteger, SurfaceSymbol)):
        # Frozen constructors already enforce every leaf invariant.
        return
    if not isinstance(node, SurfaceCall):
        raise SurfaceSchemaError("root is not a typed surface AST node")
    if len(node.children) > MAX_CHILDREN:
        raise SurfaceSchemaError(
            f"call may contain at most {MAX_CHILDREN} children"
        )
    for child in node.children:
        validate_surface_ast(child)


def ast_to_json_value(node: SurfaceNode) -> dict[str, Any]:
    validate_surface_ast(node)
    if isinstance(node, SurfaceInteger):
        return {"i": node.value}
    if isinstance(node, SurfaceSymbol):
        return {"s": node.value}
    return {
        "a": [ast_to_json_value(child) for child in node.children],
        "h": node.head,
    }


def ast_from_json_value(value: object) -> SurfaceNode:
    """Validate and convert one plain JSON value to the typed AST."""

    if type(value) is not dict:
        raise SurfaceSchemaError("each AST node must be a JSON object")
    keys = set(value)
    if keys == {"i"}:
        return SurfaceInteger(value["i"])
    if keys == {"s"}:
        return SurfaceSymbol(value["s"])
    if keys != {"a", "h"}:
        raise SurfaceSchemaError("node has missing or additional properties")
    children = value["a"]
    if type(children) is not list:
        raise SurfaceSchemaError("call property a must be an array")
    if len(children) > MAX_CHILDREN:
        raise SurfaceSchemaError(
            f"call may contain at most {MAX_CHILDREN} children"
        )
    return SurfaceCall(
        value["h"],
        tuple(ast_from_json_value(child) for child in children),
    )


def canonical_json_bytes(node: SurfaceNode) -> bytes:
    """Render the exact strict-ASCII canonical JSON document, including LF."""

    payload = json.dumps(
        ast_to_json_value(node),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (payload + "\n").encode("ascii")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SurfaceCodecError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_float(_: str) -> float:
    raise SurfaceCodecError("JSON floats and exponent notation are forbidden")


def _reject_json_constant(value: str) -> object:
    raise SurfaceCodecError(f"nonfinite JSON constant {value!r} is forbidden")


def _require_bytes(payload: object) -> bytes:
    if type(payload) is not bytes:
        raise TypeError("surface documents must be immutable bytes")
    return payload


def _ascii(payload: bytes) -> str:
    try:
        return payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise SurfaceCodecError("surface document is not strict ASCII") from exc


def parse_canonical_json(payload: bytes) -> SurfaceNode:
    payload = _require_bytes(payload)
    text = _ascii(payload)
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_float=_reject_json_float,
            parse_constant=_reject_json_constant,
        )
        node = ast_from_json_value(value)
    except SurfaceError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise SurfaceCodecError("invalid surface JSON document") from exc
    if canonical_json_bytes(node) != payload:
        raise SurfaceCodecError("surface JSON document is parseable but noncanonical")
    return node


def render_prefix(node: SurfaceNode) -> bytes:
    validate_surface_ast(node)

    def visit(current: SurfaceNode) -> str:
        if isinstance(current, SurfaceInteger):
            return f"#{current.value}"
        if isinstance(current, SurfaceSymbol):
            return f"@{current.value}"
        suffix = "".join(f" {visit(child)}" for child in current.children)
        return f"({current.head}{suffix})"

    return (visit(node) + "\n").encode("ascii")


def _parse_decimal(
    text: str,
    *,
    label: str,
    maximum: int | None = None,
) -> int:
    if _DECIMAL_RE.fullmatch(text) is None:
        raise SurfaceCodecError(f"{label} is not canonical unsigned decimal")
    value = int(text)
    if maximum is not None and value > maximum:
        raise SurfaceCodecError(f"{label} exceeds {maximum}")
    return value


class _PrefixParser:
    def __init__(self, body: str) -> None:
        self.body = body
        self.offset = 0

    def parse_node(self) -> SurfaceNode:
        if self.offset >= len(self.body):
            raise SurfaceCodecError("unexpected end of prefix document")
        marker = self.body[self.offset]
        if marker == "#":
            return self._parse_integer()
        if marker == "@":
            return self._parse_symbol()
        if marker == "(":
            return self._parse_call()
        raise SurfaceCodecError(
            f"unexpected prefix marker at byte {self.offset}"
        )

    def _leaf_end(self, start: int) -> int:
        end = start
        while end < len(self.body) and self.body[end] not in " )":
            end += 1
        return end

    def _parse_integer(self) -> SurfaceInteger:
        start = self.offset + 1
        end = self._leaf_end(start)
        self.offset = end
        return SurfaceInteger(
            _parse_decimal(
                self.body[start:end],
                label="prefix integer",
                maximum=MAX_INTEGER,
            )
        )

    def _parse_symbol(self) -> SurfaceSymbol:
        start = self.offset + 1
        end = self._leaf_end(start)
        self.offset = end
        return SurfaceSymbol(self.body[start:end])

    def _parse_call(self) -> SurfaceCall:
        self.offset += 1
        start = self.offset
        while self.offset < len(self.body) and self.body[self.offset].isdigit():
            self.offset += 1
        head = _parse_decimal(
            self.body[start:self.offset],
            label="prefix head",
            maximum=MAX_HEAD,
        )
        children: list[SurfaceNode] = []
        while True:
            if self.offset >= len(self.body):
                raise SurfaceCodecError("unterminated prefix call")
            marker = self.body[self.offset]
            if marker == ")":
                self.offset += 1
                return SurfaceCall(head, tuple(children))
            if marker != " ":
                raise SurfaceCodecError("prefix call requires one child separator")
            if len(children) >= MAX_CHILDREN:
                raise SurfaceCodecError("prefix call exceeds maximum arity")
            self.offset += 1
            children.append(self.parse_node())


def parse_prefix(payload: bytes) -> SurfaceNode:
    payload = _require_bytes(payload)
    text = _ascii(payload)
    if not text.endswith("\n") or text.endswith("\n\n") or "\r" in text:
        raise SurfaceCodecError("prefix document must end in exactly one LF")
    parser = _PrefixParser(text[:-1])
    node = parser.parse_node()
    if parser.offset != len(parser.body):
        raise SurfaceCodecError("trailing bytes follow the prefix root")
    validate_surface_ast(node)
    if render_prefix(node) != payload:
        raise SurfaceCodecError("prefix document is parseable but noncanonical")
    return node


def _preorder(
    node: SurfaceNode,
) -> tuple[list[SurfaceNode], dict[int, tuple[int, ...]]]:
    nodes: list[SurfaceNode] = []
    child_ids: dict[int, tuple[int, ...]] = {}

    def visit(current: SurfaceNode) -> int:
        node_id = len(nodes)
        nodes.append(current)
        if isinstance(current, SurfaceCall):
            child_ids[node_id] = tuple(visit(child) for child in current.children)
        return node_id

    visit(node)
    return nodes, child_ids


def render_infix(node: SurfaceNode) -> bytes:
    validate_surface_ast(node)
    nodes, child_ids = _preorder(node)
    records: list[bytes] = [b"V2"]
    for node_id, current in enumerate(nodes):
        if isinstance(current, SurfaceInteger):
            record = f"I{node_id}={current.value}"
        elif isinstance(current, SurfaceSymbol):
            record = f"S{node_id}={current.value}"
        else:
            fields = [str(current.head), str(len(current.children))]
            fields.extend(str(child_id) for child_id in child_ids[node_id])
            record = f"N{node_id}=" + "%".join(fields)
        records.append(record.encode("ascii"))
    records.append(b"R=0")
    return _RS.join(records) + _RS


@dataclass(frozen=True, slots=True)
class _InfixCall:
    head: int
    child_ids: tuple[int, ...]


def _split_infix_record(record: str) -> tuple[str, str]:
    if record.count("=") != 1:
        raise SurfaceCodecError("infix node record must contain one equals sign")
    return tuple(record.split("=", 1))  # type: ignore[return-value]


def parse_infix(payload: bytes) -> SurfaceNode:
    payload = _require_bytes(payload)
    _ascii(payload)
    parts = payload.split(_RS)
    if len(parts) < 4 or parts[0] != b"V2" or parts[-1] != b"":
        raise SurfaceCodecError("infix document has invalid V2/RS framing")
    if any(part == b"" for part in parts[1:-1]):
        raise SurfaceCodecError("infix document contains an empty record")
    if parts[-2] != b"R=0":
        raise SurfaceCodecError("infix root record must be last and exactly R=0")
    records = [_ascii(part) for part in parts[1:-2]]
    if not records:
        raise SurfaceCodecError("infix document contains no AST nodes")

    definitions: list[SurfaceInteger | SurfaceSymbol | _InfixCall] = []
    for expected_id, record in enumerate(records):
        kind = record[:1]
        if kind not in {"I", "S", "N"}:
            raise SurfaceCodecError("infix node record has an invalid kind")
        left, right = _split_infix_record(record)
        node_id = _parse_decimal(left[1:], label="infix node id")
        if node_id != expected_id:
            raise SurfaceCodecError(
                "infix node IDs must be contiguous and increasing from zero"
            )
        if kind == "I":
            definitions.append(
                SurfaceInteger(
                    _parse_decimal(
                        right,
                        label="infix integer",
                        maximum=MAX_INTEGER,
                    )
                )
            )
        elif kind == "S":
            definitions.append(SurfaceSymbol(right))
        else:
            fields = right.split("%")
            if len(fields) < 2:
                raise SurfaceCodecError("infix call omits head or arity")
            head = _parse_decimal(
                fields[0],
                label="infix head",
                maximum=MAX_HEAD,
            )
            arity = _parse_decimal(
                fields[1],
                label="infix arity",
                maximum=MAX_CHILDREN,
            )
            if len(fields) != 2 + arity:
                raise SurfaceCodecError("infix call arity does not match child IDs")
            child_ids = tuple(
                _parse_decimal(field, label="infix child id")
                for field in fields[2:]
            )
            if any(child_id <= node_id for child_id in child_ids):
                raise SurfaceCodecError(
                    "every infix child ID must exceed its parent ID"
                )
            definitions.append(_InfixCall(head, child_ids))

    parent_counts = [0] * len(definitions)
    for parent_id, definition in enumerate(definitions):
        if not isinstance(definition, _InfixCall):
            continue
        for child_id in definition.child_ids:
            if child_id >= len(definitions):
                raise SurfaceCodecError("infix child ID is undefined")
            parent_counts[child_id] += 1
            if parent_counts[child_id] > 1:
                raise SurfaceCodecError("infix node has more than one parent")
    if parent_counts[0] != 0 or any(count != 1 for count in parent_counts[1:]):
        raise SurfaceCodecError(
            "infix graph must have root zero and exactly one parent per nonroot"
        )

    built: list[SurfaceNode | None] = [None] * len(definitions)
    for node_id in range(len(definitions) - 1, -1, -1):
        definition = definitions[node_id]
        if isinstance(definition, (SurfaceInteger, SurfaceSymbol)):
            built[node_id] = definition
            continue
        children = tuple(built[child_id] for child_id in definition.child_ids)
        if any(child is None for child in children):
            raise SurfaceCodecError("infix child topology is not reverse-buildable")
        built[node_id] = SurfaceCall(
            definition.head,
            tuple(child for child in children if child is not None),
        )
    node = built[0]
    if node is None:
        raise AssertionError("root construction unexpectedly failed")
    validate_surface_ast(node)
    if render_infix(node) != payload:
        raise SurfaceCodecError("infix document is parseable but noncanonical")
    return node


def render_postfix(node: SurfaceNode) -> bytes:
    validate_surface_ast(node)

    def tokens(current: SurfaceNode) -> list[str]:
        if isinstance(current, SurfaceInteger):
            return [f"#{current.value}"]
        if isinstance(current, SurfaceSymbol):
            return [f"${current.value}"]
        result: list[str] = []
        for child in reversed(current.children):
            result.extend(tokens(child))
        result.append(f"^{current.head}/{len(current.children)}")
        return result

    return (" ".join([*tokens(node), "!"]) + "\n").encode("ascii")


def parse_postfix(payload: bytes) -> SurfaceNode:
    payload = _require_bytes(payload)
    text = _ascii(payload)
    if not text.endswith("\n") or text.endswith("\n\n") or "\r" in text:
        raise SurfaceCodecError("postfix document must end in exactly one LF")
    raw_tokens = text[:-1].split(" ")
    if len(raw_tokens) < 2 or raw_tokens[-1] != "!" or any(
        token == "" for token in raw_tokens
    ):
        raise SurfaceCodecError("postfix tokens require canonical single spacing")
    stack: list[SurfaceNode] = []
    for token in raw_tokens[:-1]:
        if token.startswith("#"):
            stack.append(
                SurfaceInteger(
                    _parse_decimal(
                        token[1:],
                        label="postfix integer",
                        maximum=MAX_INTEGER,
                    )
                )
            )
        elif token.startswith("$"):
            stack.append(SurfaceSymbol(token[1:]))
        elif token.startswith("^"):
            if token.count("/") != 1:
                raise SurfaceCodecError("postfix close must contain one slash")
            head_text, arity_text = token[1:].split("/", 1)
            head = _parse_decimal(
                head_text,
                label="postfix head",
                maximum=MAX_HEAD,
            )
            arity = _parse_decimal(
                arity_text,
                label="postfix arity",
                maximum=MAX_CHILDREN,
            )
            if arity > len(stack):
                raise SurfaceCodecError("postfix close underflows the node stack")
            if arity:
                reverse_surface_order = stack[-arity:]
                del stack[-arity:]
                children = tuple(reversed(reverse_surface_order))
            else:
                children = ()
            stack.append(SurfaceCall(head, children))
        else:
            raise SurfaceCodecError("postfix token has an invalid marker")
    if len(stack) != 1:
        raise SurfaceCodecError(
            "postfix stack must contain exactly one root before bang"
        )
    node = stack[0]
    validate_surface_ast(node)
    if render_postfix(node) != payload:
        raise SurfaceCodecError("postfix document is parseable but noncanonical")
    return node


def _renderer(value: SurfaceRenderer | int) -> SurfaceRenderer:
    if isinstance(value, bool):
        raise SurfaceCodecError("renderer ID must not be Boolean")
    try:
        return SurfaceRenderer(value)
    except (TypeError, ValueError) as exc:
        raise SurfaceCodecError("renderer ID must be one of 0, 1, 2, or 3") from exc


def render_surface(
    node: SurfaceNode,
    renderer: SurfaceRenderer | int,
) -> bytes:
    selected = _renderer(renderer)
    functions = {
        SurfaceRenderer.CANONICAL_JSON: canonical_json_bytes,
        SurfaceRenderer.PREFIX_SEXPR: render_prefix,
        SurfaceRenderer.RECORD_INFIX: render_infix,
        SurfaceRenderer.REVERSE_POSTFIX: render_postfix,
    }
    return functions[selected](node)


def parse_surface(
    payload: bytes,
    renderer: SurfaceRenderer | int,
) -> SurfaceNode:
    selected = _renderer(renderer)
    functions = {
        SurfaceRenderer.CANONICAL_JSON: parse_canonical_json,
        SurfaceRenderer.PREFIX_SEXPR: parse_prefix,
        SurfaceRenderer.RECORD_INFIX: parse_infix,
        SurfaceRenderer.REVERSE_POSTFIX: parse_postfix,
    }
    return functions[selected](payload)


def _alias_pairs(node: SurfaceNode) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []

    def visit(current: SurfaceNode) -> None:
        if not isinstance(current, SurfaceCall):
            return
        if (
            current.head == 11
            and len(current.children) == 2
            and all(isinstance(child, SurfaceSymbol) for child in current.children)
        ):
            left, right = current.children
            assert isinstance(left, SurfaceSymbol)
            assert isinstance(right, SurfaceSymbol)
            pairs.append((left.value, right.value))
        for child in current.children:
            visit(child)

    visit(node)
    return tuple(pairs)


def _alias_representatives(
    pairs: Sequence[tuple[str, str]],
) -> dict[str, str]:
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        representative = min(left_root, right_root)
        other = max(left_root, right_root)
        parent[other] = representative

    for left, right in pairs:
        SurfaceSymbol(left)
        SurfaceSymbol(right)
        union(left, right)
    return {value: find(value) for value in tuple(parent)}


def semantic_canonicalize(
    node: SurfaceNode,
    *,
    aliases: Mapping[str, str] | None = None,
) -> SurfaceNode:
    """Resolve aliases, then sort only semantically unordered collections.

    Well-formed ``h=11`` two-symbol declarations are treated as undirected
    equivalences.  Additional aliases may be supplied explicitly.  The
    lexicographically smallest opaque name is the deterministic representative
    of each equivalence class.  Alias declarations remain in the tree; this
    function canonicalizes the generic AST rather than interpreting a stage
    document's declaration lifecycle.
    """

    validate_surface_ast(node)
    pairs = list(_alias_pairs(node))
    if aliases is not None:
        if not isinstance(aliases, Mapping):
            raise SurfaceSchemaError("aliases must be a mapping")
        for left, right in aliases.items():
            if type(left) is not str or type(right) is not str:
                raise SurfaceSchemaError("alias names must be strings")
            pairs.append((left, right))
    representatives = _alias_representatives(pairs)

    def visit(current: SurfaceNode) -> SurfaceNode:
        if isinstance(current, SurfaceInteger):
            return current
        if isinstance(current, SurfaceSymbol):
            return SurfaceSymbol(representatives.get(current.value, current.value))
        children = tuple(visit(child) for child in current.children)
        if current.head in {1, 2}:
            children = tuple(sorted(children, key=canonical_json_bytes))
        return SurfaceCall(current.head, children)

    return visit(node)


def assign_opaque_symbols(
    count: int,
    *,
    prf: PRFCallback,
    context: OpaqueNameContext,
) -> tuple[SurfaceSymbol, ...]:
    """Assign collision-free opaque names using the normative PRF context."""

    if type(count) is not int or count < 0:
        raise SurfaceSchemaError("symbol count must be a nonnegative exact int")
    if not callable(prf):
        raise TypeError("prf must be callable")
    if not isinstance(context, OpaqueNameContext):
        raise TypeError("context must be an OpaqueNameContext")

    used: set[str] = set()
    result: list[SurfaceSymbol] = []
    for ordinal in range(count):
        counter = 0
        while True:
            prf_context = _canonical_plain_json_bytes(
                {
                    "cell_salt": context.cell_salt,
                    "counter": counter,
                    "fold": context.fold,
                    "presentation": context.presentation,
                    "semantic_core_id": context.semantic_core_id,
                    "split": context.split,
                    "symbol_ordinal": ordinal,
                }
            )
            digest = prf("opaque-name", prf_context)
            if type(digest) is not bytes or len(digest) != 32:
                raise SurfaceSchemaError(
                    "opaque-name PRF must return exactly 32 immutable bytes"
                )
            name = "x" + digest[:8].hex()
            if name not in used:
                used.add(name)
                result.append(SurfaceSymbol(name))
                break
            counter += 1
    return tuple(result)


def _canonical_plain_json_bytes(value: object) -> bytes:
    """Canonical JSON for non-AST PRF contexts, including the normative LF."""

    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise SurfaceSchemaError("PRF context is not canonical ASCII JSON") from exc


__all__ = [
    "CallNode",
    "IntegerNode",
    "MAX_CHILDREN",
    "MAX_HEAD",
    "MAX_INTEGER",
    "OpaqueNameContext",
    "PRFCallback",
    "SurfaceCall",
    "SurfaceCodecError",
    "SurfaceError",
    "SurfaceInteger",
    "SurfaceNode",
    "SurfaceRenderer",
    "SurfaceSchemaError",
    "SurfaceSymbol",
    "SymbolNode",
    "assign_opaque_symbols",
    "ast_from_json_value",
    "ast_to_json_value",
    "call",
    "canonical_json_bytes",
    "integer",
    "parse_canonical_json",
    "parse_infix",
    "parse_postfix",
    "parse_prefix",
    "parse_surface",
    "render_infix",
    "render_postfix",
    "render_prefix",
    "render_surface",
    "semantic_canonicalize",
    "symbol",
    "validate_surface_ast",
]
