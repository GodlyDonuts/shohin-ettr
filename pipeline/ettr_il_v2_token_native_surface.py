"""Bounded token-native byte codec for ETTR surface ASTs.

The model-visible payload contains two shared structural atoms followed by
tokenizer-atomic structural codewords.  Ordinary nodes use one codeword.
Canonical reified-incidence bundles use one fused start code, one relation
symbol, and their endpoint subtrees; parsing reconstructs every ``h=12``
incidence exactly.  The four renderers share one semantic codebook and differ
only in prefix/postfix and child-order traversal.  Opaque 64-bit surface names
live in a canonical per-document sidecar: a symbol node carries only its local
ordinal.  This preserves exact names without spending at least five Shohin
tokens per unique 64-bit value.

The codebook is public and deterministic.  It is derived solely from the exact
tokenizer vocabulary by retaining strict-ASCII, leading-space alphanumeric
strings that round-trip to one and only one token.  Every complete document is
also re-tokenized and checked against the expected token IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
import re
from typing import Final

from tokenizers import Tokenizer

from ettr_il_v2_surface import (
    MAX_HEAD,
    SurfaceCall,
    SurfaceInteger,
    SurfaceNode,
    SurfaceRenderer,
    SurfaceSymbol,
    validate_surface_ast,
)


DEFAULT_TOKENIZER_PATH: Final[Path] = Path(
    "/Users/sairamen/projects/shohin/artifacts/tokenizer/tokenizer.json"
)
DEFAULT_TOKENIZER_SHA256: Final[str] = (
    "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4"
)
CODEC_SCHEMA: Final[str] = "r12-ettr-il-v2-token-native-surface-v2"

MAX_DOCUMENT_NODES: Final[int] = 4096
MAX_DOCUMENT_DEPTH: Final[int] = 256
CODEWORD_BYTES: Final[int] = 8
MIN_CODEBOOK_SIZE: Final[int] = 2048
MAX_NATIVE_ARITY: Final[int] = 32

_CODEWORD_RE = re.compile(r" [A-Za-z][A-Za-z0-9]*\Z")
_DOCUMENT_RE = re.compile(r"(?: [A-Za-z][A-Za-z0-9]*)+\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

_CALL_BASE = 0
_CALL_COUNT = (MAX_HEAD + 1) * (MAX_NATIVE_ARITY + 1)
_CALL_END = _CALL_BASE + _CALL_COUNT
_FRAME_A = _CALL_END
_FRAME_B = _FRAME_A + 1
_FRAME_END = _FRAME_B + 1
_FRAME_FILL = _FRAME_END + 1
_REIFY_BASE = _FRAME_FILL + 1
_REIFY_COUNT = MAX_NATIVE_ARITY + 1
_REIFY_END = _REIFY_BASE + _REIFY_COUNT
_INTEGER_BASE = _REIFY_END
_RENDERER_PREAMBLES = {
    SurfaceRenderer.CANONICAL_JSON: (_FRAME_A, _FRAME_A),
    SurfaceRenderer.PREFIX_SEXPR: (_FRAME_A, _FRAME_B),
    SurfaceRenderer.RECORD_INFIX: (_FRAME_B, _FRAME_A),
    SurfaceRenderer.REVERSE_POSTFIX: (_FRAME_B, _FRAME_B),
}
_PREFIX_RENDERERS = frozenset(
    {
        SurfaceRenderer.CANONICAL_JSON,
        SurfaceRenderer.PREFIX_SEXPR,
    }
)
_REVERSE_CHILD_RENDERERS = frozenset(
    {
        SurfaceRenderer.PREFIX_SEXPR,
        SurfaceRenderer.REVERSE_POSTFIX,
    }
)


class TokenNativeSurfaceError(ValueError):
    """Base class for token-native codec rejection."""


class TokenNativeCodebookError(TokenNativeSurfaceError):
    """The tokenizer cannot provide the required public atomic codebook."""


class TokenNativeDocumentError(TokenNativeSurfaceError):
    """A document or its per-document symbol sidecar is invalid."""


class TokenNativeBoundError(TokenNativeSurfaceError):
    """A valid SurfaceNode exceeds the deliberately bounded codec."""


@dataclass(frozen=True, slots=True)
class TokenNativeCodebook:
    """Public tokenizer-derived codebook in ascending token-ID order."""

    tokenizer_sha256: str
    codebook_sha256: str
    token_ids: tuple[int, ...]
    atoms: tuple[str, ...]

    def __post_init__(self) -> None:
        if _SHA256_RE.fullmatch(self.tokenizer_sha256) is None:
            raise TokenNativeCodebookError("tokenizer SHA-256 differs")
        if _SHA256_RE.fullmatch(self.codebook_sha256) is None:
            raise TokenNativeCodebookError("codebook SHA-256 differs")
        if len(self.token_ids) != len(self.atoms):
            raise TokenNativeCodebookError("codebook ID and atom counts differ")
        if len(self.atoms) < MIN_CODEBOOK_SIZE:
            raise TokenNativeCodebookError("atomic codebook is too small")
        if tuple(sorted(self.token_ids)) != self.token_ids:
            raise TokenNativeCodebookError("codebook token IDs are not sorted")
        if len(set(self.token_ids)) != len(self.token_ids):
            raise TokenNativeCodebookError("codebook token IDs repeat")
        if len(set(self.atoms)) != len(self.atoms):
            raise TokenNativeCodebookError("codebook atoms repeat")
        if any(_CODEWORD_RE.fullmatch(atom) is None for atom in self.atoms):
            raise TokenNativeCodebookError("codebook contains a noncanonical atom")


@dataclass(frozen=True, slots=True)
class TokenNativeSymbolContext:
    """Canonical assessor-owned local ordinal mapping for one document."""

    symbols: tuple[SurfaceSymbol, ...]
    context: str = ""

    def __post_init__(self) -> None:
        _validate_symbol_table(self.symbols)
        if type(self.context) is not str:
            raise TokenNativeDocumentError("symbol context must be a string")
        try:
            self.context.encode("ascii")
        except UnicodeEncodeError as exc:
            raise TokenNativeDocumentError(
                "symbol context must be strict ASCII"
            ) from exc


@dataclass(frozen=True, slots=True)
class TokenNativeDocument:
    """Model-visible bytes plus the exact assessor-owned symbol sidecar."""

    payload: bytes
    token_ids: tuple[int, ...]
    renderer: SurfaceRenderer
    symbol_context: TokenNativeSymbolContext
    tokenizer_sha256: str
    codebook_sha256: str

    def __post_init__(self) -> None:
        if type(self.payload) is not bytes:
            raise TokenNativeDocumentError("payload must be immutable bytes")
        if (
            type(self.token_ids) is not tuple
            or any(type(token_id) is not int for token_id in self.token_ids)
        ):
            raise TokenNativeDocumentError("token IDs must be an exact int tuple")
        if type(self.renderer) is not SurfaceRenderer:
            raise TokenNativeDocumentError("renderer must be a SurfaceRenderer")
        if type(self.symbol_context) is not TokenNativeSymbolContext:
            raise TokenNativeDocumentError(
                "document requires a TokenNativeSymbolContext"
            )
        if _SHA256_RE.fullmatch(self.tokenizer_sha256) is None:
            raise TokenNativeDocumentError("document tokenizer SHA-256 differs")
        if _SHA256_RE.fullmatch(self.codebook_sha256) is None:
            raise TokenNativeDocumentError("document codebook SHA-256 differs")

    @property
    def symbol_map(self) -> tuple[tuple[int, str], ...]:
        """Return the canonical local-ordinal-to-opaque-name binding."""

        return tuple(
            (ordinal, symbol.value)
            for ordinal, symbol in enumerate(self.symbol_context.symbols)
        )

    @property
    def symbols(self) -> tuple[SurfaceSymbol, ...]:
        return self.symbol_context.symbols


@dataclass(frozen=True, slots=True)
class TokenNativeTransport:
    """A fixed-token, fixed-byte model-visible transport envelope."""

    payload: bytes
    token_ids: tuple[int, ...]
    document: TokenNativeDocument
    width: int

    def __post_init__(self) -> None:
        if type(self.payload) is not bytes:
            raise TokenNativeDocumentError(
                "transport payload must be immutable bytes"
            )
        if (
            type(self.token_ids) is not tuple
            or any(type(token_id) is not int for token_id in self.token_ids)
        ):
            raise TokenNativeDocumentError(
                "transport token IDs must be an exact int tuple"
            )
        if type(self.document) is not TokenNativeDocument:
            raise TokenNativeDocumentError(
                "transport document identity differs"
            )
        if type(self.width) is not int or self.width < 1:
            raise TokenNativeDocumentError(
                "transport width must be a positive exact int"
            )
        if len(self.token_ids) != self.width:
            raise TokenNativeDocumentError(
                "transport token count differs from width"
            )
        if len(self.payload) != self.width * CODEWORD_BYTES:
            raise TokenNativeDocumentError(
                "transport byte count differs from fixed-width codewords"
            )


@dataclass(frozen=True, slots=True)
class TokenNativeMeasurement:
    """Auditable size and atomicity measurements for one document."""

    node_count: int
    symbol_count: int
    token_count: int
    ast_token_count: int
    tokens_per_node: float


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _renderer(value: SurfaceRenderer | int) -> SurfaceRenderer:
    if isinstance(value, bool):
        raise TokenNativeDocumentError("renderer ID must not be Boolean")
    try:
        return SurfaceRenderer(value)
    except (TypeError, ValueError) as exc:
        raise TokenNativeDocumentError(
            "renderer ID must be one of 0, 1, 2, or 3"
        ) from exc


def _validate_symbol_table(
    symbols: tuple[SurfaceSymbol, ...],
) -> None:
    if type(symbols) is not tuple:
        raise TokenNativeDocumentError("symbol sidecar must be an exact tuple")
    if any(type(symbol) is not SurfaceSymbol for symbol in symbols):
        raise TokenNativeDocumentError(
            "symbol sidecar values must be exact SurfaceSymbol instances"
        )
    values = tuple(symbol.value for symbol in symbols)
    if values != tuple(sorted(values)) or len(values) != len(set(values)):
        raise TokenNativeDocumentError(
            "symbol sidecar must be unique and sorted by opaque value"
        )


def _surface_inventory(
    node: SurfaceNode,
) -> tuple[tuple[SurfaceNode, ...], tuple[SurfaceSymbol, ...], int]:
    """Validate a bounded tree and return preorder, symbols, and max depth."""

    if not isinstance(node, (SurfaceInteger, SurfaceSymbol, SurfaceCall)):
        raise TypeError("root must be a SurfaceNode")
    preorder: list[SurfaceNode] = []
    symbols: set[SurfaceSymbol] = set()
    maximum_depth = 0
    stack: list[tuple[SurfaceNode, int]] = [(node, 1)]
    while stack:
        current, depth = stack.pop()
        if depth > MAX_DOCUMENT_DEPTH:
            raise TokenNativeBoundError(
                f"surface depth exceeds {MAX_DOCUMENT_DEPTH}"
            )
        maximum_depth = max(maximum_depth, depth)
        preorder.append(current)
        if len(preorder) > MAX_DOCUMENT_NODES:
            raise TokenNativeBoundError(
                f"surface node count exceeds {MAX_DOCUMENT_NODES}"
            )
        if isinstance(current, SurfaceSymbol):
            symbols.add(current)
        elif isinstance(current, SurfaceCall):
            stack.extend((child, depth + 1) for child in reversed(current.children))
    validate_surface_ast(node)
    return (
        tuple(preorder),
        tuple(sorted(symbols, key=lambda symbol: symbol.value)),
        maximum_depth,
    )


def canonical_symbol_table(node: SurfaceNode) -> tuple[SurfaceSymbol, ...]:
    """Return the exact sorted per-document symbol binding."""

    _, symbols, _ = _surface_inventory(node)
    return symbols


def count_surface_nodes(node: SurfaceNode) -> int:
    """Count AST nodes under the codec's explicit bounds."""

    preorder, _, _ = _surface_inventory(node)
    return len(preorder)


def _reified_endpoints(
    node: SurfaceNode,
) -> tuple[SurfaceSymbol, tuple[SurfaceNode, ...]] | None:
    if (
        not isinstance(node, SurfaceCall)
        or node.head != 2
        or not node.children
        or len(node.children) > MAX_NATIVE_ARITY
    ):
        return None
    relation: SurfaceSymbol | None = None
    by_role: dict[int, SurfaceNode] = {}
    for child in node.children:
        if (
            not isinstance(child, SurfaceCall)
            or child.head != 12
            or len(child.children) != 3
            or not isinstance(child.children[0], SurfaceSymbol)
            or not isinstance(child.children[1], SurfaceInteger)
        ):
            return None
        if relation is None:
            relation = child.children[0]
        elif relation != child.children[0]:
            return None
        role = child.children[1].value
        if role in by_role:
            return None
        by_role[role] = child.children[2]
    if tuple(sorted(by_role)) != tuple(range(len(node.children))):
        return None
    assert relation is not None
    return relation, tuple(by_role[role] for role in range(len(by_role)))


def _reified_ast(
    relation: SurfaceSymbol,
    endpoints: tuple[SurfaceNode, ...],
) -> SurfaceCall:
    if not endpoints or len(endpoints) > MAX_NATIVE_ARITY:
        raise TokenNativeDocumentError("fused reification arity differs")
    return SurfaceCall(
        2,
        tuple(
            SurfaceCall(
                12,
                (relation, SurfaceInteger(role), endpoint),
            )
            for role, endpoint in enumerate(endpoints)
        ),
    )


def _codebook_fingerprint(
    token_ids: tuple[int, ...],
    atoms: tuple[str, ...],
) -> str:
    rows = (
        f"{token_id}\t{atom[1:]}\n".encode("ascii")
        for token_id, atom in zip(token_ids, atoms, strict=True)
    )
    digest = hashlib.sha256()
    digest.update((CODEC_SCHEMA + "\n").encode("ascii"))
    for row in rows:
        digest.update(row)
    return digest.hexdigest()


def _derive_codebook(
    tokenizer: Tokenizer,
    tokenizer_sha256: str,
) -> TokenNativeCodebook:
    token_ids: list[int] = []
    atoms: list[str] = []
    seen_atoms: set[str] = set()
    for token_id in range(tokenizer.get_vocab_size()):
        atom = tokenizer.decode([token_id], skip_special_tokens=False)
        if (
            atom in seen_atoms
            or len(atom.encode("ascii", errors="ignore")) != CODEWORD_BYTES
            or _CODEWORD_RE.fullmatch(atom) is None
            or tokenizer.encode(atom, add_special_tokens=False).ids != [token_id]
        ):
            continue
        seen_atoms.add(atom)
        token_ids.append(token_id)
        atoms.append(atom)
    token_id_tuple = tuple(token_ids)
    atom_tuple = tuple(atoms)
    if len(atom_tuple) < MIN_CODEBOOK_SIZE:
        raise TokenNativeCodebookError(
            f"tokenizer supplies only {len(atom_tuple)} atomic ASCII codewords"
        )
    concatenated = "".join(atom_tuple)
    if tokenizer.encode(
        concatenated,
        add_special_tokens=False,
    ).ids != list(token_id_tuple):
        raise TokenNativeCodebookError(
            "atomic codewords are not stable when concatenated"
        )
    return TokenNativeCodebook(
        tokenizer_sha256=tokenizer_sha256,
        codebook_sha256=_codebook_fingerprint(token_id_tuple, atom_tuple),
        token_ids=token_id_tuple,
        atoms=atom_tuple,
    )


@lru_cache(maxsize=4)
def _load_tokenizer_and_codebook(
    resolved_path: str,
    tokenizer_sha256: str,
) -> tuple[Tokenizer, TokenNativeCodebook]:
    tokenizer = Tokenizer.from_file(resolved_path)
    return tokenizer, _derive_codebook(tokenizer, tokenizer_sha256)


class TokenNativeSurfaceCodec:
    """Strict bounded codec tied to one exact tokenizer artifact."""

    def __init__(
        self,
        tokenizer: Tokenizer | str | Path = DEFAULT_TOKENIZER_PATH,
        *,
        required_tokenizer_sha256: str | None = DEFAULT_TOKENIZER_SHA256,
    ) -> None:
        if isinstance(tokenizer, Tokenizer):
            exact = Tokenizer.from_file(str(DEFAULT_TOKENIZER_PATH))
            same_as_exact = tokenizer.to_str() == exact.to_str()
            if (
                required_tokenizer_sha256 == DEFAULT_TOKENIZER_SHA256
                and not same_as_exact
            ):
                raise TokenNativeCodebookError(
                    "tokenizer object differs from Shohin's exact artifact"
                )
            tokenizer_sha256 = (
                DEFAULT_TOKENIZER_SHA256
                if same_as_exact
                else _sha256(tokenizer.to_str().encode("utf-8"))
            )
            if (
                required_tokenizer_sha256 is not None
                and tokenizer_sha256 != required_tokenizer_sha256
            ):
                raise TokenNativeCodebookError(
                    "tokenizer SHA-256 does not match the required artifact"
                )
            self.tokenizer = tokenizer
            self.codebook = _derive_codebook(tokenizer, tokenizer_sha256)
        else:
            path = Path(tokenizer).expanduser().resolve()
            try:
                tokenizer_bytes = path.read_bytes()
            except OSError as exc:
                raise TokenNativeCodebookError(
                    f"cannot read tokenizer artifact: {path}"
                ) from exc
            tokenizer_sha256 = _sha256(tokenizer_bytes)
            if (
                required_tokenizer_sha256 is not None
                and tokenizer_sha256 != required_tokenizer_sha256
            ):
                raise TokenNativeCodebookError(
                    "tokenizer SHA-256 does not match the required artifact"
                )
            self.tokenizer, self.codebook = _load_tokenizer_and_codebook(
                str(path),
                tokenizer_sha256,
            )
        self._atom_to_index = {
            atom: index for index, atom in enumerate(self.codebook.atoms)
        }
        if len(self.codebook.atoms) <= _INTEGER_BASE + 1:
            raise TokenNativeCodebookError(
                "codebook cannot encode calls, integers, and symbols"
            )

    @property
    def tokenizer_sha256(self) -> str:
        return self.codebook.tokenizer_sha256

    @property
    def codebook_sha256(self) -> str:
        return self.codebook.codebook_sha256

    def _symbol_base(self, symbols: tuple[SurfaceSymbol, ...]) -> int:
        symbol_base = len(self.codebook.atoms) - len(symbols)
        if symbol_base <= _INTEGER_BASE:
            raise TokenNativeBoundError(
                "document has too many unique symbols for the codebook"
            )
        return symbol_base

    def _logical_document(
        self,
        node: SurfaceNode,
        symbols: tuple[SurfaceSymbol, ...],
        renderer: SurfaceRenderer,
    ) -> tuple[int, ...]:
        _, observed_symbols, _ = _surface_inventory(node)
        _validate_symbol_table(symbols)
        if symbols != observed_symbols:
            raise TokenNativeDocumentError(
                "symbol sidecar is not the exact canonical document binding"
            )
        symbol_base = self._symbol_base(symbols)
        symbol_ordinals = {
            symbol: ordinal for ordinal, symbol in enumerate(symbols)
        }
        direct_integer_max = symbol_base - _INTEGER_BASE - 1
        logical: list[int] = list(_RENDERER_PREAMBLES[renderer])

        def code(current: SurfaceNode) -> int:
            if isinstance(current, SurfaceCall):
                if len(current.children) > MAX_NATIVE_ARITY:
                    raise TokenNativeBoundError(
                        "call arity exceeds the token-native bound "
                        f"{MAX_NATIVE_ARITY}"
                    )
                return (
                    _CALL_BASE
                    + current.head * (MAX_NATIVE_ARITY + 1)
                    + len(current.children)
                )
            if isinstance(current, SurfaceInteger):
                if current.value <= direct_integer_max:
                    return _INTEGER_BASE + current.value
                raise TokenNativeBoundError(
                    "integer exceeds the document's one-token direct range "
                    f"[0,{direct_integer_max}]"
                )
            return symbol_base + symbol_ordinals[current]

        def visit(current: SurfaceNode) -> None:
            reified = _reified_endpoints(current)
            if reified is not None:
                relation, endpoints = reified
                ordered_endpoints = (
                    tuple(reversed(endpoints))
                    if renderer in _REVERSE_CHILD_RENDERERS
                    else endpoints
                )
                fused = _REIFY_BASE + len(endpoints)
                if renderer in _PREFIX_RENDERERS:
                    logical.append(fused)
                    logical.append(code(relation))
                    for endpoint in ordered_endpoints:
                        visit(endpoint)
                else:
                    logical.append(code(relation))
                    for endpoint in ordered_endpoints:
                        visit(endpoint)
                    logical.append(fused)
                return
            children = (
                tuple(reversed(current.children))
                if (
                    isinstance(current, SurfaceCall)
                    and renderer in _REVERSE_CHILD_RENDERERS
                )
                else current.children
                if isinstance(current, SurfaceCall)
                else ()
            )
            if renderer in _PREFIX_RENDERERS:
                logical.append(code(current))
            for child in children:
                visit(child)
            if renderer not in _PREFIX_RENDERERS:
                logical.append(code(current))

        visit(node)
        return tuple(logical)

    def _render_logical(
        self,
        logical: tuple[int, ...],
    ) -> bytes:
        text = "".join(self.codebook.atoms[index] for index in logical)
        payload = text.encode("ascii")
        expected_ids = [self.codebook.token_ids[index] for index in logical]
        observed_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
        ).ids
        if observed_ids != expected_ids:
            raise TokenNativeCodebookError(
                "complete token-native document is not atomically re-encoded"
            )
        return payload

    def render(
        self,
        node: SurfaceNode,
        renderer: SurfaceRenderer | int,
        *,
        symbols: tuple[SurfaceSymbol, ...] | None = None,
        symbol_context: TokenNativeSymbolContext | None = None,
    ) -> bytes:
        """Render model-visible bytes under an exact symbol sidecar."""

        selected = _renderer(renderer)
        if symbols is not None and symbol_context is not None:
            raise TokenNativeDocumentError(
                "provide symbols or symbol_context, not both"
            )
        binding = (
            symbol_context.symbols
            if symbol_context is not None
            else canonical_symbol_table(node)
            if symbols is None
            else symbols
        )
        logical = self._logical_document(node, binding, selected)
        return self._render_logical(logical)

    def serialize(
        self,
        node: SurfaceNode,
        renderer: SurfaceRenderer | int,
        *,
        symbol_context: TokenNativeSymbolContext | None = None,
    ) -> TokenNativeDocument:
        """Build model-visible bytes and the lossless document sidecar."""

        selected = _renderer(renderer)
        context = (
            TokenNativeSymbolContext(canonical_symbol_table(node))
            if symbol_context is None
            else symbol_context
        )
        payload = self.render(
            node,
            selected,
            symbol_context=context,
        )
        return TokenNativeDocument(
            payload=payload,
            token_ids=self.token_ids(payload),
            renderer=selected,
            symbol_context=context,
            tokenizer_sha256=self.tokenizer_sha256,
            codebook_sha256=self.codebook_sha256,
        )

    def _payload_indices(self, payload: bytes) -> tuple[int, ...]:
        if type(payload) is not bytes:
            raise TypeError("token-native payload must be immutable bytes")
        try:
            text = payload.decode("ascii")
        except UnicodeDecodeError as exc:
            raise TokenNativeDocumentError(
                "token-native payload is not strict ASCII"
            ) from exc
        if _DOCUMENT_RE.fullmatch(text) is None:
            raise TokenNativeDocumentError(
                "payload requires one leading space per public codeword"
            )
        atoms = tuple(" " + word for word in text[1:].split(" "))
        try:
            physical = tuple(self._atom_to_index[atom] for atom in atoms)
        except KeyError as exc:
            raise TokenNativeDocumentError(
                "payload contains a word outside the public codebook"
            ) from exc
        expected_ids = [self.codebook.token_ids[index] for index in physical]
        observed_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
        ).ids
        if observed_ids != expected_ids:
            raise TokenNativeDocumentError(
                "payload does not exactly re-encode to its codebook tokens"
            )
        if len(physical) > MAX_DOCUMENT_NODES:
            raise TokenNativeBoundError("payload exceeds the document token bound")
        return physical

    def parse(
        self,
        payload: bytes,
        renderer: SurfaceRenderer | int,
        *,
        symbols: tuple[SurfaceSymbol, ...],
    ) -> SurfaceNode:
        """Parse strict bytes with the canonical per-document symbol binding."""

        selected = _renderer(renderer)
        _validate_symbol_table(symbols)
        physical = self._payload_indices(payload)
        expected_preamble = _RENDERER_PREAMBLES[selected]
        if physical[:2] != expected_preamble:
            raise TokenNativeDocumentError(
                "payload grammar preamble differs from its renderer"
            )
        logical = physical[2:]
        symbol_base = self._symbol_base(symbols)

        def leaf_or_call(
            code: int,
        ) -> tuple[SurfaceNode | None, int | None, int | None]:
            if _CALL_BASE <= code < _CALL_END:
                call_code = code - _CALL_BASE
                return None, call_code, None
            if _REIFY_BASE <= code < _REIFY_END:
                arity = code - _REIFY_BASE
                if arity < 1:
                    raise TokenNativeDocumentError(
                        "fused reification arity differs"
                    )
                return None, None, arity
            if _INTEGER_BASE <= code < symbol_base:
                return SurfaceInteger(code - _INTEGER_BASE), None, None
            if symbol_base <= code < len(self.codebook.atoms):
                return symbols[code - symbol_base], None, None
            raise TokenNativeDocumentError("payload contains an invalid AST code")

        if selected in _PREFIX_RENDERERS:
            offset = 0

            def consume(depth: int) -> SurfaceNode:
                nonlocal offset
                if depth > MAX_DOCUMENT_DEPTH:
                    raise TokenNativeBoundError(
                        f"surface depth exceeds {MAX_DOCUMENT_DEPTH}"
                    )
                if offset >= len(logical):
                    raise TokenNativeDocumentError(
                        "payload ends inside the AST"
                    )
                value, call_code, reify_arity = leaf_or_call(logical[offset])
                offset += 1
                if value is not None:
                    return value
                if reify_arity is not None:
                    relation = consume(depth + 1)
                    if not isinstance(relation, SurfaceSymbol):
                        raise TokenNativeDocumentError(
                            "fused reification lacks its relation symbol"
                        )
                    endpoints = tuple(
                        consume(depth + 1) for _ in range(reify_arity)
                    )
                    if selected in _REVERSE_CHILD_RENDERERS:
                        endpoints = tuple(reversed(endpoints))
                    return _reified_ast(relation, endpoints)
                assert call_code is not None
                head, arity = divmod(call_code, MAX_NATIVE_ARITY + 1)
                children = tuple(consume(depth + 1) for _ in range(arity))
                if selected in _REVERSE_CHILD_RENDERERS:
                    children = tuple(reversed(children))
                return SurfaceCall(head, children)

            try:
                node = consume(1)
            except RecursionError as exc:
                raise TokenNativeBoundError(
                    "surface nesting exceeds parser bounds"
                ) from exc
            if offset != len(logical):
                raise TokenNativeDocumentError(
                    "trailing tokens follow the AST root"
                )
        else:
            stack: list[SurfaceNode] = []
            for code in logical:
                value, call_code, reify_arity = leaf_or_call(code)
                if value is not None:
                    stack.append(value)
                    continue
                if reify_arity is not None:
                    needed = reify_arity + 1
                    if len(stack) < needed:
                        raise TokenNativeDocumentError(
                            "fused reification lacks relation or endpoints"
                        )
                    values = tuple(stack[-needed:])
                    del stack[-needed:]
                    relation = values[0]
                    endpoints = values[1:]
                    if not isinstance(relation, SurfaceSymbol):
                        raise TokenNativeDocumentError(
                            "fused reification lacks its relation symbol"
                        )
                    if selected in _REVERSE_CHILD_RENDERERS:
                        endpoints = tuple(reversed(endpoints))
                    stack.append(_reified_ast(relation, endpoints))
                    continue
                assert call_code is not None
                head, arity = divmod(call_code, MAX_NATIVE_ARITY + 1)
                if len(stack) < arity:
                    raise TokenNativeDocumentError(
                        "postfix call lacks child nodes"
                    )
                children = tuple(stack[-arity:]) if arity else ()
                if arity:
                    del stack[-arity:]
                if selected in _REVERSE_CHILD_RENDERERS:
                    children = tuple(reversed(children))
                stack.append(SurfaceCall(head, children))
            if len(stack) != 1:
                raise TokenNativeDocumentError(
                    "postfix payload does not contain one AST root"
                )
            node = stack[0]
        _surface_inventory(node)
        if self.render(node, selected, symbols=symbols) != payload:
            raise TokenNativeDocumentError(
                "payload is parseable but not canonical"
            )
        return node

    def deserialize(self, document: TokenNativeDocument) -> SurfaceNode:
        """Verify artifact identity and parse a complete sidecar document."""

        if type(document) is not TokenNativeDocument:
            raise TypeError("document must be a TokenNativeDocument")
        if document.tokenizer_sha256 != self.tokenizer_sha256:
            raise TokenNativeDocumentError("document tokenizer identity differs")
        if document.codebook_sha256 != self.codebook_sha256:
            raise TokenNativeDocumentError("document codebook identity differs")
        if document.token_ids != self.token_ids(document.payload):
            raise TokenNativeDocumentError("document token IDs differ from bytes")
        return self.parse(
            document.payload,
            document.renderer,
            symbols=document.symbols,
        )

    def token_ids(self, payload: bytes) -> tuple[int, ...]:
        """Return exact IDs after enforcing atomic model-visible encoding."""

        physical = self._payload_indices(payload)
        return tuple(self.codebook.token_ids[index] for index in physical)

    def _cover_indices(
        self,
        document: TokenNativeDocument,
        *,
        width: int,
        count: int,
    ) -> tuple[int, ...]:
        if count < 0:
            raise TokenNativeDocumentError("cover count differs")
        seed = hashlib.sha256(
            b"R12-ETTR-IL-v2\0transport-cover\0"
            + document.payload
            + width.to_bytes(4, "big")
        ).digest()
        result: list[int] = []
        counter = 0
        while len(result) < count:
            block = hashlib.sha256(
                seed + counter.to_bytes(8, "big")
            ).digest()
            counter += 1
            for offset in range(0, len(block), 2):
                result.append(
                    int.from_bytes(block[offset : offset + 2], "big")
                    % len(self.codebook.atoms)
                )
                if len(result) == count:
                    break
        return tuple(result)

    def pack(
        self,
        document: TokenNativeDocument,
        *,
        width: int,
    ) -> TokenNativeTransport:
        """Seal one AST in a fixed-width deterministic cover envelope."""

        if type(document) is not TokenNativeDocument:
            raise TypeError("document must be a TokenNativeDocument")
        if document.tokenizer_sha256 != self.tokenizer_sha256:
            raise TokenNativeDocumentError(
                "transport tokenizer identity differs"
            )
        if document.codebook_sha256 != self.codebook_sha256:
            raise TokenNativeDocumentError(
                "transport codebook identity differs"
            )
        if type(width) is not int or width < 1:
            raise TokenNativeDocumentError(
                "transport width must be a positive exact int"
            )
        required = len(document.token_ids)
        if required > width:
            raise TokenNativeBoundError(
                f"document needs {required} tokens, "
                f"exceeding width {width}"
            )
        logical = (
            *self._payload_indices(document.payload),
            *self._cover_indices(
                document,
                width=width,
                count=width - required,
            ),
        )
        payload = self._render_logical(logical)
        token_ids = self.token_ids(payload)
        transport = TokenNativeTransport(
            payload=payload,
            token_ids=token_ids,
            document=document,
            width=width,
        )
        if self.unpack(transport) != self.deserialize(document):
            raise TokenNativeDocumentError(
                "transport round-trip changes the AST"
            )
        return transport

    def unpack(self, transport: TokenNativeTransport) -> SurfaceNode:
        """Validate deterministic cover bytes and recover the exact AST."""

        if type(transport) is not TokenNativeTransport:
            raise TypeError("transport must be a TokenNativeTransport")
        if self.token_ids(transport.payload) != transport.token_ids:
            raise TokenNativeDocumentError(
                "transport token IDs differ from bytes"
            )
        physical = self._payload_indices(transport.payload)
        document_physical = self._payload_indices(
            transport.document.payload
        )
        end = len(document_physical)
        if physical[:end] != document_physical:
            raise TokenNativeDocumentError(
                "transport AST bytes differ from its document"
            )
        expected_cover = self._cover_indices(
            transport.document,
            width=transport.width,
            count=transport.width - end,
        )
        if physical[end:] != expected_cover:
            raise TokenNativeDocumentError(
                "transport deterministic cover differs"
            )
        return self.deserialize(transport.document)

    def measure(
        self,
        node: SurfaceNode,
        renderer: SurfaceRenderer | int,
    ) -> TokenNativeMeasurement:
        """Measure exact per-node token usage."""

        document = self.serialize(node, renderer)
        preorder, _, _ = _surface_inventory(node)
        token_count = len(self.token_ids(document.payload))
        ast_tokens = token_count - 2
        return TokenNativeMeasurement(
            node_count=len(preorder),
            symbol_count=len(document.symbols),
            token_count=token_count,
            ast_token_count=ast_tokens,
            tokens_per_node=ast_tokens / len(preorder),
        )


def encode_token_native_surface(
    node: SurfaceNode,
    renderer: SurfaceRenderer | int,
    tokenizer: Tokenizer | str | Path = DEFAULT_TOKENIZER_PATH,
    *,
    symbol_context: TokenNativeSymbolContext | None = None,
    required_tokenizer_sha256: str | None = DEFAULT_TOKENIZER_SHA256,
) -> TokenNativeDocument:
    """Encode one AST and return bytes, exact IDs, and symbol context."""

    codec = TokenNativeSurfaceCodec(
        tokenizer,
        required_tokenizer_sha256=required_tokenizer_sha256,
    )
    return codec.serialize(
        node,
        renderer,
        symbol_context=symbol_context,
    )


def parse_token_native_surface(
    payload: bytes,
    renderer: SurfaceRenderer | int,
    tokenizer: Tokenizer | str | Path,
    *,
    symbol_context: TokenNativeSymbolContext,
    required_tokenizer_sha256: str | None = DEFAULT_TOKENIZER_SHA256,
) -> SurfaceNode:
    """Parse one strict payload using its explicit per-document context."""

    codec = TokenNativeSurfaceCodec(
        tokenizer,
        required_tokenizer_sha256=required_tokenizer_sha256,
    )
    return codec.parse(
        payload,
        renderer,
        symbols=symbol_context.symbols,
    )


__all__ = [
    "CODEC_SCHEMA",
    "CODEWORD_BYTES",
    "DEFAULT_TOKENIZER_PATH",
    "DEFAULT_TOKENIZER_SHA256",
    "MAX_DOCUMENT_DEPTH",
    "MAX_DOCUMENT_NODES",
    "MAX_NATIVE_ARITY",
    "TokenNativeBoundError",
    "TokenNativeCodebook",
    "TokenNativeCodebookError",
    "TokenNativeDocument",
    "TokenNativeDocumentError",
    "TokenNativeMeasurement",
    "TokenNativeSymbolContext",
    "TokenNativeSurfaceCodec",
    "TokenNativeSurfaceError",
    "TokenNativeTransport",
    "canonical_symbol_table",
    "count_surface_nodes",
    "encode_token_native_surface",
    "parse_token_native_surface",
]
