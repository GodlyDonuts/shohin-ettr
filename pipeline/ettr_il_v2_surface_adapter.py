"""Ontology-neutral semantic-to-surface adapter for ETTR-IL v2.

The adapter turns assessor-side Horn, rewrite, and resource factors into the
common surface AST, applies one frozen presentation transform, assigns
cell-local opaque names, renders one of the four frozen codecs, and immediately
proves an assessor-side parse/presentation-inversion/canonicalization round
trip.

This module is CPU-only.  It performs no dataset selection, model access,
training, job submission, or checkpoint I/O.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
import hashlib
import re
from typing import TypeAlias

from cross_ontology_horn_board import (
    OBJECT_TYPES,
    PREDICATES,
    RULE_LIBRARY as HORN_RULE_LIBRARY,
    THEORIES as HORN_THEORIES,
    AtomPattern,
    GroundAtom,
)
from cross_ontology_resource_board import (
    OPERATOR_LIBRARY,
    OPERATOR_SYMBOL_COUNT,
    PLACE_SPECS,
    THEORIES as RESOURCE_THEORIES,
    GuardedOperator,
    Marking,
    ResourceQuantity,
)
from cross_ontology_rewrite_board import (
    CONSTRUCTORS,
    RULE_LIBRARY as REWRITE_RULE_LIBRARY,
    THEORIES as REWRITE_THEORIES,
    GroundTerm,
    PatternTerm,
)
from ettr_il_v2_semantics import (
    Command,
    HornCommand,
    HornPolicy,
    HornWorld,
    QueryOp,
    ResourceCommand,
    ResourcePolicy,
    ResourceWorld,
    RewriteCommand,
    RewritePolicy,
    RewriteWorld,
    SemanticQuery,
    World,
    query_surface_value,
)
from ettr_il_v2_surface import (
    OpaqueNameContext,
    PRFCallback,
    SurfaceCall,
    SurfaceInteger,
    SurfaceNode,
    SurfaceRenderer,
    SurfaceSymbol,
    assign_opaque_symbols,
    ast_from_json_value,
    call,
    canonical_json_bytes,
    integer,
    parse_surface,
    render_surface,
    semantic_canonicalize,
)


_ASCII_RE = re.compile(r"[\x20-\x7e]+\Z")
_SPLITS = frozenset({"train", "development", "confirmation"})
_PRESENTATIONS = (
    "base",
    "alpha_reorder",
    "alias_split",
    "relation_reification",
    "type_twin",
    "execution_semantics_twin",
)
IMPLEMENTED_PRESENTATIONS = (
    "base",
    "alpha_reorder",
    "alias_split",
    "relation_reification",
    "type_twin",
    "execution_semantics_twin",
)
QUERY_OP_ORDER = tuple(QueryOp)


class SurfaceAdapterError(ValueError):
    """A semantic factor cannot be represented by the frozen adapter."""


class SurfaceStage(StrEnum):
    WORLD = "WORLD"
    COMMAND = "COMMAND"
    QUERY = "QUERY"


SymbolKey: TypeAlias = tuple[str, int]
PresentationBuilder: TypeAlias = Callable[[World | Command, "_SymbolCatalog"], SurfaceNode]


@dataclass(frozen=True, slots=True)
class _SymbolCatalog:
    keys: tuple[SymbolKey, ...]

    def __post_init__(self) -> None:
        if type(self.keys) is not tuple or len(set(self.keys)) != len(self.keys):
            raise SurfaceAdapterError("semantic symbol catalog differs")
        for key in self.keys:
            if (
                type(key) is not tuple
                or len(key) != 2
                or type(key[0]) is not str
                or type(key[1]) is not int
            ):
                raise SurfaceAdapterError("semantic symbol key differs")

    def ordinal(self, key: SymbolKey) -> int:
        try:
            return self.keys.index(key)
        except ValueError as exc:
            raise SurfaceAdapterError(
                f"semantic symbol is outside the frozen catalog: {key!r}"
            ) from exc

    def canonical(self, key: SymbolKey) -> SurfaceSymbol:
        return _canonical_symbol(self.ordinal(key))


@dataclass(frozen=True, slots=True)
class SurfaceAdapterContext:
    """Assessor-owned context shared by one semantic rectangle."""

    fold: int
    split: str
    semantic_core_id: str
    semantic_rectangle_id: str
    renderer: SurfaceRenderer
    prf: PRFCallback
    presentation: str = "base"

    def __post_init__(self) -> None:
        if type(self.fold) is not int or self.fold not in {0, 1, 2}:
            raise SurfaceAdapterError("fold must be the exact int 0, 1, or 2")
        if type(self.split) is not str or self.split not in _SPLITS:
            raise SurfaceAdapterError("split differs")
        for label, value in (
            ("semantic_core_id", self.semantic_core_id),
            ("semantic_rectangle_id", self.semantic_rectangle_id),
        ):
            if (
                type(value) is not str
                or _ASCII_RE.fullmatch(value) is None
            ):
                raise SurfaceAdapterError(f"{label} must be nonempty ASCII")
        if type(self.renderer) is not SurfaceRenderer:
            raise SurfaceAdapterError("renderer must be a SurfaceRenderer")
        if not callable(self.prf):
            raise SurfaceAdapterError("prf must be callable")
        if type(self.presentation) is not str:
            raise SurfaceAdapterError("presentation must be a string")
        if self.presentation not in _PRESENTATIONS:
            raise SurfaceAdapterError("presentation is outside the v2 protocol")
        if self.presentation not in IMPLEMENTED_PRESENTATIONS:
            raise SurfaceAdapterError(
                f"presentation {self.presentation!r} is not implemented"
            )


@dataclass(frozen=True, slots=True)
class SemanticSurfaceDocument:
    """One rendered factor plus its sealed inverse symbol binding."""

    stage: SurfaceStage
    presentation: str
    cell_salt: str
    layout: int
    renderer: SurfaceRenderer
    ast: SurfaceNode
    source: bytes
    semantic_ast: SurfaceNode
    opaque_to_canonical: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if type(self.stage) is not SurfaceStage:
            raise SurfaceAdapterError("surface stage differs")
        if self.presentation not in IMPLEMENTED_PRESENTATIONS:
            raise SurfaceAdapterError("surface presentation is not implemented")
        if type(self.cell_salt) is not str:
            raise SurfaceAdapterError("cell salt differs")
        if type(self.layout) is not int or self.layout not in {0, 1}:
            raise SurfaceAdapterError("surface layout differs")
        if type(self.renderer) is not SurfaceRenderer:
            raise SurfaceAdapterError("document renderer differs")
        if type(self.source) is not bytes:
            raise SurfaceAdapterError("surface source must be immutable bytes")
        if render_surface(self.ast, self.renderer) != self.source:
            raise SurfaceAdapterError("surface source and AST differ")
        if len(dict(self.opaque_to_canonical)) != len(
            self.opaque_to_canonical
        ):
            raise SurfaceAdapterError("inverse opaque binding is not functional")

    def parsed_semantics(self) -> SurfaceNode:
        """Parse all source bytes and recover the canonical factor AST."""

        return parse_and_semantic_canonicalize(self)


@dataclass(frozen=True, slots=True)
class QuerySurfacePrefix:
    """One shared query document and its exact answer prefix."""

    query: SemanticQuery
    paraphrase: int
    document: SemanticSurfaceDocument
    prefix: bytes

    def __post_init__(self) -> None:
        if type(self.query) is not SemanticQuery:
            raise SurfaceAdapterError("query type differs")
        if type(self.paraphrase) is not int or self.paraphrase not in {0, 1}:
            raise SurfaceAdapterError("query paraphrase differs")
        if self.document.stage != SurfaceStage.QUERY:
            raise SurfaceAdapterError("query document has the wrong stage")
        if self.prefix != self.document.source + b"R=":
            raise SurfaceAdapterError("query prefix framing differs")


@dataclass(frozen=True, slots=True)
class CornerSurface:
    """Candidate-visible sources at one ``(world, command)`` corner."""

    world_index: int
    command_index: int
    world: SemanticSurfaceDocument
    command: SemanticSurfaceDocument
    query_prefixes: tuple[
        tuple[QuerySurfacePrefix, QuerySurfacePrefix],
        tuple[QuerySurfacePrefix, QuerySurfacePrefix],
    ]


@dataclass(frozen=True, slots=True)
class BaseSurfaceBundle:
    """All cell-local sources for one base semantic rectangle.

    ``world_variants[w][c]`` uses ``world-<c>``.
    ``command_variants[c][w]`` uses ``command-<w>``.
    """

    world_variants: tuple[
        tuple[SemanticSurfaceDocument, SemanticSurfaceDocument],
        tuple[SemanticSurfaceDocument, SemanticSurfaceDocument],
    ]
    command_variants: tuple[
        tuple[SemanticSurfaceDocument, SemanticSurfaceDocument],
        tuple[SemanticSurfaceDocument, SemanticSurfaceDocument],
    ]
    query_prefixes: tuple[
        tuple[QuerySurfacePrefix, QuerySurfacePrefix],
        tuple[QuerySurfacePrefix, QuerySurfacePrefix],
    ]

    def corner(self, world_index: int, command_index: int) -> CornerSurface:
        if (
            type(world_index) is not int
            or type(command_index) is not int
            or world_index not in {0, 1}
            or command_index not in {0, 1}
        ):
            raise SurfaceAdapterError("corner indices must be exact bits")
        return CornerSurface(
            world_index=world_index,
            command_index=command_index,
            world=self.world_variants[world_index][command_index],
            command=self.command_variants[command_index][world_index],
            query_prefixes=self.query_prefixes,
        )


def _canonical_symbol(ordinal: int) -> SurfaceSymbol:
    if type(ordinal) is not int or ordinal < 0:
        raise SurfaceAdapterError("symbol ordinal differs")
    return SurfaceSymbol(f"x{ordinal + 1:016x}")


def _horn_catalog() -> _SymbolCatalog:
    return _SymbolCatalog(
        (
            *(("type", index) for index in range(2)),
            *(("object", index) for index in range(len(OBJECT_TYPES))),
            *(("operator", index) for index in range(len(PREDICATES))),
            ("policy", 0),
            ("policy", 1),
        )
    )


def _rewrite_catalog() -> _SymbolCatalog:
    return _SymbolCatalog(
        (
            *(("type", index) for index in range(3)),
            *(("operator", index) for index in range(len(CONSTRUCTORS))),
            ("policy", 0),
            ("policy", 1),
        )
    )


def _resource_catalog() -> _SymbolCatalog:
    return _SymbolCatalog(
        (
            ("type", 0),
            ("type", 1),
            *(("object", index) for index in range(len(PLACE_SPECS))),
            *(("operator", index) for index in range(OPERATOR_SYMBOL_COUNT)),
            ("policy", 0),
            ("policy", 1),
        )
    )


def _query_catalog() -> _SymbolCatalog:
    return _SymbolCatalog(
        tuple(("operator", index) for index in range(len(QUERY_OP_ORDER)))
    )


def _catalog_for_factor(
    factor: World | Command,
    *,
    presentation: str = "base",
) -> _SymbolCatalog:
    if type(factor) in {HornWorld, HornCommand}:
        base = _horn_catalog()
    elif type(factor) in {RewriteWorld, RewriteCommand}:
        base = _rewrite_catalog()
    elif type(factor) in {ResourceWorld, ResourceCommand}:
        base = _resource_catalog()
    elif type(factor) not in {HornWorld, HornCommand}:
        raise SurfaceAdapterError("factor type is outside the three ontologies")
    if presentation == "alias_split":
        return _SymbolCatalog((*base.keys, ("alias", 0)))
    if presentation == "type_twin":
        return _SymbolCatalog((*base.keys, ("type_twin", 0)))
    if presentation == "relation_reification":
        base_ast = _base_factor_ast(factor, base)
        return _SymbolCatalog(
            (
                *base.keys,
                *(
                    ("relation", index)
                    for index in range(_application_count(base_ast))
                ),
            )
        )
    return base


def _declaration(
    symbol_node: SurfaceSymbol,
    ordinal: int,
    payload: SurfaceNode,
) -> SurfaceCall:
    return call(3, symbol_node, integer(ordinal), payload)


def _application(
    operator: SurfaceSymbol,
    *arguments: SurfaceNode,
) -> SurfaceCall:
    return call(4, operator, *arguments)


def _horn_atom(atom: GroundAtom, catalog: _SymbolCatalog) -> SurfaceCall:
    return _application(
        catalog.canonical(("operator", atom.predicate)),
        *(catalog.canonical(("object", value)) for value in atom.arguments),
    )


def _horn_pattern(
    pattern: AtomPattern,
    catalog: _SymbolCatalog,
) -> SurfaceCall:
    predicate = PREDICATES[pattern.predicate]
    return _application(
        catalog.canonical(("operator", pattern.predicate)),
        *(
            call(
                5,
                integer(variable),
                catalog.canonical(("type", required_type)),
            )
            for variable, required_type in zip(
                pattern.variables,
                predicate.argument_types,
                strict=True,
            )
        ),
    )


def _horn_declarations(catalog: _SymbolCatalog) -> tuple[SurfaceCall, ...]:
    result: list[SurfaceCall] = []
    for type_index in range(2):
        result.append(
            _declaration(
                catalog.canonical(("type", type_index)),
                type_index,
                call(0),
            )
        )
    for object_index, type_index in enumerate(OBJECT_TYPES):
        result.append(
            _declaration(
                catalog.canonical(("object", object_index)),
                object_index,
                catalog.canonical(("type", type_index)),
            )
        )
    for predicate in PREDICATES:
        result.append(
            _declaration(
                catalog.canonical(("operator", predicate.index)),
                predicate.index,
                call(
                    0,
                    *(
                        catalog.canonical(("type", type_index))
                        for type_index in predicate.argument_types
                    ),
                ),
            )
        )
    return tuple(result)


def _horn_world(world: HornWorld, catalog: _SymbolCatalog) -> SurfaceNode:
    theory = HORN_THEORIES[world.theory_index]
    laws = tuple(
        call(
            6,
            call(
                1,
                *(
                    _horn_pattern(premise, catalog)
                    for premise in HORN_RULE_LIBRARY[rule_index].premises
                ),
            ),
            _horn_pattern(
                HORN_RULE_LIBRARY[rule_index].conclusion,
                catalog,
            ),
        )
        for rule_index in theory.rule_indices
    )
    policy_index = 0 if world.policy == HornPolicy.PERSISTENT else 1
    return call(
        14,
        integer(2),
        call(1, *_horn_declarations(catalog)),
        call(
            0,
            call(1, *laws),
            call(7, *(_horn_atom(atom, catalog) for atom in world.initial)),
            call(8, catalog.canonical(("policy", policy_index))),
        ),
    )


def _horn_command(
    command: HornCommand,
    catalog: _SymbolCatalog,
) -> SurfaceNode:
    return call(
        14,
        integer(2),
        call(1, *_horn_declarations(catalog)),
        call(
            13,
            *(_horn_atom(atom, catalog) for atom in command.operations),
        ),
    )


def _rewrite_ground(
    term: GroundTerm,
    catalog: _SymbolCatalog,
) -> SurfaceCall:
    return _application(
        catalog.canonical(("operator", term.constructor_index)),
        *(_rewrite_ground(child, catalog) for child in term.children),
    )


def _rewrite_pattern(
    term: PatternTerm,
    catalog: _SymbolCatalog,
) -> SurfaceNode:
    if term.variable_index is not None:
        return call(
            5,
            integer(term.variable_index),
            catalog.canonical(("type", term.type_index)),
        )
    assert term.constructor_index is not None
    return _application(
        catalog.canonical(("operator", term.constructor_index)),
        *(_rewrite_pattern(child, catalog) for child in term.children),
    )


def _rewrite_declarations(
    catalog: _SymbolCatalog,
) -> tuple[SurfaceCall, ...]:
    result: list[SurfaceCall] = []
    for type_index in range(3):
        result.append(
            _declaration(
                catalog.canonical(("type", type_index)),
                type_index,
                call(0),
            )
        )
    for constructor in CONSTRUCTORS:
        result.append(
            _declaration(
                catalog.canonical(("operator", constructor.index)),
                constructor.index,
                call(
                    0,
                    catalog.canonical(("type", constructor.result_type)),
                    call(
                        0,
                        *(
                            catalog.canonical(("type", type_index))
                            for type_index in constructor.argument_types
                        ),
                    ),
                ),
            )
        )
    return tuple(result)


def _rewrite_world(
    world: RewriteWorld,
    catalog: _SymbolCatalog,
) -> SurfaceNode:
    theory = REWRITE_THEORIES[world.theory_index]
    laws = tuple(
        call(
            6,
            _rewrite_pattern(REWRITE_RULE_LIBRARY[index].lhs, catalog),
            _rewrite_pattern(REWRITE_RULE_LIBRARY[index].rhs, catalog),
        )
        for index in theory.rule_indices
    )
    policy_index = 0 if world.policy == RewritePolicy.CONTEXTUAL else 1
    return call(
        14,
        integer(2),
        call(1, *_rewrite_declarations(catalog)),
        call(
            0,
            call(1, *laws),
            call(7, _rewrite_ground(world.initial, catalog)),
            call(8, catalog.canonical(("policy", policy_index))),
        ),
    )


def _rewrite_command(
    command: RewriteCommand,
    catalog: _SymbolCatalog,
) -> SurfaceNode:
    return call(
        14,
        integer(2),
        call(1, *_rewrite_declarations(catalog)),
        call(
            13,
            *(
                _application(catalog.canonical(("operator", operation)))
                for operation in command.operations
            ),
        ),
    )


def _resource_quantity(
    quantity: ResourceQuantity,
    catalog: _SymbolCatalog,
) -> SurfaceCall:
    return call(
        0,
        catalog.canonical(("object", quantity.place)),
        integer(quantity.multiplicity),
    )


def _resource_law(
    operator_symbol: int,
    operator: GuardedOperator,
    catalog: _SymbolCatalog,
) -> SurfaceCall:
    return call(
        6,
        _application(catalog.canonical(("operator", operator_symbol))),
        call(
            0,
            call(
                1,
                *(
                    _resource_quantity(quantity, catalog)
                    for quantity in operator.guards
                ),
            ),
            call(
                1,
                *(
                    _resource_quantity(quantity, catalog)
                    for quantity in operator.consumes
                ),
            ),
            call(
                1,
                *(
                    _resource_quantity(quantity, catalog)
                    for quantity in operator.produces
                ),
            ),
        ),
    )


def _resource_declarations(
    catalog: _SymbolCatalog,
) -> tuple[SurfaceCall, ...]:
    result: list[SurfaceCall] = []
    for kind_index in range(2):
        result.append(
            _declaration(
                catalog.canonical(("type", kind_index)),
                kind_index,
                call(0),
            )
        )
    for place in PLACE_SPECS:
        result.append(
            _declaration(
                catalog.canonical(("object", place.index)),
                place.index,
                call(
                    0,
                    catalog.canonical(("type", place.resource_kind)),
                    integer(place.capacity),
                ),
            )
        )
    for operator_index in range(OPERATOR_SYMBOL_COUNT):
        result.append(
            _declaration(
                catalog.canonical(("operator", operator_index)),
                operator_index,
                call(0),
            )
        )
    return tuple(result)


def _resource_state(
    marking: Marking,
    catalog: _SymbolCatalog,
) -> SurfaceCall:
    return call(
        7,
        call(
            2,
            *(
                call(
                    0,
                    catalog.canonical(("object", place_index)),
                    integer(value),
                )
                for place_index, value in enumerate(marking.multiplicities)
            ),
        ),
    )


def _resource_world(
    world: ResourceWorld,
    catalog: _SymbolCatalog,
) -> SurfaceNode:
    theory = RESOURCE_THEORIES[world.theory_index]
    laws = tuple(
        _resource_law(
            operator_symbol,
            OPERATOR_LIBRARY[operator_index],
            catalog,
        )
        for operator_symbol, operator_index in enumerate(
            theory.operator_indices
        )
    )
    policy_index = (
        0 if world.policy == ResourcePolicy.ATOMIC_DEADLOCK else 1
    )
    return call(
        14,
        integer(2),
        call(1, *_resource_declarations(catalog)),
        call(
            0,
            call(1, *laws),
            _resource_state(world.initial, catalog),
            call(8, catalog.canonical(("policy", policy_index))),
        ),
    )


def _resource_command(
    command: ResourceCommand,
    catalog: _SymbolCatalog,
) -> SurfaceNode:
    return call(
        14,
        integer(2),
        call(1, *_resource_declarations(catalog)),
        call(
            13,
            *(
                _application(catalog.canonical(("operator", operation)))
                for operation in command.operations
            ),
        ),
    )


def _base_factor_ast(
    factor: World | Command,
    catalog: _SymbolCatalog,
) -> SurfaceNode:
    if type(factor) is HornWorld:
        result = _horn_world(factor, catalog)
    elif type(factor) is HornCommand:
        result = _horn_command(factor, catalog)
    elif type(factor) is RewriteWorld:
        result = _rewrite_world(factor, catalog)
    elif type(factor) is RewriteCommand:
        result = _rewrite_command(factor, catalog)
    elif type(factor) is ResourceWorld:
        result = _resource_world(factor, catalog)
    elif type(factor) is ResourceCommand:
        result = _resource_command(factor, catalog)
    else:
        raise SurfaceAdapterError("factor type is outside the three ontologies")
    return semantic_canonicalize(result)


_ALPHA_VARIABLE_CEILING = 255


def _alpha_reorder_ast(node: SurfaceNode) -> SurfaceNode:
    if isinstance(node, (SurfaceInteger, SurfaceSymbol)):
        return node
    children = tuple(_alpha_reorder_ast(child) for child in node.children)
    if node.head in {1, 2}:
        children = tuple(reversed(children))
    if (
        node.head == 5
        and len(children) == 2
        and isinstance(children[0], SurfaceInteger)
    ):
        children = (
            integer(_ALPHA_VARIABLE_CEILING - children[0].value),
            children[1],
        )
    return SurfaceCall(node.head, children)


def _symbol_occurrences(node: SurfaceNode) -> dict[str, int]:
    counts: dict[str, int] = {}

    def visit(current: SurfaceNode) -> None:
        if isinstance(current, SurfaceSymbol):
            counts[current.value] = counts.get(current.value, 0) + 1
        elif isinstance(current, SurfaceCall):
            for child in current.children:
                visit(child)

    visit(node)
    return counts


def _declaration_candidates(
    node: SurfaceNode,
) -> tuple[tuple[bytes, SurfaceSymbol], ...]:
    if (
        not isinstance(node, SurfaceCall)
        or node.head != 14
        or len(node.children) != 3
        or not isinstance(node.children[1], SurfaceCall)
        or node.children[1].head != 1
    ):
        raise SurfaceAdapterError("factor AST lacks a declaration collection")
    occurrences = _symbol_occurrences(node)
    candidates: list[tuple[bytes, SurfaceSymbol]] = []
    for declaration in node.children[1].children:
        if (
            isinstance(declaration, SurfaceCall)
            and declaration.head == 3
            and len(declaration.children) == 3
            and isinstance(declaration.children[0], SurfaceSymbol)
            and occurrences.get(declaration.children[0].value, 0) >= 2
        ):
            candidates.append(
                (
                    hashlib.sha256(canonical_json_bytes(declaration)).digest(),
                    declaration.children[0],
                )
            )
    return tuple(candidates)


def _alias_target(node: SurfaceNode) -> SurfaceSymbol:
    candidates = _declaration_candidates(node)
    if not candidates:
        raise SurfaceAdapterError("factor has no alias-splittable symbol")
    return min(candidates, key=lambda item: (item[0], item[1].value))[1]


def _type_target(
    node: SurfaceNode,
    catalog: _SymbolCatalog,
) -> SurfaceSymbol:
    type_symbols = {
        catalog.canonical(key).value
        for key in catalog.keys
        if key[0] == "type"
    }
    candidates = tuple(
        item
        for item in _declaration_candidates(node)
        if item[1].value in type_symbols
    )
    if not candidates:
        raise SurfaceAdapterError("factor has no type-twin symbol")
    return min(candidates, key=lambda item: (item[0], item[1].value))[1]


def _replace_alternating_occurrences(
    node: SurfaceNode,
    *,
    target: SurfaceSymbol,
    alias: SurfaceSymbol,
) -> SurfaceNode:
    occurrence = 0

    def visit(current: SurfaceNode, *, declaration_name: bool = False) -> SurfaceNode:
        nonlocal occurrence
        if isinstance(current, SurfaceInteger):
            return current
        if isinstance(current, SurfaceSymbol):
            if current != target or declaration_name:
                return current
            replace = occurrence % 2 == 0
            occurrence += 1
            return alias if replace else current
        return SurfaceCall(
            current.head,
            tuple(
                visit(
                    child,
                    declaration_name=(
                        current.head == 3
                        and child_index == 0
                    ),
                )
                for child_index, child in enumerate(current.children)
            ),
        )

    return visit(node)


def _alias_split_ast(
    factor: World | Command,
    catalog: _SymbolCatalog,
) -> SurfaceNode:
    base_catalog = _catalog_for_factor(factor, presentation="base")
    base = _base_factor_ast(factor, base_catalog)
    target = _alias_target(base)
    alias = catalog.canonical(("alias", 0))
    replaced = _replace_alternating_occurrences(
        base,
        target=target,
        alias=alias,
    )
    assert isinstance(replaced, SurfaceCall)
    declarations = replaced.children[1]
    assert isinstance(declarations, SurfaceCall)
    return SurfaceCall(
        replaced.head,
        (
            replaced.children[0],
            SurfaceCall(
                1,
                (*declarations.children, call(11, target, alias)),
            ),
            replaced.children[2],
        ),
    )


def _type_twin_ast(
    factor: World | Command,
    catalog: _SymbolCatalog,
) -> SurfaceNode:
    base_catalog = _catalog_for_factor(factor, presentation="base")
    base = _base_factor_ast(factor, base_catalog)
    target = _type_target(base, base_catalog)
    twin = catalog.canonical(("type_twin", 0))
    replaced = _replace_alternating_occurrences(
        base,
        target=target,
        alias=twin,
    )
    assert isinstance(replaced, SurfaceCall)
    declarations = replaced.children[1]
    assert isinstance(declarations, SurfaceCall)
    return SurfaceCall(
        replaced.head,
        (
            replaced.children[0],
            SurfaceCall(
                1,
                (*declarations.children, call(11, target, twin)),
            ),
            replaced.children[2],
        ),
    )


def _application_count(node: SurfaceNode) -> int:
    if isinstance(node, (SurfaceInteger, SurfaceSymbol)):
        return 0
    return int(node.head == 4) + sum(
        _application_count(child) for child in node.children
    )


def _relation_reify(
    node: SurfaceNode,
    relations: tuple[SurfaceSymbol, ...],
) -> SurfaceNode:
    relation_index = 0

    def visit(current: SurfaceNode) -> SurfaceNode:
        nonlocal relation_index
        if isinstance(current, (SurfaceInteger, SurfaceSymbol)):
            return current
        children = tuple(visit(child) for child in current.children)
        if current.head != 4:
            return SurfaceCall(current.head, children)
        relation = relations[relation_index]
        relation_index += 1
        return call(
            2,
            call(12, relation, integer(0), children[0]),
            *(
                call(12, relation, integer(role), endpoint)
                for role, endpoint in enumerate(children[1:], start=1)
            ),
        )

    result = visit(node)
    if relation_index != len(relations):
        raise SurfaceAdapterError("relation-node inventory differs")
    return result


def _relation_reification_ast(
    factor: World | Command,
    catalog: _SymbolCatalog,
) -> SurfaceNode:
    base = _base_factor_ast(
        factor,
        _catalog_for_factor(factor, presentation="base"),
    )
    relations = tuple(
        catalog.canonical(key)
        for key in catalog.keys
        if key[0] == "relation"
    )
    return _relation_reify(base, relations)


def _execution_semantics_twin_ast(
    factor: World | Command,
    catalog: _SymbolCatalog,
) -> SurfaceNode:
    if isinstance(factor, HornWorld):
        factor = replace(factor, policy=HornPolicy.DERIVED_ONLY)
    elif isinstance(factor, RewriteWorld):
        factor = replace(factor, policy=RewritePolicy.ROOT_ONLY)
    elif isinstance(factor, ResourceWorld):
        factor = replace(factor, policy=ResourcePolicy.SKIP_BLOCKED)
    return _base_factor_ast(factor, catalog)


def _alpha_factor_ast(
    factor: World | Command,
    catalog: _SymbolCatalog,
) -> SurfaceNode:
    return _alpha_reorder_ast(_base_factor_ast(factor, catalog))


_PRESENTATION_BUILDERS: dict[str, PresentationBuilder] = {
    "base": _base_factor_ast,
    "alpha_reorder": _alpha_factor_ast,
    "alias_split": _alias_split_ast,
    "relation_reification": _relation_reification_ast,
    "type_twin": _type_twin_ast,
    "execution_semantics_twin": _execution_semantics_twin_ast,
}


def canonical_factor_ast(
    factor: World | Command,
    *,
    presentation: str = "base",
) -> SurfaceNode:
    """Return the assessor-side canonical surface semantics for one factor."""

    presented = presented_factor_ast(factor, presentation=presentation)
    return _decode_presentation(presented, presentation)


def presented_factor_ast(
    factor: World | Command,
    *,
    presentation: str = "base",
) -> SurfaceNode:
    """Return the exact pre-opaque AST shown under one presentation."""

    if presentation not in _PRESENTATIONS:
        raise SurfaceAdapterError("presentation is outside the v2 protocol")
    try:
        builder = _PRESENTATION_BUILDERS[presentation]
    except KeyError as exc:
        raise SurfaceAdapterError(
            f"presentation {presentation!r} is not implemented"
        ) from exc
    catalog = _catalog_for_factor(factor, presentation=presentation)
    return builder(factor, catalog)


def _strip_alias_declarations(node: SurfaceNode) -> SurfaceNode:
    if isinstance(node, (SurfaceInteger, SurfaceSymbol)):
        return node
    return SurfaceCall(
        node.head,
        tuple(
            _strip_alias_declarations(child)
            for child in node.children
            if not (
                isinstance(child, SurfaceCall)
                and child.head == 11
            )
        ),
    )


def _decode_reified_applications(node: SurfaceNode) -> SurfaceNode:
    if isinstance(node, (SurfaceInteger, SurfaceSymbol)):
        return node
    children = tuple(_decode_reified_applications(child) for child in node.children)
    if (
        node.head == 2
        and children
        and all(
            isinstance(child, SurfaceCall)
            and child.head == 12
            and len(child.children) == 3
            and isinstance(child.children[0], SurfaceSymbol)
            and isinstance(child.children[1], SurfaceInteger)
            for child in children
        )
    ):
        incidences = tuple(child for child in children if isinstance(child, SurfaceCall))
        relation_names = {
            child.children[0].value
            for child in incidences
            if isinstance(child.children[0], SurfaceSymbol)
        }
        by_role = {
            child.children[1].value: child.children[2]
            for child in incidences
            if isinstance(child.children[1], SurfaceInteger)
        }
        if (
            len(relation_names) != 1
            or len(by_role) != len(incidences)
            or tuple(sorted(by_role)) != tuple(range(len(by_role)))
        ):
            raise SurfaceAdapterError("reified incidence bundle differs")
        return call(
            4,
            *(by_role[role] for role in range(len(by_role))),
        )
    return SurfaceCall(node.head, children)


def _decode_presentation(node: SurfaceNode, presentation: str) -> SurfaceNode:
    if presentation == "base":
        return semantic_canonicalize(node)
    if presentation == "alpha_reorder":
        return semantic_canonicalize(_alpha_reorder_ast(node))
    if presentation in {"alias_split", "type_twin"}:
        quotient = semantic_canonicalize(node)
        return semantic_canonicalize(_strip_alias_declarations(quotient))
    if presentation == "relation_reification":
        return semantic_canonicalize(_decode_reified_applications(node))
    if presentation == "execution_semantics_twin":
        return semantic_canonicalize(node)
    raise SurfaceAdapterError(
        f"presentation {presentation!r} is not implemented"
    )


def _replace_symbols(
    node: SurfaceNode,
    replacements: dict[str, str],
    *,
    require_known: bool,
) -> SurfaceNode:
    if isinstance(node, SurfaceInteger):
        return node
    if isinstance(node, SurfaceSymbol):
        replacement = replacements.get(node.value)
        if replacement is None:
            if require_known:
                raise SurfaceAdapterError(
                    f"surface contains an unbound opaque name {node.value!r}"
                )
            return node
        return SurfaceSymbol(replacement)
    return SurfaceCall(
        node.head,
        tuple(
            _replace_symbols(
                child,
                replacements,
                require_known=require_known,
            )
            for child in node.children
        ),
    )


def _apply_layout(node: SurfaceNode, layout: int) -> SurfaceNode:
    if layout == 0:
        return node
    if (
        not isinstance(node, SurfaceCall)
        or node.head != 14
        or len(node.children) != 3
        or not isinstance(node.children[1], SurfaceCall)
        or node.children[1].head != 1
    ):
        raise SurfaceAdapterError(
            "factor AST lacks the base declaration collection"
        )
    declarations = node.children[1]
    assert isinstance(declarations, SurfaceCall)
    return SurfaceCall(
        node.head,
        (
            node.children[0],
            SurfaceCall(1, tuple(reversed(declarations.children))),
            node.children[2],
        ),
    )


def _layout_bit(semantic_rectangle_id: str) -> int:
    return (
        hashlib.sha256(
            semantic_rectangle_id.encode("ascii") + b"|layout"
        ).digest()[0]
        >> 7
    )


def _render_factor(
    factor: World | Command,
    *,
    stage: SurfaceStage,
    cell_salt: str,
    layout: int,
    context: SurfaceAdapterContext,
) -> SemanticSurfaceDocument:
    catalog = _catalog_for_factor(
        factor,
        presentation=context.presentation,
    )
    presentation_ast = presented_factor_ast(
        factor,
        presentation=context.presentation,
    )
    semantic_ast = _decode_presentation(
        presentation_ast,
        context.presentation,
    )
    opaque = assign_opaque_symbols(
        len(catalog.keys),
        prf=context.prf,
        context=OpaqueNameContext(
            cell_salt=cell_salt,
            fold=context.fold,
            presentation=context.presentation,
            semantic_core_id=context.semantic_core_id,
            split=context.split,
        ),
    )
    canonical_names = tuple(
        catalog.canonical(key).value for key in catalog.keys
    )
    canonical_to_opaque = {
        canonical: opaque_node.value
        for canonical, opaque_node in zip(
            canonical_names,
            opaque,
            strict=True,
        )
    }
    surfaced = _replace_symbols(
        presentation_ast,
        canonical_to_opaque,
        require_known=True,
    )
    surfaced = _apply_layout(surfaced, layout)
    document = SemanticSurfaceDocument(
        stage=stage,
        presentation=context.presentation,
        cell_salt=cell_salt,
        layout=layout,
        renderer=context.renderer,
        ast=surfaced,
        source=render_surface(surfaced, context.renderer),
        semantic_ast=semantic_ast,
        opaque_to_canonical=tuple(
            (opaque_node.value, canonical)
            for canonical, opaque_node in zip(
                canonical_names,
                opaque,
                strict=True,
            )
        ),
    )
    if document.parsed_semantics() != semantic_ast:
        raise SurfaceAdapterError(
            f"{stage.value} surface does not recover its factor semantics"
        )
    return document


def _render_query(
    query: SemanticQuery,
    *,
    paraphrase: int,
    context: SurfaceAdapterContext,
) -> QuerySurfacePrefix:
    catalog = _query_catalog()
    semantic_operator = catalog.canonical(
        ("operator", QUERY_OP_ORDER.index(query.op))
    )
    semantic_ast = semantic_canonicalize(
        ast_from_json_value(
            query_surface_value(
                query,
                operator_symbol=semantic_operator.value,
                paraphrase=paraphrase,
            )
        )
    )
    opaque = assign_opaque_symbols(
        len(catalog.keys),
        prf=context.prf,
        context=OpaqueNameContext(
            cell_salt="shared-query",
            fold=context.fold,
            presentation=context.presentation,
            semantic_core_id=context.semantic_core_id,
            split=context.split,
        ),
    )
    canonical_names = tuple(
        catalog.canonical(key).value for key in catalog.keys
    )
    canonical_to_opaque = {
        canonical: opaque_node.value
        for canonical, opaque_node in zip(
            canonical_names,
            opaque,
            strict=True,
        )
    }
    surfaced = _replace_symbols(
        semantic_ast,
        canonical_to_opaque,
        require_known=True,
    )
    document = SemanticSurfaceDocument(
        stage=SurfaceStage.QUERY,
        presentation=context.presentation,
        cell_salt="shared-query",
        layout=0,
        renderer=context.renderer,
        ast=surfaced,
        source=render_surface(surfaced, context.renderer),
        semantic_ast=semantic_ast,
        opaque_to_canonical=tuple(
            (opaque_node.value, canonical)
            for canonical, opaque_node in zip(
                canonical_names,
                opaque,
                strict=True,
            )
        ),
    )
    if document.parsed_semantics() != semantic_ast:
        raise SurfaceAdapterError(
            "QUERY surface does not recover its factor semantics"
        )
    return QuerySurfacePrefix(
        query=query,
        paraphrase=paraphrase,
        document=document,
        prefix=document.source + b"R=",
    )


def parse_and_semantic_canonicalize(
    document: SemanticSurfaceDocument,
) -> SurfaceNode:
    """Fully parse a document and reverse its assessor-owned opaque binding."""

    if type(document) is not SemanticSurfaceDocument:
        raise TypeError("document must be a SemanticSurfaceDocument")
    parsed = parse_surface(document.source, document.renderer)
    normalized = _decode_presentation(
        _replace_symbols(
            parsed,
            dict(document.opaque_to_canonical),
            require_known=True,
        ),
        document.presentation,
    )
    if normalized != document.semantic_ast:
        raise SurfaceAdapterError(
            f"{document.stage.value} parsed semantics differ"
        )
    return normalized


def _matched_ontology(
    worlds: tuple[World, World],
    commands: tuple[Command, Command],
    queries: tuple[SemanticQuery, SemanticQuery],
) -> None:
    world_types = {type(world) for world in worlds}
    command_types = {type(command) for command in commands}
    if world_types == {HornWorld}:
        expected_command = HornCommand
        expected_query_prefix = "horn_"
    elif world_types == {RewriteWorld}:
        expected_command = RewriteCommand
        expected_query_prefix = "rewrite_"
    elif world_types == {ResourceWorld}:
        expected_command = ResourceCommand
        expected_query_prefix = "resource_"
    else:
        raise SurfaceAdapterError("world pair mixes or leaves ontologies")
    if command_types != {expected_command}:
        raise SurfaceAdapterError("command pair ontology differs")
    if any(not query.op.value.startswith(expected_query_prefix) for query in queries):
        raise SurfaceAdapterError("query ontology differs")


def build_base_surface_bundle(
    worlds: tuple[World, World],
    commands: tuple[Command, Command],
    queries: tuple[SemanticQuery, SemanticQuery],
    *,
    context: SurfaceAdapterContext,
) -> BaseSurfaceBundle:
    """Construct and audit all base-presentation rectangle sources."""

    if type(worlds) is not tuple or len(worlds) != 2:
        raise SurfaceAdapterError("worlds must be an exact pair")
    if type(commands) is not tuple or len(commands) != 2:
        raise SurfaceAdapterError("commands must be an exact pair")
    if type(queries) is not tuple or len(queries) != 2:
        raise SurfaceAdapterError("queries must be an exact pair")
    if type(context) is not SurfaceAdapterContext:
        raise TypeError("context must be a SurfaceAdapterContext")
    _matched_ontology(worlds, commands, queries)

    base_layout = _layout_bit(context.semantic_rectangle_id)
    world_variants = tuple(
        tuple(
            _render_factor(
                world,
                stage=SurfaceStage.WORLD,
                cell_salt=f"world-{command_index}",
                layout=base_layout ^ command_index,
                context=context,
            )
            for command_index in range(2)
        )
        for world in worlds
    )
    command_variants = tuple(
        tuple(
            _render_factor(
                command,
                stage=SurfaceStage.COMMAND,
                cell_salt=f"command-{world_index}",
                layout=base_layout ^ world_index,
                context=context,
            )
            for world_index in range(2)
        )
        for command in commands
    )
    query_prefixes = tuple(
        tuple(
            _render_query(
                query,
                paraphrase=paraphrase,
                context=context,
            )
            for paraphrase in range(2)
        )
        for query in queries
    )
    bundle = BaseSurfaceBundle(
        world_variants=world_variants,  # type: ignore[arg-type]
        command_variants=command_variants,  # type: ignore[arg-type]
        query_prefixes=query_prefixes,  # type: ignore[arg-type]
    )

    for world_index in range(2):
        left, right = bundle.world_variants[world_index]
        if left.source == right.source:
            raise SurfaceAdapterError(
                f"WORLD {world_index} variants are not byte-distinct"
            )
        if left.parsed_semantics() != right.parsed_semantics():
            raise SurfaceAdapterError(
                f"WORLD {world_index} variants change factor semantics"
            )
    for command_index in range(2):
        left, right = bundle.command_variants[command_index]
        if left.source == right.source:
            raise SurfaceAdapterError(
                f"COMMAND {command_index} variants are not byte-distinct"
            )
        if left.parsed_semantics() != right.parsed_semantics():
            raise SurfaceAdapterError(
                f"COMMAND {command_index} variants change factor semantics"
            )
    for world_index in range(2):
        for command_index in range(2):
            if (
                bundle.corner(world_index, command_index).query_prefixes
                is not bundle.query_prefixes
            ):
                raise AssertionError("query prefix sharing unexpectedly failed")
    return bundle


__all__ = [
    "BaseSurfaceBundle",
    "CornerSurface",
    "IMPLEMENTED_PRESENTATIONS",
    "QuerySurfacePrefix",
    "SemanticSurfaceDocument",
    "SurfaceAdapterContext",
    "SurfaceAdapterError",
    "SurfaceStage",
    "build_base_surface_bundle",
    "canonical_factor_ast",
    "parse_and_semantic_canonicalize",
    "presented_factor_ast",
]
