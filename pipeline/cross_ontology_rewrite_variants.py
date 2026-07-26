"""Exact structural and semantic variants for the typed-rewrite gate.

This module is an offline qualification generator.  It materializes the seven
rewrite variants frozen in the ETTR preregistration as typed graph objects:

* ``base`` is the board theory and its exact identifying evidence;
* ``alpha_reorder`` alpha-renames variables, reverses rule order, and
  permutes graph node identifiers;
* ``alias_split`` replaces one constructor symbol with two explicitly
  equivalent symbols and distributes occurrences across both;
* ``relation_reification`` replaces every direct child edge with a first-class
  relation node and parent/child incidence edges;
* ``type_twin`` lifts one rule from type 0 to a disjoint type 2 signature;
* ``execution_semantics_twin`` keeps syntax fixed but changes contextual
  closure to root-only reduction; and
* ``ambiguity_deleted_twin`` removes a decisive demonstration so that the
  exact behavioral version space becomes non-singleton.

The implementation imports only the finite offline rewrite board and the
standard library.  It contains no candidate model, tokenizer, parser,
assessor callback, or deployed runtime dependency.  Candidate-neutral
source-deleted theory packets exclude alignments, oracle outputs, evidence,
challenge terms, and variant labels.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from functools import lru_cache
from hashlib import sha256
import json
from typing import Iterable

from cross_ontology_rewrite_board import (
    CONSTRUCTORS,
    HELDOUT_THEORY_INDICES,
    RULE_LIBRARY,
    THEORIES,
    Demonstration,
    GroundTerm,
    PatternTerm,
    RewriteRule,
    challenge_terms,
    exact_version_space,
    execute_normal_forms,
    identifying_evidence,
)


CHALLENGES_PER_VARIANT = 16


class RewriteVariantKind(StrEnum):
    BASE = "base"
    ALPHA_REORDER = "alpha_reorder"
    ALIAS_SPLIT = "alias_split"
    RELATION_REIFICATION = "relation_reification"
    TYPE_TWIN = "type_twin"
    EXECUTION_SEMANTICS_TWIN = "execution_semantics_twin"
    AMBIGUITY_DELETED_TWIN = "ambiguity_deleted_twin"


VARIANT_ORDER = tuple(RewriteVariantKind)


class VariantExpectation(StrEnum):
    BASELINE = "baseline"
    EXACT_INVARIANCE = "exact_invariance"
    EXACT_TYPE_SEPARATION = "exact_type_separation"
    EXACT_EXECUTION_SEPARATION = "exact_execution_separation"
    EXACT_AMBIGUITY_SEPARATION = "exact_ambiguity_separation"


class ExecutionSemantics(StrEnum):
    CONTEXTUAL_CLOSURE = "contextual_closure"
    ROOT_ONLY = "root_only"


class QualificationDecision(StrEnum):
    EXECUTE = "execute"
    ABSTAIN_AMBIGUOUS = "abstain_ambiguous"


class GraphNodeKind(StrEnum):
    CONSTRUCTOR = "constructor"
    VARIABLE = "variable"
    RELATION = "relation"


class GraphEdgeKind(StrEnum):
    CHILD = "child"
    RELATION_PARENT = "relation_parent"
    RELATION_CHILD = "relation_child"


@dataclass(frozen=True, order=True, slots=True)
class VariantConstructor:
    """One local symbol and its explicit alias/type signature."""

    symbol_index: int
    equivalence_class: int
    result_type: int
    argument_types: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.symbol_index < 0
            or self.equivalence_class < 0
            or self.result_type < 0
            or any(type_index < 0 for type_index in self.argument_types)
        ):
            raise ValueError("variant constructor differs")


@dataclass(frozen=True, order=True, slots=True)
class VariantGraphNode:
    node_index: int
    kind: GraphNodeKind
    type_index: int | None = None
    symbol_index: int | None = None
    variable_index: int | None = None
    role: int | None = None

    def __post_init__(self) -> None:
        if self.node_index < 0:
            raise ValueError("graph node index differs")
        constructor = (
            self.type_index is not None
            and self.symbol_index is not None
            and self.variable_index is None
            and self.role is None
        )
        variable = (
            self.type_index is not None
            and self.symbol_index is None
            and self.variable_index is not None
            and self.role is None
        )
        relation = (
            self.type_index is None
            and self.symbol_index is None
            and self.variable_index is None
            and self.role is not None
        )
        expected = {
            GraphNodeKind.CONSTRUCTOR: constructor,
            GraphNodeKind.VARIABLE: variable,
            GraphNodeKind.RELATION: relation,
        }[self.kind]
        if not expected:
            raise ValueError("graph node payload differs")


@dataclass(frozen=True, order=True, slots=True)
class VariantGraphEdge:
    kind: GraphEdgeKind
    source: int
    target: int
    role: int | None = None

    def __post_init__(self) -> None:
        if self.source < 0 or self.target < 0:
            raise ValueError("graph edge endpoint differs")
        if (self.kind == GraphEdgeKind.CHILD) != (self.role is not None):
            raise ValueError("graph edge role differs")


@dataclass(frozen=True, slots=True)
class VariantTermGraph:
    root: int
    nodes: tuple[VariantGraphNode, ...]
    edges: tuple[VariantGraphEdge, ...]
    reified_relations: bool

    def __post_init__(self) -> None:
        indices = tuple(node.node_index for node in self.nodes)
        if (
            self.root not in indices
            or len(indices) != len(set(indices))
            or len(self.edges) != len(set(self.edges))
        ):
            raise ValueError("term graph identity differs")


@dataclass(frozen=True, slots=True)
class VariantRule:
    local_index: int
    lhs: VariantTermGraph
    rhs: VariantTermGraph


@dataclass(frozen=True, slots=True)
class VariantDemonstration:
    initial: VariantTermGraph
    normal_forms: tuple[VariantTermGraph, ...]


@dataclass(frozen=True, slots=True)
class CanonicalAlignment:
    """Offline-only alignment; never included in source-deleted packets."""

    symbol_to_base: tuple[tuple[int, int], ...]
    class_to_base: tuple[tuple[int, int], ...]
    rule_to_base: tuple[tuple[int, int], ...]
    type_to_base: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class SeparationWitness:
    challenge_offset: int
    canonical_term: GroundTerm
    base_outcome: tuple[GroundTerm, ...]
    variant_outcomes: tuple[tuple[GroundTerm, ...], ...]
    base_decision: QualificationDecision
    variant_decision: QualificationDecision


@dataclass(frozen=True, slots=True)
class VariantOracle:
    decision: QualificationDecision
    aligned_outputs: tuple[tuple[GroundTerm, ...], ...]
    possible_outputs: tuple[
        tuple[tuple[GroundTerm, ...], ...],
        ...,
    ]
    behavioral_class_count: int
    witness: SeparationWitness | None


@dataclass(frozen=True, slots=True)
class RewriteQualificationVariant:
    kind: RewriteVariantKind
    theory_index: int
    constructors: tuple[VariantConstructor, ...]
    rules: tuple[VariantRule, ...]
    evidence: tuple[VariantDemonstration, ...]
    challenges: tuple[VariantTermGraph, ...]
    challenge_indices: tuple[int, ...]
    execution_semantics: ExecutionSemantics
    expectation: VariantExpectation
    alignment: CanonicalAlignment
    oracle: VariantOracle

    def compiler_source_bytes(self) -> bytes:
        """Return demonstrations only; never expose latent rules or oracles."""

        return _canonical_json_bytes(
            {
                "constructors": [
                    _constructor_payload(constructor)
                    for constructor in self.constructors
                ],
                "demonstrations": [
                    _demonstration_payload(demo)
                    for demo in self.evidence
                ],
                "schema": "ettr-anonymous-evidence-v1",
            }
        )

    def late_challenge_bytes(self, offset: int) -> bytes:
        """Return one challenge graph without alignment or expected output."""

        if not 0 <= offset < len(self.challenges):
            raise IndexError("challenge offset differs")
        return _canonical_json_bytes(
            {
                "challenge": _graph_payload(self.challenges[offset]),
                "schema": "ettr-anonymous-challenge-v1",
            }
        )

    def source_deleted_theory_bytes(self) -> bytes:
        """Return an offline reference theory packet for mechanics audits."""

        return _canonical_json_bytes(
            {
                "constructors": [
                    _constructor_payload(constructor)
                    for constructor in self.constructors
                ],
                "execution_semantics": self.execution_semantics.value,
                "rules": [
                    _rule_payload(rule)
                    for rule in self.rules
                ],
                "schema": "cross_ontology_rewrite_theory_v1",
            }
        )

    def material_sha256(self) -> str:
        """Hash actual transformed material, excluding its variant label."""

        payload = {
            "challenges": [
                _graph_payload(graph)
                for graph in self.challenges
            ],
            "constructors": [
                _constructor_payload(constructor)
                for constructor in self.constructors
            ],
            "evidence": [
                _demonstration_payload(demo)
                for demo in self.evidence
            ],
            "execution_semantics": self.execution_semantics.value,
            "rules": [_rule_payload(rule) for rule in self.rules],
            "schema": "cross_ontology_rewrite_variant_material_v1",
        }
        return sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class RewriteVariantAuditReceipt:
    theory_index: int
    variant_count: int
    challenges_per_variant: int
    invariant_variant_count: int
    exact_invariance_cases: int
    type_separation_witnesses: int
    execution_separation_witnesses: int
    ambiguity_separation_witnesses: int
    unique_material_hashes: int
    all_contracts_pass: bool


@dataclass(frozen=True, order=True, slots=True)
class _RuntimeGroundTerm:
    type_index: int
    equivalence_class: int
    children: tuple[_RuntimeGroundTerm, ...] = ()

    @property
    def node_count(self) -> int:
        return 1 + sum(child.node_count for child in self.children)


@dataclass(frozen=True, order=True, slots=True)
class _RuntimePattern:
    type_index: int
    equivalence_class: int | None = None
    variable_index: int | None = None
    children: tuple[_RuntimePattern, ...] = ()


@dataclass(frozen=True, slots=True)
class _RuntimeRule:
    local_index: int
    lhs: _RuntimePattern
    rhs: _RuntimePattern


@dataclass(frozen=True, slots=True)
class _VariantParts:
    kind: RewriteVariantKind
    constructors: tuple[VariantConstructor, ...]
    rules: tuple[VariantRule, ...]
    semantics: ExecutionSemantics
    alignment: CanonicalAlignment


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _constructor_payload(constructor: VariantConstructor) -> list[object]:
    return [
        constructor.symbol_index,
        constructor.equivalence_class,
        constructor.result_type,
        list(constructor.argument_types),
    ]


def _node_payload(node: VariantGraphNode) -> list[object]:
    return [
        node.node_index,
        node.kind.value,
        node.type_index,
        node.symbol_index,
        node.variable_index,
        node.role,
    ]


def _edge_payload(edge: VariantGraphEdge) -> list[object]:
    return [
        edge.kind.value,
        edge.source,
        edge.target,
        edge.role,
    ]


def _graph_payload(graph: VariantTermGraph) -> dict[str, object]:
    return {
        "edges": [_edge_payload(edge) for edge in graph.edges],
        "nodes": [_node_payload(node) for node in graph.nodes],
        "reified_relations": graph.reified_relations,
        "root": graph.root,
    }


def _rule_payload(rule: VariantRule) -> dict[str, object]:
    return {
        "lhs": _graph_payload(rule.lhs),
        "local_index": rule.local_index,
        "rhs": _graph_payload(rule.rhs),
    }


def _demonstration_payload(
    demonstration: VariantDemonstration,
) -> dict[str, object]:
    return {
        "initial": _graph_payload(demonstration.initial),
        "normal_forms": [
            _graph_payload(graph)
            for graph in demonstration.normal_forms
        ],
    }


def _base_constructors() -> tuple[VariantConstructor, ...]:
    return tuple(
        VariantConstructor(
            symbol_index=constructor.index,
            equivalence_class=constructor.index,
            result_type=constructor.result_type,
            argument_types=constructor.argument_types,
        )
        for constructor in CONSTRUCTORS
    )


def _pattern_constructor_indices(pattern: PatternTerm) -> set[int]:
    if pattern.variable_index is not None:
        return set()
    assert pattern.constructor_index is not None
    result = {pattern.constructor_index}
    for child in pattern.children:
        result.update(_pattern_constructor_indices(child))
    return result


def _type_twin_constructors(
    selected_rule: RewriteRule,
) -> tuple[VariantConstructor, ...]:
    constructors = list(_base_constructors())
    used = sorted(
        _pattern_constructor_indices(selected_rule.lhs)
        | _pattern_constructor_indices(selected_rule.rhs)
    )
    for constructor_index in used:
        constructor = CONSTRUCTORS[constructor_index]
        result_type = 2 if constructor.result_type == 0 else 1
        argument_types = tuple(
            2 if type_index == 0 else 1
            for type_index in constructor.argument_types
        )
        if (
            result_type,
            argument_types,
        ) == (
            constructor.result_type,
            constructor.argument_types,
        ):
            continue
        constructors.append(
            VariantConstructor(
                symbol_index=len(constructors),
                equivalence_class=constructor.index,
                result_type=result_type,
                argument_types=argument_types,
            )
        )
    return tuple(constructors)


def _symbol_lookup(
    constructors: tuple[VariantConstructor, ...],
) -> dict[tuple[int, int, tuple[int, ...]], int]:
    result: dict[tuple[int, int, tuple[int, ...]], int] = {}
    for constructor in constructors:
        key = (
            constructor.equivalence_class,
            constructor.result_type,
            constructor.argument_types,
        )
        result.setdefault(key, constructor.symbol_index)
    return result


def _remap_graph_node_ids(
    graph: VariantTermGraph,
    *,
    offset: int,
) -> VariantTermGraph:
    ordered = sorted(node.node_index for node in graph.nodes)
    mapping = {
        old: offset + len(ordered) - 1 - position
        for position, old in enumerate(ordered)
    }
    nodes = tuple(
        replace(node, node_index=mapping[node.node_index])
        for node in reversed(graph.nodes)
    )
    edges = tuple(
        replace(
            edge,
            source=mapping[edge.source],
            target=mapping[edge.target],
        )
        for edge in reversed(graph.edges)
    )
    return VariantTermGraph(
        root=mapping[graph.root],
        nodes=nodes,
        edges=edges,
        reified_relations=graph.reified_relations,
    )


def _encode_tree(
    value: GroundTerm | PatternTerm,
    *,
    constructors: tuple[VariantConstructor, ...],
    reified: bool,
    variable_map: dict[int, int],
    lift_type_zero: bool,
    alias_target: int | None,
    alias_symbol: int | None,
    salt: int,
    remap_ids: bool,
) -> VariantTermGraph:
    nodes: list[VariantGraphNode] = []
    edges: list[VariantGraphEdge] = []
    lookup = _symbol_lookup(constructors)
    occurrence = [0]

    def mapped_type(type_index: int) -> int:
        if lift_type_zero and type_index == 0:
            return 2
        return type_index

    def append(node: GroundTerm | PatternTerm) -> int:
        node_index = len(nodes)
        if isinstance(node, PatternTerm) and node.variable_index is not None:
            nodes.append(
                VariantGraphNode(
                    node_index=node_index,
                    kind=GraphNodeKind.VARIABLE,
                    type_index=mapped_type(node.type_index),
                    variable_index=variable_map.get(
                        node.variable_index,
                        node.variable_index,
                    ),
                )
            )
            return node_index

        constructor_index = node.constructor_index
        assert constructor_index is not None
        constructor = CONSTRUCTORS[constructor_index]
        result_type = mapped_type(constructor.result_type)
        argument_types = tuple(
            mapped_type(type_index)
            for type_index in constructor.argument_types
        )
        symbol_index = lookup[
            (constructor.index, result_type, argument_types)
        ]
        if (
            alias_target == constructor.index
            and alias_symbol is not None
            and (salt + occurrence[0]) % 2 == 0
        ):
            symbol_index = alias_symbol
        occurrence[0] += 1
        nodes.append(
            VariantGraphNode(
                node_index=node_index,
                kind=GraphNodeKind.CONSTRUCTOR,
                type_index=result_type,
                symbol_index=symbol_index,
            )
        )
        for role, child in enumerate(node.children):
            child_index = append(child)
            if reified:
                relation_index = len(nodes)
                nodes.append(
                    VariantGraphNode(
                        node_index=relation_index,
                        kind=GraphNodeKind.RELATION,
                        role=role,
                    )
                )
                edges.extend(
                    (
                        VariantGraphEdge(
                            GraphEdgeKind.RELATION_PARENT,
                            relation_index,
                            node_index,
                        ),
                        VariantGraphEdge(
                            GraphEdgeKind.RELATION_CHILD,
                            relation_index,
                            child_index,
                        ),
                    )
                )
            else:
                edges.append(
                    VariantGraphEdge(
                        GraphEdgeKind.CHILD,
                        node_index,
                        child_index,
                        role,
                    )
                )
        return node_index

    root = append(value)
    graph = VariantTermGraph(
        root=root,
        nodes=tuple(nodes),
        edges=tuple(edges),
        reified_relations=reified,
    )
    if remap_ids:
        return _remap_graph_node_ids(graph, offset=1000 + salt * 97)
    return graph


def _graph_children(
    graph: VariantTermGraph,
) -> dict[int, tuple[tuple[int, int], ...]]:
    nodes = {node.node_index: node for node in graph.nodes}
    if any(
        edge.source not in nodes or edge.target not in nodes
        for edge in graph.edges
    ):
        raise ValueError("graph edge references absent node")
    child_map: dict[int, list[tuple[int, int]]] = {}
    relation_nodes = {
        index
        for index, node in nodes.items()
        if node.kind == GraphNodeKind.RELATION
    }
    if graph.reified_relations:
        if any(edge.kind == GraphEdgeKind.CHILD for edge in graph.edges):
            raise ValueError("reified graph retains direct child edge")
        for relation_index in sorted(relation_nodes):
            relation = nodes[relation_index]
            parents = [
                edge.target
                for edge in graph.edges
                if edge.source == relation_index
                and edge.kind == GraphEdgeKind.RELATION_PARENT
            ]
            children = [
                edge.target
                for edge in graph.edges
                if edge.source == relation_index
                and edge.kind == GraphEdgeKind.RELATION_CHILD
            ]
            if len(parents) != 1 or len(children) != 1:
                raise ValueError("reified relation incidence differs")
            assert relation.role is not None
            child_map.setdefault(parents[0], []).append(
                (relation.role, children[0])
            )
    else:
        if relation_nodes or any(
            edge.kind != GraphEdgeKind.CHILD
            for edge in graph.edges
        ):
            raise ValueError("direct graph contains reified relation")
        for edge in graph.edges:
            assert edge.role is not None
            child_map.setdefault(edge.source, []).append(
                (edge.role, edge.target)
            )
    result: dict[int, tuple[tuple[int, int], ...]] = {}
    for parent, children in child_map.items():
        ordered = tuple(sorted(children))
        if tuple(role for role, _ in ordered) != tuple(
            range(len(ordered))
        ):
            raise ValueError("child roles are not contiguous")
        result[parent] = ordered
    return result


def _decode_graph(
    graph: VariantTermGraph,
    constructors: tuple[VariantConstructor, ...],
    *,
    allow_variables: bool,
) -> _RuntimeGroundTerm | _RuntimePattern:
    nodes = {node.node_index: node for node in graph.nodes}
    symbols = {
        constructor.symbol_index: constructor
        for constructor in constructors
    }
    if len(symbols) != len(constructors):
        raise ValueError("constructor symbol identity differs")
    child_map = _graph_children(graph)
    visited: set[int] = set()
    active: set[int] = set()

    def decode(
        node_index: int,
    ) -> _RuntimeGroundTerm | _RuntimePattern:
        if node_index in active:
            raise ValueError("term graph is cyclic")
        if node_index in visited:
            raise ValueError("term graph is not a tree")
        active.add(node_index)
        visited.add(node_index)
        node = nodes[node_index]
        if node.kind == GraphNodeKind.RELATION:
            raise ValueError("relation node used as term")
        children = tuple(
            decode(child_index)
            for _, child_index in child_map.get(node_index, ())
        )
        active.remove(node_index)
        if node.kind == GraphNodeKind.VARIABLE:
            if not allow_variables or children:
                raise ValueError("variable node differs")
            assert node.type_index is not None
            assert node.variable_index is not None
            return _RuntimePattern(
                type_index=node.type_index,
                variable_index=node.variable_index,
            )
        assert node.symbol_index is not None
        assert node.type_index is not None
        symbol = symbols[node.symbol_index]
        if (
            node.type_index != symbol.result_type
            or len(children) != len(symbol.argument_types)
            or any(
                child.type_index != expected
                for child, expected in zip(
                    children,
                    symbol.argument_types,
                    strict=True,
                )
            )
        ):
            raise ValueError("typed constructor graph differs")
        if allow_variables:
            return _RuntimePattern(
                type_index=node.type_index,
                equivalence_class=symbol.equivalence_class,
                children=children,
            )
        return _RuntimeGroundTerm(
            type_index=node.type_index,
            equivalence_class=symbol.equivalence_class,
            children=children,
        )

    decoded = decode(graph.root)
    semantic_nodes = {
        index
        for index, node in nodes.items()
        if node.kind != GraphNodeKind.RELATION
    }
    if visited != semantic_nodes:
        raise ValueError("term graph contains unreachable semantic node")
    return decoded


def _compile_rules(
    rules: tuple[VariantRule, ...],
    constructors: tuple[VariantConstructor, ...],
) -> tuple[_RuntimeRule, ...]:
    compiled = []
    for rule in rules:
        lhs = _decode_graph(
            rule.lhs,
            constructors,
            allow_variables=True,
        )
        rhs = _decode_graph(
            rule.rhs,
            constructors,
            allow_variables=True,
        )
        assert isinstance(lhs, _RuntimePattern)
        assert isinstance(rhs, _RuntimePattern)
        compiled.append(_RuntimeRule(rule.local_index, lhs, rhs))
    return tuple(compiled)


def _match(
    pattern: _RuntimePattern,
    term: _RuntimeGroundTerm,
    bindings: dict[int, _RuntimeGroundTerm],
) -> bool:
    if pattern.type_index != term.type_index:
        return False
    if pattern.variable_index is not None:
        previous = bindings.setdefault(pattern.variable_index, term)
        return previous == term
    if (
        pattern.equivalence_class != term.equivalence_class
        or len(pattern.children) != len(term.children)
    ):
        return False
    return all(
        _match(child_pattern, child, bindings)
        for child_pattern, child in zip(
            pattern.children,
            term.children,
            strict=True,
        )
    )


def _instantiate(
    pattern: _RuntimePattern,
    bindings: dict[int, _RuntimeGroundTerm],
) -> _RuntimeGroundTerm:
    if pattern.variable_index is not None:
        return bindings[pattern.variable_index]
    assert pattern.equivalence_class is not None
    return _RuntimeGroundTerm(
        type_index=pattern.type_index,
        equivalence_class=pattern.equivalence_class,
        children=tuple(
            _instantiate(child, bindings)
            for child in pattern.children
        ),
    )


def _root_reducts(
    rules: tuple[_RuntimeRule, ...],
    term: _RuntimeGroundTerm,
) -> tuple[_RuntimeGroundTerm, ...]:
    reducts: set[_RuntimeGroundTerm] = set()
    for rule in rules:
        bindings: dict[int, _RuntimeGroundTerm] = {}
        if _match(rule.lhs, term, bindings):
            reduct = _instantiate(rule.rhs, bindings)
            if reduct.node_count >= term.node_count:
                raise ValueError("variant rewrite did not decrease")
            reducts.add(reduct)
    return tuple(sorted(reducts))


def _one_step_reducts(
    rules: tuple[_RuntimeRule, ...],
    term: _RuntimeGroundTerm,
    semantics: ExecutionSemantics,
) -> tuple[_RuntimeGroundTerm, ...]:
    reducts = set(_root_reducts(rules, term))
    if semantics == ExecutionSemantics.CONTEXTUAL_CLOSURE:
        for child_index, child in enumerate(term.children):
            for child_reduct in _one_step_reducts(
                rules,
                child,
                semantics,
            ):
                children = list(term.children)
                children[child_index] = child_reduct
                reducts.add(
                    _RuntimeGroundTerm(
                        term.type_index,
                        term.equivalence_class,
                        tuple(children),
                    )
                )
    return tuple(sorted(reducts))


def _execute_runtime(
    rules: tuple[_RuntimeRule, ...],
    initial: _RuntimeGroundTerm,
    semantics: ExecutionSemantics,
) -> tuple[_RuntimeGroundTerm, ...]:
    frontier = [initial]
    visited = {initial}
    terminals: set[_RuntimeGroundTerm] = set()
    while frontier:
        term = frontier.pop(0)
        reducts = _one_step_reducts(rules, term, semantics)
        if not reducts:
            terminals.add(term)
        for reduct in reducts:
            if reduct not in visited:
                visited.add(reduct)
                frontier.append(reduct)
    return tuple(sorted(terminals))


def _align_ground(
    term: _RuntimeGroundTerm,
    alignment: CanonicalAlignment,
) -> GroundTerm:
    class_to_base = dict(alignment.class_to_base)
    constructor_index = class_to_base[term.equivalence_class]
    constructor = CONSTRUCTORS[constructor_index]
    if len(term.children) != len(constructor.argument_types):
        raise ValueError("canonical constructor arity differs")
    children = tuple(
        _align_ground(child, alignment)
        for child in term.children
    )
    if any(
        child.type_index != expected
        for child, expected in zip(
            children,
            constructor.argument_types,
            strict=True,
        )
    ):
        raise ValueError("canonical child alignment differs")
    return GroundTerm(
        constructor.result_type,
        constructor.index,
        children,
    )


def _execute_graph(
    parts: _VariantParts,
    graph: VariantTermGraph,
) -> tuple[GroundTerm, ...]:
    initial = _decode_graph(
        graph,
        parts.constructors,
        allow_variables=False,
    )
    assert isinstance(initial, _RuntimeGroundTerm)
    rules = _compile_rules(parts.rules, parts.constructors)
    return tuple(
        sorted(
            _align_ground(term, parts.alignment)
            for term in _execute_runtime(
                rules,
                initial,
                parts.semantics,
            )
        )
    )


def _kind_configuration(
    kind: RewriteVariantKind,
) -> tuple[bool, bool, bool]:
    return (
        kind == RewriteVariantKind.RELATION_REIFICATION,
        kind == RewriteVariantKind.ALPHA_REORDER,
        kind == RewriteVariantKind.ALIAS_SPLIT,
    )


def _build_parts(
    theory_index: int,
    kind: RewriteVariantKind,
) -> _VariantParts:
    if theory_index not in HELDOUT_THEORY_INDICES:
        raise ValueError("qualification target is not held out")
    theory = THEORIES[theory_index]
    reified, remap_ids, alias_split = _kind_configuration(kind)
    selected_type_rule = theory.rule_indices[0]
    if kind == RewriteVariantKind.TYPE_TWIN:
        constructors = _type_twin_constructors(
            RULE_LIBRARY[selected_type_rule]
        )
    else:
        constructors = _base_constructors()
    alias_symbol: int | None = None
    if alias_split:
        base = CONSTRUCTORS[4]
        alias_symbol = len(constructors)
        constructors = (
            *constructors,
            VariantConstructor(
                symbol_index=alias_symbol,
                equivalence_class=base.index,
                result_type=base.result_type,
                argument_types=base.argument_types,
            ),
        )
    ordered_rules = list(theory.rule_indices)
    if kind == RewriteVariantKind.ALPHA_REORDER:
        ordered_rules.reverse()
    rules: list[VariantRule] = []
    rule_alignment: list[tuple[int, int]] = []
    for local_index, rule_index in enumerate(ordered_rules):
        rule = RULE_LIBRARY[rule_index]
        lift = (
            kind == RewriteVariantKind.TYPE_TWIN
            and rule_index == selected_type_rule
        )
        variable_indices = sorted(
            {
                variable
                for pattern in (rule.lhs, rule.rhs)
                for variable in _pattern_variable_indices(pattern)
            }
        )
        variable_map = {
            variable: (
                100 + theory_index * 10 + position
                if kind == RewriteVariantKind.ALPHA_REORDER
                else variable
            )
            for position, variable in enumerate(variable_indices)
        }
        lhs = _encode_tree(
            rule.lhs,
            constructors=constructors,
            reified=reified,
            variable_map=variable_map,
            lift_type_zero=lift,
            alias_target=4 if alias_split else None,
            alias_symbol=alias_symbol,
            salt=theory_index * 31 + local_index * 2,
            remap_ids=remap_ids,
        )
        rhs = _encode_tree(
            rule.rhs,
            constructors=constructors,
            reified=reified,
            variable_map=variable_map,
            lift_type_zero=lift,
            alias_target=4 if alias_split else None,
            alias_symbol=alias_symbol,
            salt=theory_index * 31 + local_index * 2 + 1,
            remap_ids=remap_ids,
        )
        rules.append(VariantRule(local_index, lhs, rhs))
        rule_alignment.append((local_index, rule_index))
    semantics = (
        ExecutionSemantics.ROOT_ONLY
        if kind == RewriteVariantKind.EXECUTION_SEMANTICS_TWIN
        else ExecutionSemantics.CONTEXTUAL_CLOSURE
    )
    symbol_to_base = tuple(
        (
            constructor.symbol_index,
            constructor.equivalence_class,
        )
        for constructor in constructors
    )
    class_to_base = tuple(
        sorted(
            {
                (
                    constructor.equivalence_class,
                    constructor.equivalence_class,
                )
                for constructor in constructors
            }
        )
    )
    type_to_base = ((0, 0), (1, 1))
    if kind == RewriteVariantKind.TYPE_TWIN:
        type_to_base = (*type_to_base, (2, 0))
    return _VariantParts(
        kind=kind,
        constructors=tuple(constructors),
        rules=tuple(rules),
        semantics=semantics,
        alignment=CanonicalAlignment(
            symbol_to_base=symbol_to_base,
            class_to_base=class_to_base,
            rule_to_base=tuple(rule_alignment),
            type_to_base=type_to_base,
        ),
    )


def _pattern_variable_indices(pattern: PatternTerm) -> set[int]:
    if pattern.variable_index is not None:
        return {pattern.variable_index}
    result: set[int] = set()
    for child in pattern.children:
        result.update(_pattern_variable_indices(child))
    return result


def _encode_ground_for_parts(
    term: GroundTerm,
    parts: _VariantParts,
    *,
    salt: int,
) -> VariantTermGraph:
    reified, remap_ids, alias_split = _kind_configuration(parts.kind)
    alias_symbol = (
        max(
            constructor.symbol_index
            for constructor in parts.constructors
        )
        if alias_split
        else None
    )
    return _encode_tree(
        term,
        constructors=parts.constructors,
        reified=reified,
        variable_map={},
        lift_type_zero=False,
        alias_target=4 if alias_split else None,
        alias_symbol=alias_symbol,
        salt=salt,
        remap_ids=remap_ids,
    )


def _exhaustive_outputs(
    parts: _VariantParts,
) -> tuple[tuple[GroundTerm, ...], ...]:
    return tuple(
        _execute_graph(
            parts,
            _encode_ground_for_parts(
                term,
                parts,
                salt=10000 + term_index,
            ),
        )
        for term_index, term in enumerate(challenge_terms())
    )


def _delete_for_ambiguity(
    theory_index: int,
) -> tuple[tuple[Demonstration, ...], tuple[int, ...], int]:
    evidence = identifying_evidence(theory_index)
    candidates = []
    for deleted_index in range(len(evidence)):
        retained = (
            evidence[:deleted_index]
            + evidence[deleted_index + 1 :]
        )
        version_space = exact_version_space(retained)
        if (
            theory_index in version_space.theory_indices
            and version_space.behavioral_class_count >= 2
        ):
            witness_index = next(
                index
                for index, term in enumerate(challenge_terms())
                if len(
                    {
                        execute_normal_forms(candidate, term)
                        for candidate in version_space.theory_indices
                    }
                )
                >= 2
            )
            candidates.append(
                (
                    version_space.behavioral_class_count,
                    len(version_space.theory_indices),
                    deleted_index,
                    retained,
                    version_space.theory_indices,
                    witness_index,
                )
            )
    if not candidates:
        raise ValueError("no decisive ambiguity deletion exists")
    _, _, _, retained, survivors, witness = min(
        candidates,
        key=lambda item: item[:3],
    )
    return retained, survivors, witness


def _select_challenge_indices(
    theory_index: int,
    mandatory: Iterable[int],
) -> tuple[int, ...]:
    selected: list[int] = []
    for index in mandatory:
        if index not in selected:
            selected.append(index)
    terms = challenge_terms()
    cursor = (theory_index * 17 + 11) % len(terms)
    while len(selected) < CHALLENGES_PER_VARIANT:
        if cursor not in selected:
            selected.append(cursor)
        cursor = (cursor + 19) % len(terms)
    return tuple(selected)


def _expectation(kind: RewriteVariantKind) -> VariantExpectation:
    return {
        RewriteVariantKind.BASE: VariantExpectation.BASELINE,
        RewriteVariantKind.ALPHA_REORDER: (
            VariantExpectation.EXACT_INVARIANCE
        ),
        RewriteVariantKind.ALIAS_SPLIT: (
            VariantExpectation.EXACT_INVARIANCE
        ),
        RewriteVariantKind.RELATION_REIFICATION: (
            VariantExpectation.EXACT_INVARIANCE
        ),
        RewriteVariantKind.TYPE_TWIN: (
            VariantExpectation.EXACT_TYPE_SEPARATION
        ),
        RewriteVariantKind.EXECUTION_SEMANTICS_TWIN: (
            VariantExpectation.EXACT_EXECUTION_SEPARATION
        ),
        RewriteVariantKind.AMBIGUITY_DELETED_TWIN: (
            VariantExpectation.EXACT_AMBIGUITY_SEPARATION
        ),
    }[kind]


def _variant_evidence(
    theory_index: int,
    parts: _VariantParts,
    *,
    retained_ambiguity: tuple[Demonstration, ...],
) -> tuple[VariantDemonstration, ...]:
    evidence = (
        retained_ambiguity
        if parts.kind == RewriteVariantKind.AMBIGUITY_DELETED_TWIN
        else identifying_evidence(theory_index)
    )
    if parts.kind == RewriteVariantKind.EXECUTION_SEMANTICS_TWIN:
        semantics_probe = next(
            (
                Demonstration(term, twin_output)
                for term_index, term in enumerate(challenge_terms())
                if (
                    twin_output := _execute_graph(
                        parts,
                        _encode_ground_for_parts(
                            term,
                            parts,
                            salt=19000 + term_index,
                        ),
                    )
                )
                != execute_normal_forms(theory_index, term)
            ),
            None,
        )
        if semantics_probe is None:
            raise ValueError(
                "execution twin lacks an evidence-level separator"
            )
        if all(
            demo.initial != semantics_probe.initial
            for demo in evidence
        ):
            evidence = (*evidence, semantics_probe)
    result = []
    for evidence_index, demo in enumerate(evidence):
        initial = _encode_ground_for_parts(
            demo.initial,
            parts,
            salt=20000 + evidence_index * 101,
        )
        aligned_normal_forms = _execute_graph(parts, initial)
        result.append(
            VariantDemonstration(
                initial=initial,
                normal_forms=tuple(
                    _encode_ground_for_parts(
                        term,
                        parts,
                        salt=21000
                        + evidence_index * 101
                        + output_index,
                    )
                    for output_index, term in enumerate(
                        aligned_normal_forms
                    )
                ),
            )
        )
    return tuple(result)


def _make_witness(
    *,
    challenge_indices: tuple[int, ...],
    canonical_index: int,
    base_output: tuple[GroundTerm, ...],
    variant_outcomes: tuple[tuple[GroundTerm, ...], ...],
    variant_decision: QualificationDecision,
) -> SeparationWitness:
    return SeparationWitness(
        challenge_offset=challenge_indices.index(canonical_index),
        canonical_term=challenge_terms()[canonical_index],
        base_outcome=base_output,
        variant_outcomes=variant_outcomes,
        base_decision=QualificationDecision.EXECUTE,
        variant_decision=variant_decision,
    )


@lru_cache(maxsize=None)
def build_rewrite_variant_family(
    theory_index: int,
) -> tuple[RewriteQualificationVariant, ...]:
    """Build and validate all seven variants for one held-out theory."""

    if theory_index not in HELDOUT_THEORY_INDICES:
        raise ValueError("qualification target is not held out")
    parts_by_kind = {
        kind: _build_parts(theory_index, kind)
        for kind in VARIANT_ORDER
    }
    exhaustive = {
        kind: _exhaustive_outputs(parts)
        for kind, parts in parts_by_kind.items()
    }
    base_exhaustive = exhaustive[RewriteVariantKind.BASE]
    board_exhaustive = tuple(
        execute_normal_forms(theory_index, term)
        for term in challenge_terms()
    )
    if base_exhaustive != board_exhaustive:
        raise ValueError("base variant differs from board oracle")
    for kind in (
        RewriteVariantKind.ALPHA_REORDER,
        RewriteVariantKind.ALIAS_SPLIT,
        RewriteVariantKind.RELATION_REIFICATION,
    ):
        if exhaustive[kind] != base_exhaustive:
            raise ValueError(f"{kind.value} is not exactly invariant")

    type_witness = next(
        (
            index
            for index, (base, twin) in enumerate(
                zip(
                    base_exhaustive,
                    exhaustive[RewriteVariantKind.TYPE_TWIN],
                    strict=True,
                )
            )
            if base != twin
        ),
        None,
    )
    execution_witness = next(
        (
            index
            for index, (base, twin) in enumerate(
                zip(
                    base_exhaustive,
                    exhaustive[
                        RewriteVariantKind.EXECUTION_SEMANTICS_TWIN
                    ],
                    strict=True,
                )
            )
            if base != twin
        ),
        None,
    )
    if type_witness is None:
        raise ValueError("type twin lacks exact separation witness")
    if execution_witness is None:
        raise ValueError("execution twin lacks exact separation witness")

    retained_ambiguity, ambiguity_survivors, ambiguity_witness = (
        _delete_for_ambiguity(theory_index)
    )
    challenge_indices = _select_challenge_indices(
        theory_index,
        (type_witness, execution_witness, ambiguity_witness),
    )
    family = []
    for kind in VARIANT_ORDER:
        parts = parts_by_kind[kind]
        encoded_challenges = tuple(
            _encode_ground_for_parts(
                challenge_terms()[canonical_index],
                parts,
                salt=30000 + offset * 103,
            )
            for offset, canonical_index in enumerate(challenge_indices)
        )
        aligned_outputs = tuple(
            _execute_graph(parts, graph)
            for graph in encoded_challenges
        )
        decision = QualificationDecision.EXECUTE
        behavioral_class_count = 1
        possible_outputs = tuple(
            (output,)
            for output in aligned_outputs
        )
        witness: SeparationWitness | None = None
        if kind == RewriteVariantKind.TYPE_TWIN:
            witness = _make_witness(
                challenge_indices=challenge_indices,
                canonical_index=type_witness,
                base_output=base_exhaustive[type_witness],
                variant_outcomes=(
                    exhaustive[kind][type_witness],
                ),
                variant_decision=QualificationDecision.EXECUTE,
            )
        elif kind == RewriteVariantKind.EXECUTION_SEMANTICS_TWIN:
            witness = _make_witness(
                challenge_indices=challenge_indices,
                canonical_index=execution_witness,
                base_output=base_exhaustive[execution_witness],
                variant_outcomes=(
                    exhaustive[kind][execution_witness],
                ),
                variant_decision=QualificationDecision.EXECUTE,
            )
        elif kind == RewriteVariantKind.AMBIGUITY_DELETED_TWIN:
            decision = QualificationDecision.ABSTAIN_AMBIGUOUS
            behavioral_class_count = len(
                {
                    tuple(
                        execute_normal_forms(candidate, term)
                        for term in challenge_terms()
                    )
                    for candidate in ambiguity_survivors
                }
            )
            possible_outputs = tuple(
                tuple(
                    sorted(
                        {
                            execute_normal_forms(
                                candidate,
                                challenge_terms()[canonical_index],
                            )
                            for candidate in ambiguity_survivors
                        }
                    )
                )
                for canonical_index in challenge_indices
            )
            witness_outcomes = possible_outputs[
                challenge_indices.index(ambiguity_witness)
            ]
            if len(witness_outcomes) < 2:
                raise ValueError("ambiguity twin witness does not separate")
            witness = _make_witness(
                challenge_indices=challenge_indices,
                canonical_index=ambiguity_witness,
                base_output=base_exhaustive[ambiguity_witness],
                variant_outcomes=witness_outcomes,
                variant_decision=decision,
            )
        family.append(
            RewriteQualificationVariant(
                kind=kind,
                theory_index=theory_index,
                constructors=parts.constructors,
                rules=parts.rules,
                evidence=_variant_evidence(
                    theory_index,
                    parts,
                    retained_ambiguity=retained_ambiguity,
                ),
                challenges=encoded_challenges,
                challenge_indices=challenge_indices,
                execution_semantics=parts.semantics,
                expectation=_expectation(kind),
                alignment=parts.alignment,
                oracle=VariantOracle(
                    decision=decision,
                    aligned_outputs=aligned_outputs,
                    possible_outputs=possible_outputs,
                    behavioral_class_count=behavioral_class_count,
                    witness=witness,
                ),
            )
        )
    result = tuple(family)
    audit_rewrite_variant_family(result)
    return result


def execute_variant_challenge(
    variant: RewriteQualificationVariant,
    challenge_offset: int,
) -> tuple[GroundTerm, ...]:
    """Execute one encoded challenge with the module's offline oracle."""

    if not 0 <= challenge_offset < len(variant.challenges):
        raise IndexError("challenge offset differs")
    parts = _VariantParts(
        kind=variant.kind,
        constructors=variant.constructors,
        rules=variant.rules,
        semantics=variant.execution_semantics,
        alignment=variant.alignment,
    )
    return _execute_graph(parts, variant.challenges[challenge_offset])


def canonical_rule_signature(
    variant: RewriteQualificationVariant,
) -> tuple[tuple[object, ...], ...]:
    """Return rules aligned to base identities and alpha-normalized."""

    compiled = {
        rule.local_index: rule
        for rule in _compile_rules(
            variant.rules,
            variant.constructors,
        )
    }
    rule_to_base = dict(variant.alignment.rule_to_base)

    def pattern_signature(
        pattern: _RuntimePattern,
        variables: dict[int, int],
    ) -> tuple[object, ...]:
        if pattern.variable_index is not None:
            canonical = variables.setdefault(
                pattern.variable_index,
                len(variables),
            )
            return ("variable", pattern.type_index, canonical)
        return (
            "constructor",
            pattern.type_index,
            pattern.equivalence_class,
            tuple(
                pattern_signature(child, variables)
                for child in pattern.children
            ),
        )

    signatures = []
    for local_index, base_index in sorted(
        rule_to_base.items(),
        key=lambda item: item[1],
    ):
        rule = compiled[local_index]
        variables: dict[int, int] = {}
        signatures.append(
            (
                base_index,
                pattern_signature(rule.lhs, variables),
                pattern_signature(rule.rhs, variables),
            )
        )
    return tuple(signatures)


def audit_rewrite_variant_family(
    family: tuple[RewriteQualificationVariant, ...],
) -> RewriteVariantAuditReceipt:
    """Fail closed unless every preregistered variant contract is exact."""

    if tuple(variant.kind for variant in family) != VARIANT_ORDER:
        raise ValueError("variant order or membership differs")
    theory_indices = {variant.theory_index for variant in family}
    if len(theory_indices) != 1:
        raise ValueError("variant family crosses theories")
    if any(
        len(variant.challenges) != CHALLENGES_PER_VARIANT
        or len(variant.challenge_indices) != CHALLENGES_PER_VARIANT
        for variant in family
    ):
        raise ValueError("variant challenge count differs")
    base = family[0]
    base_signature = canonical_rule_signature(base)
    invariant_cases = 0
    for variant in family[1:4]:
        if (
            variant.expectation != VariantExpectation.EXACT_INVARIANCE
            or canonical_rule_signature(variant) != base_signature
            or variant.oracle.aligned_outputs
            != base.oracle.aligned_outputs
        ):
            raise ValueError(
                f"{variant.kind.value} invariance contract differs"
            )
        invariant_cases += len(variant.challenges)
    alias_variant = family[2]
    alias_classes: dict[int, set[int]] = {}
    for constructor in alias_variant.constructors:
        alias_classes.setdefault(
            constructor.equivalence_class,
            set(),
        ).add(constructor.symbol_index)
    split_classes = tuple(
        symbols
        for symbols in alias_classes.values()
        if len(symbols) > 1
    )
    material_graphs = [
        graph
        for rule in alias_variant.rules
        for graph in (rule.lhs, rule.rhs)
    ]
    for demonstration in alias_variant.evidence:
        material_graphs.append(demonstration.initial)
        material_graphs.extend(demonstration.normal_forms)
    material_graphs.extend(alias_variant.challenges)
    used_symbols = {
        node.symbol_index
        for graph in material_graphs
        for node in graph.nodes
        if node.kind == GraphNodeKind.CONSTRUCTOR
    }
    if (
        len(split_classes) != 1
        or not split_classes[0] <= used_symbols
    ):
        raise ValueError("alias split does not materialize both symbols")
    type_variant = family[4]
    execution_variant = family[5]
    ambiguity_variant = family[6]
    if (
        type_variant.expectation
        != VariantExpectation.EXACT_TYPE_SEPARATION
        or type_variant.oracle.witness is None
        or type_variant.oracle.witness.base_outcome
        in type_variant.oracle.witness.variant_outcomes
    ):
        raise ValueError("type twin separation contract differs")
    if (
        execution_variant.expectation
        != VariantExpectation.EXACT_EXECUTION_SEPARATION
        or execution_variant.oracle.witness is None
        or execution_variant.oracle.witness.base_outcome
        in execution_variant.oracle.witness.variant_outcomes
    ):
        raise ValueError("execution twin separation contract differs")
    ambiguity_witness = ambiguity_variant.oracle.witness
    if (
        ambiguity_variant.expectation
        != VariantExpectation.EXACT_AMBIGUITY_SEPARATION
        or ambiguity_variant.oracle.decision
        != QualificationDecision.ABSTAIN_AMBIGUOUS
        or ambiguity_variant.oracle.behavioral_class_count < 2
        or ambiguity_witness is None
        or len(ambiguity_witness.variant_outcomes) < 2
        or ambiguity_witness.base_decision
        == ambiguity_witness.variant_decision
    ):
        raise ValueError("ambiguity twin separation contract differs")
    material_hashes = {
        variant.material_sha256()
        for variant in family
    }
    if len(material_hashes) != len(VARIANT_ORDER):
        raise ValueError("variant material is not structurally unique")
    return RewriteVariantAuditReceipt(
        theory_index=base.theory_index,
        variant_count=len(family),
        challenges_per_variant=CHALLENGES_PER_VARIANT,
        invariant_variant_count=3,
        exact_invariance_cases=invariant_cases,
        type_separation_witnesses=1,
        execution_separation_witnesses=1,
        ambiguity_separation_witnesses=1,
        unique_material_hashes=len(material_hashes),
        all_contracts_pass=True,
    )


__all__ = [
    "CHALLENGES_PER_VARIANT",
    "VARIANT_ORDER",
    "CanonicalAlignment",
    "ExecutionSemantics",
    "GraphEdgeKind",
    "GraphNodeKind",
    "QualificationDecision",
    "RewriteQualificationVariant",
    "RewriteVariantAuditReceipt",
    "RewriteVariantKind",
    "SeparationWitness",
    "VariantConstructor",
    "VariantDemonstration",
    "VariantExpectation",
    "VariantGraphEdge",
    "VariantGraphNode",
    "VariantOracle",
    "VariantRule",
    "VariantTermGraph",
    "audit_rewrite_variant_family",
    "build_rewrite_variant_family",
    "canonical_rule_signature",
    "execute_variant_challenge",
]
