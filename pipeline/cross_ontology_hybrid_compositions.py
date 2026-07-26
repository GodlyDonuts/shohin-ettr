"""Frozen offline mechanics for the three preregistered ETTR hybrids.

This module is assessor-side qualification infrastructure.  It defines exactly
the three hybrid couplings named in the ETTR preregistration, executes every
case through two independently implemented mechanics paths, and emits a
hash-bound receipt.  Candidate-visible source and challenge bytes contain no
hybrid identity, board-family label, theory index, intermediate selector,
expected output, or oracle record.

Passing this module proves only that the bounded hybrid board and its causal
interventions are exact.  It is not learned, model-owned, architecture-native,
or general-reasoning evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, StrEnum
from functools import lru_cache
from hashlib import sha256
from itertools import product
import json
from typing import Any, TypeAlias

from cross_ontology_horn_board import (
    GroundAtom,
    OBJECT_TYPES,
    PREDICATES,
    RULE_LIBRARY as HORN_RULE_LIBRARY,
    THEORIES as HORN_THEORIES,
)
from cross_ontology_resource_board import (
    Marking,
    OPERATOR_LIBRARY,
    PLACE_SPECS,
    ProcessOutcome,
    ProcessStatus,
    THEORIES as RESOURCE_THEORIES,
)
from cross_ontology_rewrite_board import (
    GroundTerm,
    PatternTerm,
    RULE_LIBRARY as REWRITE_RULE_LIBRARY,
    THEORIES as REWRITE_THEORIES,
    challenge_terms,
)


CLAIM_BOUNDARY = (
    "Exact offline hybrid-board mechanics and causal-intervention receipt "
    "only; this is not learned capability, model-owned execution, native "
    "reasoning, or an unrestricted general-reasoning claim."
)
CASES_PER_HYBRID = 16
HYBRID_COUNT = 3
TOTAL_CASES = CASES_PER_HYBRID * HYBRID_COUNT
TOTAL_EXECUTIONS = TOTAL_CASES * 2

_CANDIDATE_SCHEMA = "ettr-coupled-world-v1"
_CHALLENGE_SCHEMA = "ettr-coupled-challenge-v1"
_REWRITE_LOCATION_COUNT = 2
_FORBIDDEN_CANDIDATE_TOKENS = (
    b"answer",
    b"expected",
    b"family",
    b"horn",
    b"hybrid",
    b"oracle",
    b"resource",
    b"rewrite",
    b"theory_index",
)


class HybridKind(StrEnum):
    """The three and only three preregistered hybrid couplings."""

    ARITHMETIC_SELECTS_REWRITE_LOCATION = "arithmetic_index_selects_rewrite_location"
    HORN_RELATION_SELECTS_RESOURCE_OPERATOR = "horn_relation_selects_resource_operator"
    RESOURCE_STATE_CONTROLS_HORN_QUERY = "resource_state_controls_horn_query"


HYBRID_ORDER = tuple(HybridKind)


@dataclass(frozen=True, slots=True)
class ArithmeticRewriteSpec:
    rewrite_theory_index: int
    term: GroundTerm
    operands: tuple[int, int, int]
    intervention_operands: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class HornResourceSpec:
    horn_theory_index: int
    horn_initial: tuple[GroundAtom, ...]
    intervention_horn_initial: tuple[GroundAtom, ...]
    selector_atom: GroundAtom
    resource_theory_index: int
    resource_initial: Marking
    true_symbol: int
    false_symbol: int


@dataclass(frozen=True, slots=True)
class ResourceHornSpec:
    resource_theory_index: int
    resource_initial: Marking
    intervention_resource_initial: Marking
    resource_sequence: tuple[int, ...]
    control_place: int
    control_threshold: int
    horn_theory_index: int
    horn_initial: tuple[GroundAtom, ...]
    query_if_true: GroundAtom
    query_if_false: GroundAtom


HybridSpec: TypeAlias = ArithmeticRewriteSpec | HornResourceSpec | ResourceHornSpec


@dataclass(frozen=True, order=True, slots=True)
class RewriteEvent:
    path: tuple[int, ...]
    rule_index: int
    terminal: GroundTerm


@dataclass(frozen=True, slots=True)
class ArithmeticRewriteResult:
    selected_index: int
    selected_path: tuple[int, ...]
    selected_rule_index: int
    terminal: GroundTerm


@dataclass(frozen=True, slots=True)
class HornResourceResult:
    selector_holds: bool
    selected_symbol: int
    outcome: ProcessOutcome


@dataclass(frozen=True, slots=True)
class ResourceHornResult:
    control_holds: bool
    selected_query: GroundAtom
    query_holds: bool
    resource_outcome: ProcessOutcome


HybridResult: TypeAlias = (
    ArithmeticRewriteResult | HornResourceResult | ResourceHornResult
)


@dataclass(frozen=True, slots=True)
class HybridCase:
    """One assessor record with paired candidate-visible challenges."""

    kind: HybridKind
    case_index: int
    spec: HybridSpec
    compiler_source: bytes
    challenge: bytes
    intervention_challenge: bytes

    def compiler_source_bytes(self) -> bytes:
        return self.compiler_source

    def late_challenge_bytes(self, *, intervention: bool = False) -> bytes:
        return self.intervention_challenge if intervention else self.challenge


@dataclass(frozen=True, slots=True)
class HybridCaseReceipt:
    kind: HybridKind
    case_index: int
    source_sha256: str
    challenge_sha256: str
    intervention_challenge_sha256: str
    expected_sha256: str
    intervention_expected_sha256: str
    row_sha256: str


@dataclass(frozen=True, slots=True)
class HybridQualificationReceipt:
    hybrid_count: int
    cases_per_hybrid: int
    case_count: int
    execution_count: int
    independent_oracle_agreement_count: int
    causal_intervention_count: int
    causal_signal_change_count: int
    causal_output_change_count: int
    candidate_label_leak_count: int
    unique_row_count: int
    payload_sha256: str
    claim_boundary: str
    all_contracts_pass: bool


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            field.name: _canonicalize(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: str(pair[0]),
            )
        }
    if isinstance(value, (tuple, list)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"hybrid value is not canonical: {type(value)!r}")


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _canonicalize(value),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _digest(value: Any) -> str:
    payload = value if isinstance(value, bytes) else _canonical_bytes(value)
    return sha256(payload).hexdigest()


def _atom(predicate: int, *arguments: int) -> GroundAtom:
    return GroundAtom(predicate, tuple(arguments))


def _pattern_payload(pattern: PatternTerm) -> Any:
    if pattern.variable_index is not None:
        return ["v", pattern.type_index, pattern.variable_index]
    return [
        "c",
        pattern.type_index,
        pattern.constructor_index,
        *(_pattern_payload(child) for child in pattern.children),
    ]


def _term_payload(term: GroundTerm) -> Any:
    return [
        term.type_index,
        term.constructor_index,
        *(_term_payload(child) for child in term.children),
    ]


def _atom_payload(atom: GroundAtom) -> Any:
    return [atom.predicate, *atom.arguments]


def _horn_law_payload(theory_index: int) -> list[Any]:
    return [
        [
            [
                [premise.predicate, *premise.variables]
                for premise in HORN_RULE_LIBRARY[rule_index].premises
            ],
            [
                HORN_RULE_LIBRARY[rule_index].conclusion.predicate,
                *HORN_RULE_LIBRARY[rule_index].conclusion.variables,
            ],
        ]
        for rule_index in HORN_THEORIES[theory_index].rule_indices
    ]


def _rewrite_law_payload(theory_index: int) -> list[Any]:
    return [
        [
            _pattern_payload(REWRITE_RULE_LIBRARY[rule_index].lhs),
            _pattern_payload(REWRITE_RULE_LIBRARY[rule_index].rhs),
        ]
        for rule_index in REWRITE_THEORIES[theory_index].rule_indices
    ]


def _quantity_payload(quantities: tuple[Any, ...]) -> list[Any]:
    return [
        [quantity.place, quantity.resource_kind, quantity.multiplicity]
        for quantity in quantities
    ]


def _resource_law_payload(theory_index: int) -> list[Any]:
    payload = []
    for operator_index in RESOURCE_THEORIES[theory_index].operator_indices:
        operator = OPERATOR_LIBRARY[operator_index]
        payload.append(
            [
                _quantity_payload(operator.guards),
                _quantity_payload(operator.consumes),
                _quantity_payload(operator.produces),
            ]
        )
    return payload


def _candidate_source(spec: HybridSpec) -> bytes:
    if isinstance(spec, ArithmeticRewriteSpec):
        body = {
            "a": [["*", "+", "%"], _REWRITE_LOCATION_COUNT],
            "b": _rewrite_law_payload(spec.rewrite_theory_index),
        }
    elif isinstance(spec, HornResourceSpec):
        body = {
            "a": _horn_law_payload(spec.horn_theory_index),
            "b": _resource_law_payload(spec.resource_theory_index),
            "c": [spec.true_symbol, spec.false_symbol],
        }
    else:
        body = {
            "a": _resource_law_payload(spec.resource_theory_index),
            "b": _horn_law_payload(spec.horn_theory_index),
            "c": [
                spec.control_place,
                spec.control_threshold,
                _atom_payload(spec.query_if_true),
                _atom_payload(spec.query_if_false),
            ],
        }
    return _canonical_bytes({"schema": _CANDIDATE_SCHEMA, "world": body})


def _candidate_challenge(
    spec: HybridSpec,
    *,
    intervention: bool,
) -> bytes:
    if isinstance(spec, ArithmeticRewriteSpec):
        body = {
            "a": (spec.intervention_operands if intervention else spec.operands),
            "b": _term_payload(spec.term),
        }
    elif isinstance(spec, HornResourceSpec):
        body = {
            "a": [
                _atom_payload(atom)
                for atom in (
                    spec.intervention_horn_initial
                    if intervention
                    else spec.horn_initial
                )
            ],
            "b": list(spec.resource_initial.multiplicities),
        }
    else:
        initial = (
            spec.intervention_resource_initial
            if intervention
            else spec.resource_initial
        )
        body = {
            "a": list(initial.multiplicities),
            "b": list(spec.resource_sequence),
            "c": [_atom_payload(atom) for atom in spec.horn_initial],
        }
    return _canonical_bytes({"schema": _CHALLENGE_SCHEMA, "input": body})


def _primary_pattern_match(
    pattern: PatternTerm,
    term: GroundTerm,
    bindings: dict[int, GroundTerm],
) -> bool:
    if pattern.type_index != term.type_index:
        return False
    if pattern.variable_index is not None:
        previous = bindings.setdefault(pattern.variable_index, term)
        return previous == term
    if pattern.constructor_index != term.constructor_index or len(
        pattern.children
    ) != len(term.children):
        return False
    return all(
        _primary_pattern_match(child_pattern, child, bindings)
        for child_pattern, child in zip(
            pattern.children,
            term.children,
            strict=True,
        )
    )


def _primary_instantiate(
    pattern: PatternTerm,
    bindings: dict[int, GroundTerm],
) -> GroundTerm:
    if pattern.variable_index is not None:
        return bindings[pattern.variable_index]
    assert pattern.constructor_index is not None
    return GroundTerm(
        pattern.type_index,
        pattern.constructor_index,
        tuple(_primary_instantiate(child, bindings) for child in pattern.children),
    )


def _primary_occurrences(
    term: GroundTerm,
    path: tuple[int, ...] = (),
) -> tuple[tuple[tuple[int, ...], GroundTerm], ...]:
    result = [(path, term)]
    for child_index, child in enumerate(term.children):
        result.extend(_primary_occurrences(child, (*path, child_index)))
    return tuple(result)


def _primary_replace(
    term: GroundTerm,
    path: tuple[int, ...],
    replacement: GroundTerm,
) -> GroundTerm:
    if not path:
        return replacement
    child_index = path[0]
    children = list(term.children)
    children[child_index] = _primary_replace(
        children[child_index],
        path[1:],
        replacement,
    )
    return GroundTerm(
        term.type_index,
        term.constructor_index,
        tuple(children),
    )


def _primary_rewrite_events(
    theory_index: int,
    term: GroundTerm,
) -> tuple[RewriteEvent, ...]:
    events = []
    for path, redex in _primary_occurrences(term):
        for rule_index in REWRITE_THEORIES[theory_index].rule_indices:
            rule = REWRITE_RULE_LIBRARY[rule_index]
            bindings: dict[int, GroundTerm] = {}
            if _primary_pattern_match(rule.lhs, redex, bindings):
                events.append(
                    RewriteEvent(
                        path,
                        rule_index,
                        _primary_replace(
                            term,
                            path,
                            _primary_instantiate(rule.rhs, bindings),
                        ),
                    )
                )
    return tuple(sorted(events))


def _independent_nodes(
    term: GroundTerm,
) -> tuple[tuple[tuple[int, ...], GroundTerm], ...]:
    pending = [((), term)]
    nodes = []
    while pending:
        path, node = pending.pop(0)
        nodes.append((path, node))
        pending[0:0] = [
            ((*path, child_index), child)
            for child_index, child in reversed(tuple(enumerate(node.children)))
        ]
    return tuple(sorted(nodes))


def _independent_pattern_match(
    pattern: PatternTerm,
    term: GroundTerm,
) -> dict[int, GroundTerm] | None:
    bindings: dict[int, GroundTerm] = {}
    pending = [(pattern, term)]
    while pending:
        left, right = pending.pop()
        if left.type_index != right.type_index:
            return None
        if left.variable_index is not None:
            if (
                left.variable_index in bindings
                and bindings[left.variable_index] != right
            ):
                return None
            bindings[left.variable_index] = right
            continue
        if left.constructor_index != right.constructor_index or len(
            left.children
        ) != len(right.children):
            return None
        pending.extend(zip(left.children, right.children, strict=True))
    return bindings


def _independent_instantiate(
    pattern: PatternTerm,
    bindings: dict[int, GroundTerm],
) -> GroundTerm:
    if pattern.variable_index is not None:
        return bindings[pattern.variable_index]
    assert pattern.constructor_index is not None
    children = tuple(
        _independent_instantiate(child, bindings) for child in pattern.children
    )
    return GroundTerm(
        pattern.type_index,
        pattern.constructor_index,
        children,
    )


def _independent_replace(
    term: GroundTerm,
    path: tuple[int, ...],
    replacement: GroundTerm,
) -> GroundTerm:
    replacements = {path: replacement}

    def rebuild(node: GroundTerm, prefix: tuple[int, ...]) -> GroundTerm:
        if prefix in replacements:
            return replacements[prefix]
        return GroundTerm(
            node.type_index,
            node.constructor_index,
            tuple(
                rebuild(child, (*prefix, child_index))
                for child_index, child in enumerate(node.children)
            ),
        )

    return rebuild(term, ())


def _independent_rewrite_events(
    theory_index: int,
    term: GroundTerm,
) -> tuple[RewriteEvent, ...]:
    events = set()
    for path, redex in _independent_nodes(term):
        for rule_index in REWRITE_THEORIES[theory_index].rule_indices:
            rule = REWRITE_RULE_LIBRARY[rule_index]
            bindings = _independent_pattern_match(rule.lhs, redex)
            if bindings is None:
                continue
            replacement = _independent_instantiate(rule.rhs, bindings)
            events.add(
                RewriteEvent(
                    path,
                    rule_index,
                    _independent_replace(term, path, replacement),
                )
            )
    return tuple(sorted(events))


def _primary_horn_closure(
    theory_index: int,
    initial: tuple[GroundAtom, ...],
) -> tuple[GroundAtom, ...]:
    facts = set(initial)

    def extend(
        premises: tuple[Any, ...],
        offset: int,
        bindings: dict[int, int],
    ) -> list[dict[int, int]]:
        if offset == len(premises):
            return [bindings]
        premise = premises[offset]
        matches = []
        for fact in facts:
            if fact.predicate != premise.predicate or len(fact.arguments) != len(
                premise.variables
            ):
                continue
            updated = dict(bindings)
            valid = True
            for variable, argument in zip(
                premise.variables,
                fact.arguments,
                strict=True,
            ):
                if variable in updated and updated[variable] != argument:
                    valid = False
                    break
                updated[variable] = argument
            if valid:
                matches.extend(extend(premises, offset + 1, updated))
        return matches

    while True:
        before = len(facts)
        for rule_index in HORN_THEORIES[theory_index].rule_indices:
            rule = HORN_RULE_LIBRARY[rule_index]
            for bindings in extend(rule.premises, 0, {}):
                facts.add(
                    GroundAtom(
                        rule.conclusion.predicate,
                        tuple(
                            bindings[variable] for variable in rule.conclusion.variables
                        ),
                    )
                )
        if len(facts) == before:
            return tuple(sorted(facts))


def _independent_horn_closure(
    theory_index: int,
    initial: tuple[GroundAtom, ...],
) -> tuple[GroundAtom, ...]:
    facts = set(initial)
    while True:
        derived = set(facts)
        for rule_index in HORN_THEORIES[theory_index].rule_indices:
            rule = HORN_RULE_LIBRARY[rule_index]
            variable_types: dict[int, int] = {}
            for pattern in (*rule.premises, rule.conclusion):
                signature = PREDICATES[pattern.predicate].argument_types
                for variable, type_index in zip(
                    pattern.variables,
                    signature,
                    strict=True,
                ):
                    if (
                        variable in variable_types
                        and variable_types[variable] != type_index
                    ):
                        raise ValueError("Horn variable typing differs")
                    variable_types[variable] = type_index
            variables = tuple(sorted(variable_types))
            domains = [
                tuple(
                    object_index
                    for object_index, object_type in enumerate(OBJECT_TYPES)
                    if object_type == variable_types[variable]
                )
                for variable in variables
            ]
            for values in product(*domains):
                assignment = dict(zip(variables, values, strict=True))
                premises = {
                    GroundAtom(
                        premise.predicate,
                        tuple(assignment[variable] for variable in premise.variables),
                    )
                    for premise in rule.premises
                }
                if premises <= facts:
                    derived.add(
                        GroundAtom(
                            rule.conclusion.predicate,
                            tuple(
                                assignment[variable]
                                for variable in rule.conclusion.variables
                            ),
                        )
                    )
        if derived == facts:
            return tuple(sorted(facts))
        facts = derived


def _primary_resource_execution(
    theory_index: int,
    initial: Marking,
    sequence: tuple[int, ...],
) -> ProcessOutcome:
    counts = {place.index: initial.multiplicities[place.index] for place in PLACE_SPECS}
    for cursor, symbol in enumerate(sequence):
        operator_index = RESOURCE_THEORIES[theory_index].operator_indices[symbol]
        operator = OPERATOR_LIBRARY[operator_index]
        required = {
            quantity.place: max(
                quantity.multiplicity,
                next(
                    (
                        item.multiplicity
                        for item in operator.consumes
                        if item.place == quantity.place
                    ),
                    0,
                ),
            )
            for quantity in operator.guards
        }
        for quantity in operator.consumes:
            required[quantity.place] = max(
                required.get(quantity.place, 0),
                quantity.multiplicity,
            )
        if any(counts[place] < amount for place, amount in required.items()):
            return ProcessOutcome(
                Marking(tuple(counts[index] for index in range(4))),
                cursor,
                ProcessStatus.DEADLOCK,
            )
        updated = dict(counts)
        for quantity in operator.consumes:
            updated[quantity.place] -= quantity.multiplicity
        for quantity in operator.produces:
            updated[quantity.place] += quantity.multiplicity
        if any(updated[place.index] > place.capacity for place in PLACE_SPECS):
            return ProcessOutcome(
                Marking(tuple(counts[index] for index in range(4))),
                cursor,
                ProcessStatus.DEADLOCK,
            )
        counts = updated
    return ProcessOutcome(
        Marking(tuple(counts[index] for index in range(4))),
        len(sequence),
        ProcessStatus.HALT,
    )


def _independent_resource_execution(
    theory_index: int,
    initial: Marking,
    sequence: tuple[int, ...],
) -> ProcessOutcome:
    state = initial.multiplicities
    for cursor, symbol in enumerate(sequence):
        operator = OPERATOR_LIBRARY[
            RESOURCE_THEORIES[theory_index].operator_indices[symbol]
        ]
        guard = [0] * len(PLACE_SPECS)
        consume = [0] * len(PLACE_SPECS)
        produce = [0] * len(PLACE_SPECS)
        for quantity in operator.guards:
            guard[quantity.place] = quantity.multiplicity
        for quantity in operator.consumes:
            consume[quantity.place] = quantity.multiplicity
        for quantity in operator.produces:
            produce[quantity.place] = quantity.multiplicity
        if any(
            state[index] < max(guard[index], consume[index])
            for index in range(len(PLACE_SPECS))
        ):
            return ProcessOutcome(
                Marking(state),
                cursor,
                ProcessStatus.DEADLOCK,
            )
        candidate = tuple(
            state[index] - consume[index] + produce[index]
            for index in range(len(PLACE_SPECS))
        )
        if any(
            candidate[index] > PLACE_SPECS[index].capacity
            for index in range(len(PLACE_SPECS))
        ):
            return ProcessOutcome(
                Marking(state),
                cursor,
                ProcessStatus.DEADLOCK,
            )
        state = candidate
    return ProcessOutcome(
        Marking(state),
        len(sequence),
        ProcessStatus.HALT,
    )


def _primary_arithmetic_index(
    operands: tuple[int, int, int],
    modulus: int,
) -> int:
    left, right, bias = operands
    return (left * right + bias) % modulus


def _independent_arithmetic_index(
    operands: tuple[int, int, int],
    modulus: int,
) -> int:
    left, right, bias = operands
    value = bias + sum(right for _ in range(left))
    while value >= modulus:
        value -= modulus
    return value


def execute_hybrid_case(
    case: HybridCase,
    *,
    intervention: bool = False,
) -> HybridResult:
    """Execute one hybrid through the primary independent mechanics."""

    spec = case.spec
    if isinstance(spec, ArithmeticRewriteSpec):
        events = _primary_rewrite_events(
            spec.rewrite_theory_index,
            spec.term,
        )
        if len(events) != _REWRITE_LOCATION_COUNT:
            raise ValueError("frozen rewrite location count differs")
        operands = spec.intervention_operands if intervention else spec.operands
        selected = _primary_arithmetic_index(operands, len(events))
        event = events[selected]
        return ArithmeticRewriteResult(
            selected,
            event.path,
            event.rule_index,
            event.terminal,
        )
    if isinstance(spec, HornResourceSpec):
        initial = spec.intervention_horn_initial if intervention else spec.horn_initial
        selector_holds = spec.selector_atom in _primary_horn_closure(
            spec.horn_theory_index,
            initial,
        )
        symbol = spec.true_symbol if selector_holds else spec.false_symbol
        return HornResourceResult(
            selector_holds,
            symbol,
            _primary_resource_execution(
                spec.resource_theory_index,
                spec.resource_initial,
                (symbol,),
            ),
        )
    initial = (
        spec.intervention_resource_initial if intervention else spec.resource_initial
    )
    resource_outcome = _primary_resource_execution(
        spec.resource_theory_index,
        initial,
        spec.resource_sequence,
    )
    control_holds = (
        resource_outcome.marking.multiplicities[spec.control_place]
        > spec.control_threshold
    )
    query = spec.query_if_true if control_holds else spec.query_if_false
    closure = _primary_horn_closure(
        spec.horn_theory_index,
        spec.horn_initial,
    )
    return ResourceHornResult(
        control_holds,
        query,
        query in closure,
        resource_outcome,
    )


def independent_hybrid_oracle(
    case: HybridCase,
    *,
    intervention: bool = False,
) -> HybridResult:
    """Execute one hybrid through separately implemented oracle mechanics."""

    spec = case.spec
    if isinstance(spec, ArithmeticRewriteSpec):
        events = _independent_rewrite_events(
            spec.rewrite_theory_index,
            spec.term,
        )
        if len(events) != _REWRITE_LOCATION_COUNT:
            raise ValueError("frozen rewrite location count differs")
        operands = spec.intervention_operands if intervention else spec.operands
        selected = _independent_arithmetic_index(
            operands,
            len(events),
        )
        event = events[selected]
        return ArithmeticRewriteResult(
            selected,
            event.path,
            event.rule_index,
            event.terminal,
        )
    if isinstance(spec, HornResourceSpec):
        initial = spec.intervention_horn_initial if intervention else spec.horn_initial
        selector_holds = spec.selector_atom in _independent_horn_closure(
            spec.horn_theory_index,
            initial,
        )
        symbol = spec.true_symbol if selector_holds else spec.false_symbol
        return HornResourceResult(
            selector_holds,
            symbol,
            _independent_resource_execution(
                spec.resource_theory_index,
                spec.resource_initial,
                (symbol,),
            ),
        )
    initial = (
        spec.intervention_resource_initial if intervention else spec.resource_initial
    )
    resource_outcome = _independent_resource_execution(
        spec.resource_theory_index,
        initial,
        spec.resource_sequence,
    )
    control_holds = (
        resource_outcome.marking.multiplicities[spec.control_place]
        > spec.control_threshold
    )
    query = spec.query_if_true if control_holds else spec.query_if_false
    closure = _independent_horn_closure(
        spec.horn_theory_index,
        spec.horn_initial,
    )
    return ResourceHornResult(
        control_holds,
        query,
        query in closure,
        resource_outcome,
    )


def _coupling_signal(result: HybridResult) -> Any:
    if isinstance(result, ArithmeticRewriteResult):
        return result.selected_index
    if isinstance(result, HornResourceResult):
        return result.selector_holds
    return result.control_holds


def _semantic_output(result: HybridResult) -> Any:
    if isinstance(result, ArithmeticRewriteResult):
        return result.terminal
    if isinstance(result, HornResourceResult):
        return result.outcome
    return result.query_holds


_ARITHMETIC_ROWS = (
    (6, 30, 1, 1, 0),
    (6, 31, 1, 2, 0),
    (8, 42, 1, 3, 0),
    (8, 50, 1, 4, 0),
    (8, 52, 1, 5, 0),
    (6, 30, 2, 1, 0),
    (6, 31, 2, 2, 1),
    (8, 42, 2, 3, 0),
    (8, 50, 2, 4, 1),
    (8, 52, 2, 5, 0),
    (6, 30, 3, 1, 1),
    (6, 31, 3, 2, 0),
    (8, 42, 3, 3, 1),
    (8, 50, 3, 4, 0),
    (8, 52, 3, 5, 1),
    (6, 30, 4, 3, 1),
)


_HORN_RESOURCE_ROWS = (
    (0, ((_atom(0, 0)),), _atom(1, 0), 0, (0, 1, 1, 0)),
    (1, ((_atom(0, 0)),), _atom(1, 0), 7, (1, 0, 0, 0)),
    (2, ((_atom(0, 0)),), _atom(1, 0), 14, (0, 1, 1, 0)),
    (3, ((_atom(0, 0)),), _atom(1, 0), 21, (0, 1, 1, 0)),
    (4, ((_atom(0, 0)),), _atom(1, 0), 28, (0, 0, 0, 2)),
    (5, ((_atom(0, 0)),), _atom(1, 0), 35, (0, 0, 0, 2)),
    (6, ((_atom(0, 0)),), _atom(1, 0), 42, (0, 0, 0, 2)),
    (7, ((_atom(0, 0)),), _atom(1, 0), 49, (0, 2, 0, 0)),
    (8, ((_atom(0, 0)),), _atom(1, 0), 56, (0, 0, 0, 2)),
    (9, ((_atom(0, 0)),), _atom(1, 0), 3, (0, 0, 0, 2)),
    (10, ((_atom(3, 0, 3)),), _atom(0, 0), 10, (0, 2, 0, 0)),
    (11, ((_atom(4, 3, 0)),), _atom(0, 0), 17, (0, 0, 0, 2)),
    (
        12,
        (_atom(1, 0), _atom(3, 0, 3)),
        _atom(4, 3, 0),
        24,
        (0, 0, 0, 2),
    ),
    (13, ((_atom(3, 0, 3)),), _atom(0, 0), 31, (0, 0, 0, 2)),
    (14, ((_atom(3, 0, 3)),), _atom(0, 0), 38, (1, 0, 0, 0)),
    (15, ((_atom(4, 3, 0)),), _atom(2, 3), 45, (0, 2, 0, 0)),
)


_RESOURCE_HORN_ROWS = (
    (0, (2, 0, 1, 0), (1, 0, 1, 0), (0, 1), 0, 0),
    (11, (2, 1, 0, 0), (1, 1, 0, 0), (0, 1), 0, 3),
    (22, (0, 2, 1, 0), (0, 1, 1, 0), (0, 2), 1, 6),
    (33, (1, 0, 0, 2), (0, 0, 0, 2), (0, 2), 0, 9),
    (44, (1, 1, 0, 2), (1, 0, 0, 2), (0, 0), 1, 12),
    (55, (0, 2, 0, 2), (0, 1, 0, 2), (1, 2), 1, 15),
    (6, (2, 0, 1, 0), (1, 0, 1, 0), (0, 2), 0, 18),
    (17, (0, 2, 1, 0), (0, 1, 1, 0), (0, 1), 1, 1),
    (28, (0, 2, 0, 2), (0, 1, 0, 2), (0, 1), 1, 4),
    (39, (2, 0, 0, 1), (1, 0, 0, 1), (0, 2), 0, 7),
    (50, (2, 2, 0, 0), (1, 2, 0, 0), (0, 1), 0, 10),
    (1, (2, 0, 1, 0), (1, 0, 1, 0), (0, 1), 0, 13),
    (12, (2, 1, 1, 0), (1, 1, 1, 0), (0, 1), 0, 16),
    (23, (1, 2, 1, 0), (1, 1, 1, 0), (0, 2), 1, 19),
    (34, (0, 2, 0, 2), (0, 1, 0, 2), (0, 2), 1, 2),
    (45, (2, 0, 0, 1), (1, 0, 0, 1), (0, 2), 0, 5),
)

_RESOURCE_HORN_DERIVED_P1_THEORIES = frozenset({0, 1, 2, 3, 4, 5, 6, 7, 9})


def _frozen_specs() -> tuple[tuple[HybridKind, HybridSpec], ...]:
    specs: list[tuple[HybridKind, HybridSpec]] = []
    terms = challenge_terms()
    for theory, term_index, left, right, bias in _ARITHMETIC_ROWS:
        specs.append(
            (
                HybridKind.ARITHMETIC_SELECTS_REWRITE_LOCATION,
                ArithmeticRewriteSpec(
                    theory,
                    terms[term_index],
                    (left, right, bias),
                    (left, right, bias + 1),
                ),
            )
        )
    for (
        horn_theory,
        horn_initial,
        selector,
        resource_theory,
        marking,
    ) in _HORN_RESOURCE_ROWS:
        specs.append(
            (
                HybridKind.HORN_RELATION_SELECTS_RESOURCE_OPERATOR,
                HornResourceSpec(
                    horn_theory,
                    tuple(horn_initial),
                    (),
                    selector,
                    resource_theory,
                    Marking(tuple(marking)),
                    0,
                    1,
                ),
            )
        )
    for (
        resource_theory,
        marking,
        intervention_marking,
        sequence,
        control_place,
        horn_theory,
    ) in _RESOURCE_HORN_ROWS:
        query_if_true = (
            _atom(1, 0)
            if horn_theory in _RESOURCE_HORN_DERIVED_P1_THEORIES
            else _atom(0, 0)
        )
        specs.append(
            (
                HybridKind.RESOURCE_STATE_CONTROLS_HORN_QUERY,
                ResourceHornSpec(
                    resource_theory,
                    Marking(tuple(marking)),
                    Marking(tuple(intervention_marking)),
                    tuple(sequence),
                    control_place,
                    0,
                    horn_theory,
                    (_atom(0, 0),),
                    query_if_true,
                    _atom(0, 1),
                ),
            )
        )
    return tuple(specs)


@lru_cache(maxsize=1)
def build_hybrid_cases() -> tuple[HybridCase, ...]:
    counts = {kind: 0 for kind in HYBRID_ORDER}
    cases = []
    for kind, spec in _frozen_specs():
        case_index = counts[kind]
        counts[kind] += 1
        cases.append(
            HybridCase(
                kind,
                case_index,
                spec,
                _candidate_source(spec),
                _candidate_challenge(spec, intervention=False),
                _candidate_challenge(spec, intervention=True),
            )
        )
    if counts != {kind: CASES_PER_HYBRID for kind in HYBRID_ORDER}:
        raise ValueError("hybrid case geometry differs")
    return tuple(cases)


def audit_hybrid_cases(
    cases: tuple[HybridCase, ...],
) -> tuple[
    tuple[HybridCaseReceipt, ...],
    HybridQualificationReceipt,
]:
    if len(cases) != TOTAL_CASES:
        raise ValueError("hybrid case count differs")
    counts = {kind: sum(case.kind == kind for case in cases) for kind in HYBRID_ORDER}
    if counts != {kind: CASES_PER_HYBRID for kind in HYBRID_ORDER}:
        raise ValueError("hybrid balance differs")
    if {(case.kind, case.case_index) for case in cases} != {
        (kind, case_index)
        for kind in HYBRID_ORDER
        for case_index in range(CASES_PER_HYBRID)
    }:
        raise ValueError("hybrid case identities differ")

    records = []
    agreement = 0
    signal_changes = 0
    output_changes = 0
    leak_count = 0
    for case in cases:
        source = case.compiler_source_bytes()
        challenge = case.late_challenge_bytes()
        intervention_challenge = case.late_challenge_bytes(intervention=True)
        candidate_bytes = source + challenge + intervention_challenge
        leak_count += sum(
            token in candidate_bytes.lower() for token in _FORBIDDEN_CANDIDATE_TOKENS
        )
        if challenge == intervention_challenge:
            raise ValueError("hybrid intervention is byte-invariant")

        expected = execute_hybrid_case(case)
        intervention_expected = execute_hybrid_case(
            case,
            intervention=True,
        )
        independent = independent_hybrid_oracle(case)
        independent_intervention = independent_hybrid_oracle(
            case,
            intervention=True,
        )
        agreement += expected == independent
        agreement += intervention_expected == independent_intervention
        signal_changes += _coupling_signal(expected) != _coupling_signal(
            intervention_expected
        )
        output_changes += _semantic_output(expected) != _semantic_output(
            intervention_expected
        )
        material = {
            "kind": case.kind,
            "case_index": case.case_index,
            "source_sha256": _digest(source),
            "challenge_sha256": _digest(challenge),
            "intervention_challenge_sha256": _digest(intervention_challenge),
            "expected_sha256": _digest(expected),
            "intervention_expected_sha256": _digest(intervention_expected),
        }
        records.append(
            HybridCaseReceipt(
                kind=case.kind,
                case_index=case.case_index,
                source_sha256=material["source_sha256"],
                challenge_sha256=material["challenge_sha256"],
                intervention_challenge_sha256=material["intervention_challenge_sha256"],
                expected_sha256=material["expected_sha256"],
                intervention_expected_sha256=material["intervention_expected_sha256"],
                row_sha256=_digest(material),
            )
        )

    if agreement != TOTAL_EXECUTIONS:
        raise ValueError("independent hybrid oracle disagrees")
    if signal_changes != TOTAL_CASES:
        raise ValueError("hybrid intervention does not change coupling")
    if output_changes != TOTAL_CASES:
        raise ValueError("hybrid intervention does not change output")
    if leak_count:
        raise ValueError("candidate bytes contain assessor labels")
    row_hashes = {record.row_sha256 for record in records}
    if len(row_hashes) != TOTAL_CASES:
        raise ValueError("hybrid row hashes are not unique")
    payload_sha256 = _digest(
        [
            _canonicalize(record)
            for record in sorted(
                records,
                key=lambda item: item.row_sha256,
            )
        ]
    )
    receipt = HybridQualificationReceipt(
        hybrid_count=HYBRID_COUNT,
        cases_per_hybrid=CASES_PER_HYBRID,
        case_count=TOTAL_CASES,
        execution_count=TOTAL_EXECUTIONS,
        independent_oracle_agreement_count=agreement,
        causal_intervention_count=TOTAL_CASES,
        causal_signal_change_count=signal_changes,
        causal_output_change_count=output_changes,
        candidate_label_leak_count=leak_count,
        unique_row_count=len(row_hashes),
        payload_sha256=payload_sha256,
        claim_boundary=CLAIM_BOUNDARY,
        all_contracts_pass=True,
    )
    return tuple(records), receipt


@lru_cache(maxsize=1)
def build_hybrid_qualification_receipt() -> tuple[
    tuple[HybridCaseReceipt, ...],
    HybridQualificationReceipt,
]:
    return audit_hybrid_cases(build_hybrid_cases())


__all__ = [
    "ArithmeticRewriteResult",
    "ArithmeticRewriteSpec",
    "CASES_PER_HYBRID",
    "CLAIM_BOUNDARY",
    "HYBRID_COUNT",
    "HYBRID_ORDER",
    "HornResourceResult",
    "HornResourceSpec",
    "HybridCase",
    "HybridCaseReceipt",
    "HybridKind",
    "HybridQualificationReceipt",
    "ResourceHornResult",
    "ResourceHornSpec",
    "TOTAL_CASES",
    "TOTAL_EXECUTIONS",
    "audit_hybrid_cases",
    "build_hybrid_cases",
    "build_hybrid_qualification_receipt",
    "execute_hybrid_case",
    "independent_hybrid_oracle",
]
