"""Assessor-side semantic execution and query mechanics for ETTR-IL v2.

This module owns no candidate renderer, model, optimizer, checkpoint, or
training asset.  It reuses only the finite catalogs and immutable value types
from the three cross-ontology boards.  Primary execution and independent
replay are deliberately separate implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
from itertools import product
import json
import re
from typing import Any, TypeAlias

from cross_ontology_horn_board import (
    OBJECT_TYPES,
    PREDICATES,
    RULE_LIBRARY as HORN_RULE_LIBRARY,
    THEORIES as HORN_THEORIES,
    GroundAtom,
    all_ground_atoms,
)
from cross_ontology_resource_board import (
    OPERATOR_LIBRARY,
    OPERATOR_SYMBOL_COUNT,
    PLACE_SPECS,
    THEORIES as RESOURCE_THEORIES,
    Marking,
    ProcessStatus,
)
from cross_ontology_rewrite_board import (
    CONSTRUCTORS,
    RULE_LIBRARY as REWRITE_RULE_LIBRARY,
    THEORIES as REWRITE_THEORIES,
    GroundTerm,
    PatternTerm,
)


MASTER_SEED_PREIMAGE = (
    b"R12_ETTR_ISOLATED_LEARNABILITY_V2|2026-07-26|semantic-generator"
)
MASTER_SEED = hashlib.sha256(MASTER_SEED_PREIMAGE).digest()
MASTER_SEED_HEX = (
    "f6edaccd75ba80763540b990fcd0d1c85016e2d62a79cc3bbe328a206db925dd"
)
if MASTER_SEED.hex() != MASTER_SEED_HEX:
    raise RuntimeError("ETTR-IL v2 master seed differs")

V2_MIN_DEPTH = 1
V2_MAX_DEPTH = 6
V2_RESOURCE_MAX_DEPTH = 6
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_OPAQUE_SYMBOL = re.compile(r"^x[0-9a-f]{16}$")


class SemanticError(ValueError):
    """Base error for malformed or inadmissible assessor semantics."""


class SemanticAdmissionError(SemanticError):
    """A well-formed semantic object fails a causal admission invariant."""


class Ontology(StrEnum):
    HORN = "horn"
    REWRITE = "rewrite"
    RESOURCE = "resource"


class HornPolicy(StrEnum):
    PERSISTENT = "persistent"
    DERIVED_ONLY = "derived_only"


class RewritePolicy(StrEnum):
    CONTEXTUAL = "contextual"
    ROOT_ONLY = "root_only"


class ResourcePolicy(StrEnum):
    ATOMIC_DEADLOCK = "atomic_deadlock"
    SKIP_BLOCKED = "skip_blocked"


class TerminalDisposition(StrEnum):
    ANSWER = "ANSWER"
    ABSTAIN = "ABSTAIN"
    REJECT = "REJECT"


class StepOutcome(StrEnum):
    APPLIED = "applied"
    SKIPPED = "skipped"
    DEADLOCK = "deadlock"
    AMBIGUOUS = "ambiguous"
    REJECTED = "rejected"


class QueryOp(StrEnum):
    HORN_HAS = "horn_has"
    HORN_COUNT_GE = "horn_count_ge"
    REWRITE_ROOT_IS = "rewrite_root_is"
    REWRITE_CONTAINS = "rewrite_contains"
    REWRITE_NODES_GE = "rewrite_nodes_ge"
    REWRITE_CHILD_ROOT_IS = "rewrite_child_root_is"
    RESOURCE_PLACE_GE = "resource_place_ge"
    RESOURCE_CURSOR_GE = "resource_cursor_ge"
    RESOURCE_HALT = "resource_halt"


def _require_exact_int(value: object, name: str, lower: int, upper: int) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise SemanticError(f"{name} differs")
    return value


def _require_evidence_id(value: object) -> str:
    if type(value) is not str or _HEX_64.fullmatch(value) is None:
        raise SemanticError("evidence_id differs")
    return value


def _require_exact_tuple(value: object, name: str) -> tuple[Any, ...]:
    if type(value) is not tuple:
        raise SemanticError(f"{name} is not an exact tuple")
    return value


def _validate_ground_atom(atom: object) -> GroundAtom:
    if type(atom) is not GroundAtom:
        raise SemanticError("Horn atom type differs")
    predicate = _require_exact_int(
        atom.predicate,
        "Horn predicate",
        0,
        len(PREDICATES) - 1,
    )
    arguments = _require_exact_tuple(atom.arguments, "Horn arguments")
    required_types = PREDICATES[predicate].argument_types
    if len(arguments) != len(required_types):
        raise SemanticError("Horn atom arity differs")
    for argument, required_type in zip(
        arguments,
        required_types,
        strict=True,
    ):
        object_index = _require_exact_int(
            argument,
            "Horn object",
            0,
            len(OBJECT_TYPES) - 1,
        )
        if OBJECT_TYPES[object_index] != required_type:
            raise SemanticError("Horn atom typing differs")
    return atom


def _validate_ground_term(
    term: object,
    *,
    max_nodes: int | None = None,
) -> GroundTerm:
    if type(term) is not GroundTerm:
        raise SemanticError("rewrite term type differs")
    constructor_index = _require_exact_int(
        term.constructor_index,
        "rewrite constructor",
        0,
        len(CONSTRUCTORS) - 1,
    )
    constructor = CONSTRUCTORS[constructor_index]
    if type(term.type_index) is not int or term.type_index != constructor.result_type:
        raise SemanticError("rewrite term typing differs")
    children = _require_exact_tuple(term.children, "rewrite children")
    if len(children) != len(constructor.argument_types):
        raise SemanticError("rewrite term arity differs")
    for child, required_type in zip(
        children,
        constructor.argument_types,
        strict=True,
    ):
        validated = _validate_ground_term(child)
        if validated.type_index != required_type:
            raise SemanticError("rewrite child typing differs")
    if max_nodes is not None and term.node_count > max_nodes:
        raise SemanticError("rewrite world exceeds the bounded initial domain")
    return term


def _validate_marking(marking: object) -> Marking:
    if type(marking) is not Marking:
        raise SemanticError("resource marking type differs")
    values = _require_exact_tuple(
        marking.multiplicities,
        "resource multiplicities",
    )
    if len(values) != len(PLACE_SPECS):
        raise SemanticError("resource marking width differs")
    for value, place in zip(values, PLACE_SPECS, strict=True):
        _require_exact_int(
            value,
            "resource multiplicity",
            0,
            place.capacity,
        )
    return marking


def _validate_depth_operations(
    depth: object,
    operations: object,
    *,
    name: str,
) -> tuple[Any, ...]:
    _require_exact_int(depth, f"{name} depth", V2_MIN_DEPTH, V2_MAX_DEPTH)
    values = _require_exact_tuple(operations, f"{name} operations")
    if len(values) != depth:
        raise SemanticError(f"{name} depth and operation count differ")
    return values


@dataclass(frozen=True, slots=True)
class HornWorld:
    evidence_id: str
    theory_index: int
    initial: tuple[GroundAtom, ...]
    policy: HornPolicy

    def __post_init__(self) -> None:
        _require_evidence_id(self.evidence_id)
        _require_exact_int(
            self.theory_index,
            "Horn theory index",
            0,
            len(HORN_THEORIES) - 1,
        )
        initial = _require_exact_tuple(self.initial, "Horn initial state")
        if not initial or len(initial) > len(all_ground_atoms()):
            raise SemanticError("Horn initial state cardinality differs")
        for atom in initial:
            _validate_ground_atom(atom)
        if tuple(sorted(set(initial))) != initial:
            raise SemanticError("Horn initial state is not sorted unique")
        if type(self.policy) is not HornPolicy:
            raise SemanticError("Horn policy differs")


@dataclass(frozen=True, slots=True)
class HornCommand:
    depth: int
    operations: tuple[GroundAtom, ...]

    def __post_init__(self) -> None:
        operations = _validate_depth_operations(
            self.depth,
            self.operations,
            name="Horn command",
        )
        for atom in operations:
            _validate_ground_atom(atom)


@dataclass(frozen=True, slots=True)
class RewriteWorld:
    evidence_id: str
    theory_index: int
    initial: GroundTerm
    policy: RewritePolicy

    def __post_init__(self) -> None:
        _require_evidence_id(self.evidence_id)
        _require_exact_int(
            self.theory_index,
            "rewrite theory index",
            0,
            len(REWRITE_THEORIES) - 1,
        )
        initial = _validate_ground_term(self.initial, max_nodes=4)
        if initial.type_index != 0:
            raise SemanticError("rewrite command world root is not type zero")
        if type(self.policy) is not RewritePolicy:
            raise SemanticError("rewrite policy differs")


@dataclass(frozen=True, slots=True)
class RewriteCommand:
    depth: int
    operations: tuple[int, ...]

    def __post_init__(self) -> None:
        operations = _validate_depth_operations(
            self.depth,
            self.operations,
            name="rewrite command",
        )
        for operation in operations:
            _require_exact_int(operation, "rewrite operation", 0, 1)


@dataclass(frozen=True, slots=True)
class ResourceWorld:
    evidence_id: str
    theory_index: int
    initial: Marking
    policy: ResourcePolicy

    def __post_init__(self) -> None:
        _require_evidence_id(self.evidence_id)
        _require_exact_int(
            self.theory_index,
            "resource theory index",
            0,
            len(RESOURCE_THEORIES) - 1,
        )
        _validate_marking(self.initial)
        if type(self.policy) is not ResourcePolicy:
            raise SemanticError("resource policy differs")


@dataclass(frozen=True, slots=True)
class ResourceCommand:
    depth: int
    operations: tuple[int, ...]

    def __post_init__(self) -> None:
        operations = _validate_depth_operations(
            self.depth,
            self.operations,
            name="resource command",
        )
        if self.depth > V2_RESOURCE_MAX_DEPTH:
            raise SemanticError("resource command exceeds the v2 depth guard")
        for operation in operations:
            _require_exact_int(
                operation,
                "resource operation",
                0,
                OPERATOR_SYMBOL_COUNT - 1,
            )


@dataclass(frozen=True, slots=True)
class HornStep:
    index: int
    operation: GroundAtom
    before: tuple[GroundAtom, ...]
    after: tuple[GroundAtom, ...]
    outcome: StepOutcome
    prefix_dependent: bool


@dataclass(frozen=True, slots=True)
class RewriteSnapshot:
    index: int
    normal_forms: tuple[GroundTerm, ...]


@dataclass(frozen=True, slots=True)
class RewriteStep:
    index: int
    operation: int
    before: GroundTerm
    wrapped: GroundTerm
    normal_forms: tuple[GroundTerm, ...]
    outcome: StepOutcome
    prefix_dependent: bool


@dataclass(frozen=True, slots=True)
class ResourceStep:
    index: int
    operation: int
    before: Marking
    after: Marking
    cursor_before: int
    cursor_after: int
    outcome: StepOutcome
    prefix_dependent: bool


@dataclass(frozen=True, slots=True)
class HornExecution:
    world: HornWorld
    command: HornCommand
    snapshots: tuple[tuple[GroundAtom, ...], ...]
    steps: tuple[HornStep, ...]
    disposition: TerminalDisposition

    @property
    def terminal(self) -> tuple[GroundAtom, ...]:
        return self.snapshots[-1]


@dataclass(frozen=True, slots=True)
class RewriteExecution:
    world: RewriteWorld
    command: RewriteCommand
    snapshots: tuple[RewriteSnapshot, ...]
    steps: tuple[RewriteStep, ...]
    disposition: TerminalDisposition

    @property
    def terminal_normal_forms(self) -> tuple[GroundTerm, ...]:
        return self.snapshots[-1].normal_forms


@dataclass(frozen=True, slots=True)
class ResourceExecution:
    world: ResourceWorld
    command: ResourceCommand
    snapshots: tuple[Marking, ...]
    steps: tuple[ResourceStep, ...]
    cursor: int
    status: ProcessStatus
    disposition: TerminalDisposition

    @property
    def terminal(self) -> Marking:
        return self.snapshots[-1]


World: TypeAlias = HornWorld | RewriteWorld | ResourceWorld
Command: TypeAlias = HornCommand | RewriteCommand | ResourceCommand
Execution: TypeAlias = HornExecution | RewriteExecution | ResourceExecution


def _horn_rule_variable_types(rule_index: int) -> dict[int, int]:
    rule = HORN_RULE_LIBRARY[rule_index]
    result: dict[int, int] = {}
    for pattern in (*rule.premises, rule.conclusion):
        spec = PREDICATES[pattern.predicate]
        for variable, required_type in zip(
            pattern.variables,
            spec.argument_types,
            strict=True,
        ):
            previous = result.setdefault(variable, required_type)
            if previous != required_type:
                raise SemanticError("Horn rule variable typing differs")
    return result


def _ground_horn_pattern(pattern: Any, assignment: dict[int, int]) -> GroundAtom:
    return GroundAtom(
        pattern.predicate,
        tuple(assignment[variable] for variable in pattern.variables),
    )


def _horn_closure_primary(
    theory_index: int,
    initial: tuple[GroundAtom, ...],
) -> tuple[GroundAtom, ...]:
    facts = set(initial)
    theory = HORN_THEORIES[theory_index]
    while True:
        changed = False
        for rule_index in theory.rule_indices:
            rule = HORN_RULE_LIBRARY[rule_index]
            variable_types = _horn_rule_variable_types(rule_index)
            variables = tuple(sorted(variable_types))
            domains = tuple(
                tuple(
                    object_index
                    for object_index, object_type in enumerate(OBJECT_TYPES)
                    if object_type == variable_types[variable]
                )
                for variable in variables
            )
            for values in product(*domains):
                assignment = dict(zip(variables, values, strict=True))
                if all(
                    _ground_horn_pattern(premise, assignment) in facts
                    for premise in rule.premises
                ):
                    conclusion = _ground_horn_pattern(
                        rule.conclusion,
                        assignment,
                    )
                    if conclusion not in facts:
                        facts.add(conclusion)
                        changed = True
        if not changed:
            return tuple(sorted(facts))


def _horn_closure_replay(
    theory_index: int,
    initial: tuple[GroundAtom, ...],
) -> tuple[GroundAtom, ...]:
    implications: list[tuple[frozenset[GroundAtom], GroundAtom]] = []
    theory = HORN_THEORIES[theory_index]
    for rule_index in theory.rule_indices:
        rule = HORN_RULE_LIBRARY[rule_index]
        variable_types: dict[int, int] = {}
        for pattern in (*rule.premises, rule.conclusion):
            spec = PREDICATES[pattern.predicate]
            for variable, required_type in zip(
                pattern.variables,
                spec.argument_types,
                strict=True,
            ):
                previous = variable_types.setdefault(variable, required_type)
                if previous != required_type:
                    raise SemanticError("Horn replay rule typing differs")
        variables = tuple(sorted(variable_types))
        domains = tuple(
            tuple(
                object_index
                for object_index, object_type in enumerate(OBJECT_TYPES)
                if object_type == variable_types[variable]
            )
            for variable in variables
        )
        for values in product(*domains):
            assignment = dict(zip(variables, values, strict=True))
            implications.append(
                (
                    frozenset(
                        _ground_horn_pattern(premise, assignment)
                        for premise in rule.premises
                    ),
                    _ground_horn_pattern(rule.conclusion, assignment),
                )
            )
    facts = set(initial)
    pending = list(implications)
    while True:
        fired = False
        retained: list[tuple[frozenset[GroundAtom], GroundAtom]] = []
        for premises, conclusion in pending:
            if premises <= facts:
                if conclusion not in facts:
                    facts.add(conclusion)
                    fired = True
            else:
                retained.append((premises, conclusion))
        pending = retained
        if not fired:
            return tuple(sorted(facts))


def _project_horn_state(
    policy: HornPolicy,
    closure: tuple[GroundAtom, ...],
    asserted: frozenset[GroundAtom],
) -> tuple[GroundAtom, ...]:
    if policy == HornPolicy.PERSISTENT:
        return closure
    return tuple(atom for atom in closure if atom not in asserted)


def _execute_horn_with(
    world: HornWorld,
    command: HornCommand,
    closure_helper: Any,
    *,
    require_dependent: bool,
) -> HornExecution:
    full_state = tuple(world.initial)
    asserted = frozenset(world.initial)
    snapshots: list[tuple[GroundAtom, ...]] = [tuple(world.initial)]
    steps: list[HornStep] = []
    for index, operation in enumerate(command.operations, start=1):
        if operation in full_state:
            raise SemanticAdmissionError(
                f"Horn operation {index} is already present"
            )
        asserted = asserted | {operation}
        full_state = closure_helper(
            world.theory_index,
            tuple(sorted((*full_state, operation))),
        )
        after = _project_horn_state(world.policy, full_state, asserted)
        if after == snapshots[-1]:
            raise SemanticAdmissionError(
                f"Horn operation {index} changes no semantic state"
            )
        direct_full = closure_helper(
            world.theory_index,
            tuple(sorted((*world.initial, operation))),
        )
        direct_asserted = frozenset((*world.initial, operation))
        direct = _project_horn_state(
            world.policy,
            direct_full,
            direct_asserted,
        )
        dependent = index == 1 or direct != after
        if require_dependent and not dependent:
            raise SemanticAdmissionError(
                f"Horn operation {index} is prefix independent"
            )
        steps.append(
            HornStep(
                index=index,
                operation=operation,
                before=snapshots[-1],
                after=after,
                outcome=StepOutcome.APPLIED,
                prefix_dependent=dependent,
            )
        )
        snapshots.append(after)
    return HornExecution(
        world=world,
        command=command,
        snapshots=tuple(snapshots),
        steps=tuple(steps),
        disposition=TerminalDisposition.ANSWER,
    )


def execute_horn(
    world: HornWorld,
    command: HornCommand,
    *,
    require_dependent: bool = True,
) -> HornExecution:
    """Execute Horn command semantics with iterative rule saturation."""

    return _execute_horn_with(
        world,
        command,
        _horn_closure_primary,
        require_dependent=require_dependent,
    )


def replay_horn(
    world: HornWorld,
    command: HornCommand,
    *,
    require_dependent: bool = True,
) -> HornExecution:
    """Replay Horn semantics after independently grounding all implications."""

    full_state = tuple(world.initial)
    asserted = frozenset(world.initial)
    snapshots: list[tuple[GroundAtom, ...]] = [tuple(world.initial)]
    steps: list[HornStep] = []

    def project_state(
        closure: tuple[GroundAtom, ...],
        asserted_facts: frozenset[GroundAtom],
    ) -> tuple[GroundAtom, ...]:
        if world.policy == HornPolicy.PERSISTENT:
            return closure
        return tuple(
            atom for atom in closure if atom not in asserted_facts
        )

    for index, operation in enumerate(command.operations, start=1):
        if operation in full_state:
            raise SemanticAdmissionError(
                f"Horn operation {index} is already present"
            )
        asserted = asserted.union((operation,))
        full_state = _horn_closure_replay(
            world.theory_index,
            tuple(sorted((*full_state, operation))),
        )
        after = project_state(full_state, asserted)
        if after == snapshots[-1]:
            raise SemanticAdmissionError(
                f"Horn operation {index} changes no semantic state"
            )
        direct_asserted = frozenset((*world.initial, operation))
        direct = project_state(
            _horn_closure_replay(
                world.theory_index,
                tuple(sorted((*world.initial, operation))),
            ),
            direct_asserted,
        )
        dependent = index == 1 or direct != after
        if require_dependent and not dependent:
            raise SemanticAdmissionError(
                f"Horn operation {index} is prefix independent"
            )
        steps.append(
            HornStep(
                index=index,
                operation=operation,
                before=snapshots[-1],
                after=after,
                outcome=StepOutcome.APPLIED,
                prefix_dependent=dependent,
            )
        )
        snapshots.append(after)
    return HornExecution(
        world=world,
        command=command,
        snapshots=tuple(snapshots),
        steps=tuple(steps),
        disposition=TerminalDisposition.ANSWER,
    )


def _rewrite_match_primary(
    pattern: PatternTerm,
    term: GroundTerm,
    bindings: dict[int, GroundTerm],
) -> bool:
    if pattern.type_index != term.type_index:
        return False
    if pattern.variable_index is not None:
        previous = bindings.setdefault(pattern.variable_index, term)
        return previous == term
    if (
        pattern.constructor_index != term.constructor_index
        or len(pattern.children) != len(term.children)
    ):
        return False
    return all(
        _rewrite_match_primary(child_pattern, child, bindings)
        for child_pattern, child in zip(
            pattern.children,
            term.children,
            strict=True,
        )
    )


def _rewrite_instantiate_primary(
    pattern: PatternTerm,
    bindings: dict[int, GroundTerm],
) -> GroundTerm:
    if pattern.variable_index is not None:
        return bindings[pattern.variable_index]
    assert pattern.constructor_index is not None
    return GroundTerm(
        pattern.type_index,
        pattern.constructor_index,
        tuple(
            _rewrite_instantiate_primary(child, bindings)
            for child in pattern.children
        ),
    )


def _rewrite_occurrence_paths(term: GroundTerm) -> tuple[tuple[int, ...], ...]:
    paths: list[tuple[int, ...]] = [()]
    for child_index, child in enumerate(term.children):
        paths.extend(
            (child_index, *suffix)
            for suffix in _rewrite_occurrence_paths(child)
        )
    return tuple(paths)


def _rewrite_at_path(term: GroundTerm, path: tuple[int, ...]) -> GroundTerm:
    result = term
    for child_index in path:
        result = result.children[child_index]
    return result


def _rewrite_replace_path(
    term: GroundTerm,
    path: tuple[int, ...],
    replacement: GroundTerm,
) -> GroundTerm:
    if not path:
        return replacement
    child_index = path[0]
    children = list(term.children)
    children[child_index] = _rewrite_replace_path(
        children[child_index],
        path[1:],
        replacement,
    )
    return GroundTerm(
        term.type_index,
        term.constructor_index,
        tuple(children),
    )


def _rewrite_reducts_primary(
    theory_index: int,
    term: GroundTerm,
    policy: RewritePolicy,
) -> tuple[GroundTerm, ...]:
    paths = ((),) if policy == RewritePolicy.ROOT_ONLY else _rewrite_occurrence_paths(term)
    reducts: set[GroundTerm] = set()
    for path in paths:
        redex = _rewrite_at_path(term, path)
        for rule_index in REWRITE_THEORIES[theory_index].rule_indices:
            rule = REWRITE_RULE_LIBRARY[rule_index]
            bindings: dict[int, GroundTerm] = {}
            if _rewrite_match_primary(rule.lhs, redex, bindings):
                replacement = _rewrite_instantiate_primary(
                    rule.rhs,
                    bindings,
                )
                reduct = _rewrite_replace_path(term, path, replacement)
                if reduct.node_count >= term.node_count:
                    raise SemanticError("rewrite does not structurally decrease")
                reducts.add(reduct)
    return tuple(sorted(reducts))


def _rewrite_normal_forms_primary(
    theory_index: int,
    initial: GroundTerm,
    policy: RewritePolicy,
) -> tuple[GroundTerm, ...]:
    frontier = [initial]
    visited = {initial}
    terminals: set[GroundTerm] = set()
    while frontier:
        state = frontier.pop(0)
        reducts = _rewrite_reducts_primary(theory_index, state, policy)
        if not reducts:
            terminals.add(state)
        for reduct in reducts:
            if reduct not in visited:
                visited.add(reduct)
                frontier.append(reduct)
    return tuple(sorted(terminals))


def _rewrite_root_reduct_replay(
    pattern: PatternTerm,
    replacement: PatternTerm,
    term: GroundTerm,
) -> GroundTerm | None:
    stack = [(pattern, term)]
    bindings: dict[int, GroundTerm] = {}
    while stack:
        expected, observed = stack.pop()
        if expected.type_index != observed.type_index:
            return None
        if expected.variable_index is not None:
            previous = bindings.get(expected.variable_index)
            if previous is not None and previous != observed:
                return None
            bindings[expected.variable_index] = observed
            continue
        if (
            expected.constructor_index != observed.constructor_index
            or len(expected.children) != len(observed.children)
        ):
            return None
        stack.extend(
            zip(expected.children, observed.children, strict=True)
        )

    def instantiate(node: PatternTerm) -> GroundTerm:
        if node.variable_index is not None:
            return bindings[node.variable_index]
        assert node.constructor_index is not None
        return GroundTerm(
            node.type_index,
            node.constructor_index,
            tuple(instantiate(child) for child in node.children),
        )

    return instantiate(replacement)


def _rewrite_reducts_replay(
    theory_index: int,
    term: GroundTerm,
    policy: RewritePolicy,
) -> tuple[GroundTerm, ...]:
    reducts: set[GroundTerm] = set()
    for rule_index in REWRITE_THEORIES[theory_index].rule_indices:
        rule = REWRITE_RULE_LIBRARY[rule_index]
        reduct = _rewrite_root_reduct_replay(rule.lhs, rule.rhs, term)
        if reduct is not None:
            if reduct.node_count >= term.node_count:
                raise SemanticError("replay rewrite does not decrease")
            reducts.add(reduct)
    if policy == RewritePolicy.CONTEXTUAL:
        for child_index, child in enumerate(term.children):
            for child_reduct in _rewrite_reducts_replay(
                theory_index,
                child,
                policy,
            ):
                children = list(term.children)
                children[child_index] = child_reduct
                reducts.add(
                    GroundTerm(
                        term.type_index,
                        term.constructor_index,
                        tuple(children),
                    )
                )
    return tuple(sorted(reducts))


def _rewrite_normal_forms_replay(
    theory_index: int,
    initial: GroundTerm,
    policy: RewritePolicy,
    memo: dict[GroundTerm, tuple[GroundTerm, ...]] | None = None,
) -> tuple[GroundTerm, ...]:
    cache = {} if memo is None else memo
    if initial in cache:
        return cache[initial]
    reducts = _rewrite_reducts_replay(theory_index, initial, policy)
    if not reducts:
        cache[initial] = (initial,)
        return cache[initial]
    terminals: set[GroundTerm] = set()
    for reduct in reducts:
        terminals.update(
            _rewrite_normal_forms_replay(
                theory_index,
                reduct,
                policy,
                cache,
            )
        )
    cache[initial] = tuple(sorted(terminals))
    return cache[initial]


def _wrap_rewrite(term: GroundTerm, constructor_index: int) -> GroundTerm:
    _validate_ground_term(term)
    return GroundTerm(
        0,
        5,
        (term, GroundTerm(0, constructor_index, ())),
    )


def _execute_rewrite_with(
    world: RewriteWorld,
    command: RewriteCommand,
    normal_form_helper: Any,
    *,
    require_dependent: bool,
) -> RewriteExecution:
    current = world.initial
    snapshots = [RewriteSnapshot(0, (current,))]
    steps: list[RewriteStep] = []
    disposition = TerminalDisposition.ANSWER
    for index, operation in enumerate(command.operations, start=1):
        wrapped = _wrap_rewrite(current, operation)
        normal_forms = normal_form_helper(
            world.theory_index,
            wrapped,
            world.policy,
        )
        if not normal_forms:
            steps.append(
                RewriteStep(
                    index,
                    operation,
                    current,
                    wrapped,
                    (),
                    StepOutcome.REJECTED,
                    False,
                )
            )
            snapshots.append(RewriteSnapshot(index, ()))
            disposition = TerminalDisposition.REJECT
            break
        if len(normal_forms) > 1:
            steps.append(
                RewriteStep(
                    index,
                    operation,
                    current,
                    wrapped,
                    normal_forms,
                    StepOutcome.AMBIGUOUS,
                    False,
                )
            )
            snapshots.append(RewriteSnapshot(index, normal_forms))
            disposition = TerminalDisposition.ABSTAIN
            break
        successor = normal_forms[0]
        if successor == current:
            raise SemanticAdmissionError(
                f"rewrite operation {index} changes no semantic state"
            )
        direct = normal_form_helper(
            world.theory_index,
            _wrap_rewrite(world.initial, operation),
            world.policy,
        )
        dependent = index == 1 or direct != (successor,)
        if require_dependent and not dependent:
            raise SemanticAdmissionError(
                f"rewrite operation {index} is prefix independent"
            )
        steps.append(
            RewriteStep(
                index,
                operation,
                current,
                wrapped,
                normal_forms,
                StepOutcome.APPLIED,
                dependent,
            )
        )
        snapshots.append(RewriteSnapshot(index, normal_forms))
        current = successor
    return RewriteExecution(
        world=world,
        command=command,
        snapshots=tuple(snapshots),
        steps=tuple(steps),
        disposition=disposition,
    )


def execute_rewrite(
    world: RewriteWorld,
    command: RewriteCommand,
    *,
    require_dependent: bool = True,
) -> RewriteExecution:
    """Execute rewrite commands with occurrence-path breadth-first search."""

    return _execute_rewrite_with(
        world,
        command,
        _rewrite_normal_forms_primary,
        require_dependent=require_dependent,
    )


def replay_rewrite(
    world: RewriteWorld,
    command: RewriteCommand,
    *,
    require_dependent: bool = True,
) -> RewriteExecution:
    """Replay rewrite commands with recursive root/child reduction."""

    current = world.initial
    snapshots = [RewriteSnapshot(0, (current,))]
    steps: list[RewriteStep] = []
    disposition = TerminalDisposition.ANSWER
    for index, operation in enumerate(command.operations, start=1):
        wrapped = _wrap_rewrite(current, operation)
        normal_forms = _rewrite_normal_forms_replay(
            world.theory_index,
            wrapped,
            world.policy,
        )
        if not normal_forms:
            steps.append(
                RewriteStep(
                    index=index,
                    operation=operation,
                    before=current,
                    wrapped=wrapped,
                    normal_forms=(),
                    outcome=StepOutcome.REJECTED,
                    prefix_dependent=False,
                )
            )
            snapshots.append(RewriteSnapshot(index, ()))
            disposition = TerminalDisposition.REJECT
            break
        if len(normal_forms) > 1:
            steps.append(
                RewriteStep(
                    index=index,
                    operation=operation,
                    before=current,
                    wrapped=wrapped,
                    normal_forms=normal_forms,
                    outcome=StepOutcome.AMBIGUOUS,
                    prefix_dependent=False,
                )
            )
            snapshots.append(RewriteSnapshot(index, normal_forms))
            disposition = TerminalDisposition.ABSTAIN
            break
        successor = normal_forms[0]
        if successor == current:
            raise SemanticAdmissionError(
                f"rewrite operation {index} changes no semantic state"
            )
        direct = _rewrite_normal_forms_replay(
            world.theory_index,
            _wrap_rewrite(world.initial, operation),
            world.policy,
        )
        dependent = index == 1 or direct != (successor,)
        if require_dependent and not dependent:
            raise SemanticAdmissionError(
                f"rewrite operation {index} is prefix independent"
            )
        steps.append(
            RewriteStep(
                index=index,
                operation=operation,
                before=current,
                wrapped=wrapped,
                normal_forms=normal_forms,
                outcome=StepOutcome.APPLIED,
                prefix_dependent=dependent,
            )
        )
        snapshots.append(RewriteSnapshot(index, normal_forms))
        current = successor
    return RewriteExecution(
        world=world,
        command=command,
        snapshots=tuple(snapshots),
        steps=tuple(steps),
        disposition=disposition,
    )


def _resource_transition_primary(
    theory_index: int,
    marking: Marking,
    symbol: int,
) -> Marking | None:
    theory = RESOURCE_THEORIES[theory_index]
    operator = OPERATOR_LIBRARY[theory.operator_indices[symbol]]
    counts = {
        place.index: marking.multiplicities[place.index]
        for place in PLACE_SPECS
    }
    guards = {
        quantity.place: quantity.multiplicity
        for quantity in operator.guards
    }
    consumes = {
        quantity.place: quantity.multiplicity
        for quantity in operator.consumes
    }
    produces = {
        quantity.place: quantity.multiplicity
        for quantity in operator.produces
    }
    if any(
        counts[place] < required
        for place, required in (*guards.items(), *consumes.items())
    ):
        return None
    successor = dict(counts)
    for place, amount in consumes.items():
        successor[place] -= amount
    for place, amount in produces.items():
        successor[place] += amount
    if any(
        not 0 <= successor[place.index] <= place.capacity
        for place in PLACE_SPECS
    ):
        return None
    return Marking(
        tuple(successor[index] for index in range(len(PLACE_SPECS)))
    )


def _resource_transition_replay(
    theory_index: int,
    marking: Marking,
    symbol: int,
) -> Marking | None:
    operator_index = RESOURCE_THEORIES[theory_index].operator_indices[symbol]
    operator = OPERATOR_LIBRARY[operator_index]
    width = len(PLACE_SPECS)
    minimum = [0] * width
    debit = [0] * width
    credit = [0] * width
    for quantity in operator.guards:
        minimum[quantity.place] = max(
            minimum[quantity.place],
            quantity.multiplicity,
        )
    for quantity in operator.consumes:
        debit[quantity.place] = quantity.multiplicity
        minimum[quantity.place] = max(
            minimum[quantity.place],
            quantity.multiplicity,
        )
    for quantity in operator.produces:
        credit[quantity.place] = quantity.multiplicity
    if any(
        available < required
        for available, required in zip(
            marking.multiplicities,
            minimum,
            strict=True,
        )
    ):
        return None
    successor = tuple(
        available - consumed + produced
        for available, consumed, produced in zip(
            marking.multiplicities,
            debit,
            credit,
            strict=True,
        )
    )
    if any(
        value < 0 or value > place.capacity
        for value, place in zip(successor, PLACE_SPECS, strict=True)
    ):
        return None
    return Marking(successor)


def _execute_resource_with(
    world: ResourceWorld,
    command: ResourceCommand,
    transition_helper: Any,
    *,
    require_dependent: bool,
) -> ResourceExecution:
    current = world.initial
    snapshots = [current]
    steps: list[ResourceStep] = []
    cursor = 0
    status = ProcessStatus.HALT
    for index, operation in enumerate(command.operations, start=1):
        successor = transition_helper(
            world.theory_index,
            current,
            operation,
        )
        if successor is None:
            if world.policy == ResourcePolicy.ATOMIC_DEADLOCK:
                steps.append(
                    ResourceStep(
                        index=index,
                        operation=operation,
                        before=current,
                        after=current,
                        cursor_before=index - 1,
                        cursor_after=index - 1,
                        outcome=StepOutcome.DEADLOCK,
                        prefix_dependent=False,
                    )
                )
                snapshots.append(current)
                cursor = index - 1
                status = ProcessStatus.DEADLOCK
                break
            steps.append(
                ResourceStep(
                    index=index,
                    operation=operation,
                    before=current,
                    after=current,
                    cursor_before=index - 1,
                    cursor_after=index,
                    outcome=StepOutcome.SKIPPED,
                    prefix_dependent=False,
                )
            )
            snapshots.append(current)
            cursor = index
            continue
        if successor == current:
            raise SemanticAdmissionError(
                f"resource operation {index} changes no marking"
            )
        direct = transition_helper(
            world.theory_index,
            world.initial,
            operation,
        )
        dependent = index == 1 or direct != successor
        if require_dependent and not dependent:
            raise SemanticAdmissionError(
                f"resource operation {index} is prefix independent"
            )
        steps.append(
            ResourceStep(
                index=index,
                operation=operation,
                before=current,
                after=successor,
                cursor_before=index - 1,
                cursor_after=index,
                outcome=StepOutcome.APPLIED,
                prefix_dependent=dependent,
            )
        )
        snapshots.append(successor)
        current = successor
        cursor = index
    else:
        cursor = command.depth
        status = ProcessStatus.HALT
    return ResourceExecution(
        world=world,
        command=command,
        snapshots=tuple(snapshots),
        steps=tuple(steps),
        cursor=cursor,
        status=status,
        disposition=TerminalDisposition.ANSWER,
    )


def execute_resource(
    world: ResourceWorld,
    command: ResourceCommand,
    *,
    require_dependent: bool = True,
) -> ResourceExecution:
    """Execute resource commands through the independent v2 depth-six guard."""

    return _execute_resource_with(
        world,
        command,
        _resource_transition_primary,
        require_dependent=require_dependent,
    )


def replay_resource(
    world: ResourceWorld,
    command: ResourceCommand,
    *,
    require_dependent: bool = True,
) -> ResourceExecution:
    """Replay resource commands as bounded vector algebra."""

    current = world.initial
    snapshots = [current]
    steps: list[ResourceStep] = []
    cursor = 0
    status = ProcessStatus.HALT
    for index, operation in enumerate(command.operations, start=1):
        successor = _resource_transition_replay(
            world.theory_index,
            current,
            operation,
        )
        if successor is None:
            if world.policy == ResourcePolicy.ATOMIC_DEADLOCK:
                steps.append(
                    ResourceStep(
                        index=index,
                        operation=operation,
                        before=current,
                        after=current,
                        cursor_before=index - 1,
                        cursor_after=index - 1,
                        outcome=StepOutcome.DEADLOCK,
                        prefix_dependent=False,
                    )
                )
                snapshots.append(current)
                cursor = index - 1
                status = ProcessStatus.DEADLOCK
                break
            steps.append(
                ResourceStep(
                    index=index,
                    operation=operation,
                    before=current,
                    after=current,
                    cursor_before=index - 1,
                    cursor_after=index,
                    outcome=StepOutcome.SKIPPED,
                    prefix_dependent=False,
                )
            )
            snapshots.append(current)
            cursor = index
            continue
        if successor == current:
            raise SemanticAdmissionError(
                f"resource operation {index} changes no marking"
            )
        direct = _resource_transition_replay(
            world.theory_index,
            world.initial,
            operation,
        )
        dependent = index == 1 or direct != successor
        if require_dependent and not dependent:
            raise SemanticAdmissionError(
                f"resource operation {index} is prefix independent"
            )
        steps.append(
            ResourceStep(
                index=index,
                operation=operation,
                before=current,
                after=successor,
                cursor_before=index - 1,
                cursor_after=index,
                outcome=StepOutcome.APPLIED,
                prefix_dependent=dependent,
            )
        )
        snapshots.append(successor)
        current = successor
        cursor = index
    else:
        cursor = command.depth
        status = ProcessStatus.HALT
    return ResourceExecution(
        world=world,
        command=command,
        snapshots=tuple(snapshots),
        steps=tuple(steps),
        cursor=cursor,
        status=status,
        disposition=TerminalDisposition.ANSWER,
    )


def execute_semantics(
    world: World,
    command: Command,
    *,
    require_dependent: bool = True,
) -> Execution:
    """Dispatch a matched world/command pair to the primary executor."""

    if type(world) is HornWorld and type(command) is HornCommand:
        return execute_horn(
            world,
            command,
            require_dependent=require_dependent,
        )
    if type(world) is RewriteWorld and type(command) is RewriteCommand:
        return execute_rewrite(
            world,
            command,
            require_dependent=require_dependent,
        )
    if type(world) is ResourceWorld and type(command) is ResourceCommand:
        return execute_resource(
            world,
            command,
            require_dependent=require_dependent,
        )
    raise SemanticError("world and command ontologies differ")


def replay_semantics(
    world: World,
    command: Command,
    *,
    require_dependent: bool = True,
) -> Execution:
    """Dispatch a matched world/command pair to the independent replay."""

    if type(world) is HornWorld and type(command) is HornCommand:
        return replay_horn(
            world,
            command,
            require_dependent=require_dependent,
        )
    if type(world) is RewriteWorld and type(command) is RewriteCommand:
        return replay_rewrite(
            world,
            command,
            require_dependent=require_dependent,
        )
    if type(world) is ResourceWorld and type(command) is ResourceCommand:
        return replay_resource(
            world,
            command,
            require_dependent=require_dependent,
        )
    raise SemanticError("world and command ontologies differ")


_QUERY_ONTOLOGY = {
    QueryOp.HORN_HAS: Ontology.HORN,
    QueryOp.HORN_COUNT_GE: Ontology.HORN,
    QueryOp.REWRITE_ROOT_IS: Ontology.REWRITE,
    QueryOp.REWRITE_CONTAINS: Ontology.REWRITE,
    QueryOp.REWRITE_NODES_GE: Ontology.REWRITE,
    QueryOp.REWRITE_CHILD_ROOT_IS: Ontology.REWRITE,
    QueryOp.RESOURCE_PLACE_GE: Ontology.RESOURCE,
    QueryOp.RESOURCE_CURSOR_GE: Ontology.RESOURCE,
    QueryOp.RESOURCE_HALT: Ontology.RESOURCE,
}


@dataclass(frozen=True, order=True, slots=True)
class SemanticQuery:
    op: QueryOp
    args: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.op) is not QueryOp:
            raise SemanticError("query operation differs")
        args = _require_exact_tuple(self.args, "query arguments")
        for argument in args:
            if type(argument) is not int:
                raise SemanticError("query argument type differs")
        if self.op == QueryOp.HORN_HAS:
            if len(args) not in {2, 3}:
                raise SemanticError("horn_has query arity differs")
            _validate_ground_atom(GroundAtom(args[0], args[1:]))
        elif self.op == QueryOp.HORN_COUNT_GE:
            if len(args) != 1 or not 1 <= args[0] <= 27:
                raise SemanticError("horn_count_ge query differs")
        elif self.op in {
            QueryOp.REWRITE_ROOT_IS,
            QueryOp.REWRITE_CONTAINS,
        }:
            if len(args) != 1 or not 0 <= args[0] <= 7:
                raise SemanticError("rewrite constructor query differs")
        elif self.op == QueryOp.REWRITE_NODES_GE:
            if len(args) != 1 or not 1 <= args[0] <= 16:
                raise SemanticError("rewrite_nodes_ge query differs")
        elif self.op == QueryOp.REWRITE_CHILD_ROOT_IS:
            if (
                len(args) != 2
                or not 0 <= args[0] <= 1
                or not 0 <= args[1] <= 7
            ):
                raise SemanticError("rewrite_child_root_is query differs")
        elif self.op == QueryOp.RESOURCE_PLACE_GE:
            if (
                len(args) != 2
                or not 0 <= args[0] <= 3
                or not 1 <= args[1] <= 3
            ):
                raise SemanticError("resource_place_ge query differs")
        elif self.op == QueryOp.RESOURCE_CURSOR_GE:
            if len(args) != 1 or not 1 <= args[0] <= 6:
                raise SemanticError("resource_cursor_ge query differs")
        elif self.op == QueryOp.RESOURCE_HALT:
            if args:
                raise SemanticError("resource_halt query has arguments")

    @property
    def ontology(self) -> Ontology:
        return _QUERY_ONTOLOGY[self.op]

    def assessor_value(self) -> dict[str, object]:
        """Return the sealed assessor representation, never candidate bytes."""

        return {"args": list(self.args), "op": self.op.value}


def enumerate_queries(ontology: Ontology) -> tuple[SemanticQuery, ...]:
    """Enumerate the exact finite query grammar in normative order."""

    if type(ontology) is not Ontology:
        raise SemanticError("query ontology differs")
    if ontology == Ontology.HORN:
        return (
            *(
                SemanticQuery(
                    QueryOp.HORN_HAS,
                    (atom.predicate, *atom.arguments),
                )
                for atom in all_ground_atoms()
            ),
            *(
                SemanticQuery(QueryOp.HORN_COUNT_GE, (threshold,))
                for threshold in range(1, 28)
            ),
        )
    if ontology == Ontology.REWRITE:
        return (
            *(
                SemanticQuery(QueryOp.REWRITE_ROOT_IS, (constructor,))
                for constructor in range(8)
            ),
            *(
                SemanticQuery(QueryOp.REWRITE_CONTAINS, (constructor,))
                for constructor in range(8)
            ),
            *(
                SemanticQuery(QueryOp.REWRITE_NODES_GE, (threshold,))
                for threshold in range(1, 17)
            ),
            *(
                SemanticQuery(
                    QueryOp.REWRITE_CHILD_ROOT_IS,
                    (position, constructor),
                )
                for position in range(2)
                for constructor in range(8)
            ),
        )
    return (
        *(
            SemanticQuery(
                QueryOp.RESOURCE_PLACE_GE,
                (place, threshold),
            )
            for place in range(4)
            for threshold in range(1, 4)
        ),
        *(
            SemanticQuery(QueryOp.RESOURCE_CURSOR_GE, (threshold,))
            for threshold in range(1, 7)
        ),
        SemanticQuery(QueryOp.RESOURCE_HALT, ()),
    )


def _term_contains(term: GroundTerm, constructor: int) -> bool:
    return term.constructor_index == constructor or any(
        _term_contains(child, constructor) for child in term.children
    )


def evaluate_query(query: SemanticQuery, execution: Execution) -> bool:
    """Evaluate one sealed Boolean query on a terminal semantic execution."""

    if execution.disposition != TerminalDisposition.ANSWER:
        raise SemanticAdmissionError("query cannot score a non-ANSWER result")
    if query.ontology == Ontology.HORN:
        if type(execution) is not HornExecution:
            raise SemanticError("Horn query and execution differ")
        if query.op == QueryOp.HORN_HAS:
            atom = GroundAtom(query.args[0], query.args[1:])
            return atom in execution.terminal
        return len(execution.terminal) >= query.args[0]
    if query.ontology == Ontology.REWRITE:
        if type(execution) is not RewriteExecution:
            raise SemanticError("rewrite query and execution differ")
        normal_forms = execution.terminal_normal_forms
        if len(normal_forms) != 1:
            raise SemanticAdmissionError(
                "rewrite query lacks a singleton terminal"
            )
        term = normal_forms[0]
        if query.op == QueryOp.REWRITE_ROOT_IS:
            return term.constructor_index == query.args[0]
        if query.op == QueryOp.REWRITE_CONTAINS:
            return _term_contains(term, query.args[0])
        if query.op == QueryOp.REWRITE_NODES_GE:
            return term.node_count >= query.args[0]
        position, constructor = query.args
        return (
            position < len(term.children)
            and term.children[position].constructor_index == constructor
        )
    if type(execution) is not ResourceExecution:
        raise SemanticError("resource query and execution differ")
    if query.op == QueryOp.RESOURCE_PLACE_GE:
        place, threshold = query.args
        return execution.terminal.multiplicities[place] >= threshold
    if query.op == QueryOp.RESOURCE_CURSOR_GE:
        return execution.cursor >= query.args[0]
    return execution.status == ProcessStatus.HALT


def _execution_ontology(execution: Execution) -> Ontology:
    if type(execution) is HornExecution:
        return Ontology.HORN
    if type(execution) is RewriteExecution:
        return Ontology.REWRITE
    if type(execution) is ResourceExecution:
        return Ontology.RESOURCE
    raise SemanticError("execution type differs")


@dataclass(frozen=True, slots=True)
class SemanticRectangle:
    """Four cells ordered W0C0, W0C1, W1C0, W1C1."""

    cells: tuple[Execution, Execution, Execution, Execution]

    def __post_init__(self) -> None:
        cells = _require_exact_tuple(self.cells, "semantic rectangle cells")
        if len(cells) != 4:
            raise SemanticError("semantic rectangle does not have four cells")
        ontology = _execution_ontology(cells[0])
        if any(_execution_ontology(cell) != ontology for cell in cells):
            raise SemanticError("semantic rectangle mixes ontologies")
        if any(
            cell.disposition != TerminalDisposition.ANSWER
            for cell in cells
        ):
            raise SemanticAdmissionError(
                "semantic rectangle includes a non-ANSWER cell"
            )

    @property
    def ontology(self) -> Ontology:
        return _execution_ontology(self.cells[0])


CHECKERBOARD_PATTERNS = (
    (False, True, True, False),
    (True, False, False, True),
)


def checkerboard_labels(
    query: SemanticQuery,
    rectangle: SemanticRectangle,
) -> tuple[bool, bool, bool, bool]:
    """Require exact all-edge contrast and 2/2 balance."""

    if type(rectangle) is not SemanticRectangle:
        raise SemanticError("checkerboard rectangle type differs")
    if query.ontology != rectangle.ontology:
        raise SemanticError("checkerboard query ontology differs")
    labels = tuple(evaluate_query(query, cell) for cell in rectangle.cells)
    if labels not in CHECKERBOARD_PATTERNS:
        raise SemanticAdmissionError(
            f"query labels are not a strict checkerboard: {labels!r}"
        )
    return labels


def admissible_checkerboard_queries(
    rectangle: SemanticRectangle,
) -> tuple[SemanticQuery, ...]:
    """Return every finite query that passes strict checkerboard admission."""

    admitted: list[SemanticQuery] = []
    for query in enumerate_queries(rectangle.ontology):
        try:
            checkerboard_labels(query, rectangle)
        except SemanticAdmissionError:
            continue
        admitted.append(query)
    return tuple(admitted)


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _query_rank(query: SemanticQuery, slot: int) -> bytes:
    _require_exact_int(slot, "query slot", 0, 1)
    return hashlib.sha256(
        MASTER_SEED
        + f"|query|{slot}|".encode("ascii")
        + _canonical_json_bytes(query.assessor_value())
    ).digest()


def query_denotation_signature(
    query: SemanticQuery,
    bounded_terminal_universe: tuple[Execution, ...],
) -> tuple[bool, ...]:
    """Evaluate a query on a caller-frozen complete bounded universe."""

    universe = _require_exact_tuple(
        bounded_terminal_universe,
        "bounded terminal universe",
    )
    if not universe:
        raise SemanticAdmissionError("bounded terminal universe is empty")
    if any(
        _execution_ontology(execution) != query.ontology
        or execution.disposition != TerminalDisposition.ANSWER
        for execution in universe
    ):
        raise SemanticAdmissionError(
            "bounded terminal universe is mixed or non-ANSWER"
        )
    return tuple(evaluate_query(query, execution) for execution in universe)


@dataclass(frozen=True, slots=True)
class SelectedQueries:
    slot_0: SemanticQuery
    slot_1: SemanticQuery
    slot_0_labels: tuple[bool, bool, bool, bool]
    slot_1_labels: tuple[bool, bool, bool, bool]
    slot_0_denotation: tuple[bool, ...]
    slot_1_denotation: tuple[bool, ...]


def select_queries(
    rectangle: SemanticRectangle,
    *,
    bounded_terminal_universe: tuple[Execution, ...],
) -> SelectedQueries:
    """Select two checkerboards with globally non-equivalent denotations."""

    admitted = admissible_checkerboard_queries(rectangle)
    if not admitted:
        raise SemanticAdmissionError(
            "semantic rectangle has no checkerboard query"
        )
    slot_0 = min(
        admitted,
        key=lambda query: (
            _query_rank(query, 0),
            _canonical_json_bytes(query.assessor_value()),
        ),
    )
    slot_0_signature = query_denotation_signature(
        slot_0,
        bounded_terminal_universe,
    )
    complement = tuple(not value for value in slot_0_signature)
    slot_1_candidates: list[
        tuple[SemanticQuery, tuple[bool, ...]]
    ] = []
    for query in admitted:
        if query == slot_0:
            continue
        signature = query_denotation_signature(
            query,
            bounded_terminal_universe,
        )
        if signature not in {slot_0_signature, complement}:
            slot_1_candidates.append((query, signature))
    if not slot_1_candidates:
        raise SemanticAdmissionError(
            "semantic rectangle lacks a non-equivalent second query"
        )
    slot_1, slot_1_signature = min(
        slot_1_candidates,
        key=lambda item: (
            _query_rank(item[0], 1),
            _canonical_json_bytes(item[0].assessor_value()),
        ),
    )
    return SelectedQueries(
        slot_0=slot_0,
        slot_1=slot_1,
        slot_0_labels=checkerboard_labels(slot_0, rectangle),
        slot_1_labels=checkerboard_labels(slot_1, rectangle),
        slot_0_denotation=slot_0_signature,
        slot_1_denotation=slot_1_signature,
    )


def query_surface_value(
    query: SemanticQuery,
    *,
    operator_symbol: str,
    paraphrase: int,
) -> dict[str, object]:
    """Return an ontology-neutral candidate-visible query AST value.

    The caller assigns a split-local opaque symbol to the assessor operation.
    No operation name, ontology tag, theory index, label, or disposition enters
    this value.
    """

    if type(operator_symbol) is not str or _OPAQUE_SYMBOL.fullmatch(
        operator_symbol
    ) is None:
        raise SemanticError("query surface operator is not opaque")
    _require_exact_int(paraphrase, "query paraphrase", 0, 1)
    expression: dict[str, object] = {
        "a": [
            {"s": operator_symbol},
            *({"i": argument} for argument in query.args),
        ],
        "h": 4,
    }
    if paraphrase == 0:
        value: dict[str, object] = {"a": [expression], "h": 9}
    else:
        value = {"a": [expression, {"i": 1}], "h": 10}
    encoded = _canonical_json_bytes(value).lower()
    forbidden = (
        b"horn",
        b"rewrite",
        b"resource",
        b"ontology",
        b"theory",
        b"oracle",
        b"target",
        b"answer",
    )
    if any(token in encoded for token in forbidden):
        raise SemanticError("query surface leaks assessor ontology metadata")
    return value


__all__ = [
    "CHECKERBOARD_PATTERNS",
    "MASTER_SEED",
    "MASTER_SEED_HEX",
    "Ontology",
    "HornPolicy",
    "RewritePolicy",
    "ResourcePolicy",
    "TerminalDisposition",
    "StepOutcome",
    "QueryOp",
    "SemanticError",
    "SemanticAdmissionError",
    "HornWorld",
    "HornCommand",
    "RewriteWorld",
    "RewriteCommand",
    "ResourceWorld",
    "ResourceCommand",
    "HornStep",
    "RewriteSnapshot",
    "RewriteStep",
    "ResourceStep",
    "HornExecution",
    "RewriteExecution",
    "ResourceExecution",
    "SemanticQuery",
    "SemanticRectangle",
    "SelectedQueries",
    "execute_horn",
    "replay_horn",
    "execute_rewrite",
    "replay_rewrite",
    "execute_resource",
    "replay_resource",
    "execute_semantics",
    "replay_semantics",
    "enumerate_queries",
    "evaluate_query",
    "checkerboard_labels",
    "admissible_checkerboard_queries",
    "query_denotation_signature",
    "select_queries",
    "query_surface_value",
]
