"""Exact offline typed-rewrite board for the cross-ontology reactor gate.

This module is an offline generator and assessor.  It owns a deliberately
small term-rewriting ontology, exhaustive normal-form mechanics, opaque source
rendering, and exact behavioral version spaces.  Candidate-visible reference
theories are represented only by :mod:`cross_ontology_schema` transactions.

Nothing here is a candidate-time executor or host callback.  A future
candidate must infer and execute the rules itself; this module only constructs
and audits bounded qualification data before deployment.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from itertools import combinations, product
import random

from cross_ontology_schema import (
    ReactorState,
    RelationSpec,
    Transaction,
    TransactionOpcode,
    apply_transactions,
)


MAX_TERM_NODES = 4


@dataclass(frozen=True, order=True, slots=True)
class ConstructorSpec:
    index: int
    result_type: int
    argument_types: tuple[int, ...] = ()


@dataclass(frozen=True, order=True, slots=True)
class GroundTerm:
    type_index: int
    constructor_index: int
    children: tuple[GroundTerm, ...] = ()

    @property
    def node_count(self) -> int:
        return 1 + sum(child.node_count for child in self.children)


@dataclass(frozen=True, order=True, slots=True)
class PatternTerm:
    type_index: int
    constructor_index: int | None = None
    variable_index: int | None = None
    children: tuple[PatternTerm, ...] = ()

    @classmethod
    def variable(cls, index: int, type_index: int) -> PatternTerm:
        return cls(type_index=type_index, variable_index=index)

    @classmethod
    def constructor(
        cls,
        index: int,
        type_index: int,
        *children: PatternTerm,
    ) -> PatternTerm:
        return cls(
            type_index=type_index,
            constructor_index=index,
            children=tuple(children),
        )

    def __post_init__(self) -> None:
        if (self.constructor_index is None) == (
            self.variable_index is None
        ):
            raise ValueError(
                "pattern is not exactly one constructor or variable"
            )
        if self.variable_index is not None and self.children:
            raise ValueError("pattern variable has children")


@dataclass(frozen=True, order=True, slots=True)
class RewriteRule:
    index: int
    lhs: PatternTerm
    rhs: PatternTerm


@dataclass(frozen=True, order=True, slots=True)
class RewriteTheory:
    rule_indices: tuple[int, ...]


@dataclass(frozen=True, order=True, slots=True)
class Demonstration:
    initial: GroundTerm
    normal_forms: tuple[GroundTerm, ...]


class EvidenceDisposition(StrEnum):
    SINGLETON = "singleton"
    AMBIGUOUS = "ambiguous"
    CONTRADICTORY = "contradictory"
    COHERENT_ALTERNATE = "coherent_alternate"


@dataclass(frozen=True, slots=True)
class VersionSpace:
    theory_indices: tuple[int, ...]
    behavior_signatures: tuple[
        tuple[tuple[GroundTerm, ...], ...],
        ...,
    ]

    @property
    def behavioral_class_count(self) -> int:
        return len(self.behavior_signatures)


@dataclass(frozen=True, slots=True)
class RewriteEpisode:
    seed: int
    target_theory_index: int
    evidence: tuple[Demonstration, ...]
    disposition: EvidenceDisposition
    version_space: VersionSpace
    renderer: int
    source: str


# Types 0 and 1 are intentionally anonymous.  Constructor argument order is
# semantically significant, especially for constructor 5.
CONSTRUCTORS = (
    ConstructorSpec(0, 0),
    ConstructorSpec(1, 0),
    ConstructorSpec(2, 1),
    ConstructorSpec(3, 1),
    ConstructorSpec(4, 0, (0,)),
    ConstructorSpec(5, 0, (0, 0)),
    ConstructorSpec(6, 0, (1,)),
    ConstructorSpec(7, 1, (0,)),
)

_X0 = PatternTerm.variable(0, 0)
_Y0 = PatternTerm.variable(1, 0)
_A = PatternTerm.constructor(0, 0)
_B = PatternTerm.constructor(1, 0)
_F_X = PatternTerm.constructor(4, 0, _X0)

RULE_LIBRARY = (
    RewriteRule(
        0,
        PatternTerm.constructor(4, 0, _F_X),
        _F_X,
    ),
    RewriteRule(
        1,
        PatternTerm.constructor(4, 0, _A),
        _B,
    ),
    RewriteRule(
        2,
        PatternTerm.constructor(5, 0, _X0, _X0),
        _F_X,
    ),
    RewriteRule(
        3,
        PatternTerm.constructor(5, 0, _F_X, _Y0),
        PatternTerm.constructor(5, 0, _X0, _Y0),
    ),
    RewriteRule(
        4,
        PatternTerm.constructor(
            6,
            0,
            PatternTerm.constructor(7, 1, _X0),
        ),
        _X0,
    ),
    RewriteRule(
        5,
        PatternTerm.constructor(7, 1, _F_X),
        PatternTerm.constructor(7, 1, _X0),
    ),
)

THEORIES = tuple(
    RewriteTheory(pair)
    for pair in combinations(range(len(RULE_LIBRARY)), 2)
)

# Every primitive rule appears in training, while these exact compositions do
# not.  Qualification episodes draw targets only from this held-out set.
_HELDOUT_RULE_PAIRS = frozenset(
    {
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 4),
        (2, 5),
        (3, 5),
    }
)
HELDOUT_THEORY_INDICES = tuple(
    index
    for index, theory in enumerate(THEORIES)
    if theory.rule_indices in _HELDOUT_RULE_PAIRS
)
TRAIN_THEORY_INDICES = tuple(
    index
    for index in range(len(THEORIES))
    if index not in HELDOUT_THEORY_INDICES
)


def _constructor(index: int) -> ConstructorSpec:
    try:
        return CONSTRUCTORS[index]
    except IndexError as exc:
        raise ValueError("constructor index differs") from exc


def _validate_ground(term: GroundTerm) -> None:
    constructor = _constructor(term.constructor_index)
    if (
        term.type_index != constructor.result_type
        or len(term.children) != len(constructor.argument_types)
    ):
        raise ValueError("ground term typing differs")
    for child, required_type in zip(
        term.children,
        constructor.argument_types,
        strict=True,
    ):
        if child.type_index != required_type:
            raise ValueError("ground child typing differs")
        _validate_ground(child)


def _pattern_variables(pattern: PatternTerm) -> Counter[tuple[int, int]]:
    if pattern.variable_index is not None:
        return Counter({(pattern.variable_index, pattern.type_index): 1})
    result: Counter[tuple[int, int]] = Counter()
    for child in pattern.children:
        result.update(_pattern_variables(child))
    return result


def _pattern_constructor_count(pattern: PatternTerm) -> int:
    if pattern.variable_index is not None:
        return 0
    return 1 + sum(
        _pattern_constructor_count(child)
        for child in pattern.children
    )


def _validate_pattern(pattern: PatternTerm) -> None:
    if pattern.variable_index is not None:
        if pattern.variable_index < 0 or pattern.type_index not in {0, 1}:
            raise ValueError("pattern variable differs")
        return
    assert pattern.constructor_index is not None
    constructor = _constructor(pattern.constructor_index)
    if (
        pattern.type_index != constructor.result_type
        or len(pattern.children) != len(constructor.argument_types)
    ):
        raise ValueError("pattern constructor typing differs")
    for child, required_type in zip(
        pattern.children,
        constructor.argument_types,
        strict=True,
    ):
        if child.type_index != required_type:
            raise ValueError("pattern child typing differs")
        _validate_pattern(child)


def _validate_rule(rule: RewriteRule) -> None:
    _validate_pattern(rule.lhs)
    _validate_pattern(rule.rhs)
    if (
        rule.lhs.variable_index is not None
        or rule.lhs.type_index != rule.rhs.type_index
    ):
        raise ValueError("rewrite root typing differs")
    lhs_variables = _pattern_variables(rule.lhs)
    rhs_variables = _pattern_variables(rule.rhs)
    if any(
        count > lhs_variables[variable]
        for variable, count in rhs_variables.items()
    ):
        raise ValueError("RHS creates or duplicates a variable")
    lhs_constructor_count = _pattern_constructor_count(rule.lhs)
    rhs_constructor_count = _pattern_constructor_count(rule.rhs)
    strictly_decreases = (
        rhs_constructor_count < lhs_constructor_count
        or any(
            rhs_variables[variable] < count
            for variable, count in lhs_variables.items()
        )
    )
    if not strictly_decreases:
        raise ValueError("rewrite is not structurally decreasing")


for _rule in RULE_LIBRARY:
    _validate_rule(_rule)


@lru_cache(maxsize=1)
def challenge_terms() -> tuple[GroundTerm, ...]:
    """Enumerate every well-typed ground term through ``MAX_TERM_NODES``."""

    exact: dict[tuple[int, int], tuple[GroundTerm, ...]] = {}
    for size in range(1, MAX_TERM_NODES + 1):
        for type_index in (0, 1):
            terms: set[GroundTerm] = set()
            for constructor in CONSTRUCTORS:
                if constructor.result_type != type_index:
                    continue
                if not constructor.argument_types:
                    if size == 1:
                        terms.add(
                            GroundTerm(
                                type_index,
                                constructor.index,
                            )
                        )
                    continue
                remaining = size - 1
                for partition in product(
                    range(1, remaining + 1),
                    repeat=len(constructor.argument_types),
                ):
                    if sum(partition) != remaining:
                        continue
                    child_domains = tuple(
                        exact.get((child_type, child_size), ())
                        for child_type, child_size in zip(
                            constructor.argument_types,
                            partition,
                            strict=True,
                        )
                    )
                    for children in product(*child_domains):
                        terms.add(
                            GroundTerm(
                                type_index,
                                constructor.index,
                                tuple(children),
                            )
                        )
            exact[(type_index, size)] = tuple(sorted(terms))
    terms = tuple(
        sorted(
            term
            for values in exact.values()
            for term in values
        )
    )
    for term in terms:
        _validate_ground(term)
    return terms


def _match(
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
        _match(child_pattern, child, bindings)
        for child_pattern, child in zip(
            pattern.children,
            term.children,
            strict=True,
        )
    )


def _instantiate(
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
            _instantiate(child, bindings)
            for child in pattern.children
        ),
    )


def _occurrence_paths(term: GroundTerm) -> tuple[tuple[int, ...], ...]:
    paths = [()]
    for child_index, child in enumerate(term.children):
        paths.extend(
            (child_index, *path)
            for path in _occurrence_paths(child)
        )
    return tuple(paths)


def _at_path(term: GroundTerm, path: tuple[int, ...]) -> GroundTerm:
    result = term
    for child_index in path:
        result = result.children[child_index]
    return result


def _replace_at_path(
    term: GroundTerm,
    path: tuple[int, ...],
    replacement: GroundTerm,
) -> GroundTerm:
    if not path:
        return replacement
    child_index, *remaining = path
    children = list(term.children)
    children[child_index] = _replace_at_path(
        children[child_index],
        tuple(remaining),
        replacement,
    )
    return GroundTerm(
        term.type_index,
        term.constructor_index,
        tuple(children),
    )


def one_step_reducts(
    theory: RewriteTheory,
    term: GroundTerm,
) -> tuple[GroundTerm, ...]:
    """Enumerate legal occurrence/path rewrites in deterministic order."""

    reducts: set[GroundTerm] = set()
    for path in _occurrence_paths(term):
        redex = _at_path(term, path)
        for rule_index in theory.rule_indices:
            rule = RULE_LIBRARY[rule_index]
            bindings: dict[int, GroundTerm] = {}
            if _match(rule.lhs, redex, bindings):
                reduct = _replace_at_path(
                    term,
                    path,
                    _instantiate(rule.rhs, bindings),
                )
                if reduct.node_count >= term.node_count:
                    raise ValueError("rewrite did not decrease term size")
                reducts.add(reduct)
    return tuple(sorted(reducts))


@lru_cache(maxsize=None)
def execute_normal_forms(
    theory_index: int,
    initial: GroundTerm,
) -> tuple[GroundTerm, ...]:
    """Breadth-first exhaustive reduction to every reachable normal form."""

    _validate_ground(initial)
    theory = THEORIES[theory_index]
    frontier = [initial]
    visited = {initial}
    terminals: set[GroundTerm] = set()
    while frontier:
        state = frontier.pop(0)
        reducts = one_step_reducts(theory, state)
        if not reducts:
            terminals.add(state)
        for reduct in reducts:
            if reduct not in visited:
                visited.add(reduct)
                frontier.append(reduct)
    return tuple(sorted(terminals))


@lru_cache(maxsize=None)
def behavior_signature(
    theory_index: int,
) -> tuple[tuple[GroundTerm, ...], ...]:
    return tuple(
        execute_normal_forms(theory_index, term)
        for term in challenge_terms()
    )


def consistent_theories(
    evidence: tuple[Demonstration, ...],
) -> tuple[int, ...]:
    return tuple(
        theory_index
        for theory_index in range(len(THEORIES))
        if all(
            execute_normal_forms(theory_index, demo.initial)
            == demo.normal_forms
            for demo in evidence
        )
    )


def exact_version_space(
    evidence: tuple[Demonstration, ...],
) -> VersionSpace:
    theories = consistent_theories(evidence)
    signatures = tuple(
        sorted(
            {
                behavior_signature(theory_index)
                for theory_index in theories
            }
        )
    )
    return VersionSpace(theories, signatures)


def identifying_evidence(
    target_theory_index: int,
) -> tuple[Demonstration, ...]:
    """Greedily isolate the target's exact behavioral equivalence class."""

    remaining = tuple(range(len(THEORIES)))
    evidence: list[Demonstration] = []
    unused = set(range(len(challenge_terms())))
    while len(
        {
            behavior_signature(theory_index)
            for theory_index in remaining
        }
    ) > 1:
        best: tuple[int, int, tuple[GroundTerm, ...]] | None = None
        for challenge_index in sorted(unused):
            initial = challenge_terms()[challenge_index]
            terminal = execute_normal_forms(target_theory_index, initial)
            survivor_count = sum(
                execute_normal_forms(theory_index, initial) == terminal
                for theory_index in remaining
            )
            candidate = (survivor_count, challenge_index, terminal)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is None or best[0] == len(remaining):
            raise ValueError("target theory is not behaviorally identifiable")
        _, challenge_index, terminal = best
        initial = challenge_terms()[challenge_index]
        evidence.append(Demonstration(initial, terminal))
        unused.remove(challenge_index)
        remaining = tuple(
            theory_index
            for theory_index in remaining
            if execute_normal_forms(theory_index, initial) == terminal
        )
    return tuple(evidence)


def _ambiguous_evidence(
    target_theory_index: int,
) -> tuple[Demonstration, ...]:
    candidates: list[
        tuple[int, int, int, Demonstration]
    ] = []
    for challenge_index, initial in enumerate(challenge_terms()):
        demo = Demonstration(
            initial,
            execute_normal_forms(target_theory_index, initial),
        )
        version_space = exact_version_space((demo,))
        if version_space.behavioral_class_count >= 2:
            candidates.append(
                (
                    version_space.behavioral_class_count,
                    len(version_space.theory_indices),
                    challenge_index,
                    demo,
                )
            )
    if not candidates:
        raise ValueError("target has no nonempty ambiguous evidence")
    _, _, _, demonstration = min(
        candidates,
        key=lambda item: item[:3],
    )
    return (demonstration,)


def _term_text(
    term: GroundTerm,
    names: tuple[str, ...],
    renderer: int,
) -> str:
    name = names[term.constructor_index]
    children = tuple(
        _term_text(child, names, renderer)
        for child in term.children
    )
    if not children:
        return name
    if renderer == 0:
        return f"{name}({','.join(children)})"
    if renderer == 1:
        return f"[{name} {' '.join(children)}]"
    if renderer == 2:
        return f"({' '.join(children)} @{name})"
    if renderer == 3:
        roles = "|".join(
            f"{index}:{child}"
            for index, child in enumerate(children)
        )
        return f"{{{name}|{roles}}}"
    raise ValueError("renderer differs")


def render_source(
    evidence: tuple[Demonstration, ...],
    *,
    seed: int,
    renderer: int,
) -> str:
    """Render identical demonstrations through one of four opaque notations."""

    rng = random.Random((seed << 5) ^ renderer ^ 0x5A17)
    names = tuple(
        f"z{value:04x}"
        for value in rng.sample(range(0x1000, 0xFFFF), len(CONSTRUCTORS))
    )
    lines = [f"folio {renderer}:{len(evidence)}"]
    for index, demo in enumerate(evidence):
        initial = _term_text(demo.initial, names, renderer)
        separators = (
            (" => ", " ; "),
            (" ~> ", " || "),
            (" / ", " + "),
            (" :: ", " & "),
        )[renderer]
        normal_forms = separators[1].join(
            _term_text(term, names, renderer)
            for term in demo.normal_forms
        )
        lines.append(
            f"{index}{separators[0]}{initial}"
            f"{separators[0]}{normal_forms}"
        )
    return "\n".join(lines) + "\n"


def build_episode(
    seed: int,
    disposition: EvidenceDisposition,
    *,
    renderer: int,
) -> RewriteEpisode:
    target = HELDOUT_THEORY_INDICES[
        seed % len(HELDOUT_THEORY_INDICES)
    ]
    evidence = identifying_evidence(target)
    if disposition == EvidenceDisposition.AMBIGUOUS:
        evidence = _ambiguous_evidence(target)
    elif disposition == EvidenceDisposition.CONTRADICTORY:
        evidence = (
            Demonstration(
                challenge_terms()[0],
                (),
            ),
            *evidence,
        )
    elif disposition == EvidenceDisposition.COHERENT_ALTERNATE:
        target_signature = behavior_signature(target)
        alternate = next(
            theory_index
            for theory_index in TRAIN_THEORY_INDICES
            if behavior_signature(theory_index) != target_signature
        )
        evidence = identifying_evidence(alternate)
    version_space = exact_version_space(evidence)
    if disposition == EvidenceDisposition.SINGLETON:
        valid = (
            version_space.behavioral_class_count == 1
            and target in version_space.theory_indices
        )
    elif disposition == EvidenceDisposition.AMBIGUOUS:
        valid = version_space.behavioral_class_count >= 2
    elif disposition == EvidenceDisposition.CONTRADICTORY:
        valid = not version_space.theory_indices
    else:
        valid = (
            version_space.behavioral_class_count == 1
            and target not in version_space.theory_indices
        )
    if not valid:
        raise ValueError("episode disposition construction differs")
    return RewriteEpisode(
        seed=seed,
        target_theory_index=target,
        evidence=evidence,
        disposition=disposition,
        version_space=version_space,
        renderer=renderer,
        source=render_source(
            evidence,
            seed=seed,
            renderer=renderer,
        ),
    )


def _append_pattern_transactions(
    pattern: PatternTerm,
    *,
    parent_slot: int | None,
    child_role: int | None,
    variable_slots: dict[int, int],
    transactions: list[Transaction],
    next_slot: list[int],
) -> int:
    slot = next_slot[0]
    next_slot[0] += 1
    transactions.extend(
        (
            Transaction(TransactionOpcode.ALLOC, (slot, 2)),
            Transaction(
                TransactionOpcode.WRITE,
                (slot, pattern.type_index),
            ),
        )
    )
    if pattern.variable_index is not None:
        transactions.append(
            Transaction(
                TransactionOpcode.LINK,
                (5, slot, variable_slots[pattern.variable_index]),
            )
        )
    else:
        assert pattern.constructor_index is not None
        transactions.append(
            Transaction(
                TransactionOpcode.LINK,
                (2, slot, pattern.constructor_index),
            )
        )
    if parent_slot is not None and child_role is not None:
        transactions.append(
            Transaction(
                TransactionOpcode.LINK,
                (3 + child_role, parent_slot, slot),
            )
        )
    for index, child in enumerate(pattern.children):
        _append_pattern_transactions(
            child,
            parent_slot=slot,
            child_role=index,
            variable_slots=variable_slots,
            transactions=transactions,
            next_slot=next_slot,
        )
    return slot


def reference_theory_state(theory_index: int) -> ReactorState:
    """Encode rewrite syntax as generic typed objects and ordered relations."""

    theory = THEORIES[theory_index]
    relation_specs = (
        RelationSpec(0, (1, 2)),
        RelationSpec(1, (1, 2)),
        RelationSpec(2, (2, 0)),
        RelationSpec(3, (2, 2)),
        RelationSpec(4, (2, 2)),
        RelationSpec(5, (2, 3)),
        RelationSpec(6, (1, 3)),
    )
    transactions: list[Transaction] = []
    for constructor in CONSTRUCTORS:
        packed_signature = (
            constructor.result_type
            | (len(constructor.argument_types) << 4)
            | sum(
                type_index << (8 + 4 * index)
                for index, type_index in enumerate(
                    constructor.argument_types
                )
            )
        )
        transactions.extend(
            (
                Transaction(
                    TransactionOpcode.ALLOC,
                    (constructor.index, 0),
                ),
                Transaction(
                    TransactionOpcode.WRITE,
                    (constructor.index, packed_signature),
                ),
            )
        )
    next_slot = [len(CONSTRUCTORS)]
    rule_slots: list[int] = []
    for local_rule_index, rule_index in enumerate(theory.rule_indices):
        rule = RULE_LIBRARY[rule_index]
        rule_slot = next_slot[0]
        next_slot[0] += 1
        rule_slots.append(rule_slot)
        transactions.extend(
            (
                Transaction(
                    TransactionOpcode.ALLOC,
                    (rule_slot, 1),
                ),
                Transaction(
                    TransactionOpcode.WRITE,
                    (rule_slot, local_rule_index),
                ),
            )
        )
        variables = {
            variable: type_index
            for variable, type_index in _pattern_variables(
                rule.lhs
            )
        }
        variable_slots: dict[int, int] = {}
        for variable, type_index in sorted(variables.items()):
            variable_slot = next_slot[0]
            next_slot[0] += 1
            variable_slots[variable] = variable_slot
            transactions.extend(
                (
                    Transaction(
                        TransactionOpcode.ALLOC,
                        (variable_slot, 3),
                    ),
                    Transaction(
                        TransactionOpcode.WRITE,
                        (variable_slot, type_index),
                    ),
                    Transaction(
                        TransactionOpcode.LINK,
                        (6, rule_slot, variable_slot),
                    ),
                )
            )
        lhs_slot = _append_pattern_transactions(
            rule.lhs,
            parent_slot=None,
            child_role=None,
            variable_slots=variable_slots,
            transactions=transactions,
            next_slot=next_slot,
        )
        rhs_slot = _append_pattern_transactions(
            rule.rhs,
            parent_slot=None,
            child_role=None,
            variable_slots=variable_slots,
            transactions=transactions,
            next_slot=next_slot,
        )
        transactions.extend(
            (
                Transaction(
                    TransactionOpcode.LINK,
                    (0, rule_slot, lhs_slot),
                ),
                Transaction(
                    TransactionOpcode.LINK,
                    (1, rule_slot, rhs_slot),
                ),
            )
        )
    transactions.extend(
        (
            Transaction(TransactionOpcode.SET_ROOT, (rule_slots[0],)),
            Transaction(TransactionOpcode.COMMIT),
            Transaction(TransactionOpcode.HALT),
        )
    )
    return apply_transactions(
        ReactorState(
            capacity=32,
            type_count=4,
            relation_specs=relation_specs,
        ),
        transactions,
    )


__all__ = [
    "CONSTRUCTORS",
    "HELDOUT_THEORY_INDICES",
    "MAX_TERM_NODES",
    "RULE_LIBRARY",
    "TRAIN_THEORY_INDICES",
    "THEORIES",
    "ConstructorSpec",
    "Demonstration",
    "EvidenceDisposition",
    "GroundTerm",
    "PatternTerm",
    "RewriteEpisode",
    "RewriteRule",
    "RewriteTheory",
    "VersionSpace",
    "behavior_signature",
    "build_episode",
    "challenge_terms",
    "consistent_theories",
    "exact_version_space",
    "execute_normal_forms",
    "identifying_evidence",
    "one_step_reducts",
    "reference_theory_state",
    "render_source",
]
