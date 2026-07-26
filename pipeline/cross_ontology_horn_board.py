"""Exact offline Horn-law board for the cross-ontology reactor gate.

The candidate never imports this module.  It generates opaque demonstrations,
late challenges, exact version-space labels, and an assessor-only reference
packet expressed through the ontology-neutral transaction schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from itertools import combinations
import random

from cross_ontology_schema import (
    ReactorState,
    RelationSpec,
    Transaction,
    TransactionOpcode,
    apply_transactions,
)


@dataclass(frozen=True, order=True, slots=True)
class PredicateSpec:
    index: int
    argument_types: tuple[int, ...]


@dataclass(frozen=True, order=True, slots=True)
class GroundAtom:
    predicate: int
    arguments: tuple[int, ...]


@dataclass(frozen=True, order=True, slots=True)
class AtomPattern:
    predicate: int
    variables: tuple[int, ...]


@dataclass(frozen=True, order=True, slots=True)
class HornRule:
    premises: tuple[AtomPattern, ...]
    conclusion: AtomPattern


@dataclass(frozen=True, slots=True)
class HornTheory:
    rule_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Demonstration:
    initial: tuple[GroundAtom, ...]
    terminal: tuple[GroundAtom, ...]


class EvidenceDisposition(StrEnum):
    SINGLETON = "singleton"
    AMBIGUOUS = "ambiguous"
    CONTRADICTORY = "contradictory"
    COHERENT_ALTERNATE = "coherent_alternate"


@dataclass(frozen=True, slots=True)
class HornEpisode:
    seed: int
    target_theory_index: int
    evidence: tuple[Demonstration, ...]
    disposition: EvidenceDisposition
    consistent_theory_indices: tuple[int, ...]
    behavioral_class_count: int
    renderer: int
    source: str


PREDICATES = (
    PredicateSpec(0, (0,)),
    PredicateSpec(1, (0,)),
    PredicateSpec(2, (1,)),
    PredicateSpec(3, (0, 1)),
    PredicateSpec(4, (1, 0)),
)

OBJECT_TYPES = (0, 0, 0, 1, 1, 1)

RULE_LIBRARY = (
    HornRule(
        (AtomPattern(0, (0,)),),
        AtomPattern(1, (0,)),
    ),
    HornRule(
        (AtomPattern(1, (0,)), AtomPattern(3, (0, 1))),
        AtomPattern(2, (1,)),
    ),
    HornRule(
        (AtomPattern(2, (1,)), AtomPattern(4, (1, 0))),
        AtomPattern(0, (0,)),
    ),
    HornRule(
        (AtomPattern(3, (0, 1)),),
        AtomPattern(0, (0,)),
    ),
    HornRule(
        (AtomPattern(4, (1, 0)),),
        AtomPattern(2, (1,)),
    ),
    HornRule(
        (AtomPattern(1, (0,)), AtomPattern(3, (0, 1))),
        AtomPattern(4, (1, 0)),
    ),
)

THEORIES = tuple(
    HornTheory(tuple(indices))
    for indices in combinations(range(len(RULE_LIBRARY)), 3)
)


def _predicate(index: int) -> PredicateSpec:
    return PREDICATES[index]


@lru_cache(maxsize=1)
def all_ground_atoms() -> tuple[GroundAtom, ...]:
    atoms: list[GroundAtom] = []
    for predicate in PREDICATES:
        domains = [
            tuple(
                slot
                for slot, type_index in enumerate(OBJECT_TYPES)
                if type_index == required_type
            )
            for required_type in predicate.argument_types
        ]
        if len(domains) == 1:
            atoms.extend(
                GroundAtom(predicate.index, (left,))
                for left in domains[0]
            )
        else:
            atoms.extend(
                GroundAtom(predicate.index, (left, right))
                for left in domains[0]
                for right in domains[1]
            )
    return tuple(sorted(atoms))


def _variable_types(rule: HornRule) -> dict[int, int]:
    result: dict[int, int] = {}
    for pattern in (*rule.premises, rule.conclusion):
        predicate = _predicate(pattern.predicate)
        for variable, type_index in zip(
            pattern.variables,
            predicate.argument_types,
            strict=True,
        ):
            previous = result.setdefault(variable, type_index)
            if previous != type_index:
                raise ValueError("rule variable typing differs")
    return result


def _ground(
    pattern: AtomPattern,
    assignment: dict[int, int],
) -> GroundAtom:
    return GroundAtom(
        pattern.predicate,
        tuple(assignment[variable] for variable in pattern.variables),
    )


def execute_closure(
    theory: HornTheory,
    initial: tuple[GroundAtom, ...],
) -> tuple[GroundAtom, ...]:
    """Compute the assessor-side monotone least fixed point."""

    facts = set(initial)
    while True:
        before = len(facts)
        for rule_index in theory.rule_indices:
            rule = RULE_LIBRARY[rule_index]
            variable_types = _variable_types(rule)
            domains = {
                variable: tuple(
                    slot
                    for slot, object_type in enumerate(OBJECT_TYPES)
                    if object_type == type_index
                )
                for variable, type_index in variable_types.items()
            }
            variables = tuple(sorted(domains))
            assignments = [({}, 0)]
            for variable in variables:
                expanded: list[tuple[dict[int, int], int]] = []
                for assignment, depth in assignments:
                    for slot in domains[variable]:
                        updated = dict(assignment)
                        updated[variable] = slot
                        expanded.append((updated, depth + 1))
                assignments = expanded
            for assignment, _ in assignments:
                if all(
                    _ground(premise, assignment) in facts
                    for premise in rule.premises
                ):
                    facts.add(_ground(rule.conclusion, assignment))
        if len(facts) == before:
            return tuple(sorted(facts))


@lru_cache(maxsize=1)
def challenge_initials() -> tuple[tuple[GroundAtom, ...], ...]:
    atoms = all_ground_atoms()
    return tuple((atom,) for atom in atoms) + tuple(
        pair for pair in combinations(atoms, 2)
    )


@lru_cache(maxsize=None)
def behavior_signature(
    theory_index: int,
) -> tuple[tuple[GroundAtom, ...], ...]:
    return tuple(
        execute_closure(THEORIES[theory_index], initial)
        for initial in challenge_initials()
    )


def consistent_theories(
    evidence: tuple[Demonstration, ...],
) -> tuple[int, ...]:
    return tuple(
        theory_index
        for theory_index, theory in enumerate(THEORIES)
        if all(
            execute_closure(theory, demo.initial) == demo.terminal
            for demo in evidence
        )
    )


def behavioral_class_count(
    theory_indices: tuple[int, ...],
) -> int:
    return len(
        {
            behavior_signature(theory_index)
            for theory_index in theory_indices
        }
    )


def identifying_evidence(
    target_theory_index: int,
) -> tuple[Demonstration, ...]:
    remaining = tuple(range(len(THEORIES)))
    evidence: list[Demonstration] = []
    unused = set(range(len(challenge_initials())))
    while behavioral_class_count(remaining) > 1:
        best: tuple[int, int, tuple[GroundAtom, ...]] | None = None
        for challenge_index in sorted(unused):
            initial = challenge_initials()[challenge_index]
            terminal = execute_closure(
                THEORIES[target_theory_index],
                initial,
            )
            survivor_count = sum(
                execute_closure(THEORIES[index], initial) == terminal
                for index in remaining
            )
            candidate = (
                survivor_count,
                challenge_index,
                terminal,
            )
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is None or best[0] == len(remaining):
            raise ValueError("target theory is not behaviorally identifiable")
        _, challenge_index, terminal = best
        initial = challenge_initials()[challenge_index]
        evidence.append(Demonstration(initial, terminal))
        unused.remove(challenge_index)
        remaining = tuple(
            index
            for index in remaining
            if execute_closure(THEORIES[index], initial) == terminal
        )
    return tuple(evidence)


def _atom_text(
    atom: GroundAtom,
    predicate_names: tuple[str, ...],
    object_names: tuple[str, ...],
    renderer: int,
) -> str:
    name = predicate_names[atom.predicate]
    arguments = tuple(object_names[index] for index in atom.arguments)
    if renderer == 0:
        return f"{name}({','.join(arguments)})"
    if renderer == 1:
        return f"[{name}|{'|'.join(arguments)}]"
    if renderer == 2:
        return f"{' '.join(arguments)} :: {name}"
    if renderer == 3:
        return f"<{','.join(arguments)};{name}>"
    raise ValueError("renderer differs")


def render_source(
    evidence: tuple[Demonstration, ...],
    *,
    seed: int,
    renderer: int,
) -> str:
    rng = random.Random((seed << 4) ^ renderer)
    predicate_names = tuple(
        f"k{value:03x}"
        for value in rng.sample(range(256, 4096), len(PREDICATES))
    )
    object_names = tuple(
        f"v{value:03x}"
        for value in rng.sample(range(4096, 8192), len(OBJECT_TYPES))
    )
    lines = [f"ledger {renderer}:{len(evidence)}"]
    for index, demo in enumerate(evidence):
        initial = " ".join(
            _atom_text(atom, predicate_names, object_names, renderer)
            for atom in demo.initial
        )
        terminal = " ".join(
            _atom_text(atom, predicate_names, object_names, renderer)
            for atom in demo.terminal
        )
        lines.append(f"{index} / {initial} / {terminal}")
    return "\n".join(lines) + "\n"


def build_episode(
    seed: int,
    disposition: EvidenceDisposition,
    *,
    renderer: int,
) -> HornEpisode:
    target = seed % len(THEORIES)
    evidence = identifying_evidence(target)
    effective_target = target
    if disposition == EvidenceDisposition.AMBIGUOUS:
        evidence = evidence[:-1]
    elif disposition == EvidenceDisposition.CONTRADICTORY:
        first = evidence[0]
        missing_initial = tuple(
            atom
            for atom in first.terminal
            if atom not in first.initial
        )
        impossible_terminal = (
            missing_initial
            if missing_initial
            else first.terminal[1:]
        )
        evidence = (
            Demonstration(first.initial, impossible_terminal),
            *evidence[1:],
        )
    elif disposition == EvidenceDisposition.COHERENT_ALTERNATE:
        target_signature = behavior_signature(target)
        effective_target = next(
            index
            for index in range(len(THEORIES))
            if behavior_signature(index) != target_signature
        )
        evidence = identifying_evidence(effective_target)
    consistent = consistent_theories(evidence)
    classes = behavioral_class_count(consistent)
    expected = {
        EvidenceDisposition.SINGLETON: (1, True),
        EvidenceDisposition.AMBIGUOUS: (2, True),
        EvidenceDisposition.CONTRADICTORY: (0, False),
        EvidenceDisposition.COHERENT_ALTERNATE: (1, True),
    }[disposition]
    if (
        (classes < expected[0] if disposition == EvidenceDisposition.AMBIGUOUS
         else classes != expected[0])
        or (effective_target in consistent) != expected[1]
    ):
        raise ValueError("episode disposition construction differs")
    return HornEpisode(
        seed=seed,
        target_theory_index=target,
        evidence=evidence,
        disposition=disposition,
        consistent_theory_indices=consistent,
        behavioral_class_count=classes,
        renderer=renderer,
        source=render_source(
            evidence,
            seed=seed,
            renderer=renderer,
        ),
    )


def reference_theory_state(theory_index: int) -> ReactorState:
    """Encode one theory without a family opcode or executable callback."""

    theory = THEORIES[theory_index]
    relation_specs = (
        RelationSpec(0, (3, 1)),
        RelationSpec(1, (3, 0)),
        RelationSpec(2, (3, 2)),
        RelationSpec(3, (3, 2)),
        RelationSpec(4, (1, 3)),
    )
    transactions: list[Transaction] = []
    next_slot = 0
    predicate_slots: dict[int, int] = {}
    for predicate in PREDICATES:
        predicate_slots[predicate.index] = next_slot
        transactions.extend(
            (
                Transaction(TransactionOpcode.ALLOC, (next_slot, 0)),
                Transaction(
                    TransactionOpcode.WRITE,
                    (
                        next_slot,
                        len(predicate.argument_types)
                        + sum(
                            value << (4 + 4 * index)
                            for index, value in enumerate(
                                predicate.argument_types
                            )
                        ),
                    ),
                ),
            )
        )
        next_slot += 1
    for rule_index in theory.rule_indices:
        rule = RULE_LIBRARY[rule_index]
        rule_slot = next_slot
        transactions.append(
            Transaction(TransactionOpcode.ALLOC, (rule_slot, 1))
        )
        next_slot += 1
        variable_slots: dict[int, int] = {}
        for variable, type_index in sorted(_variable_types(rule).items()):
            variable_slots[variable] = next_slot
            transactions.extend(
                (
                    Transaction(TransactionOpcode.ALLOC, (next_slot, 2)),
                    Transaction(
                        TransactionOpcode.WRITE,
                        (next_slot, type_index),
                    ),
                )
            )
            next_slot += 1
        for is_conclusion, pattern in (
            *((False, premise) for premise in rule.premises),
            (True, rule.conclusion),
        ):
            occurrence_slot = next_slot
            transactions.append(
                Transaction(
                    TransactionOpcode.ALLOC,
                    (occurrence_slot, 3),
                )
            )
            transactions.append(
                Transaction(
                    TransactionOpcode.LINK,
                    (0, occurrence_slot, rule_slot),
                )
            )
            transactions.append(
                Transaction(
                    TransactionOpcode.LINK,
                    (
                        1,
                        occurrence_slot,
                        predicate_slots[pattern.predicate],
                    ),
                )
            )
            for argument_index, variable in enumerate(pattern.variables):
                transactions.append(
                    Transaction(
                        TransactionOpcode.LINK,
                        (
                            2 + argument_index,
                            occurrence_slot,
                            variable_slots[variable],
                        ),
                    )
                )
            if is_conclusion:
                transactions.append(
                    Transaction(
                        TransactionOpcode.LINK,
                        (4, rule_slot, occurrence_slot),
                    )
                )
            next_slot += 1
    transactions.extend(
        (
            Transaction(TransactionOpcode.SET_ROOT, (5,)),
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
    "AtomPattern",
    "Demonstration",
    "EvidenceDisposition",
    "GroundAtom",
    "HornEpisode",
    "HornRule",
    "HornTheory",
    "OBJECT_TYPES",
    "PREDICATES",
    "PredicateSpec",
    "RULE_LIBRARY",
    "THEORIES",
    "all_ground_atoms",
    "behavior_signature",
    "behavioral_class_count",
    "build_episode",
    "challenge_initials",
    "consistent_theories",
    "execute_closure",
    "identifying_evidence",
    "reference_theory_state",
    "render_source",
]
