"""Offline guarded resource-process board for the ETTR qualification gate.

This module is assessor-side board mechanics.  It is not a candidate runtime,
and its exact executor is not a host callback available to a candidate.  The
only candidate-packet substrate shared with other ontologies is
``cross_ontology_schema``.

The board identifies opaque guarded operators from single-step evidence and
tests them only in previously unseen ordered compositions.  A blocked guard,
insufficient consumed multiplicity, or capacity overflow deadlocks the whole
sequence at the current cursor.  Exhausting the sequence halts normally.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from itertools import permutations, product
import random

from cross_ontology_schema import (
    ReactorState,
    RelationSpec,
    Transaction,
    TransactionOpcode,
    apply_transactions,
)


CLAIM_BOUNDARY = (
    "Offline board and assessor mechanics only; no candidate-time semantic "
    "parser, resource executor, search procedure, repair loop, or host callback."
)


@dataclass(frozen=True, order=True, slots=True)
class ResourceKind:
    index: int

    def __post_init__(self) -> None:
        if not 0 <= self.index < 2:
            raise ValueError("resource kind differs")


@dataclass(frozen=True, order=True, slots=True)
class PlaceSpec:
    index: int
    resource_kind: int
    capacity: int

    def __post_init__(self) -> None:
        if (
            not 0 <= self.index < 4
            or not 0 <= self.resource_kind < 2
            or not 1 <= self.capacity <= 3
        ):
            raise ValueError("place specification differs")


@dataclass(frozen=True, order=True, slots=True)
class ResourceQuantity:
    place: int
    resource_kind: int
    multiplicity: int

    def __post_init__(self) -> None:
        if (
            not 0 <= self.place < 4
            or not 0 <= self.resource_kind < 2
            or not 1 <= self.multiplicity <= 3
        ):
            raise ValueError("resource quantity differs")


@dataclass(frozen=True, slots=True)
class GuardedOperator:
    guards: tuple[ResourceQuantity, ...]
    consumes: tuple[ResourceQuantity, ...]
    produces: tuple[ResourceQuantity, ...]

    def __post_init__(self) -> None:
        for quantities in (self.guards, self.consumes, self.produces):
            places = tuple(quantity.place for quantity in quantities)
            if len(places) != len(set(places)):
                raise ValueError("operator repeats a place in one clause")
            for quantity in quantities:
                place = PLACE_SPECS[quantity.place]
                if quantity.resource_kind != place.resource_kind:
                    raise ValueError("operator resource typing differs")


@dataclass(frozen=True, slots=True)
class ResourceTheory:
    """Map three opaque operator symbols to guarded operator laws."""

    operator_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            len(self.operator_indices) != OPERATOR_SYMBOL_COUNT
            or len(set(self.operator_indices)) != OPERATOR_SYMBOL_COUNT
            or any(
                not 0 <= index < len(OPERATOR_LIBRARY)
                for index in self.operator_indices
            )
        ):
            raise ValueError("resource theory differs")


@dataclass(frozen=True, order=True, slots=True)
class Marking:
    multiplicities: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.multiplicities) != len(PLACE_SPECS) or any(
            not 0 <= value <= place.capacity
            for value, place in zip(
                self.multiplicities,
                PLACE_SPECS,
                strict=True,
            )
        ):
            raise ValueError("resource marking differs")


class ProcessStatus(StrEnum):
    HALT = "halt"
    DEADLOCK = "deadlock"


@dataclass(frozen=True, order=True, slots=True)
class ProcessOutcome:
    marking: Marking
    cursor: int
    status: ProcessStatus


@dataclass(frozen=True, slots=True)
class Demonstration:
    initial: Marking
    sequence: tuple[int, ...]
    outcome: ProcessOutcome

    def __post_init__(self) -> None:
        if not 1 <= len(self.sequence) <= MAX_SEQUENCE_LENGTH or any(
            not 0 <= symbol < OPERATOR_SYMBOL_COUNT for symbol in self.sequence
        ):
            raise ValueError("demonstration sequence differs")


class EvidenceDisposition(StrEnum):
    SINGLETON = "singleton"
    AMBIGUOUS = "ambiguous"
    CONTRADICTORY = "contradictory"
    COHERENT_ALTERNATE = "coherent_alternate"


@dataclass(frozen=True, slots=True)
class ResourceEpisode:
    seed: int
    target_theory_index: int
    evidence: tuple[Demonstration, ...]
    disposition: EvidenceDisposition
    consistent_theory_indices: tuple[int, ...]
    behavioral_class_count: int
    renderer: int
    source: str


RESOURCE_KINDS = (ResourceKind(0), ResourceKind(1))
PLACE_SPECS = (
    PlaceSpec(0, 0, 3),
    PlaceSpec(1, 0, 3),
    PlaceSpec(2, 1, 3),
    PlaceSpec(3, 1, 3),
)
OPERATOR_SYMBOL_COUNT = 3
MAX_SEQUENCE_LENGTH = 3


def _quantity(place: int, multiplicity: int) -> ResourceQuantity:
    return ResourceQuantity(
        place=place,
        resource_kind=PLACE_SPECS[place].resource_kind,
        multiplicity=multiplicity,
    )


OPERATOR_LIBRARY = (
    GuardedOperator(
        guards=(_quantity(0, 1),),
        consumes=(_quantity(0, 1),),
        produces=(_quantity(1, 1),),
    ),
    GuardedOperator(
        guards=(_quantity(1, 1), _quantity(2, 1)),
        consumes=(_quantity(1, 1), _quantity(2, 1)),
        produces=(_quantity(3, 2),),
    ),
    GuardedOperator(
        guards=(_quantity(3, 2),),
        consumes=(_quantity(3, 2),),
        produces=(_quantity(0, 1), _quantity(2, 1)),
    ),
    GuardedOperator(
        guards=(_quantity(0, 1), _quantity(3, 1)),
        consumes=(_quantity(3, 1),),
        produces=(_quantity(2, 1),),
    ),
    GuardedOperator(
        guards=(_quantity(1, 2),),
        consumes=(_quantity(1, 2),),
        produces=(_quantity(2, 1),),
    ),
)

THEORIES = tuple(
    ResourceTheory(tuple(operator_indices))
    for operator_indices in permutations(
        range(len(OPERATOR_LIBRARY)),
        OPERATOR_SYMBOL_COUNT,
    )
)


def _quantity_map(
    quantities: tuple[ResourceQuantity, ...],
) -> dict[int, int]:
    return {quantity.place: quantity.multiplicity for quantity in quantities}


def execute_sequence(
    theory: ResourceTheory,
    initial: Marking,
    sequence: tuple[int, ...],
) -> ProcessOutcome:
    """Execute a sequence imperatively for assessor-side board generation."""

    if not 1 <= len(sequence) <= MAX_SEQUENCE_LENGTH or any(
        not 0 <= symbol < OPERATOR_SYMBOL_COUNT for symbol in sequence
    ):
        raise ValueError("resource sequence differs")
    counts = {place.index: initial.multiplicities[place.index] for place in PLACE_SPECS}
    for cursor, symbol in enumerate(sequence):
        operator = OPERATOR_LIBRARY[theory.operator_indices[symbol]]
        guards = _quantity_map(operator.guards)
        consumes = _quantity_map(operator.consumes)
        produces = _quantity_map(operator.produces)
        if any(
            counts[place] < required
            for place, required in (*guards.items(), *consumes.items())
        ):
            return ProcessOutcome(
                Marking(tuple(counts[index] for index in range(4))),
                cursor,
                ProcessStatus.DEADLOCK,
            )
        updated = dict(counts)
        for place, amount in consumes.items():
            updated[place] -= amount
        for place, amount in produces.items():
            updated[place] += amount
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


@lru_cache(maxsize=1)
def input_markings() -> tuple[Marking, ...]:
    """The bounded input board; execution may still produce multiplicity 3."""

    return tuple(
        Marking(tuple(values)) for values in product(range(3), repeat=len(PLACE_SPECS))
    )


@lru_cache(maxsize=1)
def single_step_cases() -> tuple[tuple[Marking, tuple[int, ...]], ...]:
    return tuple(
        (marking, (symbol,))
        for marking in input_markings()
        for symbol in range(OPERATOR_SYMBOL_COUNT)
    )


@lru_cache(maxsize=1)
def heldout_programs() -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(sequence)
        for length in (2, 3)
        for sequence in product(range(OPERATOR_SYMBOL_COUNT), repeat=length)
    )


@lru_cache(maxsize=1)
def challenge_cases() -> tuple[tuple[Marking, tuple[int, ...]], ...]:
    return tuple(
        (marking, sequence)
        for marking in input_markings()
        for sequence in heldout_programs()
    )


@lru_cache(maxsize=None)
def behavior_signature(
    theory_index: int,
) -> tuple[ProcessOutcome, ...]:
    return tuple(
        execute_sequence(THEORIES[theory_index], marking, sequence)
        for marking, sequence in challenge_cases()
    )


@lru_cache(maxsize=1)
def _behavior_class_ids() -> tuple[int, ...]:
    identifiers: dict[tuple[ProcessOutcome, ...], int] = {}
    result: list[int] = []
    for theory_index in range(len(THEORIES)):
        signature = behavior_signature(theory_index)
        result.append(identifiers.setdefault(signature, len(identifiers)))
    return tuple(result)


@lru_cache(maxsize=None)
def _single_case_outcome(
    theory_index: int,
    case_index: int,
) -> ProcessOutcome:
    initial, sequence = single_step_cases()[case_index]
    return execute_sequence(THEORIES[theory_index], initial, sequence)


def consistent_theories(
    evidence: tuple[Demonstration, ...],
) -> tuple[int, ...]:
    return tuple(
        theory_index
        for theory_index, theory in enumerate(THEORIES)
        if all(
            execute_sequence(theory, demo.initial, demo.sequence) == demo.outcome
            for demo in evidence
        )
    )


def behavioral_class_count(
    theory_indices: tuple[int, ...],
) -> int:
    class_ids = _behavior_class_ids()
    return len({class_ids[theory_index] for theory_index in theory_indices})


@lru_cache(maxsize=None)
def identifying_evidence(
    target_theory_index: int,
) -> tuple[Demonstration, ...]:
    if not 0 <= target_theory_index < len(THEORIES):
        raise ValueError("target theory differs")
    remaining = tuple(range(len(THEORIES)))
    unused = set(range(len(single_step_cases())))
    evidence: list[Demonstration] = []
    while behavioral_class_count(remaining) > 1:
        best: (
            tuple[
                int,
                int,
                int,
                ProcessOutcome,
                tuple[int, ...],
            ]
            | None
        ) = None
        for case_index in sorted(unused):
            outcome = _single_case_outcome(
                target_theory_index,
                case_index,
            )
            survivors = tuple(
                index
                for index in remaining
                if _single_case_outcome(index, case_index) == outcome
            )
            candidate = (
                behavioral_class_count(survivors),
                len(survivors),
                case_index,
                outcome,
                survivors,
            )
            if best is None or candidate[:3] < best[:3]:
                best = candidate
        if best is None or best[0] == behavioral_class_count(remaining):
            raise ValueError("resource theory is not behaviorally identifiable")
        _, _, case_index, outcome, remaining = best
        initial, sequence = single_step_cases()[case_index]
        evidence.append(Demonstration(initial, sequence, outcome))
        unused.remove(case_index)
    return tuple(evidence)


def _contradictory_evidence(
    target_theory_index: int,
) -> tuple[Demonstration, ...]:
    target = THEORIES[target_theory_index]
    for initial, sequence in single_step_cases():
        target_outcome = execute_sequence(target, initial, sequence)
        alternate_outcome = next(
            (
                execute_sequence(theory, initial, sequence)
                for theory in THEORIES
                if execute_sequence(theory, initial, sequence) != target_outcome
            ),
            None,
        )
        if alternate_outcome is not None:
            return (
                Demonstration(initial, sequence, target_outcome),
                Demonstration(initial, sequence, alternate_outcome),
                *identifying_evidence(target_theory_index),
            )
    raise ValueError("no contradictory resource witness exists")


def _alternate_theory(target_theory_index: int) -> int:
    target_signature = behavior_signature(target_theory_index)
    return next(
        index
        for index in range(len(THEORIES))
        if behavior_signature(index) != target_signature
    )


def _marking_text(
    marking: Marking,
    place_names: tuple[str, ...],
    kind_names: tuple[str, ...],
    renderer: int,
) -> str:
    entries = tuple(
        (
            place_names[place.index],
            kind_names[place.resource_kind],
            marking.multiplicities[place.index],
        )
        for place in PLACE_SPECS
    )
    if renderer == 0:
        return (
            "{"
            + ",".join(f"{place}@{kind}={count}" for place, kind, count in entries)
            + "}"
        )
    if renderer == 1:
        return (
            "["
            + "|".join(f"{count}*{kind}:{place}" for place, kind, count in entries)
            + "]"
        )
    if renderer == 2:
        return (
            "<"
            + ";".join(f"{place}^{count}/{kind}" for place, kind, count in entries)
            + ">"
        )
    if renderer == 3:
        return (
            "("
            + " ".join(f"{kind}[{place},{count}]" for place, kind, count in entries)
            + ")"
        )
    raise ValueError("renderer differs")


def render_source(
    evidence: tuple[Demonstration, ...],
    *,
    seed: int,
    renderer: int,
) -> str:
    if not 0 <= renderer < 4:
        raise ValueError("renderer differs")
    rng = random.Random(seed ^ 0x5A17)
    place_names = tuple(
        f"p{value:03x}" for value in rng.sample(range(256, 4096), len(PLACE_SPECS))
    )
    kind_names = tuple(
        f"r{value:03x}" for value in rng.sample(range(4096, 8192), len(RESOURCE_KINDS))
    )
    operator_names = tuple(
        f"o{value:03x}"
        for value in rng.sample(
            range(8192, 12288),
            OPERATOR_SYMBOL_COUNT,
        )
    )
    status_names = tuple(
        f"s{value:03x}" for value in rng.sample(range(12288, 14336), 2)
    )
    separators = (
        (" => ", " :: "),
        (" / ", " # "),
        (" ~~ ", " @@ "),
        (" |> ", " <| "),
    )[renderer]
    lines = [f"q{renderer}:{len(evidence)}"]
    for index, demo in enumerate(evidence):
        initial = _marking_text(
            demo.initial,
            place_names,
            kind_names,
            renderer,
        )
        sequence = ".".join(operator_names[symbol] for symbol in demo.sequence)
        terminal = _marking_text(
            demo.outcome.marking,
            place_names,
            kind_names,
            renderer,
        )
        status = status_names[0 if demo.outcome.status == ProcessStatus.HALT else 1]
        lines.append(
            f"{index}{separators[0]}{initial}{separators[0]}"
            f"{sequence}{separators[1]}{terminal}{separators[1]}"
            f"{status}:{demo.outcome.cursor}"
        )
    return "\n".join(lines) + "\n"


def build_episode(
    seed: int,
    disposition: EvidenceDisposition,
    *,
    renderer: int,
) -> ResourceEpisode:
    target = seed % len(THEORIES)
    effective_target = target
    evidence = identifying_evidence(target)
    if disposition == EvidenceDisposition.AMBIGUOUS:
        evidence = evidence[:-1]
    elif disposition == EvidenceDisposition.CONTRADICTORY:
        evidence = _contradictory_evidence(target)
    elif disposition == EvidenceDisposition.COHERENT_ALTERNATE:
        effective_target = _alternate_theory(target)
        evidence = identifying_evidence(effective_target)
    consistent = consistent_theories(evidence)
    classes = behavioral_class_count(consistent)
    if disposition == EvidenceDisposition.SINGLETON:
        valid = classes == 1 and target in consistent
    elif disposition == EvidenceDisposition.AMBIGUOUS:
        valid = classes >= 2 and target in consistent
    elif disposition == EvidenceDisposition.CONTRADICTORY:
        valid = classes == 0 and not consistent
    else:
        valid = (
            classes == 1 and target not in consistent and effective_target in consistent
        )
    if not valid:
        raise ValueError("resource episode disposition construction differs")
    return ResourceEpisode(
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
    """Encode an inert assessor reference packet in the generic schema."""

    if not 0 <= theory_index < len(THEORIES):
        raise ValueError("theory index differs")
    relation_specs = (
        RelationSpec(0, (0, 2)),
        RelationSpec(1, (2, 3)),
        RelationSpec(2, (2, 3)),
        RelationSpec(3, (2, 3)),
        RelationSpec(4, (3, 1)),
    )
    transactions: list[Transaction] = [
        Transaction(TransactionOpcode.ALLOC, (0, 0)),
        Transaction(TransactionOpcode.WRITE, (0, 1)),
    ]
    for place in PLACE_SPECS:
        slot = 1 + place.index
        transactions.extend(
            (
                Transaction(TransactionOpcode.ALLOC, (slot, 1)),
                Transaction(
                    TransactionOpcode.WRITE,
                    (
                        slot,
                        place.resource_kind | (place.capacity << 8),
                    ),
                ),
            )
        )
    next_slot = 5
    theory = THEORIES[theory_index]
    for symbol, operator_index in enumerate(theory.operator_indices):
        operator_slot = next_slot
        next_slot += 1
        transactions.extend(
            (
                Transaction(
                    TransactionOpcode.ALLOC,
                    (operator_slot, 2),
                ),
                Transaction(
                    TransactionOpcode.WRITE,
                    (operator_slot, symbol),
                ),
                Transaction(
                    TransactionOpcode.LINK,
                    (0, 0, operator_slot),
                ),
            )
        )
        operator = OPERATOR_LIBRARY[operator_index]
        for relation_index, quantities in enumerate(
            (operator.guards, operator.consumes, operator.produces),
            start=1,
        ):
            for quantity in quantities:
                quantity_slot = next_slot
                next_slot += 1
                transactions.extend(
                    (
                        Transaction(
                            TransactionOpcode.ALLOC,
                            (quantity_slot, 3),
                        ),
                        Transaction(
                            TransactionOpcode.WRITE,
                            (
                                quantity_slot,
                                quantity.resource_kind | (quantity.multiplicity << 8),
                            ),
                        ),
                        Transaction(
                            TransactionOpcode.LINK,
                            (
                                relation_index,
                                operator_slot,
                                quantity_slot,
                            ),
                        ),
                        Transaction(
                            TransactionOpcode.LINK,
                            (
                                4,
                                quantity_slot,
                                1 + quantity.place,
                            ),
                        ),
                    )
                )
    transactions.extend(
        (
            Transaction(TransactionOpcode.SET_ROOT, (0,)),
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
    "CLAIM_BOUNDARY",
    "Demonstration",
    "EvidenceDisposition",
    "GuardedOperator",
    "MAX_SEQUENCE_LENGTH",
    "Marking",
    "OPERATOR_LIBRARY",
    "OPERATOR_SYMBOL_COUNT",
    "PLACE_SPECS",
    "ProcessOutcome",
    "ProcessStatus",
    "RESOURCE_KINDS",
    "ResourceEpisode",
    "ResourceKind",
    "ResourceQuantity",
    "ResourceTheory",
    "THEORIES",
    "behavior_signature",
    "behavioral_class_count",
    "build_episode",
    "challenge_cases",
    "consistent_theories",
    "execute_sequence",
    "heldout_programs",
    "identifying_evidence",
    "input_markings",
    "reference_theory_state",
    "render_source",
    "single_step_cases",
]
