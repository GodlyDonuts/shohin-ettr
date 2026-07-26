"""Exact guarded-resource variants for the ETTR qualification matrix.

This is offline board and assessor machinery.  It never participates in a
candidate runtime.  Candidate-visible material is limited to ``source`` before
sealing and ``challenge_source`` after sealing; alignments and exact outcomes
remain assessor-side.

The seven variants are deliberately not renderer labels:

* ``base`` is the direct trace presentation.
* ``alpha/reorder`` bijectively renames every opaque symbol and permutes
  storage order while preserving ordered operator programs.
* ``alias split`` replaces one place node with two explicitly equal aliases
  and distributes occurrences between them.
* ``relation reification`` replaces direct marking tuples with quantity nodes
  and typed incidence edges.
* ``type twin`` changes one place's resource-kind membership while preserving
  the untyped transition dynamics.
* ``execution-semantics twin`` changes blocked operations from atomic
  deadlock to skip-and-continue.
* ``ambiguity-deleted twin`` deletes the final identifying demonstration,
  making the exact version space behaviorally ambiguous.

The transformed executor below is independent of the production board
executor.  Focused tests compare it exhaustively on the materialized matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
import hashlib
import random

from cross_ontology_resource_board import (
    Demonstration,
    Marking,
    OPERATOR_LIBRARY,
    PLACE_SPECS,
    ProcessOutcome,
    ProcessStatus,
    ResourceTheory,
    THEORIES,
    behavioral_class_count,
    challenge_cases,
    consistent_theories,
    identifying_evidence,
)


CLAIM_BOUNDARY = (
    "Offline qualification generation and independent assessment only; no "
    "candidate-time parser, executor, alignment, oracle, search, or callback."
)

QUALIFICATION_THEORY_INDICES = (0, 7, 14, 21, 30, 39, 48, 59)
CHALLENGES_PER_VARIANT = 16


class ResourceVariantName(StrEnum):
    BASE = "base"
    ALPHA_REORDER = "alpha/reorder"
    ALIAS_SPLIT = "alias split"
    RELATION_REIFICATION = "relation reification"
    TYPE_TWIN = "type twin"
    EXECUTION_SEMANTICS_TWIN = "execution-semantics twin"
    AMBIGUITY_DELETED_TWIN = "ambiguity-deleted twin"


class ExecutionSemantics(StrEnum):
    ATOMIC_DEADLOCK = "atomic_deadlock"
    SKIP_BLOCKED = "skip_blocked"


class PairExpectation(StrEnum):
    REFERENCE = "reference"
    EXACT_INVARIANCE = "exact_invariance"
    TYPE_ONLY_SEPARATION = "type_only_separation"
    EXECUTION_SEPARATION = "execution_separation"
    AMBIGUITY_ABSTENTION = "ambiguity_abstention"


class AnswerDirective(StrEnum):
    ANSWER = "answer"
    ABSTAIN = "abstain"


@dataclass(frozen=True, order=True, slots=True)
class PresentedQuantity:
    place_key: str
    kind_key: str
    multiplicity: int


@dataclass(frozen=True, order=True, slots=True)
class PresentedMarking:
    quantities: tuple[PresentedQuantity, ...]


@dataclass(frozen=True, order=True, slots=True)
class PresentedOutcome:
    marking: PresentedMarking
    cursor: int
    status: ProcessStatus


@dataclass(frozen=True, order=True, slots=True)
class PresentedDemonstration:
    initial: PresentedMarking
    sequence: tuple[str, ...]
    outcome: PresentedOutcome


@dataclass(frozen=True, order=True, slots=True)
class CanonicalOutcome:
    """Assessor-side outcome after aligning opaque names to base roles."""

    multiplicities: tuple[int, ...]
    resource_kinds: tuple[int, ...]
    cursor: int
    status: ProcessStatus

    @property
    def execution_projection(
        self,
    ) -> tuple[tuple[int, ...], int, ProcessStatus]:
        return self.multiplicities, self.cursor, self.status


@dataclass(frozen=True, slots=True)
class ResourcePresentation:
    """Opaque presentation plus private assessor alignment."""

    place_keys_by_base: tuple[tuple[str, ...], ...]
    place_order: tuple[int, ...]
    kind_keys_by_base: tuple[str, ...]
    operator_keys_by_symbol: tuple[str, ...]
    operator_order: tuple[int, ...]
    place_kind_by_base: tuple[int, ...]
    alias_pairs: tuple[tuple[str, str], ...]
    reified: bool

    def __post_init__(self) -> None:
        if (
            len(self.place_keys_by_base) != len(PLACE_SPECS)
            or sorted(self.place_order) != list(range(len(PLACE_SPECS)))
            or len(self.kind_keys_by_base) != 2
            or len(set(self.kind_keys_by_base)) != 2
            or len(self.operator_keys_by_symbol) != 3
            or len(set(self.operator_keys_by_symbol)) != 3
            or sorted(self.operator_order) != [0, 1, 2]
            or len(self.place_kind_by_base) != len(PLACE_SPECS)
            or any(kind not in {0, 1} for kind in self.place_kind_by_base)
            or any(not keys for keys in self.place_keys_by_base)
        ):
            raise ValueError("resource presentation differs")
        place_keys = tuple(key for keys in self.place_keys_by_base for key in keys)
        if len(place_keys) != len(set(place_keys)):
            raise ValueError("presented place keys are not unique")
        expected_aliases = tuple(
            (keys[0], alias) for keys in self.place_keys_by_base for alias in keys[1:]
        )
        if self.alias_pairs != expected_aliases:
            raise ValueError("presented alias relation differs")

    @property
    def place_key_to_base(self) -> dict[str, int]:
        return {
            key: base
            for base, keys in enumerate(self.place_keys_by_base)
            for key in keys
        }

    @property
    def operator_key_to_symbol(self) -> dict[str, int]:
        return {key: symbol for symbol, key in enumerate(self.operator_keys_by_symbol)}


@dataclass(frozen=True, slots=True)
class ResourceVariant:
    name: ResourceVariantName
    theory_index: int
    presentation: ResourcePresentation
    execution_semantics: ExecutionSemantics
    demonstrations: tuple[PresentedDemonstration, ...]
    consistent_theory_indices: tuple[int, ...]
    behavioral_class_count: int
    source: str


@dataclass(frozen=True, slots=True)
class ResourceVariantCase:
    theory_index: int
    challenge_index: int
    variant: ResourceVariantName
    initial: PresentedMarking
    sequence: tuple[str, ...]
    challenge_source: str
    directive: AnswerDirective
    expected_outcome: CanonicalOutcome | None
    possible_outcomes: tuple[CanonicalOutcome, ...]
    pair_expectation: PairExpectation


VARIANT_ORDER = tuple(ResourceVariantName)
STRUCTURAL_EQUIVALENTS = frozenset(
    {
        ResourceVariantName.ALPHA_REORDER,
        ResourceVariantName.ALIAS_SPLIT,
        ResourceVariantName.RELATION_REIFICATION,
    }
)


def _stable_rank(theory_index: int, challenge_index: int) -> bytes:
    return hashlib.sha256(
        f"resource-variant-v1:{theory_index}:{challenge_index}".encode("ascii")
    ).digest()


def _opaque_keys(seed: int, count: int) -> tuple[str, ...]:
    rng = random.Random(seed)
    return tuple(
        f"x{value:05x}" for value in rng.sample(range(0x10000, 0xFFFFF), count)
    )


def _base_presentation(theory_index: int) -> ResourcePresentation:
    keys = _opaque_keys(0x51A7 + theory_index * 101, 9)
    places = tuple((key,) for key in keys[:4])
    return ResourcePresentation(
        place_keys_by_base=places,
        place_order=(0, 1, 2, 3),
        kind_keys_by_base=keys[4:6],
        operator_keys_by_symbol=keys[6:9],
        operator_order=(0, 1, 2),
        place_kind_by_base=tuple(place.resource_kind for place in PLACE_SPECS),
        alias_pairs=(),
        reified=False,
    )


def _presentation(
    theory_index: int,
    name: ResourceVariantName,
) -> ResourcePresentation:
    base = _base_presentation(theory_index)
    if name == ResourceVariantName.ALPHA_REORDER:
        keys = _opaque_keys(0xA17A + theory_index * 131, 9)
        rng = random.Random(0xA710 + theory_index)
        place_order = list(range(4))
        operator_order = list(range(3))
        rng.shuffle(place_order)
        rng.shuffle(operator_order)
        return ResourcePresentation(
            place_keys_by_base=tuple((key,) for key in keys[:4]),
            place_order=tuple(place_order),
            kind_keys_by_base=keys[4:6],
            operator_keys_by_symbol=keys[6:9],
            operator_order=tuple(operator_order),
            place_kind_by_base=base.place_kind_by_base,
            alias_pairs=(),
            reified=False,
        )
    if name == ResourceVariantName.ALIAS_SPLIT:
        split_place = theory_index % len(PLACE_SPECS)
        alias = _opaque_keys(0xA11A5 + theory_index * 149, 1)[0]
        place_keys = list(base.place_keys_by_base)
        place_keys[split_place] = (place_keys[split_place][0], alias)
        return ResourcePresentation(
            place_keys_by_base=tuple(place_keys),
            place_order=base.place_order,
            kind_keys_by_base=base.kind_keys_by_base,
            operator_keys_by_symbol=base.operator_keys_by_symbol,
            operator_order=base.operator_order,
            place_kind_by_base=base.place_kind_by_base,
            alias_pairs=((place_keys[split_place][0], alias),),
            reified=False,
        )
    if name == ResourceVariantName.RELATION_REIFICATION:
        return ResourcePresentation(
            place_keys_by_base=base.place_keys_by_base,
            place_order=base.place_order,
            kind_keys_by_base=base.kind_keys_by_base,
            operator_keys_by_symbol=base.operator_keys_by_symbol,
            operator_order=base.operator_order,
            place_kind_by_base=base.place_kind_by_base,
            alias_pairs=(),
            reified=True,
        )
    if name == ResourceVariantName.TYPE_TWIN:
        changed_place = (theory_index * 3 + 1) % len(PLACE_SPECS)
        kinds = list(base.place_kind_by_base)
        kinds[changed_place] = 1 - kinds[changed_place]
        return ResourcePresentation(
            place_keys_by_base=base.place_keys_by_base,
            place_order=base.place_order,
            kind_keys_by_base=base.kind_keys_by_base,
            operator_keys_by_symbol=base.operator_keys_by_symbol,
            operator_order=base.operator_order,
            place_kind_by_base=tuple(kinds),
            alias_pairs=(),
            reified=False,
        )
    return base


def _alias_choice(
    keys: tuple[str, ...],
    *,
    context: int,
    base_place: int,
) -> str:
    return keys[(context + base_place) % len(keys)]


def present_marking(
    marking: Marking,
    presentation: ResourcePresentation,
    *,
    context: int,
) -> PresentedMarking:
    return PresentedMarking(
        tuple(
            PresentedQuantity(
                place_key=_alias_choice(
                    presentation.place_keys_by_base[base_place],
                    context=context,
                    base_place=base_place,
                ),
                kind_key=presentation.kind_keys_by_base[
                    presentation.place_kind_by_base[base_place]
                ],
                multiplicity=marking.multiplicities[base_place],
            )
            for base_place in presentation.place_order
        )
    )


def _decode_marking(
    marking: PresentedMarking,
    presentation: ResourcePresentation,
) -> Marking:
    place_key_to_base = presentation.place_key_to_base
    kind_key_to_base = {
        key: kind for kind, key in enumerate(presentation.kind_keys_by_base)
    }
    values: dict[int, int] = {}
    for quantity in marking.quantities:
        try:
            base_place = place_key_to_base[quantity.place_key]
            presented_kind = kind_key_to_base[quantity.kind_key]
        except KeyError as exc:
            raise ValueError("presented marking uses an unknown key") from exc
        if presented_kind != presentation.place_kind_by_base[base_place]:
            raise ValueError("presented marking typing differs")
        previous = values.setdefault(base_place, quantity.multiplicity)
        if previous != quantity.multiplicity:
            raise ValueError("equal aliases carry different multiplicities")
    if set(values) != set(range(len(PLACE_SPECS))):
        raise ValueError("presented marking is incomplete")
    return Marking(tuple(values[index] for index in range(len(PLACE_SPECS))))


def _execute_transformed(
    theory: ResourceTheory,
    initial: Marking,
    sequence: tuple[int, ...],
    semantics: ExecutionSemantics,
) -> ProcessOutcome:
    """Independent imperative executor for transformed variant semantics."""

    counts = list(initial.multiplicities)
    for cursor, symbol in enumerate(sequence):
        operator = OPERATOR_LIBRARY[theory.operator_indices[symbol]]
        guards = {quantity.place: quantity.multiplicity for quantity in operator.guards}
        consumes = {
            quantity.place: quantity.multiplicity for quantity in operator.consumes
        }
        produces = {
            quantity.place: quantity.multiplicity for quantity in operator.produces
        }
        enabled = all(
            counts[place] >= required
            for place, required in (*guards.items(), *consumes.items())
        )
        successor = list(counts)
        if enabled:
            for place, amount in consumes.items():
                successor[place] -= amount
            for place, amount in produces.items():
                successor[place] += amount
            enabled = all(
                0 <= successor[place.index] <= place.capacity for place in PLACE_SPECS
            )
        if not enabled:
            if semantics == ExecutionSemantics.SKIP_BLOCKED:
                continue
            return ProcessOutcome(
                Marking(tuple(counts)),
                cursor,
                ProcessStatus.DEADLOCK,
            )
        counts = successor
    return ProcessOutcome(
        Marking(tuple(counts)),
        len(sequence),
        ProcessStatus.HALT,
    )


def execute_presented(
    theory_index: int,
    presentation: ResourcePresentation,
    initial: PresentedMarking,
    sequence: tuple[str, ...],
    semantics: ExecutionSemantics,
    *,
    context: int,
) -> PresentedOutcome:
    """Execute opaque transformed inputs without the board executor."""

    try:
        decoded_sequence = tuple(
            presentation.operator_key_to_symbol[key] for key in sequence
        )
    except KeyError as exc:
        raise ValueError("presented program uses an unknown operator") from exc
    outcome = _execute_transformed(
        THEORIES[theory_index],
        _decode_marking(initial, presentation),
        decoded_sequence,
        semantics,
    )
    return PresentedOutcome(
        marking=present_marking(
            outcome.marking,
            presentation,
            context=context + 1,
        ),
        cursor=outcome.cursor,
        status=outcome.status,
    )


def canonicalize_outcome(
    outcome: PresentedOutcome,
    presentation: ResourcePresentation,
) -> CanonicalOutcome:
    marking = _decode_marking(outcome.marking, presentation)
    return CanonicalOutcome(
        multiplicities=marking.multiplicities,
        resource_kinds=presentation.place_kind_by_base,
        cursor=outcome.cursor,
        status=outcome.status,
    )


def _present_demo(
    theory_index: int,
    demonstration: Demonstration,
    presentation: ResourcePresentation,
    semantics: ExecutionSemantics,
    *,
    context: int,
) -> PresentedDemonstration:
    initial = present_marking(
        demonstration.initial,
        presentation,
        context=context,
    )
    sequence = tuple(
        presentation.operator_keys_by_symbol[symbol]
        for symbol in demonstration.sequence
    )
    outcome = execute_presented(
        theory_index,
        presentation,
        initial,
        sequence,
        semantics,
        context=context,
    )
    return PresentedDemonstration(initial, sequence, outcome)


def _common_semantics_probe(theory_index: int) -> Demonstration:
    initial = Marking((0, 0, 0, 0))
    sequence = (0,)
    outcome = _execute_transformed(
        THEORIES[theory_index],
        initial,
        sequence,
        ExecutionSemantics.ATOMIC_DEADLOCK,
    )
    return Demonstration(initial, sequence, outcome)


def _base_evidence(
    theory_index: int,
    name: ResourceVariantName,
) -> tuple[Demonstration, ...]:
    evidence = identifying_evidence(theory_index)
    if name == ResourceVariantName.AMBIGUITY_DELETED_TWIN:
        evidence = evidence[:-1]
    return (*evidence, _common_semantics_probe(theory_index))


def _consistent_under_semantics(
    demonstrations: tuple[PresentedDemonstration, ...],
    presentation: ResourcePresentation,
    semantics: ExecutionSemantics,
) -> tuple[int, ...]:
    result: list[int] = []
    for theory_index in range(len(THEORIES)):
        valid = True
        for context, demo in enumerate(demonstrations):
            observed = canonicalize_outcome(demo.outcome, presentation)
            candidate = execute_presented(
                theory_index,
                presentation,
                demo.initial,
                demo.sequence,
                semantics,
                context=context,
            )
            canonical = canonicalize_outcome(candidate, presentation)
            if canonical.execution_projection != observed.execution_projection:
                valid = False
                break
        if valid:
            result.append(theory_index)
    return tuple(result)


def _render_marking_direct(marking: PresentedMarking) -> str:
    return (
        "{"
        + ",".join(
            f"{quantity.place_key}@{quantity.kind_key}={quantity.multiplicity}"
            for quantity in marking.quantities
        )
        + "}"
    )


def _render_marking_reified(
    marking: PresentedMarking,
    *,
    state_key: str,
) -> tuple[str, ...]:
    lines: list[str] = []
    for index, quantity in enumerate(marking.quantities):
        node = f"{state_key}n{index}"
        lines.extend(
            (
                f"N {node} {quantity.multiplicity}",
                f"E x0 {state_key} {node}",
                f"E x1 {node} {quantity.place_key}",
                f"E x2 {node} {quantity.kind_key}",
            )
        )
    return tuple(lines)


def _render_source(
    variant: ResourceVariantName,
    presentation: ResourcePresentation,
    demonstrations: tuple[PresentedDemonstration, ...],
) -> str:
    lines = ["V 1"]
    for base_place in presentation.place_order:
        kind = presentation.kind_keys_by_base[
            presentation.place_kind_by_base[base_place]
        ]
        for key in presentation.place_keys_by_base[base_place]:
            lines.append(f"P {key} {kind} {PLACE_SPECS[base_place].capacity}")
    lines.extend(f"A {left} {right}" for left, right in presentation.alias_pairs)
    lines.append(
        "O "
        + " ".join(
            presentation.operator_keys_by_symbol[symbol]
            for symbol in presentation.operator_order
        )
    )
    order = list(range(len(demonstrations)))
    if variant == ResourceVariantName.ALPHA_REORDER:
        order.reverse()
    for output_index, demo_index in enumerate(order):
        demo = demonstrations[demo_index]
        initial_key = f"i{output_index}"
        terminal_key = f"t{output_index}"
        lines.append(f"D {output_index}")
        if presentation.reified:
            lines.extend(_render_marking_reified(demo.initial, state_key=initial_key))
        else:
            lines.append(f"I {initial_key} {_render_marking_direct(demo.initial)}")
        lines.append(f"W {' '.join(demo.sequence)}")
        if presentation.reified:
            lines.extend(
                _render_marking_reified(
                    demo.outcome.marking,
                    state_key=terminal_key,
                )
            )
        else:
            lines.append(
                f"T {terminal_key} {_render_marking_direct(demo.outcome.marking)}"
            )
        lines.append(f"Z {demo.outcome.status.value} {demo.outcome.cursor}")
    return "\n".join(lines) + "\n"


@lru_cache(maxsize=None)
def build_resource_variant(
    theory_index: int,
    name: ResourceVariantName,
) -> ResourceVariant:
    if not 0 <= theory_index < len(THEORIES):
        raise ValueError("resource theory index differs")
    presentation = _presentation(theory_index, name)
    semantics = (
        ExecutionSemantics.SKIP_BLOCKED
        if name == ResourceVariantName.EXECUTION_SEMANTICS_TWIN
        else ExecutionSemantics.ATOMIC_DEADLOCK
    )
    base_evidence = _base_evidence(theory_index, name)
    demonstrations = tuple(
        _present_demo(
            theory_index,
            demo,
            presentation,
            semantics,
            context=context,
        )
        for context, demo in enumerate(base_evidence)
    )
    if semantics == ExecutionSemantics.ATOMIC_DEADLOCK:
        consistent = consistent_theories(base_evidence)
    else:
        consistent = _consistent_under_semantics(
            demonstrations,
            presentation,
            semantics,
        )
    return ResourceVariant(
        name=name,
        theory_index=theory_index,
        presentation=presentation,
        execution_semantics=semantics,
        demonstrations=demonstrations,
        consistent_theory_indices=consistent,
        behavioral_class_count=behavioral_class_count(consistent),
        source=_render_source(name, presentation, demonstrations),
    )


@lru_cache(maxsize=None)
def build_resource_variants(
    theory_index: int,
) -> tuple[ResourceVariant, ...]:
    return tuple(build_resource_variant(theory_index, name) for name in VARIANT_ORDER)


@lru_cache(maxsize=None)
def _selected_challenge_indices(theory_index: int) -> tuple[int, ...]:
    buckets: dict[tuple[ProcessStatus, int], list[int]] = {
        (status, length): [] for status in ProcessStatus for length in (2, 3)
    }
    for challenge_index, (marking, sequence) in enumerate(challenge_cases()):
        outcome = _execute_transformed(
            THEORIES[theory_index],
            marking,
            sequence,
            ExecutionSemantics.ATOMIC_DEADLOCK,
        )
        buckets[(outcome.status, len(sequence))].append(challenge_index)
    selected: list[int] = []
    for key in (
        (ProcessStatus.HALT, 2),
        (ProcessStatus.HALT, 3),
        (ProcessStatus.DEADLOCK, 2),
        (ProcessStatus.DEADLOCK, 3),
    ):
        ranked = sorted(
            buckets[key],
            key=lambda index: _stable_rank(theory_index, index),
        )
        if len(ranked) < 4:
            raise ValueError("resource challenge balance differs")
        selected.extend(ranked[:4])

    ambiguous_evidence = _base_evidence(
        theory_index,
        ResourceVariantName.AMBIGUITY_DELETED_TWIN,
    )
    survivors = consistent_theories(ambiguous_evidence)
    witnesses = [
        challenge_index
        for challenge_index, (marking, sequence) in enumerate(challenge_cases())
        if len(
            {
                _execute_transformed(
                    THEORIES[index],
                    marking,
                    sequence,
                    ExecutionSemantics.ATOMIC_DEADLOCK,
                )
                for index in survivors
            }
        )
        > 1
    ]
    if not witnesses:
        raise ValueError("ambiguity deletion has no behavioral witness")
    witness = min(witnesses, key=lambda index: _stable_rank(theory_index, index))
    if witness not in selected:
        marking, sequence = challenge_cases()[witness]
        status = _execute_transformed(
            THEORIES[theory_index],
            marking,
            sequence,
            ExecutionSemantics.ATOMIC_DEADLOCK,
        ).status
        replaceable = [
            index
            for index in selected
            if (
                _execute_transformed(
                    THEORIES[theory_index],
                    challenge_cases()[index][0],
                    challenge_cases()[index][1],
                    ExecutionSemantics.ATOMIC_DEADLOCK,
                ).status,
                len(challenge_cases()[index][1]),
            )
            == (status, len(sequence))
        ]
        selected[selected.index(replaceable[-1])] = witness
    return tuple(selected)


def _render_challenge(
    presentation: ResourcePresentation,
    initial: PresentedMarking,
    sequence: tuple[str, ...],
) -> str:
    lines = ["C 1"]
    if presentation.reified:
        lines.extend(_render_marking_reified(initial, state_key="q"))
    else:
        lines.append(f"I q {_render_marking_direct(initial)}")
    lines.append(f"W {' '.join(sequence)}")
    return "\n".join(lines) + "\n"


def _canonical_from_process(
    outcome: ProcessOutcome,
    resource_kinds: tuple[int, ...],
) -> CanonicalOutcome:
    return CanonicalOutcome(
        multiplicities=outcome.marking.multiplicities,
        resource_kinds=resource_kinds,
        cursor=outcome.cursor,
        status=outcome.status,
    )


@lru_cache(maxsize=None)
def build_resource_variant_cases(
    theory_index: int,
) -> tuple[ResourceVariantCase, ...]:
    variants = {
        variant.name: variant for variant in build_resource_variants(theory_index)
    }
    challenge_indices = _selected_challenge_indices(theory_index)
    ambiguity = variants[ResourceVariantName.AMBIGUITY_DELETED_TWIN]
    cases: list[ResourceVariantCase] = []
    for name in VARIANT_ORDER:
        variant = variants[name]
        for position, challenge_index in enumerate(challenge_indices):
            base_initial, base_sequence = challenge_cases()[challenge_index]
            initial = present_marking(
                base_initial,
                variant.presentation,
                context=1000 + position,
            )
            sequence = tuple(
                variant.presentation.operator_keys_by_symbol[symbol]
                for symbol in base_sequence
            )
            presented_outcome = execute_presented(
                theory_index,
                variant.presentation,
                initial,
                sequence,
                variant.execution_semantics,
                context=1000 + position,
            )
            expected = canonicalize_outcome(
                presented_outcome,
                variant.presentation,
            )
            directive = AnswerDirective.ANSWER
            possible = (expected,)
            if name == ResourceVariantName.BASE:
                pair_expectation = PairExpectation.REFERENCE
            elif name in STRUCTURAL_EQUIVALENTS:
                pair_expectation = PairExpectation.EXACT_INVARIANCE
            elif name == ResourceVariantName.TYPE_TWIN:
                pair_expectation = PairExpectation.TYPE_ONLY_SEPARATION
            elif name == ResourceVariantName.EXECUTION_SEMANTICS_TWIN:
                base = _execute_transformed(
                    THEORIES[theory_index],
                    base_initial,
                    base_sequence,
                    ExecutionSemantics.ATOMIC_DEADLOCK,
                )
                pair_expectation = (
                    PairExpectation.EXECUTION_SEPARATION
                    if expected.execution_projection
                    != _canonical_from_process(
                        base,
                        tuple(place.resource_kind for place in PLACE_SPECS),
                    ).execution_projection
                    else PairExpectation.EXACT_INVARIANCE
                )
            else:
                directive = AnswerDirective.ABSTAIN
                expected = None
                pair_expectation = PairExpectation.AMBIGUITY_ABSTENTION
                possible = tuple(
                    sorted(
                        {
                            _canonical_from_process(
                                _execute_transformed(
                                    THEORIES[index],
                                    base_initial,
                                    base_sequence,
                                    ExecutionSemantics.ATOMIC_DEADLOCK,
                                ),
                                tuple(place.resource_kind for place in PLACE_SPECS),
                            )
                            for index in ambiguity.consistent_theory_indices
                        }
                    )
                )
            cases.append(
                ResourceVariantCase(
                    theory_index=theory_index,
                    challenge_index=challenge_index,
                    variant=name,
                    initial=initial,
                    sequence=sequence,
                    challenge_source=_render_challenge(
                        variant.presentation,
                        initial,
                        sequence,
                    ),
                    directive=directive,
                    expected_outcome=expected,
                    possible_outcomes=possible,
                    pair_expectation=pair_expectation,
                )
            )
    return tuple(cases)


@lru_cache(maxsize=1)
def build_resource_qualification_matrix() -> tuple[ResourceVariantCase, ...]:
    return tuple(
        case
        for theory_index in QUALIFICATION_THEORY_INDICES
        for case in build_resource_variant_cases(theory_index)
    )


__all__ = [
    "AnswerDirective",
    "CHALLENGES_PER_VARIANT",
    "CLAIM_BOUNDARY",
    "CanonicalOutcome",
    "ExecutionSemantics",
    "PairExpectation",
    "PresentedDemonstration",
    "PresentedMarking",
    "PresentedOutcome",
    "PresentedQuantity",
    "QUALIFICATION_THEORY_INDICES",
    "ResourcePresentation",
    "ResourceVariant",
    "ResourceVariantCase",
    "ResourceVariantName",
    "STRUCTURAL_EQUIVALENTS",
    "VARIANT_ORDER",
    "build_resource_qualification_matrix",
    "build_resource_variant",
    "build_resource_variant_cases",
    "build_resource_variants",
    "canonicalize_outcome",
    "execute_presented",
    "present_marking",
]
