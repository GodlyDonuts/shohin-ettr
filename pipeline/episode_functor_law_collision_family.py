#!/usr/bin/env python3
"""Generated, source-audited law-collision families for SSQAC falsification.

This module generalizes the exact mechanics in
``episode_functor_law_collision_board.py``.  It generates many four-cell
``F0L0/F0L1/F1L0/F1L1`` units rather than promoting the minimal fixture.
Every source has an exhaustively enumerated two-completion version space; the
opposite laws select opposite completions; and a separately chosen late query
has a balanced XOR or XNOR answer pattern.

The family is a CPU falsifier, not evidence of reasoning.  Its shortcut audit
attacks a preregistered vector of target-free structural and renderer features
with grouped held-out classifiers.  A family whose held-out attacks are not
compatible with chance is returned with an explicit no-go decision.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import product
import json
import math
import random
from typing import Callable, Sequence

from pipeline.episode_functor_law_collision_board import (
    DirectEvidence,
    LawCollisionSource,
    MachineCompletion,
    ObservationFact,
    PathObservationClause,
    SourceAudit,
    TransitionFact,
    audit_version_space,
    encode_non_law,
    encode_source,
    enumerate_completions,
    parse_source,
)


FAMILY_SCHEMA = "episode_functor_law_collision_family_v1"
UNIT_SCHEMA = "episode_functor_law_collision_family_unit_v1"
UNIT_RECEIPT_SCHEMA = "episode_functor_law_collision_family_unit_receipt_v1"
SHORTCUT_RECEIPT_SCHEMA = "episode_functor_law_collision_shortcut_audit_v1"
FAMILY_RECEIPT_SCHEMA = "episode_functor_law_collision_family_receipt_v1"
STATUS = "cpu_falsifier_only_no_reasoning_claim"
DEFAULT_SEED = "ssqac-generated-law-collision-family-v1"
CELL_NAMES = ("F0L0", "F0L1", "F1L0", "F1L1")
GEOMETRY_TEMPLATES = (
    (4, 2, 1),
    (4, 3, 2),
    (5, 2, 2),
    (5, 3, 1),
    (6, 2, 1),
    (6, 3, 2),
    (7, 2, 2),
    (7, 3, 1),
)


class GeneratedCollisionFamilyError(ValueError):
    """Generated family construction or verification failed closed."""


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _sha256(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _derive_keys(seed: str, count: int, domain: str) -> tuple[str, ...]:
    keys: list[str] = []
    nonce = 0
    while len(keys) < count:
        digest = sha256(
            f"{domain}\0{seed}\0{nonce}".encode("utf-8")
        ).hexdigest()
        key = f"k_{digest[:16]}"
        nonce += 1
        if key not in keys:
            keys.append(key)
    return tuple(keys)


def _seeded_rng(*parts: object) -> random.Random:
    material = "\0".join(str(part) for part in parts).encode("utf-8")
    return random.Random(int.from_bytes(sha256(material).digest(), "big"))


def _cycle_type(permutation: Sequence[int]) -> tuple[int, ...]:
    seen: set[int] = set()
    lengths: list[int] = []
    for start in range(len(permutation)):
        if start in seen:
            continue
        state = start
        length = 0
        while state not in seen:
            seen.add(state)
            state = permutation[state]
            length += 1
        lengths.append(length)
    return tuple(sorted(lengths, reverse=True))


def _word_inventory(
    action_count: int,
    minimum_depth: int,
    maximum_depth: int,
    *,
    require_action: int,
) -> list[tuple[int, ...]]:
    return [
        word
        for depth in range(minimum_depth, maximum_depth + 1)
        for word in product(range(action_count), repeat=depth)
        if require_action in word
    ]


def _answer_index(
    completion: MachineCompletion,
    *,
    start: int,
    word: Sequence[int],
    observer: int,
) -> int:
    state = start
    for action in word:
        state = completion.transitions[action][state]
    return completion.observations[observer][state]


@dataclass(frozen=True, slots=True)
class GeneratedCollisionFamilyConfig:
    seed: str = DEFAULT_SEED
    unit_count: int = 32
    maximum_generation_attempts: int = 512
    minimum_law_depth: int = 2
    maximum_law_depth: int = 4
    minimum_query_depth: int = 3
    maximum_query_depth: int = 7
    shortcut_accuracy_ceiling: float = 0.625
    shortcut_alpha: float = 0.05

    def __post_init__(self) -> None:
        if type(self.seed) is not str or not self.seed:
            raise GeneratedCollisionFamilyError("seed must be a nonempty string")
        if (
            type(self.unit_count) is not int
            or self.unit_count < 32
            or self.unit_count % 32
        ):
            raise GeneratedCollisionFamilyError(
                "unit_count must be a positive multiple of 32"
            )
        if (
            type(self.maximum_generation_attempts) is not int
            or self.maximum_generation_attempts < 1
        ):
            raise GeneratedCollisionFamilyError(
                "maximum_generation_attempts must be positive"
            )
        if not 1 <= self.minimum_law_depth <= self.maximum_law_depth <= 4:
            raise GeneratedCollisionFamilyError("law depths must lie in [1, 4]")
        if not (
            1
            <= self.minimum_query_depth
            <= self.maximum_query_depth
            <= 10
        ):
            raise GeneratedCollisionFamilyError("query depths must lie in [1, 10]")
        if not 0.5 < self.shortcut_accuracy_ceiling < 1.0:
            raise GeneratedCollisionFamilyError(
                "shortcut accuracy ceiling must lie in (0.5, 1)"
            )
        if not 0.0 < self.shortcut_alpha < 0.5:
            raise GeneratedCollisionFamilyError(
                "shortcut alpha must lie in (0, 0.5)"
            )


@dataclass(frozen=True, slots=True)
class LateQuery:
    start_index: int
    action_indices: tuple[int, ...]
    observer_index: int

    @property
    def depth(self) -> int:
        return len(self.action_indices)


@dataclass(frozen=True, slots=True)
class FactWorldInvariant:
    incomplete_action_count: int
    incomplete_action_missing_cells: tuple[int, ...]
    complete_action_cycle_types: tuple[tuple[int, ...], ...]

    def canonical_bytes(self) -> bytes:
        return _canonical_json(asdict(self))


@dataclass(frozen=True, slots=True)
class GeneratedCollisionUnitReceipt:
    schema: str
    status: str
    unit_id: str
    geometry: tuple[int, int, int, int]
    parity: int
    source_sha256s: tuple[str, ...]
    source_receipt_sha256s: tuple[str, ...]
    non_law_sha256s: tuple[str, ...]
    selected_completion_sha256s: tuple[str, ...]
    fact_invariant_sha256s: tuple[str, str]
    law_query: LateQuery
    late_query: LateQuery
    late_answer_indices: tuple[int, int, int, int]
    gates: tuple[tuple[str, bool], ...]

    def canonical_bytes(self) -> bytes:
        return _canonical_json(asdict(self))

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.canonical_bytes())


@dataclass(frozen=True, slots=True)
class GeneratedCollisionUnit:
    schema: str
    status: str
    unit_id: str
    parity: int
    sources: tuple[bytes, bytes, bytes, bytes]
    late_query: LateQuery
    source_audits: tuple[SourceAudit, SourceAudit, SourceAudit, SourceAudit]
    fact_invariants: tuple[FactWorldInvariant, FactWorldInvariant]
    receipt: GeneratedCollisionUnitReceipt

    @property
    def gates(self) -> dict[str, bool]:
        return dict(self.receipt.gates)


@dataclass(frozen=True, slots=True)
class ShallowExample:
    unit_id: str
    split: str
    cell_name: str
    features: tuple[float, ...]
    target: int


@dataclass(frozen=True, slots=True)
class ClassifierAudit:
    name: str
    evaluation_examples: int
    correct: int
    accuracy: float
    exact_unit_flip_tail_probability: float
    accuracy_gate: bool
    randomization_gate: bool


@dataclass(frozen=True, slots=True)
class ShortcutAuditReceipt:
    schema: str
    status: str
    feature_names: tuple[str, ...]
    training_unit_ids: tuple[str, ...]
    evaluation_unit_ids: tuple[str, ...]
    training_examples: int
    evaluation_examples: int
    target_counts: tuple[tuple[int, int], ...]
    classifiers: tuple[ClassifierAudit, ...]
    chance_compatible: bool
    decision: str

    def canonical_bytes(self) -> bytes:
        return _canonical_json(asdict(self))

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.canonical_bytes())


@dataclass(frozen=True, slots=True)
class GeneratedCollisionFamilyReceipt:
    schema: str
    status: str
    seed: str
    unit_count: int
    cell_count: int
    geometries: tuple[tuple[int, int, int, int], ...]
    query_address_count: int
    target_counts: tuple[tuple[int, int], ...]
    unit_receipt_sha256s: tuple[str, ...]
    shortcut_audit_receipt_sha256: str
    all_exact_unit_gates_passed: bool
    shortcut_chance_gates_passed: bool
    promotion_eligible: bool
    decision: str
    claim_boundary: str

    def canonical_bytes(self) -> bytes:
        return _canonical_json(asdict(self))

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.canonical_bytes())


@dataclass(frozen=True, slots=True)
class GeneratedCollisionFamilyAudit:
    config: GeneratedCollisionFamilyConfig
    units: tuple[GeneratedCollisionUnit, ...]
    shortcut_audit: ShortcutAuditReceipt
    receipt: GeneratedCollisionFamilyReceipt


def _fact_invariant(evidence: DirectEvidence) -> FactWorldInvariant:
    state_index = {state: index for index, state in enumerate(evidence.states)}
    action_index = {action: index for index, action in enumerate(evidence.actions)}
    rows: list[list[int | None]] = [
        [None] * len(evidence.states) for _ in evidence.actions
    ]
    for fact in evidence.transitions:
        rows[action_index[fact.action]][state_index[fact.source]] = state_index[
            fact.destination
        ]
    missing = tuple(sum(value is None for value in row) for row in rows)
    complete_types = tuple(
        sorted(
            _cycle_type(tuple(int(value) for value in row))
            for row in rows
            if all(value is not None for value in row)
        )
    )
    return FactWorldInvariant(
        incomplete_action_count=sum(count > 0 for count in missing),
        incomplete_action_missing_cells=tuple(sorted(count for count in missing if count)),
        complete_action_cycle_types=complete_types,
    )


def _observation_rows(
    rng: random.Random,
    observer_count: int,
    state_count: int,
) -> tuple[tuple[int, ...], ...]:
    rows: list[tuple[int, ...]] = []
    for _ in range(observer_count):
        row = [index % 2 for index in range(state_count)]
        rng.shuffle(row)
        rows.append(tuple(row))
    return tuple(rows)


def _build_evidence(
    *,
    states: tuple[str, ...],
    actions: tuple[str, ...],
    observers: tuple[str, ...],
    answers: tuple[str, str],
    hidden_row: tuple[int, ...],
    hidden_sources: tuple[int, int],
    complete_rows: tuple[tuple[int, ...], ...],
    observation_rows: tuple[tuple[int, ...], ...],
) -> DirectEvidence:
    transitions = [
        TransitionFact(actions[0], states[source], states[destination])
        for source, destination in enumerate(hidden_row)
        if source not in hidden_sources
    ]
    for action_index, row in enumerate(complete_rows, start=1):
        transitions.extend(
            TransitionFact(
                actions[action_index],
                states[source],
                states[destination],
            )
            for source, destination in enumerate(row)
        )
    observations = tuple(
        ObservationFact(observers[observer], states[state], answers[answer])
        for observer, row in enumerate(observation_rows)
        for state, answer in enumerate(row)
    )
    return DirectEvidence(
        states=states,
        actions=actions,
        observers=observers,
        answers=answers,
        action_rows_are_permutations=True,
        transitions=tuple(transitions),
        observations=observations,
    )


def _query_answer(
    audit: SourceAudit,
    query: LateQuery,
) -> int:
    completion = audit.selected_completion
    if completion is None:
        raise GeneratedCollisionFamilyError(
            "late query requires a uniquely selected completion"
        )
    return _answer_index(
        completion,
        start=query.start_index,
        word=query.action_indices,
        observer=query.observer_index,
    )


def _make_sources(
    evidence0: DirectEvidence,
    evidence1: DirectEvidence,
    law_query: LateQuery,
) -> tuple[bytes, bytes, bytes, bytes]:
    sources: list[bytes] = []
    for evidence in (evidence0, evidence1):
        for expected in (0, 1):
            clause = PathObservationClause(
                clause_id="determining-generated-path",
                start=evidence.states[law_query.start_index],
                actions=tuple(
                    evidence.actions[index]
                    for index in law_query.action_indices
                ),
                observer=evidence.observers[law_query.observer_index],
                expected=evidence.answers[expected],
                alternate=evidence.answers[1 - expected],
            )
            sources.append(
                encode_source(
                    LawCollisionSource(
                        evidence=evidence,
                        law_present=True,
                        clauses=(clause,),
                    )
                )
            )
    return tuple(sources)  # type: ignore[return-value]


def _unit_id(seed: str, index: int) -> str:
    return f"unit-{index:04d}-{sha256(f'{seed}:{index}'.encode()).hexdigest()[:12]}"


def _generate_unit(
    config: GeneratedCollisionFamilyConfig,
    index: int,
) -> GeneratedCollisionUnit:
    state_count, action_count, observer_count = GEOMETRY_TEMPLATES[
        index % len(GEOMETRY_TEMPLATES)
    ]
    round_index = index // len(GEOMETRY_TEMPLATES)
    parity = round_index % 2
    identifier = _unit_id(config.seed, index)
    key_count = state_count + action_count + observer_count + 2
    keys = _derive_keys(
        f"{config.seed}:{identifier}",
        key_count,
        "generated-law-collision-family",
    )
    cursor = 0
    states = keys[cursor : cursor + state_count]
    cursor += state_count
    actions = keys[cursor : cursor + action_count]
    cursor += action_count
    observers = keys[cursor : cursor + observer_count]
    cursor += observer_count
    answers = keys[cursor : cursor + 2]

    law_words = _word_inventory(
        action_count,
        config.minimum_law_depth,
        config.maximum_law_depth,
        require_action=0,
    )
    query_words = _word_inventory(
        action_count,
        config.minimum_query_depth,
        config.maximum_query_depth,
        require_action=0,
    )
    target_pattern = (0, 1, 1, 0) if parity == 0 else (1, 0, 0, 1)

    for attempt in range(config.maximum_generation_attempts):
        rng = _seeded_rng(config.seed, identifier, attempt)
        hidden_sources0 = tuple(sorted(rng.sample(range(state_count), 2)))
        hidden_sources1 = tuple(sorted(rng.sample(range(state_count), 2)))
        hidden0 = list(range(state_count))
        hidden1 = list(range(state_count))
        rng.shuffle(hidden0)
        rng.shuffle(hidden1)

        identity = tuple(range(state_count))
        f0_complete = tuple(identity for _ in range(action_count - 1))
        f1_complete = tuple(
            tuple(
                (state + shift) % state_count
                for state in range(state_count)
            )
            for shift in (
                1 if slot % 2 == 0 else state_count - 1
                for slot in range(action_count - 1)
            )
        )
        evidence0 = _build_evidence(
            states=states,
            actions=actions,
            observers=observers,
            answers=(answers[0], answers[1]),
            hidden_row=tuple(hidden0),
            hidden_sources=hidden_sources0,
            complete_rows=f0_complete,
            observation_rows=_observation_rows(rng, observer_count, state_count),
        )
        evidence1 = _build_evidence(
            states=states,
            actions=actions,
            observers=observers,
            answers=(answers[0], answers[1]),
            hidden_row=tuple(hidden1),
            hidden_sources=hidden_sources1,
            complete_rows=f1_complete,
            observation_rows=_observation_rows(rng, observer_count, state_count),
        )
        completions0 = enumerate_completions(evidence0)
        completions1 = enumerate_completions(evidence1)
        if len(completions0) != 2 or len(completions1) != 2:
            continue

        law_specs = [
            LateQuery(start, word, observer)
            for start in range(state_count)
            for word in law_words
            for observer in range(observer_count)
            if {
                _answer_index(
                    completion,
                    start=start,
                    word=word,
                    observer=observer,
                )
                for completion in completions0
            }
            == {0, 1}
            and {
                _answer_index(
                    completion,
                    start=start,
                    word=word,
                    observer=observer,
                )
                for completion in completions1
            }
            == {0, 1}
        ]
        if not law_specs:
            continue
        rng.shuffle(law_specs)
        law_query = law_specs[0]
        sources = _make_sources(evidence0, evidence1, law_query)
        audits = tuple(audit_version_space(source) for source in sources)
        if any(audit.selected_completion is None for audit in audits):
            continue

        late_specs = [
            LateQuery(start, word, observer)
            for start in range(state_count)
            for word in query_words
            for observer in range(observer_count)
        ]
        rng.shuffle(late_specs)
        late_query = next(
            (
                query
                for query in late_specs
                if query != law_query
                and tuple(_query_answer(audit, query) for audit in audits)
                == target_pattern
            ),
            None,
        )
        if late_query is None:
            continue

        invariants = (_fact_invariant(evidence0), _fact_invariant(evidence1))
        gates = {
            "four_sources": len(sources) == 4,
            "two_direct_completions_each": all(
                audit.receipt.direct_completion_count == 2 for audit in audits
            ),
            "one_law_completion_each": all(
                audit.receipt.law_completion_count == 1 for audit in audits
            ),
            "unique_completion_each": all(
                audit.receipt.resolution == "unique-completion"
                for audit in audits
            ),
            "law_twins_share_non_law_bytes": (
                encode_non_law(audits[0].source.evidence)
                == encode_non_law(audits[1].source.evidence)
                and encode_non_law(audits[2].source.evidence)
                == encode_non_law(audits[3].source.evidence)
            ),
            "opposite_laws_select_opposite_completions": (
                audits[0].receipt.selected_completion_sha256
                != audits[1].receipt.selected_completion_sha256
                and audits[2].receipt.selected_completion_sha256
                != audits[3].receipt.selected_completion_sha256
            ),
            "shared_law_query_address": all(
                isinstance(audit.source.clauses[0], PathObservationClause)
                and audit.source.evidence.states.index(
                    audit.source.clauses[0].start
                )
                == law_query.start_index
                and tuple(
                    audit.source.evidence.actions.index(action)
                    for action in audit.source.clauses[0].actions
                )
                == law_query.action_indices
                and audit.source.evidence.observers.index(
                    audit.source.clauses[0].observer
                )
                == law_query.observer_index
                for audit in audits
            ),
            "balanced_late_answers": Counter(target_pattern)
            == Counter({0: 2, 1: 2}),
            "requested_parity_realized": tuple(
                _query_answer(audit, late_query) for audit in audits
            )
            == target_pattern,
            "one_incomplete_action_each": all(
                invariant.incomplete_action_count == 1
                and invariant.incomplete_action_missing_cells == (2,)
                for invariant in invariants
            ),
            "fact_worlds_proven_nonisomorphic_by_cycle_invariant": (
                invariants[0].complete_action_cycle_types
                != invariants[1].complete_action_cycle_types
            ),
            "late_query_distinct_from_law_query": late_query != law_query,
        }
        if not all(gates.values()):
            continue
        answer_indices = tuple(
            _query_answer(audit, late_query) for audit in audits
        )
        receipt = GeneratedCollisionUnitReceipt(
            schema=UNIT_RECEIPT_SCHEMA,
            status=STATUS,
            unit_id=identifier,
            geometry=(state_count, action_count, observer_count, 2),
            parity=parity,
            source_sha256s=tuple(_sha256(source) for source in sources),
            source_receipt_sha256s=tuple(
                audit.receipt.receipt_sha256 for audit in audits
            ),
            non_law_sha256s=tuple(
                audit.receipt.non_law_sha256 for audit in audits
            ),
            selected_completion_sha256s=tuple(
                audit.receipt.selected_completion_sha256 or ""
                for audit in audits
            ),
            fact_invariant_sha256s=(
                _sha256(invariants[0].canonical_bytes()),
                _sha256(invariants[1].canonical_bytes()),
            ),
            law_query=law_query,
            late_query=late_query,
            late_answer_indices=answer_indices,  # type: ignore[arg-type]
            gates=tuple(sorted(gates.items())),
        )
        return GeneratedCollisionUnit(
            schema=UNIT_SCHEMA,
            status=STATUS,
            unit_id=identifier,
            parity=parity,
            sources=sources,
            late_query=late_query,
            source_audits=audits,  # type: ignore[arg-type]
            fact_invariants=invariants,
            receipt=receipt,
        )
    raise GeneratedCollisionFamilyError(
        f"{identifier} exhausted {config.maximum_generation_attempts} attempts"
    )


FEATURE_NAMES = (
    "states",
    "actions",
    "observers",
    "source_bytes",
    "visible_transition_count",
    "law_expected_index",
    "law_start_index",
    "law_depth",
    "law_action0_count",
    "law_action1_count",
    "law_action2_count",
    "law_observer_index",
    "query_start_index",
    "query_depth",
    "query_action0_count",
    "query_action1_count",
    "query_action2_count",
    "query_observer_index",
    "complete_action_cycle_count",
    "complete_action_fixed_points",
    "complete_action_max_cycle",
    "observer0_answer1_count",
    "observer1_answer1_count",
    "visible_source_index_sum",
    "visible_destination_index_sum",
)


def _shallow_features(
    source_payload: bytes,
    late_query: LateQuery,
    invariant: FactWorldInvariant,
) -> tuple[float, ...]:
    source = parse_source(source_payload)
    evidence = source.evidence
    clause = source.clauses[0]
    if not isinstance(clause, PathObservationClause):
        raise GeneratedCollisionFamilyError("generated law must be a path clause")
    law_actions = tuple(
        evidence.actions.index(action) for action in clause.actions
    )
    answer_index = evidence.answers.index(clause.expected)
    state_index = {state: index for index, state in enumerate(evidence.states)}
    observation_rows: list[list[int]] = [
        [] for _ in range(len(evidence.observers))
    ]
    for fact in evidence.observations:
        observation_rows[evidence.observers.index(fact.observer)].append(
            evidence.answers.index(fact.answer)
        )
    complete_cycles = [
        length
        for cycle_type in invariant.complete_action_cycle_types
        for length in cycle_type
    ]
    source_sum = sum(
        state_index[fact.source] for fact in evidence.transitions
    )
    destination_sum = sum(
        state_index[fact.destination] for fact in evidence.transitions
    )

    def action_count(word: Sequence[int], action: int) -> int:
        return sum(item == action for item in word)

    return (
        float(len(evidence.states)),
        float(len(evidence.actions)),
        float(len(evidence.observers)),
        float(len(source_payload)),
        float(len(evidence.transitions)),
        float(answer_index),
        float(evidence.states.index(clause.start)),
        float(len(law_actions)),
        float(action_count(law_actions, 0)),
        float(action_count(law_actions, 1)),
        float(action_count(law_actions, 2)),
        float(evidence.observers.index(clause.observer)),
        float(late_query.start_index),
        float(late_query.depth),
        float(action_count(late_query.action_indices, 0)),
        float(action_count(late_query.action_indices, 1)),
        float(action_count(late_query.action_indices, 2)),
        float(late_query.observer_index),
        float(len(complete_cycles)),
        float(sum(length == 1 for length in complete_cycles)),
        float(max(complete_cycles, default=0)),
        float(sum(observation_rows[0])) if observation_rows else 0.0,
        (
            float(sum(observation_rows[1]))
            if len(observation_rows) > 1
            else -1.0
        ),
        float(source_sum),
        float(destination_sum),
    )


def _family_splits(
    units: Sequence[GeneratedCollisionUnit],
) -> dict[str, str]:
    strata: dict[tuple[tuple[int, int, int, int], int], list[str]] = defaultdict(
        list
    )
    for unit in units:
        strata[(unit.receipt.geometry, unit.parity)].append(unit.unit_id)
    result: dict[str, str] = {}
    for key, identifiers in sorted(strata.items()):
        ordered = sorted(
            identifiers,
            key=lambda identifier: sha256(
                f"shortcut-split\0{key}\0{identifier}".encode()
            ).digest(),
        )
        if len(ordered) < 2:
            raise GeneratedCollisionFamilyError(
                "every geometry/parity stratum needs train and evaluation units"
            )
        cut = len(ordered) // 2
        for identifier in ordered[:cut]:
            result[identifier] = "train"
        for identifier in ordered[cut:]:
            result[identifier] = "evaluation"
    return result


def _examples(
    units: Sequence[GeneratedCollisionUnit],
) -> tuple[ShallowExample, ...]:
    splits = _family_splits(units)
    examples: list[ShallowExample] = []
    for unit in units:
        for cell_index, (name, source, target) in enumerate(
            zip(
                CELL_NAMES,
                unit.sources,
                unit.receipt.late_answer_indices,
                strict=True,
            )
        ):
            invariant = unit.fact_invariants[0 if cell_index < 2 else 1]
            examples.append(
                ShallowExample(
                    unit_id=unit.unit_id,
                    split=splits[unit.unit_id],
                    cell_name=name,
                    features=_shallow_features(
                        source,
                        unit.late_query,
                        invariant,
                    ),
                    target=target,
                )
            )
    return tuple(examples)


def _majority_predictor(
    training: Sequence[ShallowExample],
) -> Callable[[tuple[float, ...]], int]:
    counts = Counter(example.target for example in training)
    prediction = 1 if counts[1] > counts[0] else 0
    return lambda _features: prediction


def _fit_stump(
    rows: Sequence[tuple[tuple[float, ...], int]],
) -> tuple[int, float, int, int] | None:
    if not rows:
        return None
    best: tuple[int, int, float, int, int] | None = None
    feature_count = len(rows[0][0])
    for feature in range(feature_count):
        values = sorted(set(vector[feature] for vector, _ in rows))
        thresholds = values[:1] + [
            (left + right) / 2.0
            for left, right in zip(values, values[1:], strict=False)
        ]
        for threshold in thresholds:
            for low, high in ((0, 1), (1, 0)):
                correct = sum(
                    (low if vector[feature] <= threshold else high) == target
                    for vector, target in rows
                )
                candidate = (correct, -feature, -threshold, low, high)
                if best is None or candidate > best:
                    best = candidate
    if best is None:
        return None
    _, negative_feature, negative_threshold, low, high = best
    return -negative_feature, -negative_threshold, low, high


@dataclass(frozen=True, slots=True)
class _Tree:
    prediction: int
    feature: int | None = None
    threshold: float = 0.0
    low: "_Tree | None" = None
    high: "_Tree | None" = None

    def predict(self, features: tuple[float, ...]) -> int:
        if self.feature is None:
            return self.prediction
        branch = self.low if features[self.feature] <= self.threshold else self.high
        if branch is None:
            return self.prediction
        return branch.predict(features)


def _fit_tree(
    rows: Sequence[tuple[tuple[float, ...], int]],
    depth: int,
) -> _Tree:
    counts = Counter(target for _, target in rows)
    prediction = 1 if counts[1] > counts[0] else 0
    if depth == 0 or len(counts) <= 1:
        return _Tree(prediction)
    stump = _fit_stump(rows)
    if stump is None:
        return _Tree(prediction)
    feature, threshold, _, _ = stump
    low_rows = [row for row in rows if row[0][feature] <= threshold]
    high_rows = [row for row in rows if row[0][feature] > threshold]
    if not low_rows or not high_rows:
        return _Tree(prediction)
    return _Tree(
        prediction=prediction,
        feature=feature,
        threshold=threshold,
        low=_fit_tree(low_rows, depth - 1),
        high=_fit_tree(high_rows, depth - 1),
    )


def _tree_predictor(
    training: Sequence[ShallowExample],
    *,
    depth: int,
) -> Callable[[tuple[float, ...]], int]:
    tree = _fit_tree(
        [(example.features, example.target) for example in training],
        depth,
    )
    return tree.predict


def _standardizer(
    training: Sequence[ShallowExample],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    width = len(training[0].features)
    means = tuple(
        sum(example.features[index] for example in training) / len(training)
        for index in range(width)
    )
    scales = []
    for index, mean in enumerate(means):
        variance = (
            sum(
                (example.features[index] - mean) ** 2
                for example in training
            )
            / len(training)
        )
        scales.append(math.sqrt(variance) or 1.0)
    return means, tuple(scales)


def _normalize(
    vector: Sequence[float],
    means: Sequence[float],
    scales: Sequence[float],
) -> tuple[float, ...]:
    return tuple(
        (value - mean) / scale
        for value, mean, scale in zip(vector, means, scales, strict=True)
    )


def _nearest_neighbor_predictor(
    training: Sequence[ShallowExample],
) -> Callable[[tuple[float, ...]], int]:
    means, scales = _standardizer(training)
    rows = tuple(
        (_normalize(example.features, means, scales), example.target)
        for example in training
    )

    def predict(features: tuple[float, ...]) -> int:
        vector = _normalize(features, means, scales)
        _, target = min(
            (
                sum(
                    (left - right) ** 2
                    for left, right in zip(vector, row, strict=True)
                ),
                label,
            )
            for row, label in rows
        )
        return target

    return predict


def _logistic_predictor(
    training: Sequence[ShallowExample],
) -> Callable[[tuple[float, ...]], int]:
    means, scales = _standardizer(training)
    rows = tuple(
        (_normalize(example.features, means, scales), example.target)
        for example in training
    )
    weights = [0.0] * (len(means) + 1)
    for iteration in range(600):
        gradients = [0.0] * len(weights)
        for vector, target in rows:
            score = weights[0] + sum(
                weight * value
                for weight, value in zip(weights[1:], vector, strict=True)
            )
            score = max(-30.0, min(30.0, score))
            probability = 1.0 / (1.0 + math.exp(-score))
            error = probability - target
            gradients[0] += error
            for index, value in enumerate(vector, start=1):
                gradients[index] += error * value
        rate = 0.15 / math.sqrt(iteration + 1)
        weights[0] -= rate * gradients[0] / len(rows)
        for index in range(1, len(weights)):
            regularized = gradients[index] / len(rows) + 0.01 * weights[index]
            weights[index] -= rate * regularized

    def predict(features: tuple[float, ...]) -> int:
        vector = _normalize(features, means, scales)
        score = weights[0] + sum(
            weight * value
            for weight, value in zip(weights[1:], vector, strict=True)
        )
        return int(score > 0.0)

    return predict


def _exact_unit_flip_tail(
    evaluation: Sequence[ShallowExample],
    predictions: Sequence[int],
) -> float:
    by_unit: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for example, prediction in zip(evaluation, predictions, strict=True):
        by_unit[example.unit_id].append((example.target, prediction))
    distribution: dict[int, int] = {0: 1}
    observed = 0
    for rows in by_unit.values():
        correct = sum(target == prediction for target, prediction in rows)
        observed += correct
        flipped = len(rows) - correct
        updated: dict[int, int] = defaultdict(int)
        for total, multiplicity in distribution.items():
            updated[total + correct] += multiplicity
            updated[total + flipped] += multiplicity
        distribution = dict(updated)
    denominator = 2 ** len(by_unit)
    return sum(
        multiplicity
        for total, multiplicity in distribution.items()
        if total >= observed
    ) / denominator


def audit_shallow_shortcuts(
    examples: Sequence[ShallowExample],
    *,
    accuracy_ceiling: float = 0.625,
    alpha: float = 0.05,
) -> ShortcutAuditReceipt:
    """Fit preregistered attacks and evaluate only on held-out whole units."""

    if not examples:
        raise GeneratedCollisionFamilyError("shortcut audit needs examples")
    if any(len(example.features) != len(FEATURE_NAMES) for example in examples):
        raise GeneratedCollisionFamilyError("shortcut feature width drifted")
    training = tuple(example for example in examples if example.split == "train")
    evaluation = tuple(
        example for example in examples if example.split == "evaluation"
    )
    if not training or not evaluation:
        raise GeneratedCollisionFamilyError(
            "shortcut audit needs train and evaluation examples"
        )
    train_units = {example.unit_id for example in training}
    evaluation_units = {example.unit_id for example in evaluation}
    if train_units & evaluation_units:
        raise GeneratedCollisionFamilyError(
            "shortcut split leaked cells from one unit across partitions"
        )
    if Counter(example.target for example in evaluation) != Counter(
        {0: len(evaluation) // 2, 1: len(evaluation) // 2}
    ):
        raise GeneratedCollisionFamilyError(
            "shortcut evaluation partition is not exactly balanced"
        )

    factories: tuple[
        tuple[str, Callable[[Sequence[ShallowExample]], Callable[[tuple[float, ...]], int]]],
        ...,
    ] = (
        ("majority", _majority_predictor),
        ("decision-stump", lambda rows: _tree_predictor(rows, depth=1)),
        ("decision-tree-depth2", lambda rows: _tree_predictor(rows, depth=2)),
        ("nearest-neighbor", _nearest_neighbor_predictor),
        ("logistic-linear", _logistic_predictor),
    )
    audits: list[ClassifierAudit] = []
    for name, factory in factories:
        predictor = factory(training)
        predictions = tuple(
            predictor(example.features) for example in evaluation
        )
        correct = sum(
            prediction == example.target
            for prediction, example in zip(predictions, evaluation, strict=True)
        )
        accuracy = correct / len(evaluation)
        tail = _exact_unit_flip_tail(evaluation, predictions)
        audits.append(
            ClassifierAudit(
                name=name,
                evaluation_examples=len(evaluation),
                correct=correct,
                accuracy=accuracy,
                exact_unit_flip_tail_probability=tail,
                accuracy_gate=accuracy <= accuracy_ceiling,
                randomization_gate=tail >= alpha,
            )
        )
    chance_compatible = all(
        audit.accuracy_gate and audit.randomization_gate for audit in audits
    )
    return ShortcutAuditReceipt(
        schema=SHORTCUT_RECEIPT_SCHEMA,
        status=STATUS,
        feature_names=FEATURE_NAMES,
        training_unit_ids=tuple(sorted(train_units)),
        evaluation_unit_ids=tuple(sorted(evaluation_units)),
        training_examples=len(training),
        evaluation_examples=len(evaluation),
        target_counts=tuple(sorted(Counter(e.target for e in examples).items())),
        classifiers=tuple(audits),
        chance_compatible=chance_compatible,
        decision=(
            "mechanics-family-retained-no-reasoning-claim"
            if chance_compatible
            else "no-go-shallow-shortcut-leakage"
        ),
    )


def audit_generated_collision_family(
    config: GeneratedCollisionFamilyConfig | None = None,
) -> GeneratedCollisionFamilyAudit:
    """Generate and exactly audit a balanced nontrivial collision family."""

    actual = config or GeneratedCollisionFamilyConfig()
    units = tuple(_generate_unit(actual, index) for index in range(actual.unit_count))
    shortcut = audit_shallow_shortcuts(
        _examples(units),
        accuracy_ceiling=actual.shortcut_accuracy_ceiling,
        alpha=actual.shortcut_alpha,
    )
    all_unit_gates = all(all(unit.gates.values()) for unit in units)
    targets = Counter(
        answer
        for unit in units
        for answer in unit.receipt.late_answer_indices
    )
    geometries = tuple(sorted(set(unit.receipt.geometry for unit in units)))
    query_addresses = {
        (
            unit.receipt.geometry,
            unit.late_query.start_index,
            unit.late_query.action_indices,
            unit.late_query.observer_index,
        )
        for unit in units
    }
    decision = (
        "mechanics-family-retained-no-reasoning-claim"
        if all_unit_gates and shortcut.chance_compatible
        else "no-go-shallow-shortcut-leakage"
    )
    receipt = GeneratedCollisionFamilyReceipt(
        schema=FAMILY_RECEIPT_SCHEMA,
        status=STATUS,
        seed=actual.seed,
        unit_count=len(units),
        cell_count=4 * len(units),
        geometries=geometries,
        query_address_count=len(query_addresses),
        target_counts=tuple(sorted(targets.items())),
        unit_receipt_sha256s=tuple(
            unit.receipt.receipt_sha256 for unit in units
        ),
        shortcut_audit_receipt_sha256=shortcut.receipt_sha256,
        all_exact_unit_gates_passed=all_unit_gates,
        shortcut_chance_gates_passed=shortcut.chance_compatible,
        promotion_eligible=False,
        decision=decision,
        claim_boundary=(
            "Generated exact CPU mechanics and bounded held-out shortcut "
            "attacks only; this is not a neural, source-sealed, or reasoning claim."
        ),
    )
    return GeneratedCollisionFamilyAudit(
        config=actual,
        units=units,
        shortcut_audit=shortcut,
        receipt=receipt,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--unit-count", type=int, default=32)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    audit = audit_generated_collision_family(
        GeneratedCollisionFamilyConfig(
            seed=args.seed,
            unit_count=args.unit_count,
        )
    )
    print(audit.receipt.canonical_bytes().decode("ascii"), end="")
    return 0 if audit.receipt.decision != "no-go-shallow-shortcut-leakage" else 2


if __name__ == "__main__":
    raise SystemExit(main())
