#!/usr/bin/env python3
"""Exact CPU mechanics for the smallest SSQAC degree-three law collision.

The module deliberately contains no neural code and writes no artifacts.  It
defines a strict canonical JSON source, exhaustively enumerates every machine
completion admitted by the direct facts, and audits whether an episode-local
law makes one completion uniquely legal.

The minimal fixture has four anonymous states, one permutation action, one
binary observer, and two hidden transition cells.  The direct facts admit two
completions.  Their keyed behavior is identical through depth two and first
differs at depth three.  Law twins exchange the expected and alternate opaque
answer keys in one degree-three clause, preserving all key occurrence counts
while selecting opposite completions.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from itertools import permutations, product
import json
import math
import re
from typing import Iterable, Mapping, Sequence


SOURCE_SCHEMA = "episode_functor_law_collision_source_v1"
PAIR_SCHEMA = "episode_functor_law_collision_pair_v1"
QUADRUPLE_SCHEMA = "episode_functor_law_collision_quadruple_v1"
VERSION_SPACE_RECEIPT_SCHEMA = "episode_functor_version_space_receipt_v1"
COLLISION_RECEIPT_SCHEMA = "episode_functor_law_collision_receipt_v1"
QUADRUPLE_RECEIPT_SCHEMA = "episode_functor_law_collision_quadruple_receipt_v1"
MINIMALITY_RECEIPT_SCHEMA = "episode_functor_law_collision_minimality_v1"
STATUS = "exploratory_cpu_only_not_neural_not_sealed"
DEFAULT_SEED = "ssqac-law-collision-minimal-v1"
MAX_STATES = 13
MAX_ACTIONS = 5
MAX_OBSERVERS = 4
MAX_ANSWERS = 6
MAX_LAW_DEGREE = 4
MAX_COMPLETIONS = 100_000

_OPAQUE_KEY = re.compile(r"k_[0-9a-f]{16}")
_CLAUSE_ID = re.compile(r"[a-z][a-z0-9-]{0,63}")


class LawCollisionBoardError(ValueError):
    """A source, completion, intervention, or exact audit failed closed."""


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


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise LawCollisionBoardError(
            f"{context} fields differ: missing={missing}, unknown={unknown}"
        )


def _require_list(value: object, *, context: str) -> list[object]:
    if type(value) is not list:
        raise LawCollisionBoardError(f"{context} must be a JSON array")
    return value


def _require_string(value: object, *, context: str) -> str:
    if type(value) is not str:
        raise LawCollisionBoardError(f"{context} must be a JSON string")
    return value


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise LawCollisionBoardError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _decode_json(payload: bytes) -> dict[str, object]:
    if type(payload) is not bytes:
        raise LawCollisionBoardError("source must be immutable bytes")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise LawCollisionBoardError("source must be canonical ASCII JSON") from exc
    try:
        value = json.loads(text, object_pairs_hook=_object_without_duplicate_keys)
    except (json.JSONDecodeError, LawCollisionBoardError) as exc:
        if isinstance(exc, LawCollisionBoardError):
            raise
        raise LawCollisionBoardError("source is not valid JSON") from exc
    if type(value) is not dict:
        raise LawCollisionBoardError("source root must be a JSON object")
    return value


def _validate_opaque_keys(keys: Sequence[str], *, context: str) -> None:
    if not keys:
        raise LawCollisionBoardError(f"{context} inventory must not be empty")
    if len(set(keys)) != len(keys):
        raise LawCollisionBoardError(f"{context} keys must be unique")
    if any(_OPAQUE_KEY.fullmatch(key) is None for key in keys):
        raise LawCollisionBoardError(
            f"{context} keys must match {_OPAQUE_KEY.pattern!r}"
        )


def _validate_clause_id(clause_id: str) -> None:
    if _CLAUSE_ID.fullmatch(clause_id) is None:
        raise LawCollisionBoardError("clause identifier is not canonical")


@dataclass(frozen=True, slots=True)
class TransitionFact:
    action: str
    source: str
    destination: str


@dataclass(frozen=True, slots=True)
class ObservationFact:
    observer: str
    state: str
    answer: str


@dataclass(frozen=True, slots=True)
class PathObservationClause:
    clause_id: str
    start: str
    actions: tuple[str, ...]
    observer: str
    expected: str
    alternate: str

    def __post_init__(self) -> None:
        _validate_clause_id(self.clause_id)
        if not 1 <= len(self.actions) <= MAX_LAW_DEGREE:
            raise LawCollisionBoardError(
                f"path law degree must be in [1, {MAX_LAW_DEGREE}]"
            )
        if self.expected == self.alternate:
            raise LawCollisionBoardError(
                "path law expected and alternate answers must differ"
            )

    @property
    def degree(self) -> int:
        return len(self.actions)


@dataclass(frozen=True, slots=True)
class VisibleObservationClause:
    clause_id: str
    observer: str
    state: str
    answer: str

    def __post_init__(self) -> None:
        _validate_clause_id(self.clause_id)


LawClause = PathObservationClause | VisibleObservationClause


@dataclass(frozen=True, slots=True)
class DirectEvidence:
    states: tuple[str, ...]
    actions: tuple[str, ...]
    observers: tuple[str, ...]
    answers: tuple[str, ...]
    action_rows_are_permutations: bool
    transitions: tuple[TransitionFact, ...]
    observations: tuple[ObservationFact, ...]

    def __post_init__(self) -> None:
        if not 2 <= len(self.states) <= MAX_STATES:
            raise LawCollisionBoardError("state cardinality leaves the board bounds")
        if not 1 <= len(self.actions) <= MAX_ACTIONS:
            raise LawCollisionBoardError("action cardinality leaves the board bounds")
        if not 1 <= len(self.observers) <= MAX_OBSERVERS:
            raise LawCollisionBoardError("observer cardinality leaves the board bounds")
        if not 2 <= len(self.answers) <= MAX_ANSWERS:
            raise LawCollisionBoardError("answer cardinality leaves the board bounds")
        if self.action_rows_are_permutations is not True:
            raise LawCollisionBoardError(
                "the exact completion auditor requires permutation action rows"
            )
        inventories = (
            self.states,
            self.actions,
            self.observers,
            self.answers,
        )
        for name, keys in zip(
            ("state", "action", "observer", "answer"),
            inventories,
            strict=True,
        ):
            if type(keys) is not tuple:
                raise LawCollisionBoardError(f"{name} inventory must be a tuple")
            _validate_opaque_keys(keys, context=name)
        all_keys = tuple(key for inventory in inventories for key in inventory)
        if len(set(all_keys)) != len(all_keys):
            raise LawCollisionBoardError(
                "opaque keys must be disjoint across all inventories"
            )

        state_set = set(self.states)
        action_set = set(self.actions)
        observer_set = set(self.observers)
        answer_set = set(self.answers)
        transition_coordinates: set[tuple[str, str]] = set()
        transition_destinations: dict[str, set[str]] = {
            action: set() for action in self.actions
        }
        for fact in self.transitions:
            if (
                fact.action not in action_set
                or fact.source not in state_set
                or fact.destination not in state_set
            ):
                raise LawCollisionBoardError(
                    "transition fact references an unknown opaque key"
                )
            coordinate = (fact.action, fact.source)
            if coordinate in transition_coordinates:
                raise LawCollisionBoardError(
                    "transition evidence repeats an action/source coordinate"
                )
            transition_coordinates.add(coordinate)
            destinations = transition_destinations[fact.action]
            if fact.destination in destinations:
                raise LawCollisionBoardError(
                    "visible destinations violate the permutation row constraint"
                )
            destinations.add(fact.destination)

        observation_coordinates: set[tuple[str, str]] = set()
        for fact in self.observations:
            if (
                fact.observer not in observer_set
                or fact.state not in state_set
                or fact.answer not in answer_set
            ):
                raise LawCollisionBoardError(
                    "observation fact references an unknown opaque key"
                )
            coordinate = (fact.observer, fact.state)
            if coordinate in observation_coordinates:
                raise LawCollisionBoardError(
                    "observation evidence repeats an observer/state coordinate"
                )
            observation_coordinates.add(coordinate)
        expected_observations = {
            (observer, state) for observer in self.observers for state in self.states
        }
        if observation_coordinates != expected_observations:
            raise LawCollisionBoardError(
                "minimal board requires a complete observer table"
            )


@dataclass(frozen=True, slots=True)
class LawCollisionSource:
    evidence: DirectEvidence
    law_present: bool
    clauses: tuple[LawClause, ...]

    def __post_init__(self) -> None:
        if not self.law_present and self.clauses:
            raise LawCollisionBoardError("law-absent source cannot retain clauses")
        clause_ids = tuple(clause.clause_id for clause in self.clauses)
        if len(set(clause_ids)) != len(clause_ids):
            raise LawCollisionBoardError("law clause identifiers must be unique")
        state_set = set(self.evidence.states)
        action_set = set(self.evidence.actions)
        observer_set = set(self.evidence.observers)
        answer_set = set(self.evidence.answers)
        visible_observations = {
            (fact.observer, fact.state, fact.answer)
            for fact in self.evidence.observations
        }
        for clause in self.clauses:
            if isinstance(clause, PathObservationClause):
                if (
                    clause.start not in state_set
                    or clause.observer not in observer_set
                    or clause.expected not in answer_set
                    or clause.alternate not in answer_set
                    or any(action not in action_set for action in clause.actions)
                ):
                    raise LawCollisionBoardError(
                        "path law references an unknown opaque key"
                    )
            elif isinstance(clause, VisibleObservationClause):
                if (
                    clause.observer,
                    clause.state,
                    clause.answer,
                ) not in visible_observations:
                    raise LawCollisionBoardError(
                        "visible-observation clause is not redundant with direct facts"
                    )
            else:
                raise LawCollisionBoardError("unknown in-memory law clause type")


@dataclass(frozen=True, slots=True)
class MachineCompletion:
    transitions: tuple[tuple[int, ...], ...]
    observations: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not self.transitions or not self.observations:
            raise LawCollisionBoardError("completion tables must not be empty")
        state_count = len(self.transitions[0])
        if any(
            len(row) != state_count or sorted(row) != list(range(state_count))
            for row in self.transitions
        ):
            raise LawCollisionBoardError(
                "completion action rows must be full permutations"
            )
        if any(len(row) != state_count for row in self.observations):
            raise LawCollisionBoardError(
                "completion observer rows must cover every state"
            )

    def canonical_bytes(self) -> bytes:
        return _canonical_json(
            {
                "observations": self.observations,
                "transitions": self.transitions,
            }
        )

    @property
    def structural_sha256(self) -> str:
        return _sha256(self.canonical_bytes())


@dataclass(frozen=True, slots=True)
class QueryWitness:
    start_index: int
    action_indices: tuple[int, ...]
    observer_index: int
    left_answer_index: int
    right_answer_index: int

    @property
    def depth(self) -> int:
        return len(self.action_indices)


@dataclass(frozen=True, slots=True)
class VersionSpaceReceipt:
    schema: str
    status: str
    source_sha256: str
    non_law_sha256: str
    semantic_source_sha256: str
    law_present: bool
    clause_ids: tuple[str, ...]
    direct_completion_count: int
    direct_behavior_class_count: int
    law_completion_count: int
    law_behavior_class_count: int
    resolution: str
    direct_completion_sha256s: tuple[str, ...]
    law_completion_sha256s: tuple[str, ...]
    selected_completion_sha256: str | None
    direct_distinguishing_query: QueryWitness | None

    def canonical_bytes(self) -> bytes:
        return _canonical_json(asdict(self))

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.canonical_bytes())


@dataclass(frozen=True, slots=True)
class SourceAudit:
    source: LawCollisionSource
    direct_completions: tuple[MachineCompletion, ...]
    law_completions: tuple[MachineCompletion, ...]
    receipt: VersionSpaceReceipt

    @property
    def selected_completion(self) -> MachineCompletion | None:
        if len(self.law_completions) == 1:
            return self.law_completions[0]
        return None


@dataclass(frozen=True, slots=True)
class MinimalityReceipt:
    schema: str
    fixture_family: str
    searched_smaller_state_counts: tuple[int, ...]
    tested_smaller_candidates: int
    smaller_witness_count: int
    witness_state_count: int
    witness_first_distinguishing_depth: int

    def canonical_bytes(self) -> bytes:
        return _canonical_json(asdict(self))

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.canonical_bytes())


@dataclass(frozen=True, slots=True)
class LawCollisionPair:
    schema: str
    status: str
    seed: str
    left_source: bytes
    right_source: bytes

    def __post_init__(self) -> None:
        if self.schema != PAIR_SCHEMA:
            raise LawCollisionBoardError("unexpected collision pair schema")
        if self.status != STATUS:
            raise LawCollisionBoardError("collision pair status is not CPU-only")
        if not self.seed:
            raise LawCollisionBoardError("collision pair seed must not be empty")


@dataclass(frozen=True, slots=True)
class LateQuerySpec:
    start: str
    actions: tuple[str, ...]
    observer: str

    def __post_init__(self) -> None:
        if not self.actions:
            raise LawCollisionBoardError("late query must contain at least one action")


@dataclass(frozen=True, slots=True)
class LawCollisionQuadruple:
    schema: str
    status: str
    seed: str
    f0_l0_source: bytes
    f0_l1_source: bytes
    f1_l0_source: bytes
    f1_l1_source: bytes
    late_query: LateQuerySpec

    def __post_init__(self) -> None:
        if self.schema != QUADRUPLE_SCHEMA:
            raise LawCollisionBoardError("unexpected collision quadruple schema")
        if self.status != STATUS:
            raise LawCollisionBoardError("collision quadruple status is not CPU-only")
        if not self.seed:
            raise LawCollisionBoardError(
                "collision quadruple seed must not be empty"
            )


@dataclass(frozen=True, slots=True)
class CollisionPairReceipt:
    schema: str
    status: str
    seed: str
    left_source_receipt_sha256: str
    right_source_receipt_sha256: str
    low_order_signature_sha256: str
    left_selected_completion_sha256: str | None
    right_selected_completion_sha256: str | None
    first_distinguishing_query: QueryWitness | None
    minimality_receipt_sha256: str
    gates: tuple[tuple[str, bool], ...]

    def canonical_bytes(self) -> bytes:
        return _canonical_json(asdict(self))

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.canonical_bytes())


@dataclass(frozen=True, slots=True)
class CollisionPairAudit:
    left: SourceAudit
    right: SourceAudit
    low_order_signature: dict[str, object]
    minimality: MinimalityReceipt
    receipt: CollisionPairReceipt

    @property
    def gates(self) -> dict[str, bool]:
        return dict(self.receipt.gates)


@dataclass(frozen=True, slots=True)
class CollisionQuadrupleReceipt:
    schema: str
    status: str
    seed: str
    f0_pair_receipt_sha256: str
    f1_pair_receipt_sha256: str
    source_receipt_sha256s: tuple[str, ...]
    selected_completion_sha256s: tuple[str, ...]
    late_answer_indices: tuple[int, ...]
    classification: str
    known_shortcut: str
    promotion_eligible: bool
    gates: tuple[tuple[str, bool], ...]

    def canonical_bytes(self) -> bytes:
        return _canonical_json(asdict(self))

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.canonical_bytes())


@dataclass(frozen=True, slots=True)
class CollisionQuadrupleAudit:
    f0_pair: CollisionPairAudit
    f1_pair: CollisionPairAudit
    source_audits: tuple[SourceAudit, ...]
    receipt: CollisionQuadrupleReceipt

    @property
    def gates(self) -> dict[str, bool]:
        return dict(self.receipt.gates)


def _fact_to_object(fact: TransitionFact | ObservationFact) -> dict[str, str]:
    if isinstance(fact, TransitionFact):
        return {
            "action": fact.action,
            "destination": fact.destination,
            "source": fact.source,
        }
    return {
        "answer": fact.answer,
        "observer": fact.observer,
        "state": fact.state,
    }


def _clause_to_object(clause: LawClause) -> dict[str, object]:
    if isinstance(clause, PathObservationClause):
        return {
            "actions": clause.actions,
            "alternate": clause.alternate,
            "expected": clause.expected,
            "id": clause.clause_id,
            "kind": "path-observation",
            "observer": clause.observer,
            "start": clause.start,
        }
    return {
        "answer": clause.answer,
        "id": clause.clause_id,
        "kind": "visible-observation",
        "observer": clause.observer,
        "state": clause.state,
    }


def _machine_to_object(evidence: DirectEvidence) -> dict[str, object]:
    return {
        "action_rows_are_permutations": evidence.action_rows_are_permutations,
        "actions": evidence.actions,
        "answers": evidence.answers,
        "observations": tuple(_fact_to_object(fact) for fact in evidence.observations),
        "observers": evidence.observers,
        "states": evidence.states,
        "transitions": tuple(_fact_to_object(fact) for fact in evidence.transitions),
    }


def encode_non_law(evidence: DirectEvidence) -> bytes:
    """Encode the byte-identical direct-evidence portion of a source."""

    return _canonical_json({"machine": _machine_to_object(evidence)})


def encode_source(source: LawCollisionSource) -> bytes:
    """Encode one source in the only accepted canonical representation."""

    payload: dict[str, object] = {
        "machine": _machine_to_object(source.evidence),
        "schema": SOURCE_SCHEMA,
    }
    if source.law_present:
        payload["law"] = {
            "clauses": tuple(_clause_to_object(clause) for clause in source.clauses)
        }
    return _canonical_json(payload)


def _parse_string_inventory(value: object, *, context: str) -> tuple[str, ...]:
    return tuple(
        _require_string(item, context=f"{context} item")
        for item in _require_list(value, context=context)
    )


def _parse_transition(value: object) -> TransitionFact:
    if type(value) is not dict:
        raise LawCollisionBoardError("transition record must be an object")
    _require_exact_keys(
        value,
        {"action", "destination", "source"},
        context="transition record",
    )
    return TransitionFact(
        action=_require_string(value["action"], context="transition action"),
        source=_require_string(value["source"], context="transition source"),
        destination=_require_string(
            value["destination"],
            context="transition destination",
        ),
    )


def _parse_observation(value: object) -> ObservationFact:
    if type(value) is not dict:
        raise LawCollisionBoardError("observation record must be an object")
    _require_exact_keys(
        value,
        {"answer", "observer", "state"},
        context="observation record",
    )
    return ObservationFact(
        observer=_require_string(value["observer"], context="observation observer"),
        state=_require_string(value["state"], context="observation state"),
        answer=_require_string(value["answer"], context="observation answer"),
    )


def _parse_clause(value: object) -> LawClause:
    if type(value) is not dict:
        raise LawCollisionBoardError("law clause must be an object")
    kind = _require_string(value.get("kind"), context="law clause kind")
    if kind == "path-observation":
        _require_exact_keys(
            value,
            {
                "actions",
                "alternate",
                "expected",
                "id",
                "kind",
                "observer",
                "start",
            },
            context="path-observation clause",
        )
        return PathObservationClause(
            clause_id=_require_string(value["id"], context="law clause id"),
            start=_require_string(value["start"], context="law start"),
            actions=tuple(
                _require_string(action, context="law action")
                for action in _require_list(
                    value["actions"],
                    context="law actions",
                )
            ),
            observer=_require_string(value["observer"], context="law observer"),
            expected=_require_string(value["expected"], context="law expected"),
            alternate=_require_string(value["alternate"], context="law alternate"),
        )
    if kind == "visible-observation":
        _require_exact_keys(
            value,
            {"answer", "id", "kind", "observer", "state"},
            context="visible-observation clause",
        )
        return VisibleObservationClause(
            clause_id=_require_string(value["id"], context="law clause id"),
            observer=_require_string(value["observer"], context="law observer"),
            state=_require_string(value["state"], context="law state"),
            answer=_require_string(value["answer"], context="law answer"),
        )
    raise LawCollisionBoardError(f"unknown law clause kind {kind!r}")


def parse_source(payload: bytes) -> LawCollisionSource:
    """Parse a canonical source with duplicate and unknown fields rejected."""

    value = _decode_json(payload)
    if "law" in value:
        expected_root = {"law", "machine", "schema"}
    else:
        expected_root = {"machine", "schema"}
    _require_exact_keys(value, expected_root, context="source")
    if value["schema"] != SOURCE_SCHEMA:
        raise LawCollisionBoardError("unexpected source schema")

    machine = value["machine"]
    if type(machine) is not dict:
        raise LawCollisionBoardError("machine must be an object")
    _require_exact_keys(
        machine,
        {
            "action_rows_are_permutations",
            "actions",
            "answers",
            "observations",
            "observers",
            "states",
            "transitions",
        },
        context="machine",
    )
    permutation_flag = machine["action_rows_are_permutations"]
    if type(permutation_flag) is not bool:
        raise LawCollisionBoardError(
            "action_rows_are_permutations must be a JSON boolean"
        )
    evidence = DirectEvidence(
        states=_parse_string_inventory(machine["states"], context="states"),
        actions=_parse_string_inventory(machine["actions"], context="actions"),
        observers=_parse_string_inventory(machine["observers"], context="observers"),
        answers=_parse_string_inventory(machine["answers"], context="answers"),
        action_rows_are_permutations=permutation_flag,
        transitions=tuple(
            _parse_transition(record)
            for record in _require_list(
                machine["transitions"],
                context="transitions",
            )
        ),
        observations=tuple(
            _parse_observation(record)
            for record in _require_list(
                machine["observations"],
                context="observations",
            )
        ),
    )

    clauses: tuple[LawClause, ...] = ()
    law_present = "law" in value
    if law_present:
        law = value["law"]
        if type(law) is not dict:
            raise LawCollisionBoardError("law must be an object")
        _require_exact_keys(law, {"clauses"}, context="law")
        clauses = tuple(
            _parse_clause(clause)
            for clause in _require_list(law["clauses"], context="law clauses")
        )
    source = LawCollisionSource(
        evidence=evidence,
        law_present=law_present,
        clauses=clauses,
    )
    if encode_source(source) != payload:
        raise LawCollisionBoardError(
            "source is valid JSON but not the canonical source encoding"
        )
    return source


def enumerate_completions(
    evidence: DirectEvidence,
    *,
    maximum_completions: int = MAX_COMPLETIONS,
) -> tuple[MachineCompletion, ...]:
    """Exhaust every permutation-row completion consistent with direct facts."""

    if type(maximum_completions) is not int or maximum_completions < 1:
        raise LawCollisionBoardError("maximum_completions must be a positive integer")
    state_index = {key: index for index, key in enumerate(evidence.states)}
    action_index = {key: index for index, key in enumerate(evidence.actions)}
    observer_index = {key: index for index, key in enumerate(evidence.observers)}
    answer_index = {key: index for index, key in enumerate(evidence.answers)}
    state_count = len(evidence.states)

    visible_rows: list[list[int | None]] = [
        [None] * state_count for _ in evidence.actions
    ]
    for fact in evidence.transitions:
        visible_rows[action_index[fact.action]][state_index[fact.source]] = state_index[
            fact.destination
        ]

    row_options: list[tuple[tuple[int, ...], ...]] = []
    completion_count = 1
    for row in visible_rows:
        missing_sources = tuple(
            source for source, destination in enumerate(row) if destination is None
        )
        used_destinations = {
            destination for destination in row if destination is not None
        }
        missing_destinations = tuple(
            destination
            for destination in range(state_count)
            if destination not in used_destinations
        )
        if len(missing_sources) != len(missing_destinations):
            raise LawCollisionBoardError(
                "visible permutation evidence has unequal missing coordinates"
            )
        completion_count *= math.factorial(len(missing_sources))
        if completion_count > maximum_completions:
            raise LawCollisionBoardError(
                "direct version space exceeds the exhaustive completion bound"
            )
        options: list[tuple[int, ...]] = []
        for destinations in permutations(missing_destinations):
            completed = list(row)
            for source, destination in zip(
                missing_sources,
                destinations,
                strict=True,
            ):
                completed[source] = destination
            if any(destination is None for destination in completed):
                raise LawCollisionBoardError("completion left an action cell unset")
            options.append(tuple(int(destination) for destination in completed))
        row_options.append(tuple(sorted(options)))

    observations = [[-1] * state_count for _ in evidence.observers]
    for fact in evidence.observations:
        observations[observer_index[fact.observer]][state_index[fact.state]] = (
            answer_index[fact.answer]
        )
    if any(answer < 0 for row in observations for answer in row):
        raise LawCollisionBoardError("completion left an observer cell unset")
    observation_rows = tuple(tuple(row) for row in observations)

    completions = tuple(
        MachineCompletion(
            transitions=tuple(rows),
            observations=observation_rows,
        )
        for rows in product(*row_options)
    )
    if len(completions) != completion_count:
        raise LawCollisionBoardError("exhaustive completion cardinality drifted")
    return tuple(sorted(completions, key=lambda item: item.canonical_bytes()))


def execute_query(
    completion: MachineCompletion,
    evidence: DirectEvidence,
    *,
    start: str,
    actions: Sequence[str],
    observer: str,
) -> str:
    """Execute one key-addressed query against an enumerated completion."""

    try:
        state = evidence.states.index(start)
        action_indices = tuple(evidence.actions.index(action) for action in actions)
        observer_index = evidence.observers.index(observer)
    except ValueError as exc:
        raise LawCollisionBoardError("query references an unknown opaque key") from exc
    for action_index in action_indices:
        state = completion.transitions[action_index][state]
    answer_index = completion.observations[observer_index][state]
    return evidence.answers[answer_index]


def _completion_satisfies_clause(
    completion: MachineCompletion,
    evidence: DirectEvidence,
    clause: LawClause,
) -> bool:
    if isinstance(clause, PathObservationClause):
        return (
            execute_query(
                completion,
                evidence,
                start=clause.start,
                actions=clause.actions,
                observer=clause.observer,
            )
            == clause.expected
        )
    return (
        execute_query(
            completion,
            evidence,
            start=clause.state,
            actions=(),
            observer=clause.observer,
        )
        == clause.answer
    )


def filter_completions_by_law(
    source: LawCollisionSource,
    completions: Sequence[MachineCompletion],
) -> tuple[MachineCompletion, ...]:
    """Apply every structured law clause without invoking a solver."""

    if not source.law_present:
        return tuple(completions)
    return tuple(
        completion
        for completion in completions
        if all(
            _completion_satisfies_clause(completion, source.evidence, clause)
            for clause in source.clauses
        )
    )


def shortest_distinguishing_query(
    left: MachineCompletion,
    right: MachineCompletion,
) -> QueryWitness | None:
    """Find an exact shortest physical-key query distinguishing two machines."""

    if (
        len(left.transitions) != len(right.transitions)
        or len(left.observations) != len(right.observations)
        or len(left.transitions[0]) != len(right.transitions[0])
    ):
        raise LawCollisionBoardError(
            "behavior comparison requires identical machine geometry"
        )
    state_count = len(left.transitions[0])
    action_count = len(left.transitions)
    queue: deque[tuple[int, int, int, tuple[int, ...]]] = deque(
        (start, start, start, ()) for start in range(state_count)
    )
    seen = {(start, start, start) for start in range(state_count)}
    while queue:
        start, left_state, right_state, word = queue.popleft()
        for observer_index, (left_row, right_row) in enumerate(
            zip(left.observations, right.observations, strict=True)
        ):
            if left_row[left_state] != right_row[right_state]:
                return QueryWitness(
                    start_index=start,
                    action_indices=word,
                    observer_index=observer_index,
                    left_answer_index=left_row[left_state],
                    right_answer_index=right_row[right_state],
                )
        for action_index in range(action_count):
            next_left = left.transitions[action_index][left_state]
            next_right = right.transitions[action_index][right_state]
            marker = (start, next_left, next_right)
            if marker in seen:
                continue
            seen.add(marker)
            queue.append((start, next_left, next_right, (*word, action_index)))
    return None


def _behavior_classes(
    completions: Sequence[MachineCompletion],
) -> tuple[tuple[int, ...], QueryWitness | None]:
    representatives: list[int] = []
    class_ids: list[int] = []
    first_witness: QueryWitness | None = None
    for index, completion in enumerate(completions):
        assigned = None
        for class_id, representative in enumerate(representatives):
            witness = shortest_distinguishing_query(
                completion,
                completions[representative],
            )
            if witness is None:
                assigned = class_id
                break
            if first_witness is None:
                first_witness = witness
        if assigned is None:
            assigned = len(representatives)
            representatives.append(index)
        class_ids.append(assigned)
    if first_witness is None and len(representatives) > 1:
        for left_index in range(len(completions)):
            for right_index in range(left_index + 1, len(completions)):
                first_witness = shortest_distinguishing_query(
                    completions[left_index],
                    completions[right_index],
                )
                if first_witness is not None:
                    break
            if first_witness is not None:
                break
    return tuple(class_ids), first_witness


def _normalized_source_object(source: LawCollisionSource) -> dict[str, object]:
    evidence = source.evidence
    state_index = {key: index for index, key in enumerate(evidence.states)}
    action_index = {key: index for index, key in enumerate(evidence.actions)}
    observer_index = {key: index for index, key in enumerate(evidence.observers)}
    answer_index = {key: index for index, key in enumerate(evidence.answers)}
    clauses: list[dict[str, object]] = []
    for clause in source.clauses:
        if isinstance(clause, PathObservationClause):
            clauses.append(
                {
                    "actions": tuple(action_index[action] for action in clause.actions),
                    "alternate": answer_index[clause.alternate],
                    "expected": answer_index[clause.expected],
                    "id": clause.clause_id,
                    "kind": "path-observation",
                    "observer": observer_index[clause.observer],
                    "start": state_index[clause.start],
                }
            )
        else:
            clauses.append(
                {
                    "answer": answer_index[clause.answer],
                    "id": clause.clause_id,
                    "kind": "visible-observation",
                    "observer": observer_index[clause.observer],
                    "state": state_index[clause.state],
                }
            )
    return {
        "action_count": len(evidence.actions),
        "answer_count": len(evidence.answers),
        "clauses": clauses,
        "law_present": source.law_present,
        "observation_facts": tuple(
            (
                observer_index[fact.observer],
                state_index[fact.state],
                answer_index[fact.answer],
            )
            for fact in evidence.observations
        ),
        "observer_count": len(evidence.observers),
        "permutation_actions": evidence.action_rows_are_permutations,
        "state_count": len(evidence.states),
        "transition_facts": tuple(
            (
                action_index[fact.action],
                state_index[fact.source],
                state_index[fact.destination],
            )
            for fact in evidence.transitions
        ),
    }


def audit_version_space(payload: bytes) -> SourceAudit:
    """Exhaustively audit direct and law-filtered completion version spaces."""

    source = parse_source(payload)
    direct = enumerate_completions(source.evidence)
    law = filter_completions_by_law(source, direct)
    direct_classes, witness = _behavior_classes(direct)
    law_classes, _ = _behavior_classes(law)
    if not law:
        resolution = "inconsistent"
    elif len(law) == 1:
        resolution = "unique-completion"
    elif len(set(law_classes)) == 1:
        resolution = "behaviorally-unique"
    else:
        resolution = "ambiguous"
    selected = law[0].structural_sha256 if len(law) == 1 else None
    receipt = VersionSpaceReceipt(
        schema=VERSION_SPACE_RECEIPT_SCHEMA,
        status=STATUS,
        source_sha256=_sha256(payload),
        non_law_sha256=_sha256(encode_non_law(source.evidence)),
        semantic_source_sha256=_sha256(
            _canonical_json(_normalized_source_object(source))
        ),
        law_present=source.law_present,
        clause_ids=tuple(clause.clause_id for clause in source.clauses),
        direct_completion_count=len(direct),
        direct_behavior_class_count=len(set(direct_classes)),
        law_completion_count=len(law),
        law_behavior_class_count=len(set(law_classes)),
        resolution=resolution,
        direct_completion_sha256s=tuple(
            completion.structural_sha256 for completion in direct
        ),
        law_completion_sha256s=tuple(
            completion.structural_sha256 for completion in law
        ),
        selected_completion_sha256=selected,
        direct_distinguishing_query=witness,
    )
    return SourceAudit(
        source=source,
        direct_completions=direct,
        law_completions=law,
        receipt=receipt,
    )


def _derive_opaque_keys(seed: str, count: int, *, domain: str) -> tuple[str, ...]:
    keys: list[str] = []
    nonce = 0
    while len(keys) < count:
        digest = sha256(f"{domain}\0{seed}\0{nonce}".encode("utf-8")).hexdigest()
        key = f"k_{digest[:16]}"
        nonce += 1
        if key not in keys:
            keys.append(key)
    return tuple(keys)


def build_minimal_collision_pair(
    seed: str = DEFAULT_SEED,
) -> LawCollisionPair:
    """Build the deterministic four-state, first-difference-depth-three pair."""

    if type(seed) is not str or not seed:
        raise LawCollisionBoardError("seed must be a nonempty string")
    keys = _derive_opaque_keys(seed, 8, domain="law-collision-fixture")
    states = keys[:4]
    actions = (keys[4],)
    observers = (keys[5],)
    answers = keys[6:]
    evidence = DirectEvidence(
        states=states,
        actions=actions,
        observers=observers,
        answers=answers,
        action_rows_are_permutations=True,
        transitions=(
            TransitionFact(actions[0], states[1], states[2]),
            TransitionFact(actions[0], states[2], states[3]),
        ),
        observations=(
            ObservationFact(observers[0], states[0], answers[0]),
            ObservationFact(observers[0], states[1], answers[0]),
            ObservationFact(observers[0], states[2], answers[0]),
            ObservationFact(observers[0], states[3], answers[1]),
        ),
    )
    redundant = VisibleObservationClause(
        clause_id="redundant-visible",
        observer=observers[0],
        state=states[3],
        answer=answers[1],
    )
    left = LawCollisionSource(
        evidence=evidence,
        law_present=True,
        clauses=(
            PathObservationClause(
                clause_id="determining-path3",
                start=states[0],
                actions=(actions[0],) * 3,
                observer=observers[0],
                expected=answers[0],
                alternate=answers[1],
            ),
            redundant,
        ),
    )
    right = LawCollisionSource(
        evidence=evidence,
        law_present=True,
        clauses=(
            PathObservationClause(
                clause_id="determining-path3",
                start=states[0],
                actions=(actions[0],) * 3,
                observer=observers[0],
                expected=answers[1],
                alternate=answers[0],
            ),
            redundant,
        ),
    )
    return LawCollisionPair(
        schema=PAIR_SCHEMA,
        status=STATUS,
        seed=seed,
        left_source=encode_source(left),
        right_source=encode_source(right),
    )


def build_minimal_collision_quadruple(
    seed: str = DEFAULT_SEED,
) -> LawCollisionQuadruple:
    """Build the balanced F0L0/F0L1/F1L0/F1L1 degree-three falsifier.

    Both fact worlds expose two transition cells from a four-state permutation.
    The same byte-identical law twins select opposite completions in each fact
    world. A separate depth-five late query has the answer pattern 1,0,0,1.
    This balances the four cells but deliberately remains a minimal mechanics
    fixture: its two fact worlds and two laws admit a shallow XOR shortcut and
    cannot qualify a neural reasoning claim.
    """

    f0_pair = build_minimal_collision_pair(seed)
    f0_l0 = parse_source(f0_pair.left_source)
    f0_l1 = parse_source(f0_pair.right_source)
    evidence = f0_l0.evidence
    states = evidence.states
    action = evidence.actions[0]
    f1_evidence = DirectEvidence(
        states=states,
        actions=evidence.actions,
        observers=evidence.observers,
        answers=evidence.answers,
        action_rows_are_permutations=True,
        transitions=(
            TransitionFact(action, states[1], states[3]),
            TransitionFact(action, states[2], states[1]),
        ),
        observations=evidence.observations,
    )
    f1_l0 = LawCollisionSource(
        evidence=f1_evidence,
        law_present=True,
        clauses=f0_l0.clauses,
    )
    f1_l1 = LawCollisionSource(
        evidence=f1_evidence,
        law_present=True,
        clauses=f0_l1.clauses,
    )
    return LawCollisionQuadruple(
        schema=QUADRUPLE_SCHEMA,
        status=STATUS,
        seed=seed,
        f0_l0_source=f0_pair.left_source,
        f0_l1_source=f0_pair.right_source,
        f1_l0_source=encode_source(f1_l0),
        f1_l1_source=encode_source(f1_l1),
        late_query=LateQuerySpec(
            start=states[1],
            actions=(action,) * 5,
            observer=evidence.observers[0],
        ),
    )


def delete_law(payload: bytes) -> bytes:
    """Delete the complete law block while preserving direct evidence bytes."""

    source = parse_source(payload)
    return encode_source(
        LawCollisionSource(
            evidence=source.evidence,
            law_present=False,
            clauses=(),
        )
    )


def delete_clause(payload: bytes, clause_id: str) -> bytes:
    """Delete exactly one named clause, rejecting absent or ambiguous requests."""

    source = parse_source(payload)
    if not source.law_present:
        raise LawCollisionBoardError("cannot delete a clause from an absent law")
    matches = tuple(
        clause for clause in source.clauses if clause.clause_id == clause_id
    )
    if len(matches) != 1:
        raise LawCollisionBoardError(f"expected exactly one clause named {clause_id!r}")
    return encode_source(
        replace(
            source,
            clauses=tuple(
                clause for clause in source.clauses if clause.clause_id != clause_id
            ),
        )
    )


def add_redundant_visible_clause(
    payload: bytes,
    *,
    clause_id: str = "redundant-visible-copy",
) -> bytes:
    """Insert a direct observation already entailed by the evidence."""

    source = parse_source(payload)
    _validate_clause_id(clause_id)
    if any(clause.clause_id == clause_id for clause in source.clauses):
        raise LawCollisionBoardError("redundant clause id already exists")
    fact = source.evidence.observations[0]
    clause = VisibleObservationClause(
        clause_id=clause_id,
        observer=fact.observer,
        state=fact.state,
        answer=fact.answer,
    )
    return encode_source(
        LawCollisionSource(
            evidence=source.evidence,
            law_present=True,
            clauses=(*source.clauses, clause),
        )
    )


def recode_source(payload: bytes, key_map: Mapping[str, str]) -> bytes:
    """Apply an exact bijective opaque-key gauge recoding."""

    source = parse_source(payload)
    evidence = source.evidence
    all_keys = tuple(
        (
            *evidence.states,
            *evidence.actions,
            *evidence.observers,
            *evidence.answers,
        )
    )
    if set(key_map) != set(all_keys):
        raise LawCollisionBoardError(
            "key recoding domain must equal the complete opaque-key inventory"
        )
    recoded_values = tuple(key_map[key] for key in all_keys)
    _validate_opaque_keys(recoded_values, context="recoded")
    if len(set(recoded_values)) != len(recoded_values):
        raise LawCollisionBoardError("key recoding must be bijective")

    def mapped(key: str) -> str:
        return key_map[key]

    recoded_evidence = DirectEvidence(
        states=tuple(mapped(key) for key in evidence.states),
        actions=tuple(mapped(key) for key in evidence.actions),
        observers=tuple(mapped(key) for key in evidence.observers),
        answers=tuple(mapped(key) for key in evidence.answers),
        action_rows_are_permutations=evidence.action_rows_are_permutations,
        transitions=tuple(
            TransitionFact(
                mapped(fact.action),
                mapped(fact.source),
                mapped(fact.destination),
            )
            for fact in evidence.transitions
        ),
        observations=tuple(
            ObservationFact(
                mapped(fact.observer),
                mapped(fact.state),
                mapped(fact.answer),
            )
            for fact in evidence.observations
        ),
    )
    recoded_clauses: list[LawClause] = []
    for clause in source.clauses:
        if isinstance(clause, PathObservationClause):
            recoded_clauses.append(
                PathObservationClause(
                    clause_id=clause.clause_id,
                    start=mapped(clause.start),
                    actions=tuple(mapped(action) for action in clause.actions),
                    observer=mapped(clause.observer),
                    expected=mapped(clause.expected),
                    alternate=mapped(clause.alternate),
                )
            )
        else:
            recoded_clauses.append(
                VisibleObservationClause(
                    clause_id=clause.clause_id,
                    observer=mapped(clause.observer),
                    state=mapped(clause.state),
                    answer=mapped(clause.answer),
                )
            )
    return encode_source(
        LawCollisionSource(
            evidence=recoded_evidence,
            law_present=source.law_present,
            clauses=tuple(recoded_clauses),
        )
    )


def deterministic_key_recode(
    payload: bytes,
    *,
    seed: str = "ssqac-law-collision-gauge-v1",
) -> tuple[bytes, dict[str, str]]:
    """Construct and apply a deterministic source-independent key bijection."""

    source = parse_source(payload)
    all_keys = tuple(
        (
            *source.evidence.states,
            *source.evidence.actions,
            *source.evidence.observers,
            *source.evidence.answers,
        )
    )
    replacements = _derive_opaque_keys(
        seed,
        len(all_keys),
        domain="law-collision-gauge",
    )
    if set(replacements) & set(all_keys):
        replacements = _derive_opaque_keys(
            f"{seed}-disjoint",
            len(all_keys),
            domain="law-collision-gauge",
        )
    mapping = dict(zip(all_keys, replacements, strict=True))
    return recode_source(payload, mapping), mapping


def _words(action_count: int, maximum_depth: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        word
        for depth in range(maximum_depth + 1)
        for word in product(range(action_count), repeat=depth)
    )


def behavior_table(
    completion: MachineCompletion,
    *,
    maximum_depth: int,
) -> tuple[tuple[int, ...], ...]:
    """Return key-aligned answers for every word through a fixed depth."""

    if type(maximum_depth) is not int or maximum_depth < 0:
        raise LawCollisionBoardError("behavior depth must be a nonnegative integer")
    words = _words(len(completion.transitions), maximum_depth)
    rows: list[tuple[int, ...]] = []
    for start in range(len(completion.transitions[0])):
        answers: list[int] = []
        for word in words:
            state = start
            for action in word:
                state = completion.transitions[action][state]
            answers.extend(observer[state] for observer in completion.observations)
        rows.append(tuple(answers))
    return tuple(rows)


def behavior_multiset(
    completion: MachineCompletion,
    *,
    maximum_depth: int,
) -> tuple[tuple[int, ...], ...]:
    """Return the state-key-free multiset of bounded behavior rows."""

    return tuple(sorted(behavior_table(completion, maximum_depth=maximum_depth)))


def _walk_values(value: object) -> Iterable[object]:
    yield value
    if type(value) is dict:
        for key in sorted(value):
            yield from _walk_values(value[key])
    elif type(value) in {list, tuple}:
        for item in value:
            yield from _walk_values(item)


def _renderer_statistics(payload: bytes) -> dict[str, object]:
    value = _decode_json(payload)
    node_types = Counter(type(item).__name__ for item in _walk_values(value))
    byte_histogram = Counter(payload)
    return {
        "ascii_byte_histogram": tuple(sorted(byte_histogram.items())),
        "byte_length": len(payload),
        "line_count": len(payload.splitlines()),
        "node_type_counts": tuple(sorted(node_types.items())),
        "punctuation_counts": tuple(
            (character, payload.count(character.encode("ascii")))
            for character in '{}[],:"'
        ),
    }


def _key_occurrence_counts(source: LawCollisionSource) -> tuple[int, ...]:
    payload_object = _decode_json(encode_source(source))
    all_keys = tuple(
        (
            *source.evidence.states,
            *source.evidence.actions,
            *source.evidence.observers,
            *source.evidence.answers,
        )
    )
    counts = Counter(
        item
        for item in _walk_values(payload_object)
        if type(item) is str and item in set(all_keys)
    )
    return tuple(counts[key] for key in all_keys)


def _row_marginals(
    source: LawCollisionSource,
    selected: MachineCompletion,
) -> dict[str, object]:
    evidence = source.evidence
    state_index = {key: index for index, key in enumerate(evidence.states)}
    action_index = {key: index for index, key in enumerate(evidence.actions)}
    answer_index = {key: index for index, key in enumerate(evidence.answers)}
    visible_by_action: list[list[TransitionFact]] = [[] for _ in evidence.actions]
    for fact in evidence.transitions:
        visible_by_action[action_index[fact.action]].append(fact)
    transition_rows = tuple(
        {
            "full_destination_histogram": tuple(
                sorted(Counter(selected.transitions[index]).items())
            ),
            "missing_cell_count": len(evidence.states) - len(facts),
            "visible_destination_histogram": tuple(
                sorted(Counter(state_index[fact.destination] for fact in facts).items())
            ),
            "visible_source_histogram": tuple(
                sorted(Counter(state_index[fact.source] for fact in facts).items())
            ),
        }
        for index, facts in enumerate(visible_by_action)
    )
    observation_rows = tuple(
        tuple(sorted(Counter(row).items())) for row in selected.observations
    )
    direct_answer_histogram = tuple(
        sorted(
            Counter(answer_index[fact.answer] for fact in evidence.observations).items()
        )
    )
    return {
        "direct_answer_histogram": direct_answer_histogram,
        "observation_rows": observation_rows,
        "transition_rows": transition_rows,
    }


def _law_shape(source: LawCollisionSource) -> tuple[dict[str, object], ...]:
    evidence = source.evidence
    state_index = {key: index for index, key in enumerate(evidence.states)}
    action_index = {key: index for index, key in enumerate(evidence.actions)}
    observer_index = {key: index for index, key in enumerate(evidence.observers)}
    answer_index = {key: index for index, key in enumerate(evidence.answers)}
    shapes: list[dict[str, object]] = []
    for clause in source.clauses:
        if isinstance(clause, PathObservationClause):
            shapes.append(
                {
                    "action_indices": tuple(
                        action_index[action] for action in clause.actions
                    ),
                    "answer_pair": tuple(
                        sorted(
                            (
                                answer_index[clause.expected],
                                answer_index[clause.alternate],
                            )
                        )
                    ),
                    "degree": clause.degree,
                    "id": clause.clause_id,
                    "kind": "path-observation",
                    "observer_index": observer_index[clause.observer],
                    "start_index": state_index[clause.start],
                }
            )
        else:
            shapes.append(
                {
                    "answer_index": answer_index[clause.answer],
                    "id": clause.clause_id,
                    "kind": "visible-observation",
                    "observer_index": observer_index[clause.observer],
                    "state_index": state_index[clause.state],
                }
            )
    return tuple(shapes)


def low_order_collision_signature(audit: SourceAudit) -> dict[str, object]:
    """Return every target-free low-order signature used by the pair gate."""

    selected = audit.selected_completion
    if selected is None:
        raise LawCollisionBoardError(
            "low-order collision signature requires one selected completion"
        )
    source = audit.source
    evidence = source.evidence
    return {
        "behavior_multisets": tuple(
            behavior_multiset(selected, maximum_depth=depth) for depth in range(3)
        ),
        "candidate_completion_cardinality": len(audit.direct_completions),
        "cardinalities": {
            "actions": len(evidence.actions),
            "answers": len(evidence.answers),
            "observers": len(evidence.observers),
            "states": len(evidence.states),
        },
        "direct_visible_records": encode_non_law(evidence).decode("ascii"),
        "key_occurrence_counts": _key_occurrence_counts(source),
        "law_shape": _law_shape(source),
        "record_count": (
            4
            + 1
            + len(evidence.transitions)
            + len(evidence.observations)
            + len(source.clauses)
        ),
        "renderer_statistics": _renderer_statistics(encode_source(source)),
        "row_and_unary_marginals": _row_marginals(source, selected),
        "source_length": len(encode_source(source)),
    }


def _has_degree_three_first_collision(
    left: MachineCompletion,
    right: MachineCompletion,
) -> bool:
    witness = shortest_distinguishing_query(left, right)
    return (
        witness is not None
        and witness.depth == 3
        and behavior_table(left, maximum_depth=2)
        == behavior_table(right, maximum_depth=2)
        and behavior_table(left, maximum_depth=3)
        != behavior_table(right, maximum_depth=3)
    )


def audit_minimality_in_fixture_family() -> MinimalityReceipt:
    """Prove no two/three-state fixture in this exact family collides at depth 3."""

    tested = 0
    witnesses = 0
    for state_count in (2, 3):
        for observations in product((0, 1), repeat=state_count):
            if len(set(observations)) != 2 or observations[0] != 0:
                continue
            for left_row in permutations(range(state_count)):
                for left_source in range(state_count):
                    for right_source in range(left_source + 1, state_count):
                        right_row = list(left_row)
                        right_row[left_source], right_row[right_source] = (
                            right_row[right_source],
                            right_row[left_source],
                        )
                        left = MachineCompletion(
                            transitions=(tuple(left_row),),
                            observations=(tuple(observations),),
                        )
                        right = MachineCompletion(
                            transitions=(tuple(right_row),),
                            observations=(tuple(observations),),
                        )
                        tested += 1
                        if _has_degree_three_first_collision(left, right):
                            witnesses += 1
    pair = build_minimal_collision_pair()
    left = audit_version_space(pair.left_source).selected_completion
    right = audit_version_space(pair.right_source).selected_completion
    if (
        left is None
        or right is None
        or not _has_degree_three_first_collision(
            left,
            right,
        )
    ):
        raise LawCollisionBoardError("four-state minimal witness no longer qualifies")
    return MinimalityReceipt(
        schema=MINIMALITY_RECEIPT_SCHEMA,
        fixture_family=(
            "single-permutation-action_two-hidden-cells_full-binary-observer"
        ),
        searched_smaller_state_counts=(2, 3),
        tested_smaller_candidates=tested,
        smaller_witness_count=witnesses,
        witness_state_count=4,
        witness_first_distinguishing_depth=3,
    )


def _single_clause(
    source: LawCollisionSource,
    clause_type: type[PathObservationClause] | type[VisibleObservationClause],
) -> LawClause | None:
    matches = tuple(
        clause for clause in source.clauses if isinstance(clause, clause_type)
    )
    return matches[0] if len(matches) == 1 else None


def _gauge_recode_exact(
    left_payload: bytes,
    right_payload: bytes,
) -> bool:
    left_source = parse_source(left_payload)
    all_keys = tuple(
        (
            *left_source.evidence.states,
            *left_source.evidence.actions,
            *left_source.evidence.observers,
            *left_source.evidence.answers,
        )
    )
    replacements = _derive_opaque_keys(
        "ssqac-law-collision-pair-gauge-v1",
        len(all_keys),
        domain="law-collision-pair-gauge",
    )
    mapping = dict(zip(all_keys, replacements, strict=True))
    recoded_left = audit_version_space(recode_source(left_payload, mapping))
    recoded_right = audit_version_space(recode_source(right_payload, mapping))
    original_left = audit_version_space(left_payload)
    original_right = audit_version_space(right_payload)
    if (
        recoded_left.selected_completion is None
        or recoded_right.selected_completion is None
        or original_left.selected_completion is None
        or original_right.selected_completion is None
    ):
        return False
    for original, recoded in (
        (original_left, recoded_left),
        (original_right, recoded_right),
    ):
        if (
            original.receipt.semantic_source_sha256
            != recoded.receipt.semantic_source_sha256
            or original.receipt.direct_completion_sha256s
            != recoded.receipt.direct_completion_sha256s
            or original.receipt.law_completion_sha256s
            != recoded.receipt.law_completion_sha256s
            or original.receipt.resolution != recoded.receipt.resolution
        ):
            return False
    return low_order_collision_signature(recoded_left) == low_order_collision_signature(
        recoded_right
    )


def audit_collision_pair(
    pair: LawCollisionPair,
    *,
    require_all_gates: bool = True,
) -> CollisionPairAudit:
    """Run every zero-parameter collision, mutation, and recoding gate."""

    left = audit_version_space(pair.left_source)
    right = audit_version_space(pair.right_source)
    left_path = _single_clause(left.source, PathObservationClause)
    right_path = _single_clause(right.source, PathObservationClause)
    left_redundant = _single_clause(left.source, VisibleObservationClause)
    right_redundant = _single_clause(right.source, VisibleObservationClause)
    left_selected = left.selected_completion
    right_selected = right.selected_completion
    selected_ready = left_selected is not None and right_selected is not None

    left_law_deleted = audit_version_space(delete_law(pair.left_source))
    right_law_deleted = audit_version_space(delete_law(pair.right_source))
    determining_deleted: tuple[SourceAudit, SourceAudit] | None = None
    redundant_deleted: tuple[SourceAudit, SourceAudit] | None = None
    if left_path is not None and right_path is not None:
        determining_deleted = (
            audit_version_space(delete_clause(pair.left_source, left_path.clause_id)),
            audit_version_space(delete_clause(pair.right_source, right_path.clause_id)),
        )
    if left_redundant is not None and right_redundant is not None:
        redundant_deleted = (
            audit_version_space(
                delete_clause(pair.left_source, left_redundant.clause_id)
            ),
            audit_version_space(
                delete_clause(pair.right_source, right_redundant.clause_id)
            ),
        )
    redundant_inserted = (
        audit_version_space(add_redundant_visible_clause(pair.left_source)),
        audit_version_space(add_redundant_visible_clause(pair.right_source)),
    )
    left_signature = (
        low_order_collision_signature(left) if left_selected is not None else {}
    )
    right_signature = (
        low_order_collision_signature(right) if right_selected is not None else {}
    )
    minimality = audit_minimality_in_fixture_family()
    direct_hashes = set(left.receipt.direct_completion_sha256s)
    selected_hashes = {
        digest
        for digest in (
            left.receipt.selected_completion_sha256,
            right.receipt.selected_completion_sha256,
        )
        if digest is not None
    }
    opposite_laws = (
        isinstance(left_path, PathObservationClause)
        and isinstance(right_path, PathObservationClause)
        and left_path.expected == right_path.alternate
        and left_path.alternate == right_path.expected
        and left_path.start == right_path.start
        and left_path.actions == right_path.actions
        and left_path.observer == right_path.observer
    )
    direct_agreement = (
        left.receipt.direct_completion_sha256s
        == right.receipt.direct_completion_sha256s
    )
    gates = {
        "canonical_sources": (
            encode_source(left.source) == pair.left_source
            and encode_source(right.source) == pair.right_source
        ),
        "direct_completion_count_at_least_two": (
            left.receipt.direct_completion_count >= 2
            and right.receipt.direct_completion_count >= 2
        ),
        "direct_version_spaces_identical": direct_agreement,
        "direct_version_space_behaviorally_ambiguous": (
            left.receipt.direct_behavior_class_count >= 2
            and right.receipt.direct_behavior_class_count >= 2
        ),
        "degree_three_first_behavior_collision": (
            selected_ready
            and _has_degree_three_first_collision(
                left_selected,
                right_selected,
            )
        ),
        "determining_clause_deletion_ambiguous": (
            determining_deleted is not None
            and all(
                audit.receipt.resolution == "ambiguous"
                and audit.receipt.law_completion_count
                == audit.receipt.direct_completion_count
                for audit in determining_deleted
            )
        ),
        "gauge_key_recoding_exact": _gauge_recode_exact(
            pair.left_source,
            pair.right_source,
        ),
        "law_deletion_ambiguous": all(
            audit.receipt.resolution == "ambiguous"
            and audit.receipt.law_completion_count
            == audit.receipt.direct_completion_count
            for audit in (left_law_deleted, right_law_deleted)
        ),
        "law_twins_opposite": opposite_laws,
        "laws_select_exactly_one": (
            left.receipt.resolution == "unique-completion"
            and right.receipt.resolution == "unique-completion"
        ),
        "low_order_signatures_identical": (
            bool(left_signature) and left_signature == right_signature
        ),
        "minimal_four_state_fixture_in_family": (
            minimality.smaller_witness_count == 0
            and minimality.witness_state_count == 4
        ),
        "non_law_evidence_byte_identical": (
            encode_non_law(left.source.evidence)
            == encode_non_law(right.source.evidence)
        ),
        "opposite_selected_completions": (
            len(selected_hashes) == 2
            and selected_hashes == direct_hashes
            and len(direct_hashes) == 2
        ),
        "redundant_clause_deletion_no_change": (
            redundant_deleted is not None
            and redundant_deleted[0].receipt.selected_completion_sha256
            == left.receipt.selected_completion_sha256
            and redundant_deleted[1].receipt.selected_completion_sha256
            == right.receipt.selected_completion_sha256
        ),
        "redundant_clause_insertion_no_change": (
            redundant_inserted[0].receipt.selected_completion_sha256
            == left.receipt.selected_completion_sha256
            and redundant_inserted[1].receipt.selected_completion_sha256
            == right.receipt.selected_completion_sha256
        ),
    }
    ordered_gates = tuple(sorted(gates.items()))
    failed = tuple(name for name, passed in ordered_gates if not passed)
    if require_all_gates and failed:
        raise LawCollisionBoardError(
            f"law-collision pair failed gates: {', '.join(failed)}"
        )
    witness = (
        shortest_distinguishing_query(left_selected, right_selected)
        if selected_ready
        else None
    )
    signature = left_signature if left_signature == right_signature else {}
    receipt = CollisionPairReceipt(
        schema=COLLISION_RECEIPT_SCHEMA,
        status=STATUS,
        seed=pair.seed,
        left_source_receipt_sha256=left.receipt.receipt_sha256,
        right_source_receipt_sha256=right.receipt.receipt_sha256,
        low_order_signature_sha256=_sha256(_canonical_json(signature)),
        left_selected_completion_sha256=(left.receipt.selected_completion_sha256),
        right_selected_completion_sha256=(right.receipt.selected_completion_sha256),
        first_distinguishing_query=witness,
        minimality_receipt_sha256=minimality.receipt_sha256,
        gates=ordered_gates,
    )
    return CollisionPairAudit(
        left=left,
        right=right,
        low_order_signature=signature,
        minimality=minimality,
        receipt=receipt,
    )


def _law_bytes(payload: bytes) -> bytes:
    value = _decode_json(payload)
    if "law" not in value:
        raise LawCollisionBoardError("source does not contain a law block")
    return _canonical_json(value["law"])


def _late_answer(
    audit: SourceAudit,
    query: LateQuerySpec,
) -> str | None:
    selected = audit.selected_completion
    if selected is None:
        return None
    return execute_query(
        selected,
        audit.source.evidence,
        start=query.start,
        actions=query.actions,
        observer=query.observer,
    )


def audit_collision_quadruple(
    quadruple: LawCollisionQuadruple,
    *,
    require_all_gates: bool = True,
) -> CollisionQuadrupleAudit:
    """Audit the smallest balanced fact/law counterfactual quadruple."""

    f0_pair = audit_collision_pair(
        LawCollisionPair(
            schema=PAIR_SCHEMA,
            status=STATUS,
            seed=f"{quadruple.seed}:f0",
            left_source=quadruple.f0_l0_source,
            right_source=quadruple.f0_l1_source,
        ),
        require_all_gates=False,
    )
    f1_pair = audit_collision_pair(
        LawCollisionPair(
            schema=PAIR_SCHEMA,
            status=STATUS,
            seed=f"{quadruple.seed}:f1",
            left_source=quadruple.f1_l0_source,
            right_source=quadruple.f1_l1_source,
        ),
        require_all_gates=False,
    )
    audits = (
        f0_pair.left,
        f0_pair.right,
        f1_pair.left,
        f1_pair.right,
    )
    sources = tuple(audit.source for audit in audits)
    evidence = tuple(source.evidence for source in sources)
    query = quadruple.late_query
    answers = tuple(_late_answer(audit, query) for audit in audits)
    answer_inventory = sources[0].evidence.answers
    answer_indices = tuple(
        answer_inventory.index(answer) if answer in answer_inventory else -1
        for answer in answers
    )
    direct_query_answers = tuple(
        tuple(
            execute_query(
                completion,
                audit.source.evidence,
                start=query.start,
                actions=query.actions,
                observer=query.observer,
            )
            for completion in audit.direct_completions
        )
        for audit in (f0_pair.left, f1_pair.left)
    )
    path_clauses = tuple(
        _single_clause(source, PathObservationClause) for source in sources
    )
    query_is_distinct = all(
        isinstance(clause, PathObservationClause)
        and (
            clause.start,
            clause.actions,
            clause.observer,
        )
        != (query.start, query.actions, query.observer)
        for clause in path_clauses
    )
    inventories = tuple(
        (
            source.evidence.states,
            source.evidence.actions,
            source.evidence.observers,
            source.evidence.answers,
        )
        for source in sources
    )
    pair_gate_names = (
        "canonical_sources",
        "direct_completion_count_at_least_two",
        "direct_version_spaces_identical",
        "direct_version_space_behaviorally_ambiguous",
        "degree_three_first_behavior_collision",
        "determining_clause_deletion_ambiguous",
        "gauge_key_recoding_exact",
        "law_deletion_ambiguous",
        "law_twins_opposite",
        "laws_select_exactly_one",
        "low_order_signatures_identical",
        "non_law_evidence_byte_identical",
        "opposite_selected_completions",
        "redundant_clause_deletion_no_change",
        "redundant_clause_insertion_no_change",
    )
    gates = {
        "all_sources_canonical": all(
            encode_source(source) == payload
            for source, payload in zip(
                sources,
                (
                    quadruple.f0_l0_source,
                    quadruple.f0_l1_source,
                    quadruple.f1_l0_source,
                    quadruple.f1_l1_source,
                ),
                strict=True,
            )
        ),
        "both_fact_pairs_pass_collision_gates": all(
            pair.gates.get(name, False)
            for pair in (f0_pair, f1_pair)
            for name in pair_gate_names
        ),
        "common_opaque_inventory": len(set(inventories)) == 1,
        "counterfactual_facts_differ": (
            encode_non_law(evidence[0]) != encode_non_law(evidence[2])
        ),
        "laws_byte_identical_across_facts": (
            _law_bytes(quadruple.f0_l0_source)
            == _law_bytes(quadruple.f1_l0_source)
            and _law_bytes(quadruple.f0_l1_source)
            == _law_bytes(quadruple.f1_l1_source)
        ),
        "late_query_not_a_law_clause": query_is_distinct,
        "late_query_not_entailed_by_facts": all(
            len(set(fact_answers)) == 2 for fact_answers in direct_query_answers
        ),
        "late_targets_balanced": (
            len(answer_inventory) == 2
            and Counter(answers)
            == Counter({answer_inventory[0]: 2, answer_inventory[1]: 2})
        ),
        "law_alone_is_uninformative": (
            answers[0] != answers[2] and answers[1] != answers[3]
        ),
        "facts_alone_are_uninformative": (
            answers[0] != answers[1] and answers[2] != answers[3]
        ),
        "xor_counterfactual_pattern": (
            answers[0] == answers[3]
            and answers[1] == answers[2]
            and answers[0] != answers[1]
        ),
    }
    ordered_gates = tuple(sorted(gates.items()))
    failed = tuple(name for name, passed in ordered_gates if not passed)
    if require_all_gates and failed:
        raise LawCollisionBoardError(
            f"law-collision quadruple failed gates: {', '.join(failed)}"
        )
    receipt = CollisionQuadrupleReceipt(
        schema=QUADRUPLE_RECEIPT_SCHEMA,
        status=STATUS,
        seed=quadruple.seed,
        f0_pair_receipt_sha256=f0_pair.receipt.receipt_sha256,
        f1_pair_receipt_sha256=f1_pair.receipt.receipt_sha256,
        source_receipt_sha256s=tuple(
            audit.receipt.receipt_sha256 for audit in audits
        ),
        selected_completion_sha256s=tuple(
            audit.receipt.selected_completion_sha256 or "" for audit in audits
        ),
        late_answer_indices=answer_indices,
        classification="minimal_mechanics_fixture_only",
        known_shortcut=(
            "single-fixture fact-bit xor law-bit; F0/F1 state-isomorphic; "
            "fixed late-query address"
        ),
        promotion_eligible=False,
        gates=ordered_gates,
    )
    return CollisionQuadrupleAudit(
        f0_pair=f0_pair,
        f1_pair=f1_pair,
        source_audits=audits,
        receipt=receipt,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the exact CPU-only SSQAC law-collision fixture."
    )
    parser.add_argument("--seed", default=DEFAULT_SEED)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    audit = audit_collision_quadruple(
        build_minimal_collision_quadruple(args.seed)
    )
    print(audit.receipt.canonical_bytes().decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
