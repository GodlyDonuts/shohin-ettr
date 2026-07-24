#!/usr/bin/env python3
"""Gold-oracle bridge from law-collision sources to exact quotient proofs.

This module is deliberately a preparation/evaluation mechanic.  It parses raw
collision sources, exhaustively enumerates their direct machine completions,
and translates that finite version space into a Boolean one-hot quotient.  It
must never be imported into, serialized for, or exposed to a candidate model.

The source law is represented only as a selector over independently enumerated
completions.  The late query is a polynomial whose coefficient on each
completion variable is that completion's directly executed answer bit.  An
independently enumerated Boolean zero set is checked in two fields before the
exact quotient compiler is allowed to certify the claimed consequence.  The
portable result is then replayed by the standalone verifier.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import product
import json
from typing import Mapping, Sequence

from pipeline.episode_functor_law_collision_board import (
    MachineCompletion,
    PathObservationClause,
    VisibleObservationClause,
    enumerate_completions,
    execute_query,
    filter_completions_by_law,
    parse_source,
)
from pipeline.episode_functor_law_collision_family import LateQuery
from pipeline.verify_ssqac_quotient_artifact import (
    OutcomeVerification,
    verify_outcome_artifact,
)
from episode_functor_quotient_algebra import (
    BooleanFieldSemanticsReceipt,
    IntegerPolynomial,
    SparsePolynomial,
    boolean_one_hot_generators,
    certify_polynomial_consequence,
    verify_boolean_field_semantics,
)


BRIDGE_SCHEMA = "ssqac_collision_quotient_bridge_v1"
RECEIPT_SCHEMA = "ssqac_collision_quotient_bridge_receipt_v1"
QUERY_SCHEMA = "ssqac_collision_late_query_binding_v1"
COMPLETION_SCHEMA = "ssqac_collision_completion_binding_v1"
STATUS = "gold_oracle_mechanics_only"
CLAIM_BOUNDARY = (
    "Exact gold-oracle mechanics only. The completion variables, selector, "
    "query polynomial, certificate, outcome, and receipt are forbidden as "
    "candidate input and are not evidence of learned or native reasoning."
)


class CollisionQuotientBridgeError(ValueError):
    """A source translation, semantic gate, or artifact binding failed."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise CollisionQuotientBridgeError(
            "bridge value is not canonical ASCII JSON"
        ) from error


def _sha256_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _sha256_object(value: object) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _polynomial_digest(polynomial: SparsePolynomial) -> str:
    return _sha256_object(polynomial.canonical_data())


def _field_receipt_data(
    receipt: BooleanFieldSemanticsReceipt,
) -> dict[str, object]:
    return asdict(receipt)


def _validate_late_query(
    query: LateQuery,
    *,
    state_count: int,
    action_count: int,
    observer_count: int,
) -> None:
    if not isinstance(query, LateQuery):
        raise CollisionQuotientBridgeError("late query has the wrong type")
    if not 0 <= query.start_index < state_count:
        raise CollisionQuotientBridgeError("late query start is out of range")
    if not query.action_indices:
        raise CollisionQuotientBridgeError("late query action word is empty")
    if any(not 0 <= action < action_count for action in query.action_indices):
        raise CollisionQuotientBridgeError("late query action is out of range")
    if not 0 <= query.observer_index < observer_count:
        raise CollisionQuotientBridgeError("late query observer is out of range")


def _validate_variable_permutation(
    count: int,
    variable_permutation: Sequence[int] | None,
) -> tuple[int, ...]:
    if variable_permutation is None:
        return tuple(range(count))
    permutation = tuple(variable_permutation)
    if any(type(index) is not int for index in permutation):
        raise CollisionQuotientBridgeError(
            "completion-variable recoding must contain plain integers"
        )
    if sorted(permutation) != list(range(count)):
        raise CollisionQuotientBridgeError(
            "completion-variable recoding must be a complete permutation"
        )
    return permutation


def _coordinate(width: int, variable: int, degree: int = 1) -> tuple[int, ...]:
    result = [0] * width
    result[variable] = degree
    return tuple(result)


def _constant(width: int) -> tuple[int, ...]:
    return (0,) * width


def _integer_boolean_one_hot_generators(
    width: int,
) -> tuple[IntegerPolynomial, ...]:
    boolean = tuple(
        IntegerPolynomial(
            width,
            {
                _coordinate(width, variable, 2): 1,
                _coordinate(width, variable): -1,
            },
        )
        for variable in range(width)
    )
    one_hot = IntegerPolynomial(
        width,
        {
            **{
                _coordinate(width, variable): 1
                for variable in range(width)
            },
            _constant(width): -1,
        },
    )
    return (*boolean, one_hot)


def _selector_polynomial(
    width: int,
    selected_variables: Sequence[int],
) -> SparsePolynomial:
    terms = {
        _coordinate(width, variable): 1 for variable in selected_variables
    }
    terms[_constant(width)] = -1
    return SparsePolynomial(width, terms)


def _integer_selector_polynomial(
    width: int,
    selected_variables: Sequence[int],
) -> IntegerPolynomial:
    terms = {
        _coordinate(width, variable): 1 for variable in selected_variables
    }
    terms[_constant(width)] = -1
    return IntegerPolynomial(width, terms)


def _execute_late_query(
    completion: MachineCompletion,
    raw_source: bytes,
    query: LateQuery,
) -> int:
    source = parse_source(raw_source)
    evidence = source.evidence
    answer = execute_query(
        completion,
        evidence,
        start=evidence.states[query.start_index],
        actions=tuple(
            evidence.actions[action] for action in query.action_indices
        ),
        observer=evidence.observers[query.observer_index],
    )
    return evidence.answers.index(answer)


def _clause_binding(source: object) -> tuple[dict[str, object], ...]:
    clauses = []
    for clause in source.clauses:  # type: ignore[attr-defined]
        if isinstance(clause, PathObservationClause):
            clauses.append(
                {
                    "actions": list(clause.actions),
                    "alternate": clause.alternate,
                    "expected": clause.expected,
                    "id": clause.clause_id,
                    "kind": "path-observation",
                    "observer": clause.observer,
                    "start": clause.start,
                }
            )
        elif isinstance(clause, VisibleObservationClause):
            clauses.append(
                {
                    "answer": clause.answer,
                    "id": clause.clause_id,
                    "kind": "visible-observation",
                    "observer": clause.observer,
                    "state": clause.state,
                }
            )
        else:
            raise CollisionQuotientBridgeError(
                "source contains an unsupported law clause"
            )
    return tuple(clauses)


@dataclass(frozen=True, slots=True)
class CompletionVariableBinding:
    schema: str
    completion_index: int
    variable_index: int
    completion_sha256: str
    law_admissible: bool
    late_answer_bit: int

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(asdict(self))

    @property
    def binding_sha256(self) -> str:
        return _sha256_bytes(self.canonical_bytes())


@dataclass(frozen=True, slots=True)
class CollisionQuotientBridgeReceipt:
    schema: str
    status: str
    source_sha256: str
    query_sha256: str
    completion_sha256s: tuple[str, ...]
    completion_binding_sha256s: tuple[str, ...]
    completion_set_sha256: str
    generator_sha256: str
    query_polynomial_sha256: str
    intended_zero_set_sha256: str
    field_semantics_sha256: str
    certificate_sha256: str
    outcome_sha256: str
    artifact_sha256: str
    variable_permutation: tuple[int, ...]
    outcome_status: str
    outcome_value: int | None
    standalone_verification_gates: tuple[str, ...]
    gates: tuple[tuple[str, bool], ...]
    gold_oracle_only: bool
    candidate_input_allowed: bool
    reasoning_claim_allowed: bool
    promotion_eligible: bool
    claim_boundary: str

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(asdict(self))

    @property
    def receipt_sha256(self) -> str:
        return _sha256_bytes(self.canonical_bytes())


@dataclass(frozen=True, slots=True)
class CollisionQuotientBridge:
    schema: str
    status: str
    generators: tuple[SparsePolynomial, ...]
    query_polynomial: SparsePolynomial
    field_semantics: BooleanFieldSemanticsReceipt
    completion_bindings: tuple[CompletionVariableBinding, ...]
    artifact_bytes: bytes
    receipt: CollisionQuotientBridgeReceipt

    def artifact(self) -> dict[str, object]:
        value = json.loads(self.artifact_bytes)
        if type(value) is not dict:
            raise CollisionQuotientBridgeError(
                "bridge artifact root is not an object"
            )
        return value


@dataclass(frozen=True, slots=True)
class _PreparedProblem:
    generators: tuple[SparsePolynomial, ...]
    integer_generators: tuple[IntegerPolynomial, ...]
    query: SparsePolynomial
    expected_zero_set: tuple[tuple[int, ...], ...]
    completion_bindings: tuple[CompletionVariableBinding, ...]
    query_binding: dict[str, object]
    source_sha256: str
    variable_permutation: tuple[int, ...]


def _prepare_problem(
    raw_source: bytes,
    late_query: LateQuery,
    *,
    variable_permutation: Sequence[int] | None,
) -> _PreparedProblem:
    if type(raw_source) is not bytes:
        raise CollisionQuotientBridgeError("raw source must be immutable bytes")
    source = parse_source(raw_source)
    evidence = source.evidence
    if len(evidence.answers) != 2:
        raise CollisionQuotientBridgeError(
            "collision quotient bridge requires exactly two answer keys"
        )
    _validate_late_query(
        late_query,
        state_count=len(evidence.states),
        action_count=len(evidence.actions),
        observer_count=len(evidence.observers),
    )

    direct = enumerate_completions(evidence)
    if len(direct) < 2:
        raise CollisionQuotientBridgeError(
            "collision bridge requires at least two direct completions"
        )
    permutation = _validate_variable_permutation(
        len(direct),
        variable_permutation,
    )
    law_admissible = set(filter_completions_by_law(source, direct))
    selected_variables = tuple(
        permutation[index]
        for index, completion in enumerate(direct)
        if completion in law_admissible
    )
    answers = tuple(
        _execute_late_query(completion, raw_source, late_query)
        for completion in direct
    )
    if any(answer not in (0, 1) for answer in answers):
        raise CollisionQuotientBridgeError(
            "late query did not produce Boolean answer bits"
        )

    width = len(direct)
    generators = list(
        boolean_one_hot_generators(
            width,
            (tuple(range(width)),),
            boolean_variables=tuple(range(width)),
        )
    )
    integer_generators = list(_integer_boolean_one_hot_generators(width))
    if source.law_present:
        generators.append(_selector_polynomial(width, selected_variables))
        integer_generators.append(
            _integer_selector_polynomial(width, selected_variables)
        )

    query_terms: dict[tuple[int, ...], int] = {}
    for completion_index, answer in enumerate(answers):
        if answer:
            query_terms[
                _coordinate(width, permutation[completion_index])
            ] = answer
    query = SparsePolynomial(width, query_terms)

    expected_zero_set = tuple(
        assignment
        for assignment in product((0, 1), repeat=width)
        if sum(assignment) == 1
        and (
            not source.law_present
            or sum(assignment[variable] for variable in selected_variables) == 1
        )
    )
    completion_bindings = tuple(
        CompletionVariableBinding(
            schema=COMPLETION_SCHEMA,
            completion_index=index,
            variable_index=permutation[index],
            completion_sha256=completion.structural_sha256,
            law_admissible=completion in law_admissible,
            late_answer_bit=answers[index],
        )
        for index, completion in enumerate(direct)
    )
    query_binding = {
        "action_indices": list(late_query.action_indices),
        "action_keys": [
            evidence.actions[index] for index in late_query.action_indices
        ],
        "answer_bits_by_completion": list(answers),
        "observer_index": late_query.observer_index,
        "observer_key": evidence.observers[late_query.observer_index],
        "query_polynomial": query.canonical_data(),
        "schema": QUERY_SCHEMA,
        "source_law": list(_clause_binding(source)),
        "start_index": late_query.start_index,
        "start_key": evidence.states[late_query.start_index],
        "variable_permutation": list(permutation),
    }
    return _PreparedProblem(
        generators=tuple(generators),
        integer_generators=tuple(integer_generators),
        query=query,
        expected_zero_set=expected_zero_set,
        completion_bindings=completion_bindings,
        query_binding=query_binding,
        source_sha256=_sha256_bytes(raw_source),
        variable_permutation=permutation,
    )


def _artifact_bytes(
    certificate_bytes: bytes,
    outcome_bytes: bytes,
    outcome_sha256: str,
) -> bytes:
    certificate = json.loads(certificate_bytes)
    outcome = json.loads(outcome_bytes)
    return _canonical_bytes(
        {
            "certificate": certificate,
            "outcome": outcome,
            "outcome_sha256": outcome_sha256,
        }
    )


def compile_collision_quotient_bridge(
    raw_source: bytes,
    late_query: LateQuery,
    *,
    variable_permutation: Sequence[int] | None = None,
) -> CollisionQuotientBridge:
    """Compile and independently verify one gold-oracle bridge artifact."""

    prepared = _prepare_problem(
        raw_source,
        late_query,
        variable_permutation=variable_permutation,
    )
    field_semantics = verify_boolean_field_semantics(
        len(prepared.completion_bindings),
        prepared.integer_generators,
        expected_zero_set=prepared.expected_zero_set,
    )
    if not field_semantics.passed:
        raise CollisionQuotientBridgeError(
            "cross-field intended-semantics gate did not pass"
        )
    outcome = certify_polynomial_consequence(
        len(prepared.completion_bindings),
        prepared.generators,
        prepared.query,
        allowed_values=(0, 1),
        require_boolean_schema=True,
    )
    if outcome.certificate is None:
        raise CollisionQuotientBridgeError(
            f"quotient consequence did not produce a certificate: {outcome.status}"
        )
    artifact_bytes = _artifact_bytes(
        outcome.certificate.canonical_bytes(),
        outcome.canonical_bytes(),
        outcome.outcome_sha256,
    )
    artifact = json.loads(artifact_bytes)
    standalone = verify_outcome_artifact(
        artifact,
        [generator.canonical_data() for generator in prepared.generators],
        prepared.query.canonical_data(),
        expected_allowed_values=(0, 1),
    )

    completion_sha256s = tuple(
        binding.completion_sha256 for binding in prepared.completion_bindings
    )
    field_semantics_data = _field_receipt_data(field_semantics)
    gates = (
        ("candidate_input_forbidden", True),
        ("certificate_exported", True),
        ("completion_enumeration_bound", True),
        ("cross_field_semantics", field_semantics.passed),
        ("gold_oracle_only", True),
        ("intended_zero_set_bound", field_semantics.intended_zero_set_sha256 is not None),
        ("late_query_bound", True),
        ("reasoning_claim_forbidden", True),
        ("source_law_selector_bound", True),
        ("standalone_artifact_verified", bool(standalone.gates)),
    )
    receipt = CollisionQuotientBridgeReceipt(
        schema=RECEIPT_SCHEMA,
        status=STATUS,
        source_sha256=prepared.source_sha256,
        query_sha256=_sha256_object(prepared.query_binding),
        completion_sha256s=completion_sha256s,
        completion_binding_sha256s=tuple(
            binding.binding_sha256
            for binding in prepared.completion_bindings
        ),
        completion_set_sha256=_sha256_object(completion_sha256s),
        generator_sha256=_sha256_object(
            [generator.canonical_data() for generator in prepared.generators]
        ),
        query_polynomial_sha256=_polynomial_digest(prepared.query),
        intended_zero_set_sha256=_sha256_object(prepared.expected_zero_set),
        field_semantics_sha256=_sha256_object(field_semantics_data),
        certificate_sha256=standalone.certificate_sha256,
        outcome_sha256=standalone.outcome_sha256,
        artifact_sha256=_sha256_bytes(artifact_bytes),
        variable_permutation=prepared.variable_permutation,
        outcome_status=standalone.status,
        outcome_value=standalone.value,
        standalone_verification_gates=standalone.gates,
        gates=gates,
        gold_oracle_only=True,
        candidate_input_allowed=False,
        reasoning_claim_allowed=False,
        promotion_eligible=False,
        claim_boundary=CLAIM_BOUNDARY,
    )
    if not all(passed for _, passed in receipt.gates):
        raise CollisionQuotientBridgeError(
            "bridge receipt did not pass every mandatory gate"
        )
    return CollisionQuotientBridge(
        schema=BRIDGE_SCHEMA,
        status=STATUS,
        generators=prepared.generators,
        query_polynomial=prepared.query,
        field_semantics=field_semantics,
        completion_bindings=prepared.completion_bindings,
        artifact_bytes=artifact_bytes,
        receipt=receipt,
    )


def verify_collision_quotient_bridge(
    bridge: CollisionQuotientBridge,
    raw_source: bytes,
    late_query: LateQuery,
) -> OutcomeVerification:
    """Rebuild all external bindings and replay a bridge artifact fail closed."""

    if not isinstance(bridge, CollisionQuotientBridge):
        raise CollisionQuotientBridgeError("bridge has the wrong type")
    if bridge.schema != BRIDGE_SCHEMA or bridge.status != STATUS:
        raise CollisionQuotientBridgeError("bridge schema or status differs")
    receipt = bridge.receipt
    if (
        receipt.schema != RECEIPT_SCHEMA
        or receipt.status != STATUS
        or receipt.gold_oracle_only is not True
        or receipt.candidate_input_allowed is not False
        or receipt.reasoning_claim_allowed is not False
        or receipt.promotion_eligible is not False
        or receipt.claim_boundary != CLAIM_BOUNDARY
    ):
        raise CollisionQuotientBridgeError(
            "bridge claim boundary is not the exact gold-only boundary"
        )
    prepared = _prepare_problem(
        raw_source,
        late_query,
        variable_permutation=receipt.variable_permutation,
    )
    if bridge.generators != prepared.generators:
        raise CollisionQuotientBridgeError("bridge generators were tampered")
    if bridge.query_polynomial != prepared.query:
        raise CollisionQuotientBridgeError("bridge query polynomial was tampered")
    if bridge.completion_bindings != prepared.completion_bindings:
        raise CollisionQuotientBridgeError("completion bindings were tampered")

    field_semantics = verify_boolean_field_semantics(
        len(prepared.completion_bindings),
        prepared.integer_generators,
        expected_zero_set=prepared.expected_zero_set,
    )
    if bridge.field_semantics != field_semantics:
        raise CollisionQuotientBridgeError(
            "field-semantics receipt was tampered"
        )
    artifact_sha256 = _sha256_bytes(bridge.artifact_bytes)
    if artifact_sha256 != receipt.artifact_sha256:
        raise CollisionQuotientBridgeError(
            "bridge artifact digest does not match receipt"
        )
    try:
        artifact = bridge.artifact()
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise CollisionQuotientBridgeError(
            "bridge artifact is not valid JSON"
        ) from error
    standalone = verify_outcome_artifact(
        artifact,
        [generator.canonical_data() for generator in prepared.generators],
        prepared.query.canonical_data(),
        expected_allowed_values=(0, 1),
    )

    expected_hashes: Mapping[str, object] = {
        "source_sha256": prepared.source_sha256,
        "query_sha256": _sha256_object(prepared.query_binding),
        "completion_sha256s": tuple(
            binding.completion_sha256
            for binding in prepared.completion_bindings
        ),
        "completion_binding_sha256s": tuple(
            binding.binding_sha256
            for binding in prepared.completion_bindings
        ),
        "completion_set_sha256": _sha256_object(
            tuple(
                binding.completion_sha256
                for binding in prepared.completion_bindings
            )
        ),
        "generator_sha256": _sha256_object(
            [generator.canonical_data() for generator in prepared.generators]
        ),
        "query_polynomial_sha256": _polynomial_digest(prepared.query),
        "intended_zero_set_sha256": _sha256_object(prepared.expected_zero_set),
        "field_semantics_sha256": _sha256_object(
            _field_receipt_data(field_semantics)
        ),
        "certificate_sha256": standalone.certificate_sha256,
        "outcome_sha256": standalone.outcome_sha256,
        "artifact_sha256": artifact_sha256,
        "outcome_status": standalone.status,
        "outcome_value": standalone.value,
        "standalone_verification_gates": standalone.gates,
    }
    for field, expected in expected_hashes.items():
        if getattr(receipt, field) != expected:
            raise CollisionQuotientBridgeError(
                f"bridge receipt field {field} differs from independent replay"
            )
    if not receipt.gates or not all(passed for _, passed in receipt.gates):
        raise CollisionQuotientBridgeError(
            "bridge receipt contains a failed gate"
        )
    return standalone
