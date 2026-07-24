from __future__ import annotations

from dataclasses import replace
from itertools import product
import json

import pytest

from episode_functor_quotient_algebra import (
    FIELD_MODULUS,
    FIELD_SEMANTICS_SCHEMA,
    MAX_GENERATORS,
    MAX_VARIABLES,
    ConsequenceReceipt,
    IdealMembershipEvidence,
    IntegerPolynomial,
    MembershipTerm,
    QUOTIENT_VERIFICATION_SCHEMA,
    STATUS_AMBIGUOUS,
    STATUS_CERTIFIED,
    STATUS_INCOMPLETE,
    STATUS_RESOURCE_OVERFLOW,
    STATUS_UNSAT,
    SSQAC_OUTCOME_VERIFICATION_SCHEMA,
    QuotientAlgebraClosureError,
    QuotientAlgebraError,
    QuotientAlgebraLimitError,
    SparsePolynomial,
    boolean_generators,
    boolean_one_hot_generators,
    certify_polynomial_consequence,
    compile_quotient_algebra,
    one_hot_generators,
    recode_generators,
    verify_quotient_algebra_certificate,
    verify_boolean_field_semantics,
    verify_ssqac_consequence_outcome,
)


def _constant(
    num_variables: int,
    value: int,
) -> SparsePolynomial:
    return SparsePolynomial.constant(num_variables, value)


def _variable(
    num_variables: int,
    variable: int,
) -> SparsePolynomial:
    return SparsePolynomial.variable(num_variables, variable)


def _force(
    num_variables: int,
    variable: int,
    value: int,
) -> SparsePolynomial:
    return _variable(num_variables, variable) - _constant(
        num_variables,
        value,
    )


def _forced_value(receipt: ConsequenceReceipt) -> int:
    assert receipt.status == "forced"
    assert receipt.value is not None
    assert receipt.evidence is not None
    return receipt.value


def test_boolean_one_hot_coordinates_are_forced_exactly() -> None:
    num_variables = 3
    generators = (
        *boolean_one_hot_generators(
            num_variables,
            ((0, 1, 2),),
        ),
        _force(num_variables, 1, 1),
    )
    algebra = compile_quotient_algebra(
        num_variables,
        generators,
    )
    assert algebra.receipt.status == "ok"
    assert algebra.receipt.quotient_dimension == 1
    assert _forced_value(algebra.decide_coordinate(0)) == 0
    assert _forced_value(algebra.decide_coordinate(1)) == 1
    assert _forced_value(algebra.decide_coordinate(2)) == 0
    for variable in range(num_variables):
        evidence = algebra.decide_coordinate(variable).evidence
        assert evidence is not None
        assert algebra.verify_membership_evidence(evidence)


def test_ambiguous_boolean_coordinate_abstains() -> None:
    algebra = compile_quotient_algebra(
        1,
        boolean_generators(1),
    )
    decision = algebra.decide_coordinate(0)
    assert algebra.receipt.quotient_dimension == 2
    assert decision.status == "ambiguous"
    assert decision.value is None
    assert decision.evidence is None
    assert decision.normal_form == _variable(1, 0)


def test_inconsistent_ideal_is_detected_and_never_decides() -> None:
    generators = (
        *boolean_generators(1),
        _force(1, 0, 0),
        _force(1, 0, 1),
    )
    algebra = compile_quotient_algebra(1, generators)
    assert not algebra.is_consistent
    assert algebra.receipt.status == "inconsistent"
    assert algebra.receipt.quotient_dimension == 0
    decision = algebra.decide_coordinate(0)
    assert decision.status == "inconsistent"
    assert decision.value is None
    assert algebra.normal_form(SparsePolynomial.one(1)).is_zero


def test_redundant_generator_preserves_canonical_quotient() -> None:
    variable = _variable(1, 0)
    base_generators = (
        *boolean_generators(1),
        _force(1, 0, 1),
    )
    redundant = variable * variable - SparsePolynomial.one(1)
    base = compile_quotient_algebra(1, base_generators)
    extended = compile_quotient_algebra(
        1,
        (*base_generators, redundant),
    )
    assert base.receipt == extended.receipt
    assert base.normal_form(variable) == extended.normal_form(variable)
    assert _forced_value(base.decide_coordinate(0)) == 1
    assert _forced_value(extended.decide_coordinate(0)) == 1


def test_determining_generator_deletion_restores_ambiguity() -> None:
    left = _variable(2, 0)
    right = _variable(2, 1)
    shared = (
        *boolean_generators(2),
        left - right,
    )
    determined = compile_quotient_algebra(
        2,
        (*shared, _force(2, 0, 1)),
    )
    deleted = compile_quotient_algebra(2, shared)
    assert _forced_value(determined.decide_coordinate(0)) == 1
    assert _forced_value(determined.decide_coordinate(1)) == 1
    assert deleted.decide_coordinate(0).status == "ambiguous"
    assert deleted.decide_coordinate(1).status == "ambiguous"
    assert deleted.receipt.quotient_dimension == 2


def test_variable_gauge_recoding_conjugates_all_decisions() -> None:
    num_variables = 3
    permutation = (2, 0, 1)
    generators = (
        *boolean_generators(num_variables),
        *one_hot_generators(
            num_variables,
            ((0, 1, 2),),
        ),
        _force(num_variables, 0, 1),
    )
    original = compile_quotient_algebra(
        num_variables,
        generators,
    )
    recoded = compile_quotient_algebra(
        num_variables,
        recode_generators(generators, permutation),
    )
    assert (
        original.receipt.quotient_dimension == recoded.receipt.quotient_dimension == 1
    )
    for old_variable, new_variable in enumerate(permutation):
        original_value = _forced_value(original.decide_coordinate(old_variable))
        recoded_value = _forced_value(recoded.decide_coordinate(new_variable))
        assert recoded_value == original_value
    query = _variable(num_variables, 0) + _variable(
        num_variables,
        1,
    )
    assert (
        original.decide_polynomial(
            query,
            allowed_values=(0, 1, 2),
        ).value
        == recoded.decide_polynomial(
            query.recode_variables(permutation),
            allowed_values=(0, 1, 2),
        ).value
    )


def test_degree_three_relation_collision_changes_forced_result() -> None:
    num_variables = 4
    x_0 = _variable(num_variables, 0)
    x_1 = _variable(num_variables, 1)
    x_2 = _variable(num_variables, 2)
    result = _variable(num_variables, 3)
    cubic = x_0 * x_1 * x_2
    assert cubic.degree == 3
    common = (
        *boolean_generators(num_variables),
        _force(num_variables, 0, 1),
        _force(num_variables, 1, 1),
        _force(num_variables, 2, 1),
    )
    positive_law = result - cubic
    complementary_law = result + cubic - SparsePolynomial.one(num_variables)
    positive = compile_quotient_algebra(
        num_variables,
        (*common, positive_law),
    )
    complementary = compile_quotient_algebra(
        num_variables,
        (*common, complementary_law),
    )
    assert _forced_value(positive.decide_coordinate(3)) == 1
    assert _forced_value(complementary.decide_coordinate(3)) == 0
    assert positive.receipt.canonical_digest != complementary.receipt.canonical_digest


def test_field_arithmetic_is_exact_modulo_257() -> None:
    variable = _variable(1, 0)
    relation = variable + SparsePolynomial.one(1)
    algebra = compile_quotient_algebra(1, (relation,))
    decision = algebra.decide_coordinate(
        0,
        allowed_values=(0, FIELD_MODULUS - 1),
    )
    assert _forced_value(decision) == FIELD_MODULUS - 1
    assert variable.evaluate((FIELD_MODULUS - 1,)) == 256
    assert relation.evaluate((FIELD_MODULUS - 1,)) == 0


def test_membership_evidence_reconstructs_target_exactly() -> None:
    variable = _variable(1, 0)
    algebra = compile_quotient_algebra(
        1,
        (*boolean_generators(1), _force(1, 0, 1)),
    )
    target = variable * variable - SparsePolynomial.one(1)
    evidence = algebra.membership_evidence(target)
    assert evidence is not None
    assert evidence.terms
    assert algebra.verify_membership_evidence(evidence)
    first = evidence.terms[0]
    corrupted_terms = (
        MembershipTerm(
            generator_index=first.generator_index,
            multiplier=first.multiplier,
            coefficient=first.coefficient + 1,
        ),
        *evidence.terms[1:],
    )
    corrupted = IdealMembershipEvidence(
        target=evidence.target,
        terms=corrupted_terms,
    )
    assert not algebra.verify_membership_evidence(corrupted)
    assert algebra.membership_evidence(variable) is None


def test_receipts_are_deterministic_across_input_order() -> None:
    x_0 = _variable(2, 0)
    x_1 = _variable(2, 1)
    generators = (
        *boolean_generators(2),
        x_0 + x_1 - SparsePolynomial.one(2),
        _force(2, 0, 1),
    )
    forward = compile_quotient_algebra(2, generators)
    reverse = compile_quotient_algebra(2, reversed(generators))
    duplicate = compile_quotient_algebra(
        2,
        (*generators, generators[-1]),
    )
    assert forward.receipt == reverse.receipt == duplicate.receipt
    assert forward.receipt.to_json() == reverse.receipt.to_json()
    decoded = json.loads(forward.receipt.to_json())
    assert decoded["canonical_digest"] == forward.receipt.canonical_digest
    assert len(forward.receipt.canonical_digest) == 64


@pytest.mark.parametrize(
    ("num_variables", "terms"),
    [
        (2, {((1,), 1)}),
        (1, {((5,), 1)}),
        (1, {((-1,), 1)}),
        (1, {((1,), 1.5)}),
    ],
)
def test_malformed_polynomials_fail_closed(
    num_variables: int,
    terms: dict[tuple[int, ...], object],
) -> None:
    with pytest.raises(QuotientAlgebraError):
        SparsePolynomial(num_variables, terms)


def test_fixed_workspace_limits_fail_closed() -> None:
    with pytest.raises(QuotientAlgebraLimitError):
        SparsePolynomial.variable(MAX_VARIABLES + 1, 0)
    repeated = tuple(SparsePolynomial.variable(1, 0) for _ in range(MAX_GENERATORS + 1))
    with pytest.raises(QuotientAlgebraLimitError):
        compile_quotient_algebra(1, repeated)
    with pytest.raises(
        QuotientAlgebraLimitError,
        match="admitted monomial count",
    ):
        compile_quotient_algebra(20, ())
    with pytest.raises(
        QuotientAlgebraLimitError,
        match="quotient dimension",
    ):
        compile_quotient_algebra(
            10,
            boolean_generators(10),
        )


def test_malformed_or_incomplete_admitted_workspace_fails_closed() -> None:
    with pytest.raises(
        QuotientAlgebraClosureError,
        match="constant",
    ):
        compile_quotient_algebra(
            1,
            boolean_generators(1),
            admitted_monomials=((1,), (2,)),
        )
    with pytest.raises(
        QuotientAlgebraClosureError,
        match="divisor-closed",
    ):
        compile_quotient_algebra(
            1,
            boolean_generators(1),
            admitted_monomials=((0,), (2,)),
        )
    with pytest.raises(
        QuotientAlgebraClosureError,
        match="generator support",
    ):
        compile_quotient_algebra(
            1,
            boolean_generators(1),
            admitted_monomials=((0,), (1,)),
        )
    with pytest.raises(
        QuotientAlgebraClosureError,
        match="stable under complete prolongation",
    ):
        compile_quotient_algebra(
            1,
            (),
            admitted_monomials=(
                (0,),
                (1,),
                (2,),
                (3,),
                (4,),
            ),
        )


def test_boolean_schema_and_complete_prolongation_are_certified() -> None:
    algebra = compile_quotient_algebra(
        2,
        (
            *boolean_generators(2),
            _variable(2, 0) - _variable(2, 1),
        ),
        require_boolean_schema=True,
    )
    assert algebra.receipt.boolean_schema_verified
    assert algebra.receipt.prolongation_degree == algebra.receipt.degree_limit + 1
    assert (
        algebra.receipt.prolongation_monomial_count
        - algebra.receipt.prolongation_rank
        == algebra.receipt.quotient_dimension
        == 2
    )
    with pytest.raises(
        QuotientAlgebraClosureError,
        match="Boolean schema",
    ):
        compile_quotient_algebra(
            1,
            (_force(1, 0, FIELD_MODULUS - 1),),
            require_boolean_schema=True,
        )


def test_adaptive_closure_recovers_cubic_and_six_variable_one_hot_ideals() -> None:
    x = _variable(3, 0)
    y = _variable(3, 1)
    z = _variable(3, 2)
    cubic = compile_quotient_algebra(
        3,
        (
            *boolean_generators(3),
            x * y * z - SparsePolynomial.one(3),
        ),
        require_boolean_schema=True,
    )
    assert cubic.receipt.degree_limit == 5
    assert cubic.receipt.quotient_dimension == 1
    assert verify_quotient_algebra_certificate(
        cubic.export_certificate(),
        cubic.canonical_generators,
    ).passed

    variables = tuple(_variable(6, index) for index in range(6))
    one_hot = compile_quotient_algebra(
        6,
        (
            *boolean_generators(6),
            sum(variables, SparsePolynomial.zero(6))
            - SparsePolynomial.one(6),
        ),
        require_boolean_schema=True,
    )
    assert one_hot.receipt.degree_limit == 5
    assert one_hot.receipt.quotient_dimension == 6
    assert verify_quotient_algebra_certificate(
        one_hot.export_certificate(),
        one_hot.canonical_generators,
    ).passed


def test_adaptive_closure_can_be_disabled_and_never_expands_explicit_workspace() -> None:
    variables = tuple(_variable(6, index) for index in range(6))
    generators = (
        *boolean_generators(6),
        sum(variables, SparsePolynomial.zero(6)) - SparsePolynomial.one(6),
    )
    with pytest.raises(
        QuotientAlgebraClosureError,
        match="stable under complete prolongation",
    ):
        compile_quotient_algebra(
            6,
            generators,
            adaptive_closure=False,
        )

    admitted = tuple(
        exponents
        for exponents in product(range(5), repeat=3)
        if sum(exponents) <= 4 and not all(exponent >= 1 for exponent in exponents)
    )
    with pytest.raises(
        QuotientAlgebraClosureError,
        match="stable under complete prolongation",
    ):
        compile_quotient_algebra(
            3,
            boolean_generators(3),
            admitted_monomials=admitted,
        )


def test_portable_certificate_replays_independently() -> None:
    generators = (
        *boolean_generators(3),
        _variable(3, 0) - _variable(3, 1),
        _variable(3, 1) - _variable(3, 2),
    )
    algebra = compile_quotient_algebra(
        3,
        generators,
        require_boolean_schema=True,
    )
    certificate = algebra.export_certificate()
    replay = verify_quotient_algebra_certificate(certificate, generators)
    assert replay.schema == QUOTIENT_VERIFICATION_SCHEMA
    assert replay.passed
    assert replay.quotient_digest == algebra.receipt.canonical_digest
    assert replay.main_rank == algebra.receipt.rank
    assert replay.prolongation_rank == algebra.receipt.prolongation_rank
    assert replay.quotient_dimension == algebra.receipt.quotient_dimension
    assert len(certificate.certificate_sha256) == 64
    assert certificate.canonical_bytes() == algebra.export_certificate().canonical_bytes()


def test_certificate_tampering_and_source_substitution_fail_closed() -> None:
    generators = (
        *boolean_generators(2),
        _variable(2, 0) - _variable(2, 1),
    )
    algebra = compile_quotient_algebra(
        2,
        generators,
        require_boolean_schema=True,
    )
    certificate = algebra.export_certificate()
    row = certificate.rows[0]
    monomial, coefficient = row.coefficients[-1]
    corrupted_row = replace(
        row,
        coefficients=(
            *row.coefficients[:-1],
            (monomial, (coefficient % (FIELD_MODULUS - 1)) + 1),
        ),
    )
    corrupted = replace(
        certificate,
        rows=(corrupted_row, *certificate.rows[1:]),
    )
    with pytest.raises(QuotientAlgebraError):
        verify_quotient_algebra_certificate(corrupted, generators)

    substituted = (
        *boolean_generators(2),
        _force(2, 0, 1),
    )
    with pytest.raises(
        QuotientAlgebraClosureError,
        match="independently supplied source",
    ):
        verify_quotient_algebra_certificate(certificate, substituted)

    truncated = replace(
        certificate,
        prolongation_rows=certificate.prolongation_rows[:-1],
    )
    with pytest.raises(
        QuotientAlgebraClosureError,
        match="does not span",
    ):
        verify_quotient_algebra_certificate(truncated, generators)


def test_omitted_cubic_workspace_is_rejected_as_incomplete() -> None:
    admitted = tuple(
        exponents
        for exponents in product(range(5), repeat=3)
        if sum(exponents) <= 4 and not all(exponent >= 1 for exponent in exponents)
    )
    with pytest.raises(
        QuotientAlgebraClosureError,
        match="stable under complete prolongation",
    ):
        compile_quotient_algebra(
            3,
            boolean_generators(3),
            admitted_monomials=admitted,
            require_boolean_schema=True,
        )


def test_public_status_layer_keeps_all_failure_modes_distinct() -> None:
    variable = _variable(1, 0)
    certified = certify_polynomial_consequence(
        1,
        (*boolean_generators(1), _force(1, 0, 1)),
        variable,
        allowed_values=(0, 1),
    )
    assert certified.status == STATUS_CERTIFIED
    assert certified.value == 1
    assert certified.certificate is not None

    ambiguous = certify_polynomial_consequence(
        1,
        boolean_generators(1),
        variable,
        allowed_values=(0, 1),
    )
    assert ambiguous.status == STATUS_AMBIGUOUS
    assert ambiguous.value is None

    unsat = certify_polynomial_consequence(
        1,
        (*boolean_generators(1), SparsePolynomial.one(1)),
        variable,
        allowed_values=(0, 1),
    )
    assert unsat.status == STATUS_UNSAT
    assert unsat.value is None

    admitted = tuple(
        exponents
        for exponents in product(range(5), repeat=3)
        if sum(exponents) <= 4 and not all(exponent >= 1 for exponent in exponents)
    )
    incomplete = certify_polynomial_consequence(
        3,
        boolean_generators(3),
        _variable(3, 0),
        allowed_values=(0, 1),
        admitted_monomials=admitted,
    )
    assert incomplete.status == STATUS_INCOMPLETE
    assert incomplete.certificate is None

    overflow = certify_polynomial_consequence(
        20,
        (),
        _variable(20, 0),
        allowed_values=(0, 1),
    )
    assert overflow.status == STATUS_RESOURCE_OVERFLOW
    assert overflow.certificate is None


def test_nonboolean_certificate_cannot_report_a_public_pass() -> None:
    relation = _variable(1, 0) + SparsePolynomial.one(1)
    algebra = compile_quotient_algebra(1, (relation,))
    assert not algebra.receipt.boolean_schema_verified
    replay = verify_quotient_algebra_certificate(
        algebra.export_certificate(),
        (relation,),
    )
    assert not replay.passed
    assert not dict(replay.gates)["boolean_and_multiplication_structure"]
    with pytest.raises(
        QuotientAlgebraError,
        match="requires Boolean schema",
    ):
        certify_polynomial_consequence(
            1,
            (relation,),
            _variable(1, 0),
            allowed_values=(0, FIELD_MODULUS - 1),
            require_boolean_schema=False,
        )


def test_unique_out_of_domain_query_is_incomplete_not_ambiguous() -> None:
    x = _variable(2, 0)
    y = _variable(2, 1)
    generators = (
        *boolean_generators(2),
        _force(2, 0, 1),
        _force(2, 1, 1),
    )
    outcome = certify_polynomial_consequence(
        2,
        generators,
        x + y,
        allowed_values=(0, 1),
    )
    assert outcome.status == STATUS_INCOMPLETE
    assert outcome.consequence is not None
    assert outcome.consequence.status == "out_of_domain"
    assert not outcome.consequence.domain_verified


def test_claimed_query_value_and_evidence_are_bound_and_replayed() -> None:
    query = _variable(1, 0)
    generators = (
        *boolean_generators(1),
        _force(1, 0, 1),
    )
    outcome = certify_polynomial_consequence(
        1,
        generators,
        query,
        allowed_values=(0, 1),
    )
    verification = verify_ssqac_consequence_outcome(
        outcome,
        generators,
        query,
        expected_allowed_values=(0, 1),
    )
    assert verification.schema == SSQAC_OUTCOME_VERIFICATION_SCHEMA
    assert verification.passed
    assert len(outcome.outcome_sha256) == 64

    corrupted = replace(outcome, value=0)
    with pytest.raises(
        QuotientAlgebraClosureError,
        match="value differs",
    ):
        verify_ssqac_consequence_outcome(
            corrupted,
            generators,
            query,
            expected_allowed_values=(0, 1),
        )
    with pytest.raises(
        QuotientAlgebraClosureError,
        match="query differs",
    ):
        verify_ssqac_consequence_outcome(
            outcome,
            generators,
            SparsePolynomial.one(1) - query,
            expected_allowed_values=(0, 1),
        )


def test_integer_laws_have_identical_f257_f263_boolean_semantics() -> None:
    width = 3
    x = (1, 0, 0)
    y = (0, 1, 0)
    z = (0, 0, 1)
    x2 = (2, 0, 0)
    y2 = (0, 2, 0)
    z2 = (0, 0, 2)
    zero = (0, 0, 0)
    facts = (
        IntegerPolynomial(width, {x2: 1, x: -1}),
        IntegerPolynomial(width, {y2: 1, y: -1}),
        IntegerPolynomial(width, {z2: 1, z: -1}),
        IntegerPolynomial(width, {x: 1, y: -1}),
        IntegerPolynomial(width, {y: 1, z: -1}),
    )
    law_zero = IntegerPolynomial(width, {(1, 1, 1): 1})
    law_one = IntegerPolynomial(
        width,
        {
            zero: 1,
            x: -1,
            y: -1,
            z: -1,
            (1, 1, 0): 1,
            (1, 0, 1): 1,
            (0, 1, 1): 1,
            (1, 1, 1): -1,
        },
    )
    zero_receipt = verify_boolean_field_semantics(
        width,
        (*facts, law_zero),
    )
    one_receipt = verify_boolean_field_semantics(
        width,
        (*facts, law_one),
    )
    assert zero_receipt.schema == FIELD_SEMANTICS_SCHEMA
    assert zero_receipt.passed and one_receipt.passed
    assert zero_receipt.zero_set_size == one_receipt.zero_set_size == 1
    assert zero_receipt.moduli == one_receipt.moduli == (257, 263)


def test_field_semantics_rejects_coefficient_aliasing() -> None:
    aliased = IntegerPolynomial(1, {((1,), 257)})
    with pytest.raises(
        QuotientAlgebraError,
        match="vanishes in a validation field",
    ):
        verify_boolean_field_semantics(1, (aliased,))


def test_field_semantics_bind_independent_intended_zero_set() -> None:
    equality = IntegerPolynomial(
        2,
        {
            (1, 0): 1,
            (0, 1): -1,
        },
    )
    bound = verify_boolean_field_semantics(
        2,
        (equality,),
        expected_zero_set=((0, 0), (1, 1)),
    )
    assert bound.passed
    assert dict(bound.gates)["intended_semantics"]
    assert bound.intended_zero_set_sha256 is not None

    unbound = verify_boolean_field_semantics(2, (equality,))
    assert unbound.passed
    assert "intended_semantics" not in dict(unbound.gates)
    assert unbound.intended_zero_set_sha256 is None
    with pytest.raises(
        QuotientAlgebraClosureError,
        match="intended source semantics",
    ):
        verify_boolean_field_semantics(
            2,
            (equality,),
            expected_zero_set=((0, 1), (1, 0)),
        )
    with pytest.raises(
        QuotientAlgebraError,
        match="duplicate",
    ):
        verify_boolean_field_semantics(
            2,
            (equality,),
            expected_zero_set=((0, 0), (0, 0), (1, 1)),
        )


def test_three_variable_hostile_law_orbit_is_exact() -> None:
    num_variables = 3
    x = _variable(num_variables, 0)
    y = _variable(num_variables, 1)
    z = _variable(num_variables, 2)
    one = SparsePolynomial.one(num_variables)
    facts = (
        *boolean_generators(num_variables),
        x - y,
        y - z,
    )
    ambiguous = compile_quotient_algebra(
        num_variables,
        facts,
        require_boolean_schema=True,
    )
    assert ambiguous.receipt.quotient_dimension == 2
    assert ambiguous.decide_coordinate(0).status == "ambiguous"
    assert _forced_value(
        ambiguous.decide_polynomial(
            x - y,
            allowed_values=(0,),
        )
    ) == 0

    law_zero = x * y * z
    law_one = (one - x) * (one - y) * (one - z)
    zero = compile_quotient_algebra(
        num_variables,
        (*facts, law_zero),
        require_boolean_schema=True,
    )
    one_only = compile_quotient_algebra(
        num_variables,
        (*facts, law_one),
        require_boolean_schema=True,
    )
    assert _forced_value(zero.decide_coordinate(0)) == 0
    assert _forced_value(one_only.decide_coordinate(0)) == 1

    unsat = compile_quotient_algebra(
        num_variables,
        (*facts, one),
        require_boolean_schema=True,
    )
    assert unsat.receipt.status == "inconsistent"
    assert unsat.decide_coordinate(0).status == "inconsistent"


def test_invalid_one_hot_and_recoding_inputs_fail_closed() -> None:
    with pytest.raises(QuotientAlgebraError):
        one_hot_generators(2, ((),))
    with pytest.raises(QuotientAlgebraError):
        one_hot_generators(2, ((0, 0),))
    polynomial = _variable(2, 0)
    with pytest.raises(QuotientAlgebraError):
        polynomial.recode_variables((0, 0))
    with pytest.raises(QuotientAlgebraError):
        polynomial.recode_variables((0,))
