from __future__ import annotations

import json

from episode_functor_quotient_algebra import (
    SparsePolynomial,
    boolean_generators,
    certify_polynomial_consequence,
    compile_quotient_algebra,
)
from pipeline.verify_ssqac_quotient_artifact import (
    verify_outcome_artifact,
    verify_quotient_artifact,
)


def _degree_five_problem() -> tuple[
    tuple[SparsePolynomial, ...],
    SparsePolynomial,
]:
    variables = tuple(SparsePolynomial.variable(3, index) for index in range(3))
    generators = (
        *boolean_generators(3),
        variables[0] * variables[1] * variables[2]
        - SparsePolynomial.one(3),
    )
    return generators, variables[0]


def test_independent_verifier_accepts_real_adaptive_degree_five_certificate() -> None:
    generators, _ = _degree_five_problem()
    algebra = compile_quotient_algebra(
        3,
        generators,
        require_boolean_schema=True,
    )
    assert algebra.receipt.degree_limit == 5
    certificate = algebra.export_certificate()
    payload = json.loads(certificate.canonical_bytes())
    verified = verify_quotient_artifact(
        {
            "certificate": payload,
            "certificate_sha256": certificate.certificate_sha256,
        },
        [generator.canonical_data() for generator in generators],
    )
    assert verified.quotient_dimension == 1
    assert verified.certificate_sha256 == certificate.certificate_sha256


def test_independent_verifier_accepts_real_bound_consequence_outcome() -> None:
    generators, query = _degree_five_problem()
    outcome = certify_polynomial_consequence(
        3,
        generators,
        query,
        allowed_values=(0, 1),
    )
    assert outcome.status == "CERTIFIED"
    assert outcome.value == 1
    assert outcome.certificate is not None
    verified = verify_outcome_artifact(
        {
            "certificate": json.loads(outcome.certificate.canonical_bytes()),
            "outcome": json.loads(outcome.canonical_bytes()),
            "outcome_sha256": outcome.outcome_sha256,
        },
        [generator.canonical_data() for generator in generators],
        query.canonical_data(),
        expected_allowed_values=(0, 1),
    )
    assert verified.status == "CERTIFIED"
    assert verified.value == 1
