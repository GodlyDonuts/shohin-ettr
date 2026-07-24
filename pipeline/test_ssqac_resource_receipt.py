from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from pipeline.ssqac_resource_receipt import (
    PEAK_WORKSPACE_KEYS,
    REPLAYABLE_ALU_RECEIPT_SCHEMA,
    ResourceReceiptError,
    build_ssqac_resource_receipt,
    make_replayable_primitive_alu_receipt,
    verify_ssqac_resource_receipt,
)


GENERATOR = [[[2], 1], [[1], 256]]
QUERY = [[[2], 1], [[1], 256]]


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()


def _row(
    pivot: int,
    coefficients: list[tuple[int, int]],
    provenance: list[tuple[int, int]],
) -> dict[str, object]:
    return {
        "coefficients": [
            [[monomial], coefficient]
            for monomial, coefficient in coefficients
        ],
        "pivot": [pivot],
        "provenance": [
            {
                "coefficient": coefficient,
                "generator_index": 0,
                "multiplier": [multiplier],
            }
            for multiplier, coefficient in provenance
        ],
    }


def _certificate() -> dict[str, object]:
    semantic = {
        "basis": [[1], [0]],
        "boolean_schema_verified": True,
        "degree_limit": 2,
        "field_modulus": 257,
        "monomials": [[2], [1], [0]],
        "prolongation_degree": 3,
        "prolongation_monomial_count": 4,
        "prolongation_rank": 2,
        "rref": [[0, [[0, 1], [1, 256]]]],
        "status": "ok",
        "variable_count": 1,
    }
    return {
        "admitted_monomials": [[2], [1], [0]],
        "degree_limit": 2,
        "field_modulus": 257,
        "generators": [GENERATOR],
        "prolongation_degree": 3,
        "prolongation_monomials": [[3], [2], [1], [0]],
        "prolongation_rows": [
            _row(3, [(3, 1), (1, 256)], [(0, 1), (1, 1)]),
            _row(2, [(2, 1), (1, 256)], [(0, 1)]),
        ],
        "quotient_digest": _digest(semantic),
        "require_boolean_schema": True,
        "rows": [_row(2, [(2, 1), (1, 256)], [(0, 1)])],
        "schema": "ssqac_quotient_algebra_certificate_v1",
        "variable_count": 1,
    }


def _quotient_artifact() -> dict[str, object]:
    certificate = _certificate()
    return {
        "certificate": certificate,
        "certificate_sha256": _digest(certificate),
    }


def _outcome_artifact() -> dict[str, object]:
    certificate = _certificate()
    outcome = {
        "allowed_values": [0, 1],
        "certificate_sha256": _digest(certificate),
        "consequence": {
            "domain_verified": True,
            "evidence": {
                "target": QUERY,
                "terms": [
                    {
                        "coefficient": 1,
                        "generator_index": 0,
                        "multiplier": [0],
                    }
                ],
            },
            "normal_form": [],
            "quotient_digest": certificate["quotient_digest"],
            "status": "forced",
            "value": 0,
        },
        "diagnostic": None,
        "query": QUERY,
        "quotient_digest": certificate["quotient_digest"],
        "schema": "ssqac_quotient_consequence_outcome_v1",
        "status": "CERTIFIED",
        "value": 0,
    }
    return {
        "certificate": certificate,
        "outcome": outcome,
        "outcome_sha256": _digest(outcome),
    }


def _bounds(**overrides: int) -> dict[str, int]:
    result = {
        "artifact_bytes": 1_000_000,
        "monomial_slots": 4096,
        "primitive_instruction_slots": 1_000,
        "provenance_support": 100_000,
        "provenance_terms": 1_000_000,
        "quotient_dimension": 256,
        "row_support": 4096,
        "rref_nonzeros": 1_000_000,
    }
    result.update(overrides)
    assert set(result) == set(PEAK_WORKSPACE_KEYS)
    return result


def _refresh_resource_digests(receipt: dict[str, object]) -> None:
    payload = receipt["receipt"]
    assert isinstance(payload, dict)
    vector = payload["resource_vector"]
    payload["resource_vector_sha256"] = _digest(vector)
    receipt["receipt_sha256"] = _digest(payload)


def test_exact_quotient_resource_vector_and_independent_replay() -> None:
    artifact = _quotient_artifact()
    bounds = _bounds()
    receipt = build_ssqac_resource_receipt(
        artifact,
        [GENERATOR],
        declared_peak_workspace_bounds=bounds,
    )
    payload = receipt["receipt"]
    assert isinstance(payload, dict)
    vector = payload["resource_vector"]
    assert isinstance(vector, dict)
    assert vector["variable_count"] == 1
    assert vector["generator_count"] == 1
    assert vector["generator_term_count"] == 2
    assert vector["main_degree"] == 2
    assert vector["prolongation_degree"] == 3
    assert vector["main_monomial_count"] == 3
    assert vector["prolongation_monomial_count"] == 4
    assert vector["main_rank"] == 1
    assert vector["prolongation_rank"] == 2
    assert vector["main_rref_nonzeros"] == 2
    assert vector["prolongation_rref_nonzeros"] == 4
    assert vector["main_provenance_terms"] == 1
    assert vector["prolongation_provenance_terms"] == 3
    assert vector["max_row_support"] == 2
    assert vector["max_provenance_support"] == 2
    assert vector["quotient_dimension"] == 2
    assert vector["query_term_count"] == 0
    assert vector["evidence_term_count"] == 0
    assert vector["source_independent_artifact_bytes"] == len(
        json.dumps(
            artifact,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    runtime = payload["runtime_observations"]
    assert runtime == {
        "device_peak_memory_bytes": None,
        "status": (
            "not-derived: wall time and device memory require an external "
            "runtime observer"
        ),
        "wall_time_seconds": None,
    }
    verified = verify_ssqac_resource_receipt(
        receipt,
        artifact,
        [GENERATOR],
        expected_declared_peak_workspace_bounds=bounds,
    )
    assert verified.passed
    assert verified.artifact_kind == "quotient"


def test_outcome_counts_query_and_membership_evidence() -> None:
    artifact = _outcome_artifact()
    bounds = _bounds()
    receipt = build_ssqac_resource_receipt(
        artifact,
        [GENERATOR],
        expected_query=QUERY,
        expected_allowed_values=(0, 1),
        declared_peak_workspace_bounds=bounds,
    )
    payload = receipt["receipt"]
    assert isinstance(payload, dict)
    vector = payload["resource_vector"]
    assert isinstance(vector, dict)
    assert vector["query_term_count"] == 2
    assert vector["evidence_term_count"] == 1
    assert vector["total_polynomial_term_count"] == 5
    result = verify_ssqac_resource_receipt(
        receipt,
        artifact,
        [GENERATOR],
        expected_query=QUERY,
        expected_allowed_values=(0, 1),
        expected_declared_peak_workspace_bounds=bounds,
    )
    assert result.artifact_kind == "outcome"


def test_replays_primitive_opcode_counts_cycles_and_sequential_depth() -> None:
    program = make_replayable_primitive_alu_receipt(
        (
            ("LOAD", 0, 0, 0),
            ("INV", 0, 1, 0),
            ("SCALE", 0, 1, 0),
            ("LOAD", 1, 0, 0),
            ("NEG", 0, 1, 0),
            ("AXPY", 1, 0, 1),
            ("SWAP", 0, 1, 0),
            ("HALT", 0, 0, 0),
        )
    )
    artifact = _quotient_artifact()
    bounds = _bounds()
    receipt = build_ssqac_resource_receipt(
        artifact,
        [GENERATOR],
        primitive_program_receipts=(program,),
        declared_peak_workspace_bounds=bounds,
    )
    payload = receipt["receipt"]
    assert isinstance(payload, dict)
    vector = payload["resource_vector"]
    assert isinstance(vector, dict)
    assert vector["primitive_program_count"] == 1
    assert vector["primitive_cycles"] == 8
    assert vector["sequential_depth"] == 8
    assert vector["primitive_opcode_counts"] == {
        "LOAD": 2,
        "INV": 1,
        "NEG": 1,
        "SCALE": 1,
        "AXPY": 1,
        "SWAP": 1,
        "HALT": 1,
    }
    actual = payload["actual_peak_workspace"]
    assert isinstance(actual, dict)
    assert actual["primitive_instruction_slots"] == 8
    assert verify_ssqac_resource_receipt(
        receipt,
        artifact,
        [GENERATOR],
        primitive_program_receipts=(program,),
        expected_declared_peak_workspace_bounds=bounds,
    ).passed


def test_rejects_opaque_nonreplayable_primitive_receipt() -> None:
    opaque = {
        "schema": "ssqac_primitive_field_row_program_receipt_v1",
        "executed_instructions": 7,
        "trace_sha256": "0" * 64,
    }
    with pytest.raises(ResourceReceiptError, match="keys differ|replayable"):
        build_ssqac_resource_receipt(
            _quotient_artifact(),
            [GENERATOR],
            primitive_program_receipts=(opaque,),
            declared_peak_workspace_bounds=_bounds(),
        )


def test_rejects_primitive_claim_tamper_even_with_outer_rehash() -> None:
    program = make_replayable_primitive_alu_receipt(
        (("LOAD", 0, 0, 0), ("HALT", 0, 0, 0))
    )
    assert program["schema"] == REPLAYABLE_ALU_RECEIPT_SCHEMA
    counts = program["opcode_counts"]
    assert isinstance(counts, dict)
    counts["LOAD"] = 2
    with pytest.raises(ResourceReceiptError, match="opcode counts differ"):
        build_ssqac_resource_receipt(
            _quotient_artifact(),
            [GENERATOR],
            primitive_program_receipts=(program,),
            declared_peak_workspace_bounds=_bounds(),
        )


def test_rejects_resource_vector_tamper_with_refreshed_hashes() -> None:
    artifact = _quotient_artifact()
    bounds = _bounds()
    receipt = build_ssqac_resource_receipt(
        artifact,
        [GENERATOR],
        declared_peak_workspace_bounds=bounds,
    )
    payload = receipt["receipt"]
    assert isinstance(payload, dict)
    vector = payload["resource_vector"]
    assert isinstance(vector, dict)
    vector["main_rref_nonzeros"] = 999
    _refresh_resource_digests(receipt)
    with pytest.raises(ResourceReceiptError, match="independent replay"):
        verify_ssqac_resource_receipt(
            receipt,
            artifact,
            [GENERATOR],
            expected_declared_peak_workspace_bounds=bounds,
        )


def test_rejects_artifact_substitution_and_workspace_declaration_tamper() -> None:
    artifact = _quotient_artifact()
    bounds = _bounds()
    receipt = build_ssqac_resource_receipt(
        artifact,
        [GENERATOR],
        declared_peak_workspace_bounds=bounds,
    )
    changed_bounds = dict(bounds)
    changed_bounds["monomial_slots"] -= 1
    with pytest.raises(ResourceReceiptError, match="external declaration"):
        verify_ssqac_resource_receipt(
            receipt,
            artifact,
            [GENERATOR],
            expected_declared_peak_workspace_bounds=changed_bounds,
        )
    changed_artifact = deepcopy(artifact)
    changed_artifact["certificate_sha256"] = "0" * 64
    with pytest.raises(ResourceReceiptError, match="quotient replay failed"):
        verify_ssqac_resource_receipt(
            receipt,
            changed_artifact,
            [GENERATOR],
            expected_declared_peak_workspace_bounds=bounds,
        )


def test_rejects_workspace_underdeclaration_and_resource_overflow() -> None:
    artifact = _quotient_artifact()
    with pytest.raises(ResourceReceiptError, match="workspace underflow"):
        build_ssqac_resource_receipt(
            artifact,
            [GENERATOR],
            declared_peak_workspace_bounds=_bounds(monomial_slots=3),
        )
    with pytest.raises(ResourceReceiptError, match="resource overflow"):
        build_ssqac_resource_receipt(
            artifact,
            [GENERATOR],
            declared_peak_workspace_bounds=_bounds(),
            resource_caps={"variable_count": 0},
        )
    with pytest.raises(ResourceReceiptError, match="workspace overflow"):
        build_ssqac_resource_receipt(
            artifact,
            [GENERATOR],
            declared_peak_workspace_bounds=_bounds(monomial_slots=5000),
        )


def test_rejects_fabricated_runtime_observations_even_with_rehash() -> None:
    artifact = _quotient_artifact()
    bounds = _bounds()
    receipt = build_ssqac_resource_receipt(
        artifact,
        [GENERATOR],
        declared_peak_workspace_bounds=bounds,
    )
    payload = receipt["receipt"]
    assert isinstance(payload, dict)
    runtime = payload["runtime_observations"]
    assert isinstance(runtime, dict)
    runtime["wall_time_seconds"] = 0.001
    receipt["receipt_sha256"] = _digest(payload)
    with pytest.raises(ResourceReceiptError, match="cannot fabricate"):
        verify_ssqac_resource_receipt(
            receipt,
            artifact,
            [GENERATOR],
            expected_declared_peak_workspace_bounds=bounds,
        )


def test_adaptive_degree_five_real_certificate_is_counted_exactly() -> None:
    from episode_functor_quotient_algebra import (
        SparsePolynomial,
        boolean_generators,
        compile_quotient_algebra,
    )

    def variable(index: int) -> SparsePolynomial:
        exponents = [0, 0, 0]
        exponents[index] = 1
        return SparsePolynomial.monomial(3, tuple(exponents))

    x, y, z = (variable(index) for index in range(3))
    algebra = compile_quotient_algebra(
        3,
        (
            *boolean_generators(3),
            x * y * z - SparsePolynomial.one(3),
        ),
        require_boolean_schema=True,
    )
    certificate = algebra.export_certificate()
    plain = json.loads(certificate.canonical_bytes())
    artifact = {
        "certificate": plain,
        "certificate_sha256": certificate.certificate_sha256,
    }
    generators = [
        generator.canonical_data() for generator in algebra.canonical_generators
    ]
    bounds = _bounds()
    receipt = build_ssqac_resource_receipt(
        artifact,
        generators,
        declared_peak_workspace_bounds=bounds,
    )
    payload = receipt["receipt"]
    assert isinstance(payload, dict)
    vector = payload["resource_vector"]
    assert isinstance(vector, dict)
    assert vector["main_degree"] == 5
    assert vector["prolongation_degree"] == 6
    assert vector["variable_count"] == 3
    assert verify_ssqac_resource_receipt(
        receipt,
        artifact,
        generators,
        expected_declared_peak_workspace_bounds=bounds,
    ).passed
