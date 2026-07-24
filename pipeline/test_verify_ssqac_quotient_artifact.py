from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from pipeline.verify_ssqac_quotient_artifact import (
    ArtifactVerificationError,
    verify_outcome_artifact,
    verify_quotient_artifact,
)


GENERATOR = [[[2], 1], [[1], 256]]
QUERY = [[[2], 1], [[1], 256]]


def _digest(payload: dict[str, object]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _row(
    pivot: int,
    coefficients: list[tuple[int, int]],
    provenance: list[tuple[int, int]],
) -> dict[str, object]:
    return {
        "coefficients": [[[monomial], coefficient] for monomial, coefficient in coefficients],
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


def _refresh_certificate_digest(artifact: dict[str, object]) -> None:
    certificate = artifact["certificate"]
    assert isinstance(certificate, dict)
    artifact["certificate_sha256"] = _digest(certificate)


def _refresh_outcome_digest(artifact: dict[str, object]) -> None:
    outcome = artifact["outcome"]
    assert isinstance(outcome, dict)
    artifact["outcome_sha256"] = _digest(outcome)


def test_independently_verifies_complete_boolean_quotient() -> None:
    result = verify_quotient_artifact(_quotient_artifact(), [GENERATOR])
    assert result.quotient_dimension == 2
    assert result.main_rank == 1
    assert result.prolongation_rank == 2
    assert result.inconsistent is False
    assert "multiplication_commutation" in result.gates
    assert "boolean_idempotence" in result.gates


def test_independently_verifies_bound_forced_outcome_and_evidence() -> None:
    result = verify_outcome_artifact(
        _outcome_artifact(),
        [GENERATOR],
        QUERY,
        expected_allowed_values=(0, 1),
    )
    assert result.status == "CERTIFIED"
    assert result.value == 0
    assert "membership_evidence" in result.gates


def test_rejects_transport_digest_tamper() -> None:
    artifact = _quotient_artifact()
    artifact["certificate_sha256"] = "0" * 64
    with pytest.raises(ArtifactVerificationError, match="artifact digest"):
        verify_quotient_artifact(artifact, [GENERATOR])


def test_rejects_semantic_digest_tamper_even_when_transport_hash_is_refreshed() -> None:
    artifact = _quotient_artifact()
    certificate = artifact["certificate"]
    assert isinstance(certificate, dict)
    certificate["quotient_digest"] = "0" * 64
    _refresh_certificate_digest(artifact)
    with pytest.raises(ArtifactVerificationError, match="semantic digest"):
        verify_quotient_artifact(artifact, [GENERATOR])


def test_rejects_external_generator_substitution() -> None:
    with pytest.raises(ArtifactVerificationError, match="external source"):
        verify_quotient_artifact(_quotient_artifact(), [[[[1], 1]]])


def test_rejects_rref_coefficient_tamper_with_refreshed_hash() -> None:
    artifact = _quotient_artifact()
    certificate = artifact["certificate"]
    assert isinstance(certificate, dict)
    rows = certificate["rows"]
    assert isinstance(rows, list)
    rows[0]["coefficients"][1][1] = 255
    _refresh_certificate_digest(artifact)
    with pytest.raises(ArtifactVerificationError, match="provenance"):
        verify_quotient_artifact(artifact, [GENERATOR])


def test_rejects_provenance_tamper_with_refreshed_hash() -> None:
    artifact = _quotient_artifact()
    certificate = artifact["certificate"]
    assert isinstance(certificate, dict)
    rows = certificate["rows"]
    assert isinstance(rows, list)
    rows[0]["provenance"][0]["coefficient"] = 2
    _refresh_certificate_digest(artifact)
    with pytest.raises(ArtifactVerificationError, match="provenance"):
        verify_quotient_artifact(artifact, [GENERATOR])


def test_rejects_incomplete_prolongation_workspace() -> None:
    artifact = _quotient_artifact()
    certificate = artifact["certificate"]
    assert isinstance(certificate, dict)
    monomials = certificate["prolongation_monomials"]
    assert isinstance(monomials, list)
    monomials.pop()
    _refresh_certificate_digest(artifact)
    with pytest.raises(ArtifactVerificationError, match="complete canonical"):
        verify_quotient_artifact(artifact, [GENERATOR])


def test_rejects_incomplete_prolongation_span() -> None:
    artifact = _quotient_artifact()
    certificate = artifact["certificate"]
    assert isinstance(certificate, dict)
    rows = certificate["prolongation_rows"]
    assert isinstance(rows, list)
    rows.pop(0)
    _refresh_certificate_digest(artifact)
    with pytest.raises(ArtifactVerificationError, match="span every Macaulay"):
        verify_quotient_artifact(artifact, [GENERATOR])


def test_rejects_non_boolean_quotient_before_digest_gate() -> None:
    artifact = _quotient_artifact()
    certificate = artifact["certificate"]
    assert isinstance(certificate, dict)
    non_boolean = [[[2], 1], [[0], 1]]
    certificate["generators"] = [non_boolean]
    certificate["rows"] = [_row(2, [(2, 1), (0, 1)], [(0, 1)])]
    certificate["prolongation_rows"] = [
        _row(3, [(3, 1), (1, 1)], [(1, 1)]),
        _row(2, [(2, 1), (0, 1)], [(0, 1)]),
    ]
    _refresh_certificate_digest(artifact)
    with pytest.raises(ArtifactVerificationError, match="Boolean operator"):
        verify_quotient_artifact(artifact, [non_boolean])


def test_rejects_outcome_digest_tamper() -> None:
    artifact = _outcome_artifact()
    artifact["outcome_sha256"] = "f" * 64
    with pytest.raises(ArtifactVerificationError, match="outcome artifact digest"):
        verify_outcome_artifact(
            artifact,
            [GENERATOR],
            QUERY,
            expected_allowed_values=(0, 1),
        )


def test_rejects_query_tamper_even_with_refreshed_outcome_digest() -> None:
    artifact = _outcome_artifact()
    outcome = artifact["outcome"]
    assert isinstance(outcome, dict)
    outcome["query"] = [[[1], 1]]
    _refresh_outcome_digest(artifact)
    with pytest.raises(ArtifactVerificationError, match="externally supplied query"):
        verify_outcome_artifact(
            artifact,
            [GENERATOR],
            QUERY,
            expected_allowed_values=(0, 1),
        )


def test_rejects_claimed_value_tamper() -> None:
    artifact = _outcome_artifact()
    outcome = artifact["outcome"]
    assert isinstance(outcome, dict)
    outcome["value"] = 1
    consequence = outcome["consequence"]
    assert isinstance(consequence, dict)
    consequence["value"] = 1
    _refresh_outcome_digest(artifact)
    with pytest.raises(ArtifactVerificationError, match="forced consequence"):
        verify_outcome_artifact(
            artifact,
            [GENERATOR],
            QUERY,
            expected_allowed_values=(0, 1),
        )


def test_rejects_membership_evidence_tamper() -> None:
    artifact = _outcome_artifact()
    outcome = artifact["outcome"]
    assert isinstance(outcome, dict)
    consequence = outcome["consequence"]
    assert isinstance(consequence, dict)
    evidence = consequence["evidence"]
    assert isinstance(evidence, dict)
    terms = evidence["terms"]
    assert isinstance(terms, list)
    terms[0]["coefficient"] = 2
    _refresh_outcome_digest(artifact)
    with pytest.raises(ArtifactVerificationError, match="does not reconstruct"):
        verify_outcome_artifact(
            artifact,
            [GENERATOR],
            QUERY,
            expected_allowed_values=(0, 1),
        )


def test_rejects_allowed_value_domain_substitution() -> None:
    with pytest.raises(ArtifactVerificationError, match="external value domain"):
        verify_outcome_artifact(
            _outcome_artifact(),
            [GENERATOR],
            QUERY,
            expected_allowed_values=(0, 2),
        )


def test_rejects_embedded_certificate_substitution() -> None:
    artifact = _outcome_artifact()
    certificate = artifact["certificate"]
    assert isinstance(certificate, dict)
    certificate["require_boolean_schema"] = False
    with pytest.raises(ArtifactVerificationError, match="embedded certificate"):
        verify_outcome_artifact(
            artifact,
            [GENERATOR],
            QUERY,
            expected_allowed_values=(0, 1),
        )


def test_tamper_helpers_do_not_alias_clean_fixtures() -> None:
    tampered = deepcopy(_outcome_artifact())
    outcome = tampered["outcome"]
    assert isinstance(outcome, dict)
    outcome["value"] = 1
    clean = _outcome_artifact()
    clean_outcome = clean["outcome"]
    assert isinstance(clean_outcome, dict)
    assert clean_outcome["value"] == 0
