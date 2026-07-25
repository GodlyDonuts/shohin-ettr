from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json

import pytest

from pipeline.episode_functor_law_collision_board import (
    delete_law,
    deterministic_key_recode,
)
from pipeline.episode_functor_law_collision_family import (
    audit_generated_collision_family,
)
from pipeline.ssqac_collision_quotient_bridge import (
    CLAIM_BOUNDARY,
    CollisionQuotientBridgeError,
    compile_collision_quotient_bridge,
    verify_collision_quotient_bridge,
)
from pipeline.verify_ssqac_quotient_artifact import (
    ArtifactVerificationError,
    verify_outcome_artifact,
)


@pytest.fixture(scope="module")
def family():
    return audit_generated_collision_family()


def test_all_default_family_cells_are_exactly_certified(family) -> None:
    cell_count = 0
    values = {0: 0, 1: 0}
    for unit in family.units:
        for source, expected in zip(
            unit.sources,
            unit.receipt.late_answer_indices,
            strict=True,
        ):
            bridge = compile_collision_quotient_bridge(
                source,
                unit.late_query,
            )
            verification = verify_collision_quotient_bridge(
                bridge,
                source,
                unit.late_query,
            )
            assert verification.status == "CERTIFIED"
            assert verification.value == expected
            assert bridge.receipt.outcome_value == expected
            assert bridge.receipt.source_sha256
            assert bridge.receipt.query_sha256
            assert len(bridge.receipt.completion_sha256s) == 2
            assert len(bridge.receipt.completion_binding_sha256s) == 2
            assert bridge.receipt.certificate_sha256
            assert bridge.receipt.outcome_sha256
            assert bridge.receipt.gold_oracle_only is True
            assert bridge.receipt.candidate_input_allowed is False
            assert bridge.receipt.reasoning_claim_allowed is False
            assert bridge.receipt.promotion_eligible is False
            assert bridge.receipt.claim_boundary == CLAIM_BOUNDARY
            assert all(passed for _, passed in bridge.receipt.gates)
            values[expected] += 1
            cell_count += 1
    assert cell_count == 128
    assert values == {0: 64, 1: 64}


def test_law_deletion_is_ambiguous_not_certified(family) -> None:
    unit = family.units[0]
    source = delete_law(unit.sources[0])
    bridge = compile_collision_quotient_bridge(source, unit.late_query)
    verification = verify_collision_quotient_bridge(
        bridge,
        source,
        unit.late_query,
    )
    assert verification.status == "AMBIGUOUS"
    assert verification.value is None
    assert bridge.field_semantics.zero_set_size == 2
    assert sum(binding.law_admissible for binding in bridge.completion_bindings) == 2


def test_source_and_completion_variable_recoding_preserve_value(family) -> None:
    unit = family.units[9]
    source = unit.sources[3]
    baseline = compile_collision_quotient_bridge(source, unit.late_query)
    recoded_source, _ = deterministic_key_recode(
        source,
        seed="ssqac-collision-quotient-bridge-test-recode",
    )
    recoded = compile_collision_quotient_bridge(
        recoded_source,
        unit.late_query,
        variable_permutation=(1, 0),
    )
    assert baseline.receipt.outcome_status == "CERTIFIED"
    assert recoded.receipt.outcome_status == "CERTIFIED"
    assert baseline.receipt.outcome_value == recoded.receipt.outcome_value
    assert baseline.receipt.source_sha256 != recoded.receipt.source_sha256
    assert baseline.receipt.variable_permutation == (0, 1)
    assert recoded.receipt.variable_permutation == (1, 0)
    assert sorted(
        binding.variable_index for binding in recoded.completion_bindings
    ) == [0, 1]
    verify_collision_quotient_bridge(
        recoded,
        recoded_source,
        unit.late_query,
    )


def test_rejects_artifact_tampering_even_with_valid_json(family) -> None:
    unit = family.units[0]
    source = unit.sources[0]
    bridge = compile_collision_quotient_bridge(source, unit.late_query)
    artifact = json.loads(bridge.artifact_bytes)
    artifact["outcome"]["value"] = 1 - bridge.receipt.outcome_value
    tampered = replace(
        bridge,
        artifact_bytes=json.dumps(
            artifact,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii"),
    )
    with pytest.raises(
        CollisionQuotientBridgeError,
        match="artifact digest",
    ):
        verify_collision_quotient_bridge(
            tampered,
            source,
            unit.late_query,
        )


def test_standalone_verifier_rejects_refreshed_outcome_tamper(family) -> None:
    unit = family.units[0]
    bridge = compile_collision_quotient_bridge(
        unit.sources[0],
        unit.late_query,
    )
    artifact = bridge.artifact()
    artifact["outcome"]["value"] = 1 - bridge.receipt.outcome_value
    artifact["outcome_sha256"] = sha256(
        json.dumps(
            artifact["outcome"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    with pytest.raises(
        ArtifactVerificationError,
        match="forced consequence fields",
    ):
        verify_outcome_artifact(
            artifact,
            [
                generator.canonical_data()
                for generator in bridge.generators
            ],
            bridge.query_polynomial.canonical_data(),
            expected_allowed_values=(0, 1),
        )


def test_rejects_receipt_hash_and_claim_boundary_tampering(family) -> None:
    unit = family.units[1]
    bridge = compile_collision_quotient_bridge(
        unit.sources[1],
        unit.late_query,
    )
    bad_hash = replace(
        bridge,
        receipt=replace(
            bridge.receipt,
            completion_set_sha256="0" * 64,
        ),
    )
    with pytest.raises(
        CollisionQuotientBridgeError,
        match="completion_set_sha256",
    ):
        verify_collision_quotient_bridge(
            bad_hash,
            unit.sources[1],
            unit.late_query,
        )

    bad_boundary = replace(
        bridge,
        receipt=replace(
            bridge.receipt,
            candidate_input_allowed=True,
        ),
    )
    with pytest.raises(
        CollisionQuotientBridgeError,
        match="claim boundary",
    ):
        verify_collision_quotient_bridge(
            bad_boundary,
            unit.sources[1],
            unit.late_query,
        )


def test_invalid_variable_recode_and_wrong_source_fail_closed(family) -> None:
    unit = family.units[0]
    with pytest.raises(
        CollisionQuotientBridgeError,
        match="complete permutation",
    ):
        compile_collision_quotient_bridge(
            unit.sources[0],
            unit.late_query,
            variable_permutation=(0, 0),
        )

    bridge = compile_collision_quotient_bridge(
        unit.sources[0],
        unit.late_query,
    )
    with pytest.raises(CollisionQuotientBridgeError):
        verify_collision_quotient_bridge(
            bridge,
            unit.sources[1],
            unit.late_query,
        )
