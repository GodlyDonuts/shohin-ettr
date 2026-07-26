from __future__ import annotations

from collections import Counter
from dataclasses import replace
import inspect
import json

import pytest

import cross_ontology_hybrid_compositions as hybrids
from cross_ontology_hybrid_compositions import (
    ArithmeticRewriteResult,
    ArithmeticRewriteSpec,
    CASES_PER_HYBRID,
    CLAIM_BOUNDARY,
    HYBRID_COUNT,
    HYBRID_ORDER,
    HornResourceResult,
    HornResourceSpec,
    HybridKind,
    ResourceHornResult,
    ResourceHornSpec,
    TOTAL_CASES,
    TOTAL_EXECUTIONS,
    audit_hybrid_cases,
    build_hybrid_cases,
    build_hybrid_qualification_receipt,
    execute_hybrid_case,
    independent_hybrid_oracle,
)


FROZEN_PAYLOAD_SHA256 = (
    "d155f868494f9379b214028c8d7475cc2cde08192c9b3a5bbdea5a73b29f98e2"
)
FORBIDDEN_CANDIDATE_TOKENS = (
    b"answer",
    b"expected",
    b"family",
    b"horn",
    b"hybrid",
    b"oracle",
    b"resource",
    b"rewrite",
    b"theory_index",
)


def _challenge_input(case, *, intervention: bool) -> dict:
    return json.loads(case.late_challenge_bytes(intervention=intervention))["input"]


def test_registry_and_receipt_freeze_exact_hybrid_geometry() -> None:
    assert HYBRID_ORDER == (
        HybridKind.ARITHMETIC_SELECTS_REWRITE_LOCATION,
        HybridKind.HORN_RELATION_SELECTS_RESOURCE_OPERATOR,
        HybridKind.RESOURCE_STATE_CONTROLS_HORN_QUERY,
    )
    assert HYBRID_COUNT == len(HYBRID_ORDER) == 3
    assert CASES_PER_HYBRID == 16
    assert TOTAL_CASES == 48
    assert TOTAL_EXECUTIONS == 96

    cases = build_hybrid_cases()
    records, receipt = build_hybrid_qualification_receipt()
    assert len(cases) == len(records) == TOTAL_CASES
    assert Counter(case.kind for case in cases) == {
        kind: CASES_PER_HYBRID for kind in HYBRID_ORDER
    }
    assert {(case.kind, case.case_index) for case in cases} == {
        (kind, case_index)
        for kind in HYBRID_ORDER
        for case_index in range(CASES_PER_HYBRID)
    }

    assert receipt.hybrid_count == 3
    assert receipt.cases_per_hybrid == 16
    assert receipt.case_count == 48
    assert receipt.execution_count == 96
    assert receipt.independent_oracle_agreement_count == 96
    assert receipt.causal_intervention_count == 48
    assert receipt.causal_signal_change_count == 48
    assert receipt.causal_output_change_count == 48
    assert receipt.candidate_label_leak_count == 0
    assert receipt.unique_row_count == 48
    assert receipt.payload_sha256 == FROZEN_PAYLOAD_SHA256
    assert receipt.claim_boundary == CLAIM_BOUNDARY
    assert receipt.all_contracts_pass


def test_two_independent_mechanics_paths_agree_on_every_execution() -> None:
    for case in build_hybrid_cases():
        for intervention in (False, True):
            assert execute_hybrid_case(
                case,
                intervention=intervention,
            ) == independent_hybrid_oracle(
                case,
                intervention=intervention,
            )

    module_source = inspect.getsource(hybrids)
    assert "execute_closure(" not in module_source
    assert "execute_normal_forms(" not in module_source
    assert "execute_sequence(" not in module_source
    assert "one_step_reducts(" not in module_source


def test_arithmetic_index_causally_selects_a_rewrite_location() -> None:
    cases = [
        case
        for case in build_hybrid_cases()
        if case.kind == HybridKind.ARITHMETIC_SELECTS_REWRITE_LOCATION
    ]
    assert len(cases) == CASES_PER_HYBRID
    for case in cases:
        assert isinstance(case.spec, ArithmeticRewriteSpec)
        factual = execute_hybrid_case(case)
        counterfactual = execute_hybrid_case(
            case,
            intervention=True,
        )
        assert isinstance(factual, ArithmeticRewriteResult)
        assert isinstance(counterfactual, ArithmeticRewriteResult)
        assert factual.selected_index != counterfactual.selected_index
        assert (
            factual.selected_path,
            factual.selected_rule_index,
        ) != (
            counterfactual.selected_path,
            counterfactual.selected_rule_index,
        )
        assert factual.terminal != counterfactual.terminal

        factual_input = _challenge_input(case, intervention=False)
        counterfactual_input = _challenge_input(
            case,
            intervention=True,
        )
        assert factual_input["a"] != counterfactual_input["a"]
        assert factual_input["b"] == counterfactual_input["b"]


def test_horn_relation_causally_selects_a_resource_operator() -> None:
    cases = [
        case
        for case in build_hybrid_cases()
        if case.kind == HybridKind.HORN_RELATION_SELECTS_RESOURCE_OPERATOR
    ]
    assert len(cases) == CASES_PER_HYBRID
    for case in cases:
        assert isinstance(case.spec, HornResourceSpec)
        assert case.spec.selector_atom not in case.spec.horn_initial
        factual = execute_hybrid_case(case)
        counterfactual = execute_hybrid_case(
            case,
            intervention=True,
        )
        assert isinstance(factual, HornResourceResult)
        assert isinstance(counterfactual, HornResourceResult)
        assert factual.selector_holds
        assert not counterfactual.selector_holds
        assert factual.selected_symbol == case.spec.true_symbol
        assert counterfactual.selected_symbol == case.spec.false_symbol
        assert factual.outcome != counterfactual.outcome

        factual_input = _challenge_input(case, intervention=False)
        counterfactual_input = _challenge_input(
            case,
            intervention=True,
        )
        assert factual_input["a"] != counterfactual_input["a"]
        assert factual_input["b"] == counterfactual_input["b"]


def test_resource_state_causally_controls_a_horn_query() -> None:
    cases = [
        case
        for case in build_hybrid_cases()
        if case.kind == HybridKind.RESOURCE_STATE_CONTROLS_HORN_QUERY
    ]
    assert len(cases) == CASES_PER_HYBRID
    for case in cases:
        assert isinstance(case.spec, ResourceHornSpec)
        factual = execute_hybrid_case(case)
        counterfactual = execute_hybrid_case(
            case,
            intervention=True,
        )
        assert isinstance(factual, ResourceHornResult)
        assert isinstance(counterfactual, ResourceHornResult)
        assert factual.control_holds
        assert not counterfactual.control_holds
        assert factual.selected_query == case.spec.query_if_true
        assert counterfactual.selected_query == case.spec.query_if_false
        assert factual.query_holds
        assert not counterfactual.query_holds
        assert factual.resource_outcome.status.value == "halt"
        assert counterfactual.resource_outcome.status.value == "halt"

        factual_input = _challenge_input(case, intervention=False)
        counterfactual_input = _challenge_input(
            case,
            intervention=True,
        )
        assert factual_input["a"] != counterfactual_input["a"]
        assert factual_input["b"] == counterfactual_input["b"]
        assert factual_input["c"] == counterfactual_input["c"]


def test_candidate_surfaces_exclude_assessor_and_family_labels() -> None:
    for case in build_hybrid_cases():
        source = case.compiler_source_bytes()
        factual = case.late_challenge_bytes()
        intervention = case.late_challenge_bytes(intervention=True)
        candidate_bytes = (source + factual + intervention).lower()
        assert source.isascii()
        assert factual.isascii()
        assert intervention.isascii()
        assert source.endswith(b"\n")
        assert factual.endswith(b"\n")
        assert intervention.endswith(b"\n")
        assert all(token not in candidate_bytes for token in FORBIDDEN_CANDIDATE_TOKENS)
        assert case.kind.value.encode("ascii") not in candidate_bytes
        assert "kind" not in json.loads(source)
        assert "kind" not in json.loads(factual)
        assert "kind" not in json.loads(intervention)


def test_hashes_are_complete_unique_and_deterministic() -> None:
    first_cases = build_hybrid_cases()
    first_records, first_receipt = build_hybrid_qualification_receipt()
    second_records, second_receipt = audit_hybrid_cases(first_cases)
    assert first_records == second_records
    assert first_receipt == second_receipt
    assert first_receipt.payload_sha256 == FROZEN_PAYLOAD_SHA256
    assert len({record.row_sha256 for record in first_records}) == 48
    for record in first_records:
        hashes = (
            record.source_sha256,
            record.challenge_sha256,
            record.intervention_challenge_sha256,
            record.expected_sha256,
            record.intervention_expected_sha256,
            record.row_sha256,
        )
        assert all(
            len(value) == 64 and set(value) <= set("0123456789abcdef")
            for value in hashes
        )
        assert record.challenge_sha256 != (record.intervention_challenge_sha256)
        assert record.expected_sha256 != (record.intervention_expected_sha256)


def test_audit_fails_closed_on_intervention_or_label_tampering() -> None:
    cases = build_hybrid_cases()
    invariant_intervention = replace(
        cases[0],
        intervention_challenge=cases[0].challenge,
    )
    with pytest.raises(
        ValueError,
        match="intervention is byte-invariant",
    ):
        audit_hybrid_cases((invariant_intervention, *cases[1:]))

    leaked_source = replace(
        cases[0],
        compiler_source=cases[0].compiler_source + b"oracle\n",
    )
    with pytest.raises(
        ValueError,
        match="candidate bytes contain assessor labels",
    ):
        audit_hybrid_cases((leaked_source, *cases[1:]))


def test_claim_boundary_explicitly_rejects_a_reasoning_claim() -> None:
    lower = CLAIM_BOUNDARY.lower()
    assert "offline" in lower
    assert "not learned" in lower
    assert "native reasoning" in lower
    assert "general-reasoning claim" in lower
