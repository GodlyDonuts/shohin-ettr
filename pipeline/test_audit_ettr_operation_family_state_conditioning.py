from audit_ettr_operation_family_state_conditioning import (
    _conditional_summary,
    _multinomial_nb,
    _train_to_development,
    operation_features,
    state_features,
)
from audit_ettr_operation_effect_family_identifiability import (
    abstract_resolved_operation,
)
from audit_ettr_public_operation_state_delta import runtime_state_value
from test_audit_ettr_public_operation_state_delta import _state


def test_operation_features_factor_structure_without_literal_identity() -> None:
    left = operation_features(["call", 7, [["integer", 3]]])
    right = operation_features(["call", 7, [["integer", 2]]])
    different = operation_features(["call", 8, [["integer", 3]]])
    assert left == right
    assert left != different


def test_operation_features_accept_abstract_integer_nodes() -> None:
    left = abstract_resolved_operation(["call", 7, [["integer", 3]]])
    right = abstract_resolved_operation(["call", 7, [["integer", -129]]])
    assert left == ["call", 7, [["integer"]]]
    assert operation_features(left) == operation_features(right)


def test_state_features_separate_topology_from_exact_values() -> None:
    left = _state(value=3)
    right = _state(value=7)
    assert runtime_state_value(left) != runtime_state_value(right)
    assert state_features(left, exact_values=False) == state_features(
        right, exact_values=False
    )
    assert state_features(left, exact_values=True) != state_features(
        right, exact_values=True
    )


def test_conditional_and_transfer_summaries_are_exact() -> None:
    train = {
        "a": {"none": 2, "write": 0, "link": 0},
        "b": {"none": 0, "write": 1, "link": 1},
    }
    development = {
        "a": {"none": 1, "write": 1, "link": 0},
        "c": {"none": 0, "write": 0, "link": 2},
    }
    summary = _conditional_summary(development)
    assert summary["accuracy"] == 0.75
    assert summary["ambiguous_instances"] == 2
    transfer = _train_to_development(train, development)
    assert transfer["coverage"] == 0.5
    assert transfer["all_accuracy"] == 0.25
    assert transfer["seen_accuracy"] == 0.5


def test_factorized_classifier_uses_feature_evidence() -> None:
    train = {
        "family_counts": {"syntax": {"none": 2, "write": 2, "link": 2}},
        "feature_counts": {
            "syntax": {
                "n": {"none": 2},
                "w": {"write": 2},
                "l": {"link": 2},
            }
        },
        "feature_totals": {
            "syntax": {"none": 2, "write": 2, "link": 2}
        },
        "signatures": {"syntax": {}},
    }
    development = {
        "signatures": {
            "syntax": {
                "sn": {"none": 1},
                "sw": {"write": 1},
                "sl": {"link": 1},
            }
        },
        "signature_features": {
            "syntax": {"sn": ("n",), "sw": ("w",), "sl": ("l",)}
        },
    }
    result = _multinomial_nb(train, development, "syntax")
    assert result["accuracy"] == 1.0
