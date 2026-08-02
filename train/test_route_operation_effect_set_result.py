"""Deterministic failure-tree tests for the effect-set result router."""

from __future__ import annotations

from copy import deepcopy

from route_operation_effect_set_result import (
    CARDINALITY_GATED_SCHEMA,
    POST_WRITE_LINK_SCHEMA,
    RAIL_LOCAL_EFFECT_SCHEMA,
    REPORT_SCHEMA,
    ROLE_ANCHORED_SCHEMA,
    WRITE_LINK_RAIL_SCHEMA,
    route_result,
)


def _local(
    *,
    effect_set: float = 0.0,
    dense: float = 0.0,
    terminal: float = 0.0,
    noop: int = 20,
    other: int = 80,
    entity: float = 0.0,
    link: float = 0.0,
) -> dict[str, object]:
    return {
        "exact_rates": {
            "complete_dense_edit_exact": dense,
            "complete_effect_set_exact": effect_set,
        },
        "positive_exact_rates": {
            "entity": entity,
            "relation_link": link,
        },
        "predicted_kind_histogram": {"0": noop, "1": other},
        "terminal_state_exact_rate": terminal,
    }


def _causal(rate: float) -> dict[str, object]:
    return {"paired_order_joint_rate": rate}


def _report() -> dict[str, object]:
    return {
        "schema": REPORT_SCHEMA,
        "status": "pass",
        "terminal_state_receipt": {"contract_sha256": "0" * 64},
        "operation_effect_diagnostics": {
            "before": _local(),
            "after": _local(),
        },
        "evaluation": {
            phase: {
                "arms": {
                    "autonomous_program_autonomous_state": {
                        "source_deleted_causal": {
                            "world": _causal(0.0),
                            "command": _causal(0.0),
                        }
                    }
                }
            }
            for phase in ("before", "after")
        },
    }


def test_router_promotes_only_simultaneous_local_and_causal_gain() -> None:
    report = _report()
    report["operation_effect_diagnostics"]["after"] = _local(
        effect_set=0.1,
        dense=0.2,
        terminal=0.1,
    )
    arm = report["evaluation"]["after"]["arms"]["autonomous_program_autonomous_state"][
        "source_deleted_causal"
    ]
    arm["world"] = _causal(0.05)
    arm["command"] = _causal(0.1)
    assert route_result(report)["route"] == "replicate_fresh_population"


def test_router_sends_kind_collapse_to_ast_anchors() -> None:
    report = _report()
    report["operation_effect_diagnostics"]["after"] = _local(noop=95, other=5)
    assert route_result(report)["route"] == "public_ast_role_anchored_effect_queries"


def test_router_sends_role_anchored_collapse_to_cardinality_gate() -> None:
    report = _report()
    report["operation_effect_diagnostics"]["after"] = _local(noop=95, other=5)
    result = route_result(report, {"schema": ROLE_ANCHORED_SCHEMA})
    assert result["route"] == "explicit_effect_cardinality_gate"
    assert result["terminal_contract_schema"] == ROLE_ANCHORED_SCHEMA


def test_router_sends_cardinality_collapse_to_write_link_rails() -> None:
    report = _report()
    report["operation_effect_diagnostics"]["after"] = _local(noop=95, other=5)
    result = route_result(report, {"schema": CARDINALITY_GATED_SCHEMA})
    assert result["route"] == "write_link_typed_rails"


def test_router_sends_typed_rail_collapse_to_payload_islands() -> None:
    report = _report()
    report["operation_effect_diagnostics"]["after"] = _local(noop=95, other=5)
    result = route_result(report, {"schema": WRITE_LINK_RAIL_SCHEMA})
    assert result["route"] == "rail_local_pointer_payload_islands"
    result = route_result(report, {"schema": RAIL_LOCAL_EFFECT_SCHEMA})
    assert result["route"] == "rail_local_pointer_payload_islands"
    result = route_result(report, {"schema": POST_WRITE_LINK_SCHEMA})
    assert result["route"] == "rail_local_pointer_payload_islands"


def test_router_sends_relation_binding_failure_to_two_phase_algebra() -> None:
    report = _report()
    report["operation_effect_diagnostics"]["after"] = _local(
        entity=0.75,
        link=0.1,
    )
    assert route_result(report)["route"] == "two_phase_entity_then_relation_algebra"


def test_router_sends_state_only_gain_to_crossed_isolation() -> None:
    report = _report()
    report["operation_effect_diagnostics"]["after"] = _local(
        effect_set=0.05,
        dense=0.1,
        terminal=0.05,
    )
    assert route_result(report)["route"] == "crossed_state_sufficiency_isolation"


def test_router_rejects_loss_only_and_one_axis_gain() -> None:
    report = _report()
    only_world = deepcopy(report)
    only_world["evaluation"]["after"]["arms"]["autonomous_program_autonomous_state"][
        "source_deleted_causal"
    ]["world"] = _causal(0.5)
    assert route_result(report)["route"] == "reject_unordered_effect_set"
