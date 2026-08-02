"""Tests for the finite operation-effect campaign transition table."""

from __future__ import annotations

import pytest

from plan_operation_effect_successor import (
    OperationEffectSuccessorPlanError,
    V15_SCHEMA,
    V18_SCHEMA,
    V19_SCHEMA,
    V20_SCHEMA,
    V21_SCHEMA,
    plan_successor,
)
from route_operation_effect_set_result import ROUTE_SCHEMA


def _route(schema: str, route: str) -> dict[str, object]:
    return {
        "reason": "measured branch",
        "route": route,
        "schema": ROUTE_SCHEMA,
        "terminal_contract_schema": schema,
    }


@pytest.mark.parametrize(
    ("predecessor", "route", "successor", "island", "binding", "warm"),
    (
        (
            V15_SCHEMA,
            "operation_family_island_curriculum",
            V19_SCHEMA,
            True,
            False,
            False,
        ),
        (
            V19_SCHEMA,
            "joint_operation_family_rail_release",
            V18_SCHEMA,
            False,
            False,
            True,
        ),
        (
            V19_SCHEMA,
            "operation_role_state_bilinear_arbiter",
            V20_SCHEMA,
            True,
            True,
            False,
        ),
        (
            V20_SCHEMA,
            "joint_state_bound_family_rail_release",
            V21_SCHEMA,
            False,
            True,
            True,
        ),
    ),
)
def test_planner_allows_only_preregistered_transitions(
    predecessor: str,
    route: str,
    successor: str,
    island: bool,
    binding: bool,
    warm: bool,
) -> None:
    result = plan_successor(_route(predecessor, route), {"schema": predecessor})
    assert result["action"] == "submit_scientific_successor"
    assert result["successor_schema"] == successor
    assert result["family_island"] is island
    assert result["family_state_binding"] is binding
    assert result["warm_start"] is warm
    assert result["updates"] == 1000


@pytest.mark.parametrize("schema", (V15_SCHEMA, V18_SCHEMA, V19_SCHEMA, V20_SCHEMA, V21_SCHEMA))
def test_planner_stops_all_unregistered_routes(schema: str) -> None:
    result = plan_successor(_route(schema, "reject_local_family"), {"schema": schema})
    assert result == {
        "action": "stop",
        "predecessor_schema": schema,
        "reason": "measured branch",
        "route": "reject_local_family",
        "schema": "shohin-ettr-operation-effect-successor-plan-v1",
    }


def test_planner_rejects_crossed_route_receipt() -> None:
    with pytest.raises(OperationEffectSuccessorPlanError, match="contract differs"):
        plan_successor(
            _route(V15_SCHEMA, "operation_family_island_curriculum"),
            {"schema": V19_SCHEMA},
        )


def test_planner_does_not_restore_obsolete_v15_to_v18_hop() -> None:
    result = plan_successor(
        _route(V15_SCHEMA, "exclusive_operation_family_gate"),
        {"schema": V15_SCHEMA},
    )
    assert result["action"] == "stop"
