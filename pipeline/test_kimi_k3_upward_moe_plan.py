from __future__ import annotations

import pytest

from kimi_k3_upward_moe_plan import (
    KimiK3UpwardMoEPlanError,
    static_execution_plan,
    validate_execution_plan,
)


def test_kimi_distributed_graph_is_exact_and_unscheduled() -> None:
    plan = static_execution_plan()
    validate_execution_plan(plan)
    assert plan["h100s_per_scientific_task"] == 24
    assert plan["nodes_per_scientific_task"] == 3
    assert plan["scientific_gpu_tasks"] == 99
    assert plan["cpu_tasks"] == 3
    assert plan["total_tasks"] == 102
    assert plan["maximum_concurrent_allocations"] == 1
    assert plan["conservative_h100_hour_budget"] == pytest.approx(706.8)
    assert plan["launch_authorized"] is False


def test_kimi_graph_preserves_matched_arms_and_frozen_routing() -> None:
    plan = static_execution_plan()
    assert plan["matched_arms"] == [
        "unchanged",
        "self_refinement",
        "owner",
        "aligned_revision",
        "temporal_causal_gate",
    ]
    assert plan["native_backbone_mode"] == "eval"
    assert plan["native_router_expert_trainables"] == 0
    assert plan["no_duplicate_identity_outputs"] is True
    drifted = dict(plan)
    drifted["h100s_per_scientific_task"] = 16
    with pytest.raises(KimiK3UpwardMoEPlanError):
        validate_execution_plan(drifted)
