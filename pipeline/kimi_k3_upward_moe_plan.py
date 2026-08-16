"""Static multi-node execution geometry for the Kimi K3 Shohin scale point."""

from __future__ import annotations

from typing import Any

from q36_upward_moe_kimi_k3_host import (
    ACTIVE_PARAMETERS,
    MINIMUM_H100S,
    MINIMUM_NODES,
    MODEL_ID,
    MODEL_REVISION,
    TOTAL_PARAMETERS,
)

SOURCE_GRAPH_H100_HOURS = 58.90
SOURCE_GRAPH_H100S_PER_TASK = 2
SCIENTIFIC_GPU_TASKS = 99
CPU_TASKS = 3
TOTAL_TASKS = SCIENTIFIC_GPU_TASKS + CPU_TASKS
MAX_CONCURRENT_ALLOCATIONS = 1
H100_SCALE_FACTOR = MINIMUM_H100S // SOURCE_GRAPH_H100S_PER_TASK
CONSERVATIVE_H100_HOUR_BUDGET = SOURCE_GRAPH_H100_HOURS * H100_SCALE_FACTOR


class KimiK3UpwardMoEPlanError(RuntimeError):
    """The Kimi K3 distributed execution geometry differs."""


def static_execution_plan() -> dict[str, Any]:
    return {
        "schema": "shohin-kimi-k3-upward-moe-plan-v1",
        "status": "code_only_not_scheduled",
        "host": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "total_parameters": TOTAL_PARAMETERS,
        "active_parameters": ACTIVE_PARAMETERS,
        "h100s_per_scientific_task": MINIMUM_H100S,
        "nodes_per_scientific_task": MINIMUM_NODES,
        "scientific_gpu_tasks": SCIENTIFIC_GPU_TASKS,
        "cpu_tasks": CPU_TASKS,
        "total_tasks": TOTAL_TASKS,
        "maximum_concurrent_allocations": MAX_CONCURRENT_ALLOCATIONS,
        "source_graph_h100_hours": SOURCE_GRAPH_H100_HOURS,
        "source_graph_h100s_per_task": SOURCE_GRAPH_H100S_PER_TASK,
        "h100_scale_factor": H100_SCALE_FACTOR,
        "conservative_h100_hour_budget": CONSERVATIVE_H100_HOUR_BUDGET,
        "mechanics_precondition_tasks": 1,
        "stages": {
            "owner_fit": 1,
            "host_owned_drafts": 16,
            "draft_merge_cpu": 1,
            "materialize_cpu": 1,
            "aligned_revision_fit": 1,
            "temporal_fit": 1,
            "matched_evaluation": 80,
            "final_score_cpu": 1,
        },
        "matched_arms": [
            "unchanged",
            "self_refinement",
            "owner",
            "aligned_revision",
            "temporal_causal_gate",
        ],
        "native_backbone_mode": "eval",
        "native_router_expert_trainables": 0,
        "no_duplicate_identity_outputs": True,
        "launch_authorized": False,
    }


def validate_execution_plan(plan: dict[str, Any]) -> None:
    if plan != static_execution_plan():
        raise KimiK3UpwardMoEPlanError("Kimi K3 execution plan differs")
    if sum(plan["stages"].values()) != TOTAL_TASKS:
        raise KimiK3UpwardMoEPlanError("Kimi K3 task count differs")
