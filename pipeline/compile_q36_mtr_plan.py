#!/usr/bin/env python3
"""Compile the frozen Q36-MTR graph into a dry-run-only task blueprint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from q36_mtr_contract import (
    MAXIMUM_CONCURRENT_SINGLE_H100_REQUESTS,
    SCHEMA as GRAPH_SCHEMA,
    STAGES,
    graph_payload,
    validate_graph,
)

SCHEMA = "shohin-q36-mtr-dry-run-plan-v1"
GPU_ENTRYPOINTS = {
    "mechanics": "q36_mtr_mechanics",
    "owner_fit": "q36_mtr_train_role:owner",
    "draft_generate": "q36_mtr_generate_drafts",
    "aligned_fit": "q36_mtr_train_role:aligned",
    "draft_hidden_fit": "q36_mtr_train_role:draft_hidden",
    "calibration_revision": "q36_mtr_evaluate:calibration_revision",
    "calibration_unchanged": "q36_mtr_evaluate:calibration_unchanged",
    "commit_fit": "q36_mtr_train_commit",
    "development_revision": "q36_mtr_evaluate:development_revision",
    "development_unchanged": "q36_mtr_evaluate:development_unchanged",
    "development_self_refinement": "q36_mtr_evaluate:development_self_refinement",
    "development_draft_hidden": "q36_mtr_evaluate:development_draft_hidden",
}
CPU_ENTRYPOINTS = {
    "preflight_cpu": "q36_mtr_capture_environment",
    "draft_merge": "q36_mtr_merge_drafts",
    "materialize": "q36_mtr_materialize",
    "calibration_revision_merge": "q36_mtr_merge_evaluation:calibration_revision",
    "calibration_unchanged_merge": "q36_mtr_merge_evaluation:calibration_unchanged",
    "commit_pairs": "q36_mtr_commit_pairs:calibration",
    "development_revision_merge": "q36_mtr_merge_evaluation:development_revision",
    "development_unchanged_merge": "q36_mtr_merge_evaluation:development_unchanged",
    "development_self_refinement_merge": "q36_mtr_merge_evaluation:development_self_refinement",
    "development_draft_hidden_merge": "q36_mtr_merge_evaluation:development_draft_hidden",
    "development_commit_pairs": "q36_mtr_commit_pairs:development",
    "commit_apply": "q36_mtr_validate_commit_application",
    "precompute_custody": "q36_mtr_build_precompute_custody",
    "prescore_accounting": "q36_mtr_capture_accounting:prescore",
    "authorize_score": "q36_mtr_authorize_score",
    "score_once": "q36_mtr_score",
    "normalize": "q36_mtr_normalize",
    "final_accounting": "q36_mtr_capture_accounting:final",
    "evidence_mirror": "q36_mtr_mirror_evidence",
    "compute_custody": "q36_mtr_build_final_custody",
    "final_compare": "q36_mtr_compare_and_seal_terminal",
}


class Q36MTRPlanError(RuntimeError):
    """The Q36-MTR dry-run task plan differs."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_partition(stage: str, index: int, count: int) -> dict[str, Any] | None:
    if stage == "draft_generate":
        return {
            "population": 7_113,
            "row_start_formula": f"7113*{index}//{count}",
            "row_end_formula": f"7113*{index + 1}//{count}",
        }
    if stage.startswith("calibration_"):
        return {
            "population": 5_824,
            "row_start_formula": f"5824*{index}//{count}",
            "row_end_formula": f"5824*{index + 1}//{count}",
        }
    if stage.startswith("development_") and not stage.endswith("_merge"):
        return {
            "population": 1_289,
            "row_start_formula": f"1289*{index}//{count}",
            "row_end_formula": f"1289*{index + 1}//{count}",
        }
    return None


def compile_plan(graph: dict[str, Any], graph_sha256: str) -> dict[str, Any]:
    validate_graph(graph)
    if graph.get("schema") != GRAPH_SCHEMA:
        raise Q36MTRPlanError("Q36-MTR graph schema differs")
    gpu_tasks: list[dict[str, Any]] = []
    cpu_tasks: list[dict[str, Any]] = []
    for priority, stage in enumerate(STAGES, start=1):
        if stage.h100_per_task:
            entrypoint = GPU_ENTRYPOINTS.get(stage.name)
            if entrypoint is None:
                raise Q36MTRPlanError(f"missing Q36-MTR entrypoint: {stage.name}")
            per_task_hours = stage.expected_h100_hours / stage.tasks
            for index in range(stage.tasks):
                gpu_tasks.append(
                    {
                        "request_key": f"{stage.name}/{index:02d}",
                        "stage": stage.name,
                        "priority": priority,
                        "task_index": index,
                        "task_count": stage.tasks,
                        "entrypoint": entrypoint,
                        "dependencies": list(stage.dependencies),
                        "partition": "normal",
                        "gres": "gpu:nvidia_h100_pcie:1",
                        "h100s": 1,
                        "expected_h100_hours": per_task_hours,
                        "identity_partition": _identity_partition(
                            stage.name, index, stage.tasks
                        ),
                        "requeue": False,
                        "output_writers": 1,
                        "duplicate_submission_permitted": False,
                    }
                )
        else:
            entrypoint = CPU_ENTRYPOINTS.get(stage.name)
            if entrypoint is None:
                raise Q36MTRPlanError(f"missing Q36-MTR CPU entrypoint: {stage.name}")
            cpu_tasks.append(
                {
                    "request_key": stage.name,
                    "stage": stage.name,
                    "priority": priority,
                    "entrypoint": entrypoint,
                    "dependencies": list(stage.dependencies),
                    "h100s": 0,
                    "requeue": False,
                    "output_writers": 1,
                }
            )
    payload = {
        "schema": SCHEMA,
        "status": "dry_run_only",
        "source_commit": graph["source_commit"],
        "graph_sha256": graph_sha256,
        "scientific_submit_authorized": False,
        "submission_command_present": False,
        "model_acquisition_authorized": False,
        "data_materialization_authorized": False,
        "gpu_tasks": gpu_tasks,
        "cpu_tasks": cpu_tasks,
        "h100_requests": len(gpu_tasks),
        "expected_h100_hours": sum(task["expected_h100_hours"] for task in gpu_tasks),
        "maximum_concurrent_single_h100_requests": (
            MAXIMUM_CONCURRENT_SINGLE_H100_REQUESTS
        ),
        "one_output_per_identity": True,
        "no_duplicate": True,
        "no_orphan": True,
        "cancel_dead_dependencies_at_terminal": True,
        "temporary_shard_cleanup_after_verified_merge_and_mirror_only": True,
    }
    validate_plan(payload)
    return payload


def validate_plan(payload: dict[str, Any]) -> None:
    if payload.get("schema") != SCHEMA or payload.get("status") != "dry_run_only":
        raise Q36MTRPlanError("Q36-MTR plan schema/status differs")
    for field in (
        "scientific_submit_authorized",
        "submission_command_present",
        "model_acquisition_authorized",
        "data_materialization_authorized",
    ):
        if payload.get(field) is not False:
            raise Q36MTRPlanError(f"Q36-MTR plan authorization differs: {field}")
    gpu_tasks = payload.get("gpu_tasks")
    if not isinstance(gpu_tasks, list) or len(gpu_tasks) != 61:
        raise Q36MTRPlanError("Q36-MTR plan H100 task count differs")
    keys = [task.get("request_key") for task in gpu_tasks]
    if len(keys) != len(set(keys)):
        raise Q36MTRPlanError("Q36-MTR plan duplicates a GPU request")
    if any(
        task.get("h100s") != 1
        or task.get("partition") != "normal"
        or task.get("gres") != "gpu:nvidia_h100_pcie:1"
        or task.get("requeue") is not False
        or task.get("output_writers") != 1
        or task.get("duplicate_submission_permitted") is not False
        for task in gpu_tasks
    ):
        raise Q36MTRPlanError("Q36-MTR single-H100 request differs")
    cpu_tasks = payload.get("cpu_tasks")
    expected_cpu = [stage for stage in STAGES if stage.h100_per_task == 0]
    if (
        not isinstance(cpu_tasks, list)
        or len(cpu_tasks) != len(expected_cpu)
        or [task.get("request_key") for task in cpu_tasks]
        != [stage.name for stage in expected_cpu]
        or any(
            task.get("entrypoint") != CPU_ENTRYPOINTS.get(task.get("stage"))
            or task.get("h100s") != 0
            or task.get("requeue") is not False
            or task.get("output_writers") != 1
            for task in cpu_tasks
        )
    ):
        raise Q36MTRPlanError("Q36-MTR CPU entrypoint plan differs")
    expected_hours = sum(stage.expected_h100_hours for stage in STAGES)
    if abs(float(payload.get("expected_h100_hours", -1)) - expected_hours) > 1e-12:
        raise Q36MTRPlanError("Q36-MTR task-hour projection differs")
    if payload.get("maximum_concurrent_single_h100_requests") != 32:
        raise Q36MTRPlanError("Q36-MTR maximum concurrency differs")
    for field in (
        "one_output_per_identity",
        "no_duplicate",
        "no_orphan",
        "cancel_dead_dependencies_at_terminal",
        "temporary_shard_cleanup_after_verified_merge_and_mirror_only",
    ):
        if payload.get(field) is not True:
            raise Q36MTRPlanError(f"Q36-MTR plan safety differs: {field}")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTRPlanError(f"refusing existing Q36-MTR plan: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--graph", type=Path)
    source.add_argument("--source-commit")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.graph is not None:
        graph = json.loads(args.graph.read_text(encoding="utf-8"))
        graph_sha256 = sha256_file(args.graph)
    else:
        graph = graph_payload(args.source_commit)
        encoded = (json.dumps(graph, indent=2, sort_keys=True) + "\n").encode()
        graph_sha256 = hashlib.sha256(encoded).hexdigest()
    payload = compile_plan(graph, graph_sha256)
    _atomic_json(args.output, payload)
    print(json.dumps({"h100_requests": 61, "status": "dry_run_only"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
