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
    "preflight_cpu": "q36_mtr_live_preflight",
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


def _hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_partition(stage: str, index: int, count: int) -> dict[str, Any] | None:
    population: int | None = None
    if stage == "draft_generate":
        population = 7_113
    elif stage.startswith("calibration_"):
        population = 5_824
    elif stage.startswith("development_") and not stage.endswith("_merge"):
        population = 1_289
    if population is None:
        return None
    row_start = population * index // count
    row_end = population * (index + 1) // count
    return {
        "population": population,
        "row_start": row_start,
        "row_end": row_end,
        "identity_count": row_end - row_start,
        "row_start_formula": f"{population}*{index}//{count}",
        "row_end_formula": f"{population}*{index + 1}//{count}",
    }


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
                    "duplicate_submission_permitted": False,
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
    if not _hex(payload.get("source_commit"), 40) or not _hex(
        payload.get("graph_sha256"), 64
    ):
        raise Q36MTRPlanError("Q36-MTR plan source binding differs")
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
    expected_gpu: dict[str, dict[str, Any]] = {}
    for priority, stage in enumerate(STAGES, start=1):
        if not stage.h100_per_task:
            continue
        for index in range(stage.tasks):
            key = f"{stage.name}/{index:02d}"
            expected_gpu[key] = {
                "stage": stage.name,
                "priority": priority,
                "task_index": index,
                "task_count": stage.tasks,
                "entrypoint": GPU_ENTRYPOINTS[stage.name],
                "dependencies": list(stage.dependencies),
                "identity_partition": _identity_partition(
                    stage.name, index, stage.tasks
                ),
                "expected_h100_hours": stage.expected_h100_hours / stage.tasks,
            }
    if set(keys) != set(expected_gpu):
        raise Q36MTRPlanError("Q36-MTR GPU request identity differs")
    for task in gpu_tasks:
        expected = expected_gpu[task["request_key"]]
        if {field: task.get(field) for field in expected} != expected:
            raise Q36MTRPlanError("Q36-MTR GPU dependency or partition differs")
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
            or task.get("duplicate_submission_permitted") is not False
            for task in cpu_tasks
        )
    ):
        raise Q36MTRPlanError("Q36-MTR CPU entrypoint plan differs")
    for task, stage in zip(cpu_tasks, expected_cpu, strict=True):
        stage_priority = next(
            index for index, value in enumerate(STAGES, start=1) if value == stage
        )
        if task.get("priority") != stage_priority or task.get("dependencies") != list(
            stage.dependencies
        ):
            raise Q36MTRPlanError("Q36-MTR CPU dependency plan differs")
    partitioned = {
        stage: sorted(
            (
                task["identity_partition"]
                for task in gpu_tasks
                if task["stage"] == stage
            ),
            key=lambda value: value["row_start"],
        )
        for stage in {
            task["stage"]
            for task in gpu_tasks
            if task["identity_partition"] is not None
        }
    }
    for partitions in partitioned.values():
        if (
            not partitions
            or partitions[0]["row_start"] != 0
            or partitions[-1]["row_end"] != partitions[0]["population"]
            or any(
                row["population"] != partitions[0]["population"]
                or row["identity_count"] != row["row_end"] - row["row_start"]
                or row["identity_count"] <= 0
                or (index and partitions[index - 1]["row_end"] != row["row_start"])
                for index, row in enumerate(partitions)
            )
        ):
            raise Q36MTRPlanError("Q36-MTR identity coverage differs")
    dependencies = {stage.name: set(stage.dependencies) for stage in STAGES}
    reachable: set[str] = set()
    pending = ["final_compare"]
    while pending:
        stage = pending.pop()
        if stage in reachable:
            continue
        reachable.add(stage)
        pending.extend(dependencies[stage])
    if reachable != set(dependencies):
        raise Q36MTRPlanError("Q36-MTR graph contains orphan work")
    expected_hours = sum(stage.expected_h100_hours for stage in STAGES)
    if (
        payload.get("h100_requests") != 61
        or isinstance(payload.get("expected_h100_hours"), bool)
        or not isinstance(payload.get("expected_h100_hours"), (int, float))
        or abs(float(payload["expected_h100_hours"]) - expected_hours) > 1e-12
    ):
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
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as error:
        raise Q36MTRPlanError(f"refusing existing Q36-MTR plan: {path}") from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


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
