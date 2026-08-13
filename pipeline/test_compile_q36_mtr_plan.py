from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from compile_q36_mtr_plan import (
    Q36MTRPlanError,
    compile_plan,
    main,
    validate_plan,
)
from q36_mtr_contract import graph_payload

COMMIT = "1" * 40
GRAPH_SHA = "2" * 64


def test_dry_run_plan_has_exact_single_h100_fanout() -> None:
    plan = compile_plan(graph_payload(COMMIT), GRAPH_SHA)
    assert plan["status"] == "dry_run_only"
    assert plan["scientific_submit_authorized"] is False
    assert plan["submission_command_present"] is False
    assert plan["h100_requests"] == 61
    assert plan["expected_h100_hours"] == pytest.approx(58.9)
    assert plan["maximum_concurrent_single_h100_requests"] == 32
    assert len(plan["gpu_tasks"]) == 61
    assert len({task["request_key"] for task in plan["gpu_tasks"]}) == 61
    assert sum(task["stage"] == "draft_generate" for task in plan["gpu_tasks"]) == 16
    assert (
        sum(task["stage"].startswith("development_") for task in plan["gpu_tasks"])
        == 32
    )
    assert all(
        task["h100s"] == task["output_writers"] == 1 for task in plan["gpu_tasks"]
    )
    assert all(task["requeue"] is False for task in plan["gpu_tasks"])
    assert all(
        task["duplicate_submission_permitted"] is False for task in plan["gpu_tasks"]
    )
    draft_partitions = [
        task["identity_partition"]
        for task in plan["gpu_tasks"]
        if task["stage"] == "draft_generate"
    ]
    assert draft_partitions[0]["row_start"] == 0
    assert draft_partitions[-1]["row_end"] == 7_113
    assert sum(row["identity_count"] for row in draft_partitions) == 7_113
    assert {task["stage"]: task["entrypoint"] for task in plan["cpu_tasks"]}[
        "commit_apply"
    ] == "q36_mtr_validate_commit_application"
    assert {task["stage"]: task["entrypoint"] for task in plan["cpu_tasks"]}[
        "final_compare"
    ] == "q36_mtr_compare_and_seal_terminal"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__("scientific_submit_authorized", True),
        lambda value: value.__setitem__("submission_command_present", True),
        lambda value: value["gpu_tasks"].pop(),
        lambda value: value["gpu_tasks"][0].__setitem__("h100s", 2),
        lambda value: value["gpu_tasks"][0].__setitem__("requeue", True),
        lambda value: value["gpu_tasks"][0].__setitem__("dependencies", []),
        lambda value: value["gpu_tasks"][3]["identity_partition"].__setitem__(
            "row_start", 0
        ),
        lambda value: value["cpu_tasks"][-1].__setitem__("dependencies", []),
        lambda value: value.__setitem__("no_duplicate", False),
        lambda value: value.__setitem__("maximum_concurrent_single_h100_requests", 31),
    ],
)
def test_plan_mutations_fail_closed(mutation) -> None:
    plan = copy.deepcopy(compile_plan(graph_payload(COMMIT), GRAPH_SHA))
    mutation(plan)
    with pytest.raises(Q36MTRPlanError):
        validate_plan(plan)


def test_cli_has_no_submit_mode_and_writes_once(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "plan.json"
    monkeypatch.setattr(
        "sys.argv",
        ["compile_q36_mtr_plan.py", "--source-commit", COMMIT, "--output", str(output)],
    )
    assert main() == 0
    validate_plan(json.loads(output.read_text(encoding="utf-8")))
    with pytest.raises(Q36MTRPlanError):
        main()
