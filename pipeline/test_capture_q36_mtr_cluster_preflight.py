from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import pytest

from capture_q36_mtr_cluster_preflight import (
    Q36MTRClusterPreflightError,
    capture,
    sha256_file,
)
from compile_q36_mtr_plan import compile_plan
from q36_mtr_contract import graph_payload

COMMIT = "1" * 40


def _fixture(tmp_path: Path) -> argparse.Namespace:
    graph = tmp_path / "graph.json"
    graph.write_text(json.dumps(graph_payload(COMMIT)) + "\n", encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(compile_plan(graph_payload(COMMIT), sha256_file(graph))) + "\n",
        encoding="utf-8",
    )
    return argparse.Namespace(
        user="sa305415",
        filesystem="/lustre/fs1",
        accounting_start="2026-08-01",
        graph_contract=graph,
        plan=plan,
        output=tmp_path / "preflight.json",
    )


def _runner(overrides: dict[str, str] | None = None):
    values = {
        "squeue": "",
        "lfs": (
            "Disk quotas for usr sa305415\n"
            "Filesystem kbytes quota limit grace files quota limit grace\n"
            "/lustre/fs1 800000000 1059061760 1059061760 - 700000 1010000 1010000 -\n"
        ),
        "sinfo": (
            "evc20|idle|gpu:nvidia_h100_pcie:2\n" "evc26|idle|gpu:nvidia_h100_pcie:2\n"
        ),
        "sacct": (
            "700001|360000|billing=1,gres/gpu=1,gres/gpu:nvidia_h100_pcie=1|0|\n"
        ),
    }
    values.update(overrides or {})

    def run(command, **_kwargs):
        return subprocess.CompletedProcess(
            command, 0, stdout=values[command[0]], stderr=""
        )

    return run


def test_q36_cluster_preflight_proves_headroom_queue_and_budget(
    tmp_path: Path,
) -> None:
    report = capture(_fixture(tmp_path), _runner())
    assert report["status"] == "pass"
    assert report["queue_empty"] is True
    assert report["quota"]["free_bytes"] >= 128 * 1024**3
    assert report["quota"]["free_inodes"] >= 150_000
    assert report["eligible_h100_node_count"] == 1
    assert report["h100_hours_charged"] == 100.0
    assert report["h100_hours_remaining_after_plan"] == pytest.approx(1_841.1)


@pytest.mark.parametrize(
    "override",
    (
        {"squeue": "700002|RUNNING|normal|evc20\n"},
        {
            "lfs": (
                "/lustre/fs1 1050000000 1059061760 1059061760 - "
                "900000 1010000 1010000 -\n"
            )
        },
        {"sinfo": "evc26|idle|gpu:nvidia_h100_pcie:2\n"},
        {
            "sacct": (
                "700001|7000000|billing=1,gres/gpu=1,"
                "gres/gpu:nvidia_h100_pcie=1|0|\n"
            )
        },
    ),
)
def test_q36_cluster_preflight_fails_closed(
    tmp_path: Path, override: dict[str, str]
) -> None:
    with pytest.raises(Q36MTRClusterPreflightError):
        capture(_fixture(tmp_path), _runner(override))
