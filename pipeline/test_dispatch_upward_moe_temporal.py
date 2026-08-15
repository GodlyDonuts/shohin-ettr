from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import dispatch_upward_moe_temporal as module


def _args(tmp_path: Path):
    runtime = tmp_path / "runtime"
    model = tmp_path / "model"
    source_root = tmp_path / "sources"
    runtime.mkdir()
    model.mkdir()
    source_root.mkdir()
    files = {}
    for name in (
        "model_manifest",
        "mechanics_report",
        "b1",
        "train_source",
        "development_source",
        "freeze_report",
        "assessor_receipt",
        "assessors",
    ):
        path = tmp_path / name
        path.write_text(name + "\n", encoding="utf-8")
        files[name] = path
    args = SimpleNamespace(
        host="mixtral-8x22b",
        runtime=runtime,
        runtime_manifest_sha256="a" * 64,
        python=Path(sys.executable),
        model_root=model,
        model_manifest=files["model_manifest"],
        mechanics_report=files["mechanics_report"],
        expected_model_manifest_sha256="b" * 64,
        overlay_root=None,
        overlay_manifest=None,
        causal_conv_root=None,
        b1=files["b1"],
        source_root=source_root,
        train_source=files["train_source"],
        development_source=files["development_source"],
        freeze_report=files["freeze_report"],
        assessor_receipt=files["assessor_receipt"],
        assessors=files["assessors"],
        run_root=tmp_path / "run",
        receipt=tmp_path / "dispatch.json",
        submit=False,
    )
    graph = module.build_graph(args)
    for stage in graph:
        script = runtime / stage["script"]
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("#!/bin/bash\n", encoding="utf-8")
    return args, graph


def test_graph_prestages_exact_upward_dependencies_and_geometry(tmp_path: Path) -> None:
    args, graph = _args(tmp_path)
    module.validate(args, graph)
    by_name = {stage["name"]: stage for stage in graph}
    assert len(graph) == 12
    assert by_name["drafts"]["array"] == "0-15%16"
    assert by_name["aligned"]["dependencies"] == ["materialize"]
    assert by_name["aligned"]["exports"]["DATA_REPORT"].endswith("/data/report.json")
    assert by_name["evaluate_unchanged"]["dependencies"] == ["aligned"]
    assert by_name["evaluate_temporal_gate"]["dependencies"] == ["temporal"]
    assert by_name["score"]["dependencies"] == [
        f"evaluate_{arm}" for arm in module.ARMS
    ]
    receipt = module.submit(args, graph)
    assert receipt["status"] == "dry_run"
    assert receipt["allocation_tasks"] == 102
    assert receipt["two_h100_tasks"] == 99
    assert receipt["eight_h100_tasks"] == 0
    assert all("--export=NONE" not in command for command in receipt["commands"])
    assert not args.run_root.exists()


def test_ultra_graph_overrides_only_gpu_stages_to_eight_h100(tmp_path: Path) -> None:
    args, _ = _args(tmp_path)
    args.host = "nemotron-ultra"
    args.overlay_root = tmp_path / "overlay"
    args.overlay_root.mkdir()
    args.causal_conv_root = tmp_path / "causal"
    args.causal_conv_root.mkdir()
    args.overlay_manifest = tmp_path / "overlay.sha256"
    args.overlay_manifest.write_text("overlay\n", encoding="utf-8")
    graph = module.build_graph(args)
    module.validate(args, graph)
    receipt = module.submit(args, graph)
    assert receipt["two_h100_tasks"] == 0
    assert receipt["eight_h100_tasks"] == 99
    by_name = {stage["name"]: stage for stage in graph}
    assert by_name["owner"]["resources"] == [
        "--partition=highgpu",
        "--gres=gpu:nvidia_h100_80gb_hbm3:8",
        "--cpus-per-task=32",
        "--mem=512G",
        "--time=24:00:00",
    ]
    assert by_name["evaluate_temporal_gate"]["resources"][-1] == "--time=08:00:00"
    assert "resources" not in by_name["score"]
    gpu_commands = [
        command
        for command in receipt["commands"]
        if "--gres=gpu:nvidia_h100_80gb_hbm3:8" in command
    ]
    assert len(gpu_commands) == 9


def test_graph_refuses_existing_output_or_missing_runtime_member(
    tmp_path: Path,
) -> None:
    args, graph = _args(tmp_path)
    args.run_root.mkdir()
    with pytest.raises(module.UpwardMoETemporalDispatchError):
        module.validate(args, graph)
    args.run_root.rmdir()
    (args.runtime / graph[-1]["script"]).unlink()
    with pytest.raises(module.UpwardMoETemporalDispatchError):
        module.validate(args, graph)


def test_slurm_exports_reject_delimiter_injection() -> None:
    with pytest.raises(module.UpwardMoETemporalDispatchError):
        module._exports({"HOST": "mixtral,ALL"})


def test_partial_submission_cancels_all_predecessors(
    tmp_path: Path, monkeypatch
) -> None:
    args, graph = _args(tmp_path)
    args.submit = True
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if command[0] == "scancel":
            return subprocess.CompletedProcess(command, 0, "", "")
        if len([call for call in calls if call[0] == "sbatch"]) == 1:
            return subprocess.CompletedProcess(command, 0, "12345\n", "")
        raise subprocess.CalledProcessError(1, command, stderr="denied")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    with pytest.raises(module.UpwardMoETemporalDispatchError):
        module.submit(args, graph)
    assert ["scancel", "12345"] in calls
