from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re

import pytest
import torch

from ettr_factorial_custody import ETTRFactorialExecutionManifest
from ettr_factorial_qualification_board import (
    TOTAL_PACKETS,
    build_ettr_factorial_qualification_board,
)
from prepare_ettr_supervisor_smoke import (
    CHECKPOINT_STEP,
    ETTRSupervisorSmokeError,
    FIXTURE_SCHEMA,
    MODEL_SEED,
    PLAN_SCHEMA,
    STAGES,
    SmokeRuntimeBindings,
    _sha256_file,
    _sealed_memfd,
    prepare_fixture,
    supervisor_command,
    validate_plan,
)


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "pipeline/jobs/smoke_ettr_stage_supervisor.sbatch"
SCRIPT = ROOT / "pipeline/prepare_ettr_supervisor_smoke.py"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _immutable(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    path.chmod(0o444)
    return path.resolve()


def _runtime(tmp_path: Path) -> SmokeRuntimeBindings:
    archive = _immutable(tmp_path / "runtime.tar", b"synthetic-runtime\n")
    claim_receipt = _immutable(
        tmp_path / "claim-receipt.json",
        b'{"schema":"synthetic-runtime-receipt"}\n',
    )
    runtime_receipts = tuple(
        (
            stage,
            _immutable(
                tmp_path / f"runtime-{stage}.json",
                (
                    '{"schema":"synthetic-stage-runtime",'
                    f'"stage":"{stage}"'
                    "}\n"
                ).encode("ascii"),
            ),
        )
        for stage in STAGES
    )
    bwrap = _immutable(tmp_path / "bwrap", b"synthetic-bwrap\n")
    return SmokeRuntimeBindings(
        runtime_archive_path=archive,
        claim_runtime_verification_receipt_path=claim_receipt,
        runtime_receipt_paths=runtime_receipts,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        archive_size=archive.stat().st_size,
        inventory_sha256="1" * 64,
        source_bundle_sha256="2" * 64,
        python_sha256="3" * 64,
        bootstrap_sha256="4" * 64,
        external_launcher_sha256="5" * 64,
        bwrap_path=bwrap,
        bwrap_sha256=hashlib.sha256(bwrap.read_bytes()).hexdigest(),
        runtime_bundle_sha256s=tuple(
            (stage, str(index) * 64)
            for index, stage in enumerate(STAGES, start=6)
        ),
        runner_sha256s=tuple(
            (stage, character * 64)
            for stage, character in zip(
                STAGES,
                ("9", "a", "b"),
                strict=True,
            )
        ),
    )


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted((root / "inputs").iterdir())
    }


def test_fixture_is_deterministic_untrained_and_uses_production_contracts(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    left = tmp_path / "left"
    right = tmp_path / "right"
    left_report = prepare_fixture(
        source_root=ROOT,
        output_root=left,
        runtime=runtime,
    )
    prepare_fixture(
        source_root=ROOT,
        output_root=right,
        runtime=runtime,
    )
    assert left_report["fixture_schema"] == FIXTURE_SCHEMA
    assert left_report["checkpoint_step"] == CHECKPOINT_STEP
    assert left_report["training_assets_read"] is False
    assert _file_hashes(left) == _file_hashes(right)

    plan = json.loads((left / "fixture-plan.json").read_text())
    validate_plan(plan)
    assert plan["schema"] == PLAN_SCHEMA
    assert plan["model_seed"] == MODEL_SEED
    assert plan["checkpoint_kind"] == "deterministic-synthetic-untrained"
    assert plan["training_assets_read"] is False
    checkpoint = torch.load(
        left / "inputs/synthetic-base.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert set(checkpoint) == {"cfg", "model", "step"}
    assert checkpoint["step"] == 0

    manifest = ETTRFactorialExecutionManifest.from_path(
        left / "inputs/execution-manifest.json"
    )
    board = build_ettr_factorial_qualification_board()
    manifest.validate(
        board,
        expected_model_sha256=manifest.model_sha256,
        expected_manifest_sha256=manifest.sha256(),
    )
    assert manifest.row_count == TOTAL_PACKETS
    assert manifest.checkpoint_step == 0
    assert manifest.executor_steps == 2


def test_plan_is_closed_and_rejects_any_inventory_widening(
    tmp_path: Path,
) -> None:
    output = tmp_path / "fixture"
    prepare_fixture(
        source_root=ROOT,
        output_root=output,
        runtime=_runtime(tmp_path),
    )
    plan = json.loads((output / "fixture-plan.json").read_text())
    validate_plan(plan)
    hostile = {**plan, "extra_candidate_argument": "--read-host"}
    with pytest.raises(ETTRSupervisorSmokeError, match="plan differs"):
        validate_plan(hostile)
    widened_inputs = {
        **plan,
        "input_paths": {
            **plan["input_paths"],
            "assessor_private_key": "/tmp/key",
        },
    }
    with pytest.raises(ETTRSupervisorSmokeError, match="plan differs"):
        validate_plan(widened_inputs)


@pytest.mark.parametrize(
    ("stage", "expected_roles"),
    [
        (
            "world",
            {
                "checkpoint",
                "compiler_weights",
                "configuration",
                "execution_manifest",
                "world_tokens",
            },
        ),
        (
            "command",
            {
                "checkpoint",
                "command_tokens",
                "compiled_state",
                "compiler_receipt",
                "configuration",
                "execution_manifest",
                "reactor_weights",
            },
        ),
        (
            "query",
            {
                "checkpoint",
                "configuration",
                "execution_manifest",
                "executor_receipt",
                "query_reader_weights",
                "query_tokens",
                "terminal_state",
            },
        ),
    ],
)
def test_supervisor_command_is_exact_and_parent_linked(
    tmp_path: Path,
    stage: str,
    expected_roles: set[str],
) -> None:
    output = tmp_path / "fixture"
    prepare_fixture(
        source_root=ROOT,
        output_root=output,
        runtime=_runtime(tmp_path),
    )
    plan = json.loads((output / "fixture-plan.json").read_text())
    parent = None if stage == "world" else "c" * 64
    command = supervisor_command(
        host_python=Path("/usr/bin/python3.11"),
        runtime_root="/proc/self/fd/7/runtime",
        plan=plan,
        stage=stage,
        run_root=Path(plan["run_root"]),
        key_descriptor=9,
        allocated_gpu_minor=2,
        run_id="d" * 64,
        parent_receipt_sha256=parent,
    )
    assert command[:5] == [
        "/usr/bin/python3.11",
        "-I",
        "-S",
        "-B",
        "-c",
    ]
    assert "--runtime-root" in command
    assert "/proc/self/fd/7/runtime" in command
    assert "--verifier-private-key-fd" in command
    assert command[command.index("--verifier-private-key-fd") + 1] == "9"
    assert (
        "--parent-launch-receipt-sha256" in command
    ) is (parent is not None)
    role_values = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--input"
    ]
    assert {value.split("=", 1)[0] for value in role_values} == expected_roles
    rendered = "\0".join(command).lower()
    assert "assessor" not in rendered
    assert "private-key-path" not in rendered


@pytest.mark.skipif(
    not hasattr(os, "memfd_create"),
    reason="sealed memfd is a Linux launch primitive",
)
def test_launch_key_memfd_is_fully_sealed() -> None:
    import fcntl

    descriptor = _sealed_memfd("fixture-key", b"k" * 32)
    try:
        required = (
            fcntl.F_SEAL_SEAL
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_WRITE
        )
        assert fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & required == required
    finally:
        os.close(descriptor)


def test_root_owned_system_executable_can_be_pinned() -> None:
    executable = Path("/usr/bin/env")
    metadata = executable.stat()
    if (
        metadata.st_uid != 0
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o022
    ):
        pytest.skip("root-owned system executable is unavailable")
    digest, size = _sha256_file(
        executable,
        root_owned_executable=True,
    )
    assert _SHA256.fullmatch(digest)
    assert size == metadata.st_size


def test_job_is_one_gpu_bounded_and_runs_all_three_phases() -> None:
    job = JOB.read_text(encoding="ascii")
    script = SCRIPT.read_text(encoding="ascii")
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in job
    assert "#SBATCH -c 2" in job
    assert "#SBATCH --export=NONE" in job
    assert "#SBATCH -t 01:00:00" in job
    assert "#SBATCH --exclude=evc33,evc34,evc43,evc44" in job
    assert "EXPECTED_ASSESSOR_PYTHON_SHA256" in job
    assert "ASSESSOR_PYTHON" in job
    assert "cryptography.__version__" in job
    assert "46.0.3" in job
    assert "ettr-claim-runtime-v2.tar" in job
    assert " prepare \\" in job
    assert "run-chain \\" in job
    assert " validate \\" in job
    assert "sbatch " not in job
    assert "extract_verified_archive(" in script
    assert "remove_after_callback=True" in script
    assert "for stage in STAGES:" in script
    assert "_sealed_memfd(" in script
    assert "validate_stage_launch_receipt_chain(" in script
    assert "materialize_signed_ettr_factorial_qualification(" in script
    for forbidden in (
        "best_step300000",
        "ckpt_0300000",
        "flagship_out",
        "SHARDS=",
    ):
        assert forbidden not in job
        assert forbidden not in script


def test_runtime_binding_mutation_is_rejected(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    with pytest.raises(ETTRSupervisorSmokeError):
        replace(runtime, archive_size=0).validate()
    with pytest.raises(ETTRSupervisorSmokeError):
        replace(runtime, bwrap_sha256="not-a-hash").validate()
    for _, digest in runtime.runner_sha256s:
        assert _SHA256.fullmatch(digest)
