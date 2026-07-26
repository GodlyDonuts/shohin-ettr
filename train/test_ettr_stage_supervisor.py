from __future__ import annotations

import hashlib
import fcntl
import os
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)

from ettr_deployment_contract import (
    ETTRRuntimeImageIdentity,
    ETTRStageLaunchReceipt,
    ETTRStagePolicySpec,
    RUNTIME_IDENTITY_SCHEMA,
    STAGE_LAUNCH_RECEIPT_SCHEMA,
    canonical_loaded_object_map_sha256,
    canonical_stage_environment_sha256,
)
from ettr_stage_supervisor import (
    ETTRStageSupervisorError,
    _build_bwrap_command,
    _duplicate_sealed_verifier_key,
    _main,
    _open_immutable_file,
    _open_immutable_file_at,
    _parse_role_paths,
    _runner_arguments,
    _sha256_descriptor,
    _sign_launch_receipt,
)


def _identity() -> ETTRRuntimeImageIdentity:
    return ETTRRuntimeImageIdentity(
        schema=RUNTIME_IDENTITY_SCHEMA,
        archive_sha256="a" * 64,
        archive_size=123,
        inventory_sha256="b" * 64,
        world_runtime_bundle_sha256="c" * 64,
        command_runtime_bundle_sha256="d" * 64,
        query_runtime_bundle_sha256="e" * 64,
        python_sha256="f" * 64,
        bootstrap_sha256="1" * 64,
        external_launcher_sha256="2" * 64,
        bwrap_sha256="3" * 64,
        network_namespace_required=True,
    )


def _manifest() -> dict[str, object]:
    return {
        "checkpoint_sha256": "4" * 64,
        "checkpoint_step": 300_000,
        "executor_steps": 6,
        "model_assembly_receipt_sha256": "5" * 64,
        "tokenization_receipt_sha256": "6" * 64,
    }


def test_role_parser_requires_exact_absolute_inventory(
    tmp_path: Path,
) -> None:
    expected = ("checkpoint", "configuration")
    parsed = _parse_role_paths(
        [
            f"checkpoint={tmp_path / 'base.pt'}",
            f"configuration={tmp_path / 'config.json'}",
        ],
        expected_roles=expected,
    )
    assert tuple(sorted(parsed)) == expected
    for rows in (
        [f"checkpoint={tmp_path / 'base.pt'}"],
        [
            f"checkpoint={tmp_path / 'base.pt'}",
            f"configuration={tmp_path / 'config.json'}",
            f"assessor={tmp_path / 'secret'}",
        ],
        ["checkpoint=relative.pt", f"configuration={tmp_path / 'config'}"],
    ):
        with pytest.raises(ETTRStageSupervisorError):
            _parse_role_paths(rows, expected_roles=expected)


def test_cli_rejects_duplicate_and_abbreviated_singletons() -> None:
    common = [
        "--stage",
        "world",
        "--manifest",
        "/inputs/manifest.json",
        "--manifest-sha256",
        "a" * 64,
        "--runtime-root",
        "/proc/self/fd/9/runtime",
        "--runtime-archive",
        "/inputs/runtime.tar",
        "--runtime-receipt",
        "/inputs/runtime-receipt.json",
        "--output-root",
        "/outputs/stage",
        "--launch-receipt-output",
        "/receipts/launch.json",
        "--allocated-gpu-minor",
        "0",
        "--verifier-private-key-fd",
        "8",
        "--run-id",
        "b" * 64,
    ]
    with pytest.raises(
        ETTRStageSupervisorError,
        match="supervisor option inventory differs",
    ):
        _main([*common, "--runtime-root", "/tmp/hostile"])
    abbreviated = list(common)
    abbreviated[abbreviated.index("--runtime-root")] = "--runtime-ro"
    with pytest.raises(
        ETTRStageSupervisorError,
        match="supervisor option inventory differs",
    ):
        _main(abbreviated)


def test_immutable_input_is_measured_through_one_descriptor(
    tmp_path: Path,
) -> None:
    path = tmp_path / "input.bin"
    path.write_bytes(b"trusted-input\n")
    path.chmod(0o444)
    descriptor = _open_immutable_file(path, "fixture")
    try:
        digest, size = _sha256_descriptor(descriptor)
    finally:
        os.close(descriptor)
    assert digest == hashlib.sha256(b"trusted-input\n").hexdigest()
    assert size == len(b"trusted-input\n")

    path.chmod(0o644)
    with pytest.raises(ETTRStageSupervisorError):
        _open_immutable_file(path, "writable")
    path.chmod(0o444)
    hardlink = tmp_path / "hardlink.bin"
    hardlink.hardlink_to(path)
    with pytest.raises(ETTRStageSupervisorError):
        _open_immutable_file(path, "hard-linked")


@pytest.mark.skipif(
    not hasattr(os, "memfd_create") or not hasattr(fcntl, "F_ADD_SEALS"),
    reason="sealed memfd is a Linux verifier primitive",
)
def test_verifier_key_requires_fully_sealed_memfd() -> None:
    descriptor = os.memfd_create(
        "ettr-launch-key",
        flags=os.MFD_ALLOW_SEALING,
    )
    try:
        os.write(descriptor, b"k" * 32)
        with pytest.raises(ETTRStageSupervisorError):
            _duplicate_sealed_verifier_key(descriptor)
        fcntl.fcntl(
            descriptor,
            fcntl.F_ADD_SEALS,
            fcntl.F_SEAL_SEAL
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_WRITE,
        )
        retained = _duplicate_sealed_verifier_key(descriptor)
        try:
            assert os.read(retained, 32) == b"k" * 32
        finally:
            os.close(retained)
    finally:
        os.close(descriptor)


def test_output_path_substitution_cannot_change_descriptor_relative_hash(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "outputs"
    output_root.mkdir(mode=0o700)
    trusted = output_root / "answer.json"
    trusted.write_bytes(b"trusted-answer\n")
    trusted.chmod(0o444)
    root_descriptor = os.open(output_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        retained = tmp_path / "retained"
        output_root.rename(retained)
        output_root.mkdir(mode=0o700)
        hostile = output_root / "answer.json"
        hostile.write_bytes(b"hostile-answer\n")
        hostile.chmod(0o444)
        descriptor = _open_immutable_file_at(
            root_descriptor,
            "answer.json",
            "answer",
        )
        try:
            digest, _ = _sha256_descriptor(descriptor)
        finally:
            os.close(descriptor)
        assert digest == hashlib.sha256(b"trusted-answer\n").hexdigest()
    finally:
        os.close(root_descriptor)


@pytest.mark.parametrize("stage", ["world", "command", "query"])
def test_supervisor_builds_descriptor_bound_closed_bwrap_command(
    stage: str,
) -> None:
    policy = ETTRStagePolicySpec.canonical(stage)
    direct_roles = tuple(
        role
        for role in policy.read_roles
        if role not in {"application_bundle", "runtime_image"}
    )
    descriptors = {
        role: index
        for index, role in enumerate(direct_roles, start=20)
    }
    input_hashes = {role: "7" * 64 for role in policy.read_roles}
    input_hashes["compiler_receipt"] = "8" * 64
    input_hashes["executor_receipt"] = "9" * 64
    command = _build_bwrap_command(
        bwrap_path=Path("/usr/bin/bwrap"),
        stage=stage,
        manifest_sha256="a" * 64,
        runtime_identity=_identity(),
        runtime_descriptor=10,
        application_descriptor=11,
        input_descriptors=descriptors,
        output_descriptor=12,
        status_descriptor=13,
        block_descriptor=14,
        allocated_gpu_minor=2,
        manifest=_manifest(),
        input_sha256s=input_hashes,
    )
    rendered = "\0".join(command)
    assert command[0] == "/usr/bin/bwrap"
    for flag in (
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-net",
        "--json-status-fd",
        "--block-fd",
        "--ro-bind-fd",
        "--bind-fd",
    ):
        assert flag in command
    assert "/runtime/app/tools/run_ettr_verified_stage.py" in command
    assert "/Users/" not in rendered
    assert "authority" not in rendered
    assert "signing" not in rendered
    assert "PYTHONPATH" not in rendered
    cuda_index = command.index("CUDA_VISIBLE_DEVICES")
    assert command[cuda_index - 1] == "--setenv"
    assert command[cuda_index + 1] == "2"
    if stage == "command":
        assert "8" * 64 in command
    if stage == "query":
        assert "9" * 64 in command


def test_runner_arguments_are_derived_not_caller_supplied() -> None:
    command = _runner_arguments(
        stage="command",
        manifest=_manifest(),
        input_sha256s={"compiler_receipt": "a" * 64},
    )
    assert "--compiler-receipt-sha256" in command
    assert "a" * 64 in command
    assert "--steps" in command
    assert "6" in command
    assert "--hard" in command


def test_supervisor_signs_with_unmounted_immutable_verifier_key(
    tmp_path: Path,
) -> None:
    private_key_bytes = b"v" * 32
    private_key = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    key_path = tmp_path / "verifier.key"
    key_path.write_bytes(private_key_bytes)
    key_path.chmod(0o400)
    key_descriptor = _open_immutable_file(key_path, "verifier key")
    identity = _identity()
    policy = ETTRStagePolicySpec.canonical("world")
    input_roles = tuple(
        (role, "7" * 64) for role in policy.read_roles
    )
    output_roles = tuple(
        (role, "8" * 64) for role in policy.write_roles
    )
    unsigned = ETTRStageLaunchReceipt(
        schema=STAGE_LAUNCH_RECEIPT_SCHEMA,
        stage="world",
        run_id="9" * 64,
        parent_launch_receipt_sha256=None,
        verifier_public_key_sha256=hashlib.sha256(public_key).hexdigest(),
        execution_manifest_sha256="a" * 64,
        runtime_identity_sha256=identity.sha256(),
        stage_policy_sha256=policy.sha256(),
        bwrap_sha256=identity.bwrap_sha256,
        input_role_sha256s=input_roles,
        output_role_sha256s=output_roles,
        parent_network_namespace="1:2",
        child_network_namespace="1:3",
        allocated_gpu_minor=0,
        exit_code=0,
        stdout_sha256="4" * 64,
        stderr_sha256="5" * 64,
        environment_sha256=canonical_stage_environment_sha256(
            stage="world",
            runtime_identity=identity,
            allocated_gpu_minor=0,
        ),
        loaded_object_map_sha256=canonical_loaded_object_map_sha256(
            stage="world",
            runtime_identity=identity,
            input_role_sha256s=input_roles,
            output_role_sha256s=output_roles,
        ),
        verifier_signature_hex="",
    )
    try:
        receipt, actual_public_key = _sign_launch_receipt(
            unsigned,
            verifier_private_key_descriptor=key_descriptor,
        )
        with pytest.raises(
            ETTRStageSupervisorError,
            match="already signed",
        ):
            _sign_launch_receipt(
                receipt,
                verifier_private_key_descriptor=key_descriptor,
            )
    finally:
        os.close(key_descriptor)
    assert actual_public_key == public_key
    receipt.validate(
        runtime_identity=identity,
        policy=policy,
        expected_execution_manifest_sha256="a" * 64,
        expected_verifier_public_key=public_key,
    )

    command = _build_bwrap_command(
        bwrap_path=Path("/usr/bin/bwrap"),
        stage="world",
        manifest_sha256="a" * 64,
        runtime_identity=identity,
        runtime_descriptor=10,
        application_descriptor=11,
        input_descriptors={
            role: index
            for index, role in enumerate(
                (
                    role
                    for role in policy.read_roles
                    if role not in {"application_bundle", "runtime_image"}
                ),
                start=20,
            )
        },
        output_descriptor=12,
        status_descriptor=13,
        block_descriptor=14,
        allocated_gpu_minor=0,
        manifest=_manifest(),
        input_sha256s={role: "7" * 64 for role in policy.read_roles},
    )
    assert str(key_path) not in "\0".join(command)


def test_supervisor_rejects_wrong_key_size(
    tmp_path: Path,
) -> None:
    key_path = tmp_path / "bad.key"
    key_path.write_bytes(b"x" * 31)
    key_path.chmod(0o400)
    descriptor = _open_immutable_file(key_path, "bad verifier key")
    try:
        with pytest.raises(ETTRStageSupervisorError):
            from ettr_stage_supervisor import _read_verifier_private_key

            _read_verifier_private_key(descriptor)
    finally:
        os.close(descriptor)
