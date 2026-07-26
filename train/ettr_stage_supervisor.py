#!/usr/bin/env python3
"""Verifier-owned Bubblewrap supervisor for one ETTR qualification stage."""

from __future__ import annotations

import argparse
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import select
import stat
import subprocess
import sys
import tempfile
from typing import Mapping, Sequence

from ettr_deployment_contract import (
    ETTRDeploymentContractError,
    ETTRRuntimeImageIdentity,
    ETTRStageLaunchReceipt,
    ETTRStagePolicySpec,
    STAGE_LAUNCH_RECEIPT_SCHEMA,
    canonical_loaded_object_map_sha256,
    canonical_stage_environment,
    canonical_stage_environment_sha256,
    ed25519_public_key_from_private_bytes,
    ed25519_sign,
)


EXECUTION_MANIFEST_SCHEMA = "ettr-factorial-execution-manifest-v4"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DESCRIPTOR_RUNTIME_ROOT = re.compile(r"^/proc/self/fd/[0-9]+/runtime$")
_STAGE_POLICY_FIELDS = {
    "world": "world_stage_policy_sha256",
    "command": "command_stage_policy_sha256",
    "query": "query_stage_policy_sha256",
}
_RUNTIME_BUNDLE_FIELDS = {
    "world": "world_runtime_bundle_sha256",
    "command": "command_runtime_bundle_sha256",
    "query": "query_runtime_bundle_sha256",
}
_DIRECT_INPUT_DESTINATIONS: dict[str, str] = {
    "checkpoint": "/inputs/checkpoint.pt",
    "command_tokens": "/inputs/command.json",
    "compiled_state": "/inputs/compiled-state.safetensors",
    "compiler_receipt": "/inputs/compiler-receipt.json",
    "compiler_weights": "/inputs/compiler.safetensors",
    "configuration": "/inputs/config.json",
    "execution_manifest": "/inputs/execution-manifest.json",
    "executor_receipt": "/inputs/executor-receipt.json",
    "query_reader_weights": "/inputs/query-reader.safetensors",
    "query_tokens": "/inputs/query.json",
    "reactor_weights": "/inputs/reactor.safetensors",
    "runtime_bundle_receipt": "/inputs/runtime-receipt.json",
    "terminal_state": "/inputs/terminal-state.safetensors",
    "world_tokens": "/inputs/world.json",
}
_OUTPUT_FILENAMES: dict[str, str] = {
    "answer_output": "answer.json",
    "compiled_state_output": "compiled-state.safetensors",
    "compiler_receipt_output": "compiler-receipt.json",
    "executor_receipt_output": "executor-receipt.json",
    "query_receipt_output": "query-receipt.json",
    "terminal_state_output": "terminal-state.safetensors",
}


class ETTRStageSupervisorError(RuntimeError):
    """A measured stage launch differed from the verifier-owned contract."""


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_immutable_file(path: Path, label: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ETTRStageSupervisorError(f"{label} cannot be opened") from exc
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o222
    ):
        os.close(descriptor)
        raise ETTRStageSupervisorError(
            f"{label} is not immutable single-link input"
        )
    return descriptor


def _open_immutable_file_at(
    directory_descriptor: int,
    name: str,
    label: str,
) -> int:
    if Path(name).name != name or name in {".", ".."}:
        raise ETTRStageSupervisorError(f"{label} name differs")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(
            name,
            flags,
            dir_fd=directory_descriptor,
        )
    except OSError as exc:
        raise ETTRStageSupervisorError(f"{label} cannot be opened") from exc
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o222
    ):
        os.close(descriptor)
        raise ETTRStageSupervisorError(
            f"{label} is not immutable single-link output"
        )
    return descriptor


def _open_immutable_directory(path: Path, label: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ETTRStageSupervisorError(f"{label} cannot be opened") from exc
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_mode & 0o222
    ):
        os.close(descriptor)
        raise ETTRStageSupervisorError(
            f"{label} is not immutable directory"
        )
    return descriptor


def _sha256_descriptor(descriptor: int) -> tuple[str, int]:
    before = os.fstat(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size = 0
    while chunk := os.read(descriptor, 8 * 1024 * 1024):
        digest.update(chunk)
        size += len(chunk)
    after = os.fstat(descriptor)
    if _identity(before) != _identity(after) or size != before.st_size:
        raise ETTRStageSupervisorError("measured input changed during read")
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest(), size


def _read_canonical_manifest(
    descriptor: int,
    *,
    expected_sha256: str,
) -> tuple[dict[str, object], bytes]:
    digest, size = _sha256_descriptor(descriptor)
    if digest != expected_sha256 or size > 1024 * 1024:
        raise ETTRStageSupervisorError("execution manifest identity differs")
    payload = os.read(descriptor, size)
    os.lseek(descriptor, 0, os.SEEK_SET)
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ETTRStageSupervisorError(
            "execution manifest is malformed"
        ) from exc
    if (
        not isinstance(value, dict)
        or payload != _canonical_json_bytes(value)
        or value.get("schema") != EXECUTION_MANIFEST_SCHEMA
    ):
        raise ETTRStageSupervisorError(
            "execution manifest geometry differs"
        )
    return value, payload


def _parse_role_paths(
    rows: Sequence[str],
    *,
    expected_roles: tuple[str, ...],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for row in rows:
        role, separator, value = row.partition("=")
        path = Path(value)
        if (
            separator != "="
            or role in paths
            or role not in expected_roles
            or not path.is_absolute()
        ):
            raise ETTRStageSupervisorError("input role declaration differs")
        paths[role] = path
    if tuple(sorted(paths)) != tuple(sorted(expected_roles)):
        raise ETTRStageSupervisorError("input role inventory differs")
    return paths


def _expected_direct_input_sha256(
    *,
    stage: str,
    role: str,
    manifest: Mapping[str, object],
    measured_sha256: str,
) -> str:
    fixed_fields = {
        "checkpoint": "checkpoint_sha256",
        "command_tokens": "command_tokens_sha256",
        "compiler_weights": "compiler_sha256",
        "configuration": "config_sha256",
        "execution_manifest": None,
        "query_reader_weights": "reader_sha256",
        "query_tokens": "query_tokens_sha256",
        "reactor_weights": "reactor_sha256",
        "runtime_bundle_receipt": _RUNTIME_BUNDLE_FIELDS[stage],
        "world_tokens": "world_tokens_sha256",
    }
    field = fixed_fields.get(role)
    if field is None:
        if role == "execution_manifest":
            return measured_sha256
        if role in {
            "compiled_state",
            "compiler_receipt",
            "executor_receipt",
            "terminal_state",
        }:
            return measured_sha256
        raise ETTRStageSupervisorError("input role hash contract differs")
    expected = manifest.get(field)
    if expected != measured_sha256:
        raise ETTRStageSupervisorError(f"{role} identity differs")
    return measured_sha256


def _runner_arguments(
    *,
    stage: str,
    manifest: Mapping[str, object],
    input_sha256s: Mapping[str, str],
) -> tuple[str, ...]:
    common = (
        "--config",
        "/inputs/config.json",
        "--checkpoint",
        "/inputs/checkpoint.pt",
        "--checkpoint-sha256",
        str(manifest["checkpoint_sha256"]),
        "--expected-step",
        str(manifest["checkpoint_step"]),
    )
    if stage == "world":
        return (
            *common,
            "--compiler",
            "/inputs/compiler.safetensors",
            "--world",
            "/inputs/world.json",
            "--output",
            "/outputs/compiled-state.safetensors",
            "--receipt-output",
            "/outputs/compiler-receipt.json",
            "--hard",
        )
    if stage == "command":
        return (
            *common,
            "--state",
            "/inputs/compiled-state.safetensors",
            "--reactor",
            "/inputs/reactor.safetensors",
            "--command",
            "/inputs/command.json",
            "--compiler-receipt",
            "/inputs/compiler-receipt.json",
            "--compiler-receipt-sha256",
            input_sha256s["compiler_receipt"],
            "--output",
            "/outputs/terminal-state.safetensors",
            "--receipt-output",
            "/outputs/executor-receipt.json",
            "--steps",
            str(manifest["executor_steps"]),
            "--hard",
        )
    return (
        *common,
        "--state",
        "/inputs/terminal-state.safetensors",
        "--reader",
        "/inputs/query-reader.safetensors",
        "--query",
        "/inputs/query.json",
        "--executor-receipt",
        "/inputs/executor-receipt.json",
        "--executor-receipt-sha256",
        input_sha256s["executor_receipt"],
        "--tokenization-receipt-sha256",
        str(manifest["tokenization_receipt_sha256"]),
        "--model-assembly-receipt-sha256",
        str(manifest["model_assembly_receipt_sha256"]),
        "--output",
        "/outputs/answer.json",
        "--receipt-output",
        "/outputs/query-receipt.json",
    )


def _network_namespace_identity(path: str) -> str:
    metadata = os.stat(path)
    return f"{metadata.st_dev}:{metadata.st_ino}"


def _read_status_line(
    descriptor: int,
    *,
    timeout_seconds: int,
) -> dict[str, object]:
    ready, _, _ = select.select(
        (descriptor,),
        (),
        (),
        timeout_seconds,
    )
    if not ready:
        raise ETTRStageSupervisorError("Bubblewrap status timed out")
    payload = bytearray()
    while len(payload) <= 64 * 1024:
        chunk = os.read(descriptor, 1)
        if not chunk:
            break
        payload.extend(chunk)
        if chunk == b"\n":
            break
    try:
        value = json.loads(bytes(payload).decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ETTRStageSupervisorError(
            "Bubblewrap status is malformed"
        ) from exc
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("child-pid"), int)
        or not isinstance(value.get("net-namespace"), int)
    ):
        raise ETTRStageSupervisorError("Bubblewrap status differs")
    return value


def _write_once(path: Path, payload: bytes) -> str:
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
    except FileExistsError as exc:
        raise ETTRStageSupervisorError(
            "launch receipt output already exists"
        ) from exc
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise ETTRStageSupervisorError(
                    "launch receipt write was short"
                )
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def _read_verifier_private_key(descriptor: int) -> bytes:
    before = os.fstat(descriptor)
    if before.st_size != 32:
        raise ETTRStageSupervisorError("verifier private key differs")
    os.lseek(descriptor, 0, os.SEEK_SET)
    private_key = os.read(descriptor, 33)
    after = os.fstat(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    if len(private_key) != 32 or _identity(before) != _identity(after):
        raise ETTRStageSupervisorError("verifier private key differs")
    return private_key


def _duplicate_sealed_verifier_key(descriptor: int) -> int:
    """Retain a trusted-launcher memfd without exposing a key pathname."""

    if (
        isinstance(descriptor, bool)
        or not isinstance(descriptor, int)
        or descriptor <= 2
    ):
        raise ETTRStageSupervisorError("verifier key descriptor differs")
    try:
        retained = os.dup(descriptor)
        metadata = os.fstat(retained)
        seals = fcntl.fcntl(retained, fcntl.F_GET_SEALS)
    except OSError as exc:
        raise ETTRStageSupervisorError(
            "verifier key descriptor differs"
        ) from exc
    required_seals = (
        fcntl.F_SEAL_SEAL
        | fcntl.F_SEAL_SHRINK
        | fcntl.F_SEAL_GROW
        | fcntl.F_SEAL_WRITE
    )
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size != 32
        or seals & required_seals != required_seals
    ):
        os.close(retained)
        raise ETTRStageSupervisorError("verifier key descriptor differs")
    return retained


def _sign_launch_receipt(
    receipt: ETTRStageLaunchReceipt,
    *,
    verifier_private_key_descriptor: int,
) -> tuple[ETTRStageLaunchReceipt, bytes]:
    """Sign only a complete unsigned receipt in the verifier process."""

    if receipt.verifier_signature_hex:
        raise ETTRStageSupervisorError("launch receipt was already signed")
    private_key = _read_verifier_private_key(
        verifier_private_key_descriptor
    )
    public_key = ed25519_public_key_from_private_bytes(private_key)
    fingerprint = hashlib.sha256(public_key).hexdigest()
    if fingerprint != receipt.verifier_public_key_sha256:
        raise ETTRStageSupervisorError(
            "verifier public key fingerprint differs"
        )
    signature = ed25519_sign(private_key, receipt.signing_bytes())
    return (
        replace(
            receipt,
            verifier_signature_hex=signature.hex(),
        ),
        public_key,
    )


def _validate_root_owned_executable(path: Path, expected_sha256: str) -> int:
    descriptor = _open_immutable_file(path, "Bubblewrap")
    metadata = os.fstat(descriptor)
    digest, _ = _sha256_descriptor(descriptor)
    if (
        metadata.st_uid != 0
        or not metadata.st_mode & 0o111
        or digest != expected_sha256
    ):
        os.close(descriptor)
        raise ETTRStageSupervisorError("Bubblewrap identity differs")
    return descriptor


def _build_bwrap_command(
    *,
    bwrap_path: Path,
    stage: str,
    manifest_sha256: str,
    runtime_identity: ETTRRuntimeImageIdentity,
    runtime_descriptor: int,
    application_descriptor: int,
    input_descriptors: Mapping[str, int],
    output_descriptor: int,
    status_descriptor: int,
    block_descriptor: int,
    allocated_gpu_minor: int,
    manifest: Mapping[str, object],
    input_sha256s: Mapping[str, str],
) -> list[str]:
    environment = canonical_stage_environment(
        stage=stage,
        runtime_identity=runtime_identity,
        allocated_gpu_minor=allocated_gpu_minor,
    )
    command = [
        str(bwrap_path),
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-net",
        "--json-status-fd",
        str(status_descriptor),
        "--block-fd",
        str(block_descriptor),
        "--ro-bind-fd",
        str(runtime_descriptor),
        "/runtime",
        "--dir",
        "/inputs",
        "--ro-bind-fd",
        str(application_descriptor),
        "/app",
        "--bind-fd",
        str(output_descriptor),
        "/outputs",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--chdir",
        "/outputs",
    ]
    for role, descriptor in input_descriptors.items():
        command.extend(
            (
                "--ro-bind-fd",
                str(descriptor),
                _DIRECT_INPUT_DESTINATIONS[role],
            )
        )
    for host_path in ("/lib64", "/usr/lib64", "/usr/lib", "/lib", "/sys"):
        if Path(host_path).exists():
            command.extend(("--ro-bind", host_path, host_path))
    if Path("/proc/driver/nvidia").is_dir():
        command.extend(
            ("--ro-bind", "/proc/driver/nvidia", "/proc/driver/nvidia")
        )
    for device in (
        "/dev/nvidiactl",
        "/dev/nvidia-modeset",
        "/dev/nvidia-uvm",
        "/dev/nvidia-uvm-tools",
        "/dev/nvidia-caps",
        f"/dev/nvidia{allocated_gpu_minor}",
    ):
        if Path(device).exists():
            command.extend(("--dev-bind", device, device))
    for name, value in environment:
        command.extend(("--setenv", name, value))
    command.extend(
        (
            "--",
            "/runtime/miniforge3/bin/python",
            "-I",
            "-S",
            "-B",
            "/runtime/app/tools/run_ettr_verified_stage.py",
            "--manifest",
            "/inputs/execution-manifest.json",
            "--manifest-sha256",
            manifest_sha256,
            "--runtime-receipt",
            "/inputs/runtime-receipt.json",
            "--bundle-root",
            "/app",
            "--stage",
            stage,
            "--",
            *_runner_arguments(
                stage=stage,
                manifest=manifest,
                input_sha256s=input_sha256s,
            ),
        )
    )
    return command


def supervise_stage(
    *,
    stage: str,
    manifest_path: Path,
    expected_manifest_sha256: str,
    runtime_root: Path,
    runtime_archive_path: Path,
    runtime_receipt_path: Path,
    direct_input_paths: Mapping[str, Path],
    output_root: Path,
    receipt_output_path: Path,
    bwrap_path: Path,
    allocated_gpu_minor: int,
    timeout_seconds: int,
    verifier_private_key_descriptor: int,
    run_id: str,
    parent_launch_receipt_sha256: str | None,
) -> ETTRStageLaunchReceipt:
    """Launch one exact stage and publish a verifier-owned receipt."""

    policy = ETTRStagePolicySpec.canonical(stage)
    policy.validate()
    if (
        _SHA256.fullmatch(expected_manifest_sha256) is None
        or isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds <= 0
        or not output_root.is_absolute()
        or not receipt_output_path.is_absolute()
        or receipt_output_path.is_relative_to(output_root)
        or _DESCRIPTOR_RUNTIME_ROOT.fullmatch(runtime_root.as_posix())
        is None
        or _SHA256.fullmatch(run_id) is None
        or (
            stage == "world"
            and parent_launch_receipt_sha256 is not None
        )
        or (
            stage != "world"
            and (
                not isinstance(parent_launch_receipt_sha256, str)
                or _SHA256.fullmatch(parent_launch_receipt_sha256) is None
            )
        )
    ):
        raise ETTRStageSupervisorError("supervisor launch input differs")
    required_direct_roles = tuple(
        role
        for role in policy.read_roles
        if role not in {"application_bundle", "runtime_image"}
    )
    if tuple(sorted(direct_input_paths)) != tuple(
        sorted(required_direct_roles)
    ):
        raise ETTRStageSupervisorError("direct input inventory differs")
    descriptors: list[int] = []
    try:
        manifest_descriptor = _open_immutable_file(
            manifest_path,
            "execution manifest",
        )
        descriptors.append(manifest_descriptor)
        retained_verifier_private_key_descriptor = (
            _duplicate_sealed_verifier_key(
                verifier_private_key_descriptor,
            )
        )
        descriptors.append(retained_verifier_private_key_descriptor)
        manifest, _ = _read_canonical_manifest(
            manifest_descriptor,
            expected_sha256=expected_manifest_sha256,
        )
        if (
            manifest.get(_STAGE_POLICY_FIELDS[stage]) != policy.sha256()
            or manifest.get("network_namespace_required") is not True
            or direct_input_paths["execution_manifest"].resolve()
            != manifest_path.resolve()
        ):
            raise ETTRStageSupervisorError("stage policy manifest differs")
        archive_descriptor = _open_immutable_file(
            runtime_archive_path,
            "runtime archive",
        )
        descriptors.append(archive_descriptor)
        archive_sha256, archive_size = _sha256_descriptor(archive_descriptor)
        if (
            archive_sha256 != manifest.get("claim_runtime_archive_sha256")
            or archive_size != manifest.get("claim_runtime_archive_size")
        ):
            raise ETTRStageSupervisorError("runtime archive differs")
        runtime_descriptor = _open_immutable_directory(
            runtime_root,
            "runtime root",
        )
        descriptors.append(runtime_descriptor)
        runtime_python = runtime_root / "miniforge3/bin/python"
        python_descriptor = _open_immutable_file(
            runtime_python,
            "runtime Python",
        )
        descriptors.append(python_descriptor)
        python_sha256, _ = _sha256_descriptor(python_descriptor)
        runtime_identity = ETTRRuntimeImageIdentity.from_manifest(
            manifest,
            python_sha256=python_sha256,
        )
        supervisor_sha256 = hashlib.sha256(
            Path(__file__).resolve().read_bytes()
        ).hexdigest()
        bootstrap_descriptor = _open_immutable_file(
            runtime_root / "app/tools/run_ettr_verified_stage.py",
            "verified bootstrap",
        )
        descriptors.append(bootstrap_descriptor)
        bootstrap_sha256, _ = _sha256_descriptor(bootstrap_descriptor)
        if (
            runtime_identity.external_launcher_sha256 != supervisor_sha256
            or runtime_identity.bootstrap_sha256 != bootstrap_sha256
        ):
            raise ETTRStageSupervisorError("launcher source identity differs")
        application_root = runtime_root / f"app/candidate/{stage}"
        application_descriptor = _open_immutable_directory(
            application_root,
            "application bundle",
        )
        descriptors.append(application_descriptor)
        bwrap_descriptor = _validate_root_owned_executable(
            bwrap_path,
            runtime_identity.bwrap_sha256,
        )
        descriptors.append(bwrap_descriptor)
        if not Path(f"/dev/nvidia{allocated_gpu_minor}").exists():
            raise ETTRStageSupervisorError("allocated GPU device is absent")

        input_descriptors: dict[str, int] = {}
        input_hashes: dict[str, str] = {
            "application_bundle": str(
                manifest[_RUNTIME_BUNDLE_FIELDS[stage]]
            ),
            "runtime_image": archive_sha256,
        }
        for role in required_direct_roles:
            path = (
                runtime_receipt_path
                if role == "runtime_bundle_receipt"
                else direct_input_paths[role]
            )
            descriptor = (
                os.dup(manifest_descriptor)
                if role == "execution_manifest"
                else _open_immutable_file(path, role)
            )
            descriptors.append(descriptor)
            measured, _ = _sha256_descriptor(descriptor)
            input_hashes[role] = _expected_direct_input_sha256(
                stage=stage,
                role=role,
                manifest=manifest,
                measured_sha256=measured,
            )
            input_descriptors[role] = descriptor
        input_role_sha256s = tuple(
            (role, input_hashes[role]) for role in policy.read_roles
        )

        try:
            output_root.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise ETTRStageSupervisorError(
                "stage output root already exists"
            ) from exc
        output_descriptor = os.open(
            output_root,
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        descriptors.append(output_descriptor)
        output_metadata_before = os.fstat(output_descriptor)
        status_read, status_write = os.pipe()
        block_read, block_write = os.pipe()
        descriptors.extend((status_read, status_write, block_read, block_write))
        parent_netns = _network_namespace_identity("/proc/self/ns/net")
        command = _build_bwrap_command(
            bwrap_path=bwrap_path,
            stage=stage,
            manifest_sha256=expected_manifest_sha256,
            runtime_identity=runtime_identity,
            runtime_descriptor=runtime_descriptor,
            application_descriptor=application_descriptor,
            input_descriptors=input_descriptors,
            output_descriptor=output_descriptor,
            status_descriptor=status_write,
            block_descriptor=block_read,
            allocated_gpu_minor=allocated_gpu_minor,
            manifest=manifest,
            input_sha256s=input_hashes,
        )
        environment = dict(
            canonical_stage_environment(
                stage=stage,
                runtime_identity=runtime_identity,
                allocated_gpu_minor=allocated_gpu_minor,
            )
        )
        with (
            tempfile.TemporaryFile() as stdout_file,
            tempfile.TemporaryFile() as stderr_file,
        ):
            process: subprocess.Popen[bytes] | None = None
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    env=environment,
                    pass_fds=(
                        runtime_descriptor,
                        application_descriptor,
                        output_descriptor,
                        status_write,
                        block_read,
                        *input_descriptors.values(),
                    ),
                    close_fds=True,
                )
                os.close(status_write)
                descriptors.remove(status_write)
                os.close(block_read)
                descriptors.remove(block_read)
                status = _read_status_line(
                    status_read,
                    timeout_seconds=min(timeout_seconds, 60),
                )
                child_pid = int(status["child-pid"])
                child_netns = _network_namespace_identity(
                    f"/proc/{child_pid}/ns/net"
                )
                if (
                    int(status["net-namespace"])
                    != int(child_netns.rsplit(":", 1)[1])
                    or child_netns == parent_netns
                ):
                    raise ETTRStageSupervisorError(
                        "child network namespace differs"
                    )
                os.write(block_write, b"1")
                os.close(block_write)
                descriptors.remove(block_write)
                try:
                    exit_code = process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired as exc:
                    raise ETTRStageSupervisorError(
                        "stage launch timed out"
                    ) from exc
                os.close(status_read)
                descriptors.remove(status_read)
                stdout_file.seek(0)
                stderr_file.seek(0)
                stdout_sha256 = hashlib.sha256(
                    stdout_file.read()
                ).hexdigest()
                stderr_sha256 = hashlib.sha256(
                    stderr_file.read()
                ).hexdigest()
            except BaseException:
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait()
                raise
        if exit_code != 0:
            raise ETTRStageSupervisorError(
                f"stage process failed with exit {exit_code}; "
                f"stderr SHA-256 {stderr_sha256}"
            )
        output_metadata_after = os.fstat(output_descriptor)
        if (
            not stat.S_ISDIR(output_metadata_after.st_mode)
            or (
                output_metadata_before.st_dev,
                output_metadata_before.st_ino,
                output_metadata_before.st_mode,
                output_metadata_before.st_nlink,
            )
            != (
                output_metadata_after.st_dev,
                output_metadata_after.st_ino,
                output_metadata_after.st_mode,
                output_metadata_after.st_nlink,
            )
        ):
            raise ETTRStageSupervisorError("stage output root differs")
        actual_names = tuple(sorted(os.listdir(output_descriptor)))
        expected_names = tuple(
            sorted(_OUTPUT_FILENAMES[role] for role in policy.write_roles)
        )
        if actual_names != expected_names:
            raise ETTRStageSupervisorError("stage output inventory differs")
        output_hashes: dict[str, str] = {}
        for role in policy.write_roles:
            descriptor = _open_immutable_file_at(
                output_descriptor,
                _OUTPUT_FILENAMES[role],
                role,
            )
            descriptors.append(descriptor)
            os.fchmod(descriptor, 0o444)
            output_hashes[role], _ = _sha256_descriptor(descriptor)
        output_role_sha256s = tuple(
            (role, output_hashes[role]) for role in policy.write_roles
        )
        verifier_private_key = _read_verifier_private_key(
            retained_verifier_private_key_descriptor
        )
        verifier_public_key = ed25519_public_key_from_private_bytes(
            verifier_private_key
        )
        unsigned_receipt = ETTRStageLaunchReceipt(
            schema=STAGE_LAUNCH_RECEIPT_SCHEMA,
            stage=stage,
            run_id=run_id,
            parent_launch_receipt_sha256=parent_launch_receipt_sha256,
            verifier_public_key_sha256=hashlib.sha256(
                verifier_public_key
            ).hexdigest(),
            execution_manifest_sha256=expected_manifest_sha256,
            runtime_identity_sha256=runtime_identity.sha256(),
            stage_policy_sha256=policy.sha256(),
            bwrap_sha256=runtime_identity.bwrap_sha256,
            input_role_sha256s=input_role_sha256s,
            output_role_sha256s=output_role_sha256s,
            parent_network_namespace=parent_netns,
            child_network_namespace=child_netns,
            allocated_gpu_minor=allocated_gpu_minor,
            exit_code=exit_code,
            stdout_sha256=stdout_sha256,
            stderr_sha256=stderr_sha256,
            environment_sha256=canonical_stage_environment_sha256(
                stage=stage,
                runtime_identity=runtime_identity,
                allocated_gpu_minor=allocated_gpu_minor,
            ),
            loaded_object_map_sha256=canonical_loaded_object_map_sha256(
                stage=stage,
                runtime_identity=runtime_identity,
                input_role_sha256s=input_role_sha256s,
                output_role_sha256s=output_role_sha256s,
            ),
            verifier_signature_hex="",
        )
        receipt, verifier_public_key = _sign_launch_receipt(
            unsigned_receipt,
            verifier_private_key_descriptor=(
                retained_verifier_private_key_descriptor
            ),
        )
        receipt.validate(
            runtime_identity=runtime_identity,
            policy=policy,
            expected_execution_manifest_sha256=expected_manifest_sha256,
            expected_verifier_public_key=verifier_public_key,
        )
        _write_once(receipt_output_path, receipt.canonical_bytes())
        return receipt
    except (ETTRDeploymentContractError, KeyError, OSError) as exc:
        if isinstance(exc, ETTRStageSupervisorError):
            raise
        raise ETTRStageSupervisorError("supervised stage failed closed") from exc
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _main(argv: Sequence[str] | None = None) -> int:
    raw_arguments = tuple(sys.argv[1:] if argv is None else argv)
    singleton_options = {
        "--stage",
        "--manifest",
        "--manifest-sha256",
        "--runtime-root",
        "--runtime-archive",
        "--runtime-receipt",
        "--output-root",
        "--launch-receipt-output",
        "--bwrap",
        "--allocated-gpu-minor",
        "--timeout-seconds",
        "--verifier-private-key-fd",
        "--run-id",
        "--parent-launch-receipt-sha256",
    }
    observed: dict[str, int] = {}
    for argument in raw_arguments:
        if not argument.startswith("--"):
            continue
        option = argument.split("=", 1)[0]
        if option != "--input" and option not in singleton_options:
            raise ETTRStageSupervisorError(
                "supervisor option inventory differs"
            )
        observed[option] = observed.get(option, 0) + 1
    required_options = singleton_options - {
        "--bwrap",
        "--timeout-seconds",
        "--parent-launch-receipt-sha256",
    }
    if (
        any(observed.get(option, 0) != 1 for option in required_options)
        or any(observed.get(option, 0) > 1 for option in singleton_options)
    ):
        raise ETTRStageSupervisorError("supervisor option inventory differs")
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--stage", choices=("world", "command", "query"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--runtime-archive", type=Path, required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--launch-receipt-output", type=Path, required=True)
    parser.add_argument("--bwrap", type=Path, default=Path("/usr/bin/bwrap"))
    parser.add_argument("--allocated-gpu-minor", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument(
        "--verifier-private-key-fd",
        type=int,
        required=True,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--parent-launch-receipt-sha256")
    arguments = parser.parse_args(raw_arguments)
    policy = ETTRStagePolicySpec.canonical(arguments.stage)
    direct_roles = tuple(
        role
        for role in policy.read_roles
        if role
        not in {
            "application_bundle",
            "runtime_image",
            "runtime_bundle_receipt",
        }
    )
    direct_paths = _parse_role_paths(
        arguments.input,
        expected_roles=direct_roles,
    )
    supervise_stage(
        stage=arguments.stage,
        manifest_path=arguments.manifest,
        expected_manifest_sha256=arguments.manifest_sha256,
        runtime_root=arguments.runtime_root,
        runtime_archive_path=arguments.runtime_archive,
        runtime_receipt_path=arguments.runtime_receipt,
        direct_input_paths={
            **direct_paths,
            "runtime_bundle_receipt": arguments.runtime_receipt,
        },
        output_root=arguments.output_root,
        receipt_output_path=arguments.launch_receipt_output,
        bwrap_path=arguments.bwrap,
        allocated_gpu_minor=arguments.allocated_gpu_minor,
        timeout_seconds=arguments.timeout_seconds,
        verifier_private_key_descriptor=(
            arguments.verifier_private_key_fd
        ),
        run_id=arguments.run_id,
        parent_launch_receipt_sha256=(
            arguments.parent_launch_receipt_sha256
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
