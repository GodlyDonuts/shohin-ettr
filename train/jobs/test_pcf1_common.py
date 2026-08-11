"""Regression tests for fail-closed PCF1 shell helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

from pcf1_code_sandbox import (
    BWRAP,
    BWRAP_SHA256,
    CANDIDATE_FAILURE_EXIT_CODE,
    CANDIDATE_POLICY_SHA256,
    CANDIDATE_RANDOM_SEED,
    ELF_CLOSURE_AUDIT_SHA256,
    EXPECTED_SANDBOX_RUNTIME_DESCRIPTOR,
    EXPECTED_SANDBOX_RUNTIME_DESCRIPTOR_SHA256,
    MEMFD_ABI,
    POLICY_REJECTION_EXIT_CODE,
    RESOURCE_LIMIT_EXIT_CODE,
    PYTHON_EXECUTABLE,
    PYTHON_SHA256,
    SANDBOX_CONFIG_SHA256,
    SANDBOX_PROBES,
    SANDBOX_RUNTIME_TREE_BYTES,
    SANDBOX_RUNTIME_TREE_DIRECTORIES,
    SANDBOX_RUNTIME_TREE_ENTRIES,
    SANDBOX_RUNTIME_TREE_FILES,
    SANDBOX_RUNTIME_TREE_SHA256,
    SETUP_FAILURE_EXIT_CODE,
    INFRASTRUCTURE_FAILURE_EXIT_CODE,
    TRUSTED_COMPLETION_EXIT_CODE,
    TEST_FAILURE_EXIT_CODE,
    expected_system_library_members,
)


def test_sandbox_receipt_validator_imports_and_checks_frozen_contract(
    tmp_path: Path,
) -> None:
    probes = {name: True for name in SANDBOX_PROBES}
    probe_sha256 = hashlib.sha256(
        json.dumps(probes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    receipt = tmp_path / "sandbox_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "shohin-pcf1-code-sandbox-receipt-v1",
                "status": "pass",
                "bwrap_path": str(BWRAP),
                "bwrap_sha256": BWRAP_SHA256,
                "bwrap_version": "bubblewrap 0.4.0",
                "python_executable": str(PYTHON_EXECUTABLE),
                "python_sha256": PYTHON_SHA256,
                "sandbox_config_sha256": SANDBOX_CONFIG_SHA256,
                "candidate_policy_sha256": CANDIDATE_POLICY_SHA256,
                "trusted_completion_exit_code": TRUSTED_COMPLETION_EXIT_CODE,
                "candidate_failure_exit_code": CANDIDATE_FAILURE_EXIT_CODE,
                "infrastructure_failure_exit_code": INFRASTRUCTURE_FAILURE_EXIT_CODE,
                "test_failure_exit_code": TEST_FAILURE_EXIT_CODE,
                "setup_failure_exit_code": SETUP_FAILURE_EXIT_CODE,
                "policy_rejection_exit_code": POLICY_REJECTION_EXIT_CODE,
                "resource_limit_exit_code": RESOURCE_LIMIT_EXIT_CODE,
                "candidate_random_seed": CANDIDATE_RANDOM_SEED,
                "python_runtime_descriptor": EXPECTED_SANDBOX_RUNTIME_DESCRIPTOR,
                "python_runtime_descriptor_sha256": (
                    EXPECTED_SANDBOX_RUNTIME_DESCRIPTOR_SHA256
                ),
                "memfd_abi": MEMFD_ABI,
                "sandbox_runtime_tree_sha256": SANDBOX_RUNTIME_TREE_SHA256,
                "sandbox_runtime_tree_entries": SANDBOX_RUNTIME_TREE_ENTRIES,
                "sandbox_runtime_tree_files": SANDBOX_RUNTIME_TREE_FILES,
                "sandbox_runtime_tree_directories": SANDBOX_RUNTIME_TREE_DIRECTORIES,
                "sandbox_runtime_tree_bytes": SANDBOX_RUNTIME_TREE_BYTES,
                "elf_closure_audit_sha256": ELF_CLOSURE_AUDIT_SHA256,
                "system_library_members": expected_system_library_members(),
                "clear_environment": True,
                "network_namespace": "isolated",
                "candidate_read_only": True,
                "candidate_direct_pid_1": True,
                "site_packages_visible": False,
                "sandbox_isolation_passed": True,
                "probe_results": probes,
                "probe_sha256": probe_sha256,
            },
            sort_keys=True,
        )
        + "\n"
    )
    common = Path(__file__).with_name("pcf1_common.sh")
    python = subprocess.run(
        ["bash", "-c", "command -v python3"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    command = r"""
set -euo pipefail
source "$1"
PYTHON=$2
actual=$(pcf1_validate_sandbox_receipt "$3" "$4")
test "$actual" = "$4"
"""
    environment = os.environ.copy()
    train_root = str(Path(__file__).parents[1])
    environment["PYTHONPATH"] = (
        f"{train_root}:{environment['PYTHONPATH']}"
        if environment.get("PYTHONPATH")
        else train_root
    )
    subprocess.run(
        [
            "bash",
            "-c",
            command,
            "pcf1-test",
            str(common),
            python,
            str(receipt),
            hashlib.sha256(receipt.read_bytes()).hexdigest(),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_stage_model_dereferences_snapshot_symlinks_and_verifies_manifest(
    tmp_path: Path,
) -> None:
    blobs = tmp_path / "blobs"
    model = tmp_path / "snapshot"
    scratch = tmp_path / "scratch"
    blobs.mkdir()
    model.mkdir()
    scratch.mkdir()
    (blobs / "config").write_bytes(b"{}\n")
    (blobs / "weights").write_bytes(b"weights")
    (model / "config.json").symlink_to(blobs / "config")
    (model / "model.safetensors").symlink_to(blobs / "weights")
    manifest = model / "SHA256SUMS"
    manifest.write_text(
        "".join(
            f"{hashlib.sha256((model / name).read_bytes()).hexdigest()}  ./{name}\n"
            for name in ("config.json", "model.safetensors")
        )
    )
    common = Path(__file__).with_name("pcf1_common.sh")
    command = r"""
set -euo pipefail
source "$1"
MODEL_ROOT=$2
MODEL_CONFIG_SHA256=$3
MODEL_MANIFEST=$4
SLURM_TMPDIR=$5
PYTHON=$6
staged=$(pcf1_stage_model_to "$SLURM_TMPDIR/staged" 2 "$7")
test -f "$staged/config.json"
test -f "$staged/model.safetensors"
test ! -L "$staged/config.json"
test ! -L "$staged/model.safetensors"
"""
    subprocess.run(
        [
            "bash",
            "-c",
            command,
            "pcf1-test",
            str(common),
            str(model),
            hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
            str(manifest),
            str(scratch),
            subprocess.run(
                ["bash", "-c", "command -v python3"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            str(
                len((blobs / "config").read_bytes())
                + len((blobs / "weights").read_bytes())
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_model_tree_rejects_extra_file(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n")
    manifest = model / "SHA256SUMS"
    manifest.write_text(
        f"{hashlib.sha256((model / 'config.json').read_bytes()).hexdigest()}  ./config.json\n"
    )
    (model / "shadow.py").write_text("raise RuntimeError\n")
    common = Path(__file__).with_name("pcf1_common.sh")
    command = r"""
set -euo pipefail
source "$1"
PYTHON=$2
pcf1_verify_model_tree "$3" "$4" 1 "$5"
"""
    rejected = subprocess.run(
        [
            "bash",
            "-c",
            command,
            "pcf1-test",
            str(common),
            subprocess.run(
                ["bash", "-c", "command -v python3"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            str(model),
            str(manifest),
            str((model / "config.json").stat().st_size),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "exact membership differs" in rejected.stderr


def test_runtime_verification_rejects_extra_shadow_file(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    member = runtime / "pipeline" / "pcf1_runtime_allowlist.txt"
    member.parent.mkdir()
    member.write_text("pipeline/pcf1_runtime_allowlist.txt\n")
    receipt = runtime / "runtime.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "shohin-pcf1-runtime-v1",
                "status": "complete",
                "source_commit": "a" * 40,
                "allowlist_sha256": hashlib.sha256(member.read_bytes()).hexdigest(),
                "allowlisted_files": ["pipeline/pcf1_runtime_allowlist.txt"],
                "extra_files_permitted": False,
            }
        )
        + "\n"
    )
    manifest = runtime / "SHA256SUMS"
    entries = ("pipeline/pcf1_runtime_allowlist.txt", "runtime.json")
    manifest.write_text(
        "".join(
            f"{hashlib.sha256((runtime / name).read_bytes()).hexdigest()}  {name}\n"
            for name in entries
        )
    )
    common = Path(__file__).with_name("pcf1_common.sh")
    command = r"""
set -euo pipefail
source "$1"
RUNTIME=$2
RUNTIME_MANIFEST_SHA256=$3
PYTHON=$4
pcf1_verify_runtime_membership
"""
    arguments = [
        "bash",
        "-c",
        command,
        "pcf1-test",
        str(common),
        str(runtime),
        hashlib.sha256(manifest.read_bytes()).hexdigest(),
        subprocess.run(
            ["bash", "-c", "command -v python3"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    ]
    subprocess.run(arguments, check=True, capture_output=True, text=True)
    (runtime / "empty-extra").mkdir()
    rejected = subprocess.run(arguments, check=False, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "exact membership differs" in rejected.stderr
    (runtime / "empty-extra").rmdir()
    (runtime / "train" / "shadow.py").parent.mkdir()
    (runtime / "train" / "shadow.py").write_text("raise RuntimeError\n")
    rejected = subprocess.run(arguments, check=False, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "exact membership differs" in rejected.stderr
