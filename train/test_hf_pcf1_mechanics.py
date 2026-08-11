"""Focused fail-closed tests for PCF1 adapter admission metadata."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from build_pcf1_data import revision_prompt
from hf_pcf1_mechanics import (
    PCF1MechanicsError,
    validate_adapter_metadata,
    validate_compute_host_receipt,
    validate_sandbox_receipt,
)
from pcf1_code_sandbox import (
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


def _metadata() -> dict[str, object]:
    return {
        "schema": "shohin-hf-product-reasoning-training-v1",
        "status": "complete",
        "updates": 1,
        "gradient_accumulation": 24,
        "lora_scope": "token_mixer",
        "lora_layers": 4,
        "lora_layer_indices": [30, 31, 32, 33],
        "lora_rank": 8,
        "lora_alpha": 16.0,
        "model_revision": "revision",
        "model_loader": "multimodal",
        "backbone_layout": "multimodal-language-model",
        "lora_projection_count": 28,
        "trainable_parameters": 4096,
        "trainable_parameter_name_sha256": "a" * 64,
        "environment_receipt_sha256": "e" * 64,
        "environment_tree_sha256": "f" * 64,
        "trace": [{"update": 1}],
    }


def test_adapter_metadata_requires_exact_final_four_layers() -> None:
    training = _metadata()
    restored = copy.deepcopy(training)
    restored["update"] = 1
    validate_adapter_metadata(training, restored, "revision")
    for payload in (training, restored):
        changed = copy.deepcopy(payload)
        changed["lora_layer_indices"] = [29, 30, 31, 32]
        with pytest.raises(PCF1MechanicsError, match="mechanics|restoration"):
            if payload is training:
                validate_adapter_metadata(changed, restored, "revision")
            else:
                validate_adapter_metadata(training, changed, "revision")


def test_revision_prompt_has_no_task_router() -> None:
    rendered = revision_prompt("same source", "same draft")
    assert "same source" in rendered
    assert "same draft" in rendered
    with pytest.raises(TypeError):
        revision_prompt("same source", "same draft", "mbpp")


def test_sandbox_receipt_requires_exact_probe_and_binary_custody(
    tmp_path: Path,
) -> None:
    probes = {name: True for name in SANDBOX_PROBES}
    payload = {
        "schema": "shohin-pcf1-code-sandbox-receipt-v1",
        "status": "pass",
        "bwrap_path": "/usr/bin/bwrap",
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
        "probe_sha256": hashlib.sha256(
            json.dumps(probes, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    receipt = tmp_path / "sandbox_receipt.json"
    receipt.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert validate_sandbox_receipt(receipt, digest) == payload

    payload["probe_results"]["network_unreachable"] = False
    receipt.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(PCF1MechanicsError, match="admission receipt"):
        validate_sandbox_receipt(
            receipt, hashlib.sha256(receipt.read_bytes()).hexdigest()
        )


def test_compute_host_receipt_binds_exact_h100_and_nvidia_binary(
    tmp_path: Path,
) -> None:
    binary = tmp_path / "nvidia-smi"
    binary.write_bytes(b"nvidia fixture")
    payload = {
        "schema": "shohin-pcf1-compute-host-receipt-v1",
        "status": "complete",
        "partition": "normal",
        "node": "evc10",
        "excluded_nodes": [
            "evc26",
            "evc29",
            "evc31",
            "evc32",
            "evc33",
            "evc38",
            "evc46",
        ],
        "nvidia_smi_invoked_path": str(binary),
        "nvidia_smi_resolved_path": str(binary.resolve()),
        "nvidia_smi_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "nvidia_smi_version": "fixture 1",
        "visible_gpu_count": 1,
        "gpu_name": "NVIDIA H100 PCIe",
        "driver_version": "fixture-driver",
        "pci_bus_id": "00000000:01:00.0",
    }
    receipt = tmp_path / "compute_host_receipt.json"
    receipt.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert validate_compute_host_receipt(receipt, digest) == payload

    payload["gpu_name"] = "NVIDIA A100"
    receipt.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(PCF1MechanicsError, match="compute-host admission"):
        validate_compute_host_receipt(
            receipt, hashlib.sha256(receipt.read_bytes()).hexdigest()
        )
