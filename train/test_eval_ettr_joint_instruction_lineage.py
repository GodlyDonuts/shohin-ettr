from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from eval_ettr_joint_instruction_model import (
    COMPOSITION_KIND,
    ETTRTriEvaluationError,
    MODEL_SCHEMA,
    RUN_SCHEMA,
    _load_initialization_contract,
    _validate_external_parent_base,
    _validate_model_lineage,
    _validate_run_lineage,
)


class _ExternalControl(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(2, 2, bias=False)
        self.compiler = torch.nn.Linear(2, 2, bias=False)


def test_external_parent_base_validation_ignores_only_ettr_state() -> None:
    model = _ExternalControl()
    payload = {
        "model": {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }
    }
    payload["model"]["compiler.weight"].add_(1.0)
    _validate_external_parent_base(model, payload)
    payload["model"]["base.weight"].add_(1.0)
    with pytest.raises(
        ETTRTriEvaluationError,
        match="external parent base weights differ",
    ):
        _validate_external_parent_base(model, payload)


def _parent_contract() -> dict[str, object]:
    return {
        "schema": RUN_SCHEMA,
        "source_commit": "1" * 40,
        "ettr_release_sha256": "2" * 64,
        "model_config": {"num_slots": 64},
        "parameter_receipt": {"complete_system_parameters": 192_779_435},
        "parent_joint_model": "/grandparent/joint-model-final.pt",
        "parent_joint_model_sha256": "a" * 64,
        "parent_run_contract_sha256": "b" * 64,
    }


def _composition() -> dict[str, object]:
    return {
        "components": {
            name: {
                "path": f"/components/{name}.safetensors",
                "sha256": digit * 64,
            }
            for name, digit in zip(
                ("compiler", "reactor", "reader"),
                ("3", "4", "5"),
                strict=True,
            )
        },
        "kind": COMPOSITION_KIND,
        "optimizer_updates": 0,
        "parent_joint_model": "/parent/model.pt",
        "parent_joint_model_sha256": "6" * 64,
        "parent_run_contract": "/parent/run-contract.json",
        "parent_run_contract_sha256": "7" * 64,
        "source_commit": "8" * 40,
    }


def _composed_contract() -> dict[str, object]:
    contract = deepcopy(_parent_contract())
    contract["source_commit"] = "8" * 40
    contract["component_composition"] = _composition()
    return contract


def _payloads() -> tuple[dict[str, object], dict[str, object]]:
    parent = {
        "base_config": {"layers": 30},
        "ettr_config": {"num_slots": 64},
        "initialization": {
            "initialization": "parent-joint-model",
            "parent_joint_model_sha256": "a" * 64,
        },
        "model": {
            "base.weight": torch.tensor([1.0]),
            "compiler.weight": torch.tensor([2.0]),
        },
        "optimizer_step": 2_000,
        "run_contract_sha256": "7" * 64,
        "schedule": {"optimizer_step": 2_000},
        "schema": MODEL_SCHEMA,
    }
    candidate = deepcopy(parent)
    candidate["initialization"] = _composition()
    candidate["model"]["compiler.weight"] = torch.tensor([3.0])
    candidate["run_contract_sha256"] = "9" * 64
    return parent, candidate


def test_zero_update_composition_is_valid_initialization_child() -> None:
    parent_contract = _parent_contract()
    run_contract = _composed_contract()
    composition = _validate_run_lineage(
        parent_contract,
        run_contract,
        release_sha256="2" * 64,
        parent_run_contract_sha256="7" * 64,
        parent_joint_model_sha256="6" * 64,
    )
    parent, candidate = _payloads()
    _validate_model_lineage(
        parent,
        candidate,
        parent_run_contract_sha256="7" * 64,
        run_contract_sha256="9" * 64,
        parent_contract=parent_contract,
        run_contract=run_contract,
        composition=composition,
    )


def test_composition_rejects_changed_base_weights() -> None:
    run_contract = _composed_contract()
    parent, candidate = _payloads()
    candidate["model"]["base.weight"].add_(1.0)
    with pytest.raises(
        ETTRTriEvaluationError,
        match="changed base weights",
    ):
        _validate_model_lineage(
            parent,
            candidate,
            parent_run_contract_sha256="7" * 64,
            run_contract_sha256="9" * 64,
            parent_contract=_parent_contract(),
            run_contract=run_contract,
            composition=_composition(),
        )


def test_composition_rejects_mutated_parent_contract() -> None:
    run_contract = _composed_contract()
    run_contract["model_config"] = {"num_slots": 32}
    with pytest.raises(
        ETTRTriEvaluationError,
        match="component composition lineage",
    ):
        _validate_run_lineage(
            _parent_contract(),
            run_contract,
            release_sha256="2" * 64,
            parent_run_contract_sha256="7" * 64,
            parent_joint_model_sha256="6" * 64,
        )


def _external_parent_contract() -> dict[str, object]:
    base_import = {
        "model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "source_model_sha256": "c" * 64,
    }
    token_transcode = {
        "schema": "shohin-ettr-token-native-transcode-v1",
        "token_id_map_sha256": "d" * 64,
    }
    return {
        "architecture_seed": 2026072801,
        "ettr_release_sha256": "2" * 64,
        "initialization": {
            "architecture_seed": 2026072801,
            "base_import": base_import,
            "initialization": "external-smollm2-135m-control",
            "token_transcode": token_transcode,
        },
        "model_config": {"num_slots": 64},
        "schema": RUN_SCHEMA,
        "source_commit": "1" * 40,
        "token_transcode": token_transcode,
    }


def test_direct_external_parent_can_be_composed_and_reconstructed() -> None:
    parent_contract = _external_parent_contract()
    run_contract = deepcopy(parent_contract)
    run_contract["source_commit"] = "8" * 40
    run_contract["component_composition"] = _composition()
    composition = _validate_run_lineage(
        parent_contract,
        run_contract,
        release_sha256="2" * 64,
        parent_run_contract_sha256="7" * 64,
        parent_joint_model_sha256="6" * 64,
    )
    assert (
        _load_initialization_contract(
            parent_contract,
            composition=composition,
        )
        == parent_contract
    )

    base_import = parent_contract["initialization"]["base_import"]
    parent = {
        "base_config": {"layers": 30},
        "base_import": base_import,
        "base_rms_norm_eps": 1e-5,
        "ettr_config": {"num_slots": 64},
        "initialization": parent_contract["initialization"],
        "model": {
            "base.weight": torch.tensor([1.0]),
            "compiler.weight": torch.tensor([2.0]),
        },
        "optimizer_step": 0,
        "run_contract_sha256": "7" * 64,
        "schedule": {"optimizer_step": 0},
        "schema": MODEL_SCHEMA,
    }
    candidate = deepcopy(parent)
    candidate["initialization"] = composition
    candidate["model"]["compiler.weight"] = torch.tensor([3.0])
    candidate["run_contract_sha256"] = "9" * 64
    _validate_model_lineage(
        parent,
        candidate,
        parent_run_contract_sha256="7" * 64,
        run_contract_sha256="9" * 64,
        parent_contract=parent_contract,
        run_contract=run_contract,
        composition=composition,
    )


def _write_json(path: Path, payload: object) -> str:
    data = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_nonstage_reader_geometry_requires_and_admits_training_lineage(
    tmp_path: Path,
) -> None:
    reader_contract = {
        "component": "reader",
        "parent_joint_model_sha256": "6" * 64,
        "reader_injection": "postnorm-scaled",
        "schema": "shohin-ettr-joint-component-island-contract-v1",
    }
    contract_path = tmp_path / "island-contract.json"
    contract_sha = _write_json(contract_path, reader_contract)
    reader_report = {
        "component": "reader",
        "contract_sha256": contract_sha,
        "final_component_sha256": "5" * 64,
        "parent_joint_model_sha256": "6" * 64,
        "reader_injection": "postnorm-scaled",
        "schema": "shohin-ettr-joint-component-island-report-v1",
    }
    report_path = tmp_path / "report.json"
    report_sha = _write_json(report_path, reader_report)

    composition = _composition()
    composition["query_readout_geometry"] = "postnorm-scaled"
    composition["reader_training"] = {
        "contract": {
            "path": str(contract_path),
            "sha256": contract_sha,
        },
        "report": {
            "path": str(report_path),
            "sha256": report_sha,
        },
    }
    parent_contract = _parent_contract()
    run_contract = deepcopy(parent_contract)
    run_contract["source_commit"] = "8" * 40
    run_contract["query_readout_geometry"] = "postnorm-scaled"
    run_contract["component_composition"] = composition
    observed = _validate_run_lineage(
        parent_contract,
        run_contract,
        release_sha256="2" * 64,
        parent_run_contract_sha256="7" * 64,
        parent_joint_model_sha256="6" * 64,
    )
    parent, candidate = _payloads()
    candidate["initialization"] = composition
    candidate["query_readout_geometry"] = "postnorm-scaled"
    _validate_model_lineage(
        parent,
        candidate,
        parent_run_contract_sha256="7" * 64,
        run_contract_sha256="9" * 64,
        parent_contract=parent_contract,
        run_contract=run_contract,
        composition=observed,
    )
