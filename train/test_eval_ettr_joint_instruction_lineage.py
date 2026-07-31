from copy import deepcopy

import pytest
import torch

from eval_ettr_joint_instruction_model import (
    COMPOSITION_KIND,
    ETTRTriEvaluationError,
    MODEL_SCHEMA,
    RUN_SCHEMA,
    _validate_model_lineage,
    _validate_run_lineage,
)


def _parent_contract() -> dict[str, object]:
    return {
        "schema": RUN_SCHEMA,
        "source_commit": "1" * 40,
        "ettr_release_sha256": "2" * 64,
        "model_config": {"num_slots": 64},
        "parameter_receipt": {"complete_system_parameters": 192_779_435},
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
