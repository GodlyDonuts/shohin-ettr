from pathlib import Path

import pytest

from compose_ettr_joint_components import (
    COMPOSITION_KIND,
    ETTRJointCompositionError,
    _composed_run_contract,
    _composition_receipt,
)
from train_ettr_joint_instruction_canary import RUN_SCHEMA


def _receipt() -> dict[str, object]:
    return _composition_receipt(
        parent_joint_model=Path("/parent/model.pt"),
        parent_joint_model_sha256="1" * 64,
        parent_run_contract=Path("/parent/run-contract.json"),
        parent_run_contract_sha256="2" * 64,
        components={
            "compiler": {"path": "/c", "sha256": "3" * 64},
            "reactor": {"path": "/r", "sha256": "4" * 64},
            "reader": {"path": "/q", "sha256": "5" * 64},
        },
        source_commit="6" * 40,
    )


def test_composition_receipt_binds_zero_update_transplant() -> None:
    receipt = _receipt()
    assert receipt["kind"] == COMPOSITION_KIND
    assert receipt["optimizer_updates"] == 0
    assert receipt["components"]["reader"]["sha256"] == "5" * 64


def test_composed_contract_preserves_parent_and_binds_receipt() -> None:
    parent = {
        "schema": RUN_SCHEMA,
        "source_commit": "0" * 40,
        "model_config": {"slots": 64},
    }
    receipt = _receipt()
    composed = _composed_run_contract(
        parent,
        composition=receipt,
        source_commit="6" * 40,
    )
    assert composed["component_composition"] == receipt
    assert composed["source_commit"] == "6" * 40
    assert composed["model_config"] == {"slots": 64}
    assert "component_composition" not in parent


def test_recursive_composition_is_rejected() -> None:
    parent = {
        "schema": RUN_SCHEMA,
        "component_composition": {},
    }
    with pytest.raises(
        ETTRJointCompositionError,
        match="recursively compose",
    ):
        _composed_run_contract(
            parent,
            composition=_receipt(),
            source_commit="6" * 40,
        )
