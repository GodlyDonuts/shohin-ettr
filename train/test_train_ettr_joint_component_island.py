from argparse import Namespace
from pathlib import Path

import pytest

from train_ettr_joint_component_island import (
    ETTRJointComponentIslandError,
    _validate_args,
    _validate_parent_lineage,
)
from train_ettr_joint_instruction_canary import (
    MODEL_SCHEMA,
    RUN_SCHEMA,
)


def _valid_parent() -> tuple[dict[str, object], dict[str, object]]:
    config = {"slots": 64}
    contract = {
        "ettr_release_sha256": "1" * 64,
        "model_config": config,
        "schema": RUN_SCHEMA,
    }
    payload = {
        "ettr_config": config,
        "run_contract_sha256": "2" * 64,
        "schema": MODEL_SCHEMA,
    }
    return contract, payload


def _valid_args(tmp_path: Path) -> Namespace:
    return Namespace(
        component="compiler",
        data_root=tmp_path / "data",
        data_seed=1,
        eval_batches=2,
        gradient_clip=1.0,
        initial_component=None,
        initial_component_sha256=None,
        learning_rate=1e-4,
        log_every=10,
        output=tmp_path / "output",
        parent_joint_model=tmp_path / "joint-model-final.pt",
        parent_joint_model_sha256="2" * 64,
        parent_run_contract=tmp_path / "run-contract.json",
        parent_run_contract_sha256="3" * 64,
        reactor_reduction="decision-mean",
        reader_injection="stage",
        reader_state_source="oracle",
        release_root=tmp_path / "release",
        release_sha256="1" * 64,
        source_commit="4" * 40,
        start_position=0,
        tokenizer=tmp_path / "tokenizer.json",
        updates=1,
        weight_decay=0.0,
    )


def test_joint_component_parent_lineage_accepts_exact_parent() -> None:
    contract, payload = _valid_parent()
    _validate_parent_lineage(
        contract,
        payload,
        release_sha256="1" * 64,
        parent_run_contract_sha256="2" * 64,
    )


def test_joint_component_parent_lineage_rejects_config_drift() -> None:
    contract, payload = _valid_parent()
    payload["ettr_config"] = {"slots": 32}
    with pytest.raises(
        ETTRJointComponentIslandError,
        match="lineage differs",
    ):
        _validate_parent_lineage(
            contract,
            payload,
            release_sha256="1" * 64,
            parent_run_contract_sha256="2" * 64,
        )


def test_joint_component_warm_start_accepts_exact_receipt(
    tmp_path: Path,
) -> None:
    args = _valid_args(tmp_path)
    args.initial_component = tmp_path / "component-final.safetensors"
    args.initial_component_sha256 = "5" * 64
    _validate_args(args)


def test_joint_component_autonomous_reader_source_is_reader_only(
    tmp_path: Path,
) -> None:
    args = _valid_args(tmp_path)
    args.reader_state_source = "autonomous"
    with pytest.raises(
        ETTRJointComponentIslandError,
        match="arguments differ",
    ):
        _validate_args(args)
    args.component = "reader"
    _validate_args(args)


@pytest.mark.parametrize(
    ("component", "sha256"),
    (
        (Path("/component-final.safetensors"), None),
        (None, "5" * 64),
        (Path("component-final.safetensors"), "5" * 64),
        (Path("/component-final.safetensors"), "not-a-sha"),
    ),
)
def test_joint_component_warm_start_rejects_incomplete_receipt(
    tmp_path: Path,
    component: Path | None,
    sha256: str | None,
) -> None:
    args = _valid_args(tmp_path)
    args.initial_component = component
    args.initial_component_sha256 = sha256
    with pytest.raises(
        ETTRJointComponentIslandError,
        match="arguments differ",
    ):
        _validate_args(args)
