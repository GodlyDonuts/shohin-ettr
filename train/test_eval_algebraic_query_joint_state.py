from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
import torch

from eval_algebraic_query_joint_state import (
    AlgebraicJointStateEvaluationError,
    _require_module_state,
    _validate_args,
)


def _args(tmp_path: Path, **overrides: object) -> Namespace:
    values: dict[str, object] = {
        "release_root": tmp_path,
        "release_sha256": "0" * 64,
        "data_root": tmp_path,
        "tokenizer": tmp_path / "tokenizer.json",
        "protected_checkpoint": tmp_path / "protected.pt",
        "joint_model": tmp_path / "joint.pt",
        "joint_model_sha256": "1" * 64,
        "joint_run_contract": tmp_path / "joint.json",
        "joint_run_contract_sha256": "2" * 64,
        "compiler": tmp_path / "compiler.safetensors",
        "compiler_sha256": "3" * 64,
        "compiler_contract": tmp_path / "compiler.json",
        "compiler_contract_sha256": "4" * 64,
        "state_run_dir": None,
        "state_run_sha256s_sha256": None,
        "output": tmp_path / "output",
        "source_commit": "5" * 40,
        "data_seed": 7,
        "max_batches": 8,
    }
    values.update(overrides)
    return Namespace(**values)


def test_state_run_arguments_must_be_supplied_together(tmp_path: Path) -> None:
    with pytest.raises(
        AlgebraicJointStateEvaluationError,
        match="arguments differ",
    ):
        _validate_args(_args(tmp_path, state_run_dir=tmp_path))


def test_state_run_arguments_accept_absolute_bound_receipt(
    tmp_path: Path,
) -> None:
    _validate_args(
        _args(
            tmp_path,
            state_run_dir=tmp_path,
            state_run_sha256s_sha256="6" * 64,
        )
    )


def test_initial_component_requires_exact_tensor_identity() -> None:
    module = torch.nn.Linear(3, 2)
    state = {
        name: tensor.detach().clone()
        for name, tensor in module.state_dict().items()
    }
    _require_module_state(module, state)
    state["weight"][0, 0] += 1
    with pytest.raises(
        AlgebraicJointStateEvaluationError,
        match="initial component differs",
    ):
        _require_module_state(module, state)
