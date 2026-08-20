from __future__ import annotations

import ast
from pathlib import Path
import sys
import types

import pytest

import hf_gpt_oss_120b_train_revision as training
from hf_gpt_oss_120b_mechanics import GptOssMechanicsError

from hf_gpt_oss_120b_train_revision import (
    CONSUMED_PRESENTATIONS,
    DATA_PRESENTATIONS,
    GRADIENT_ACCUMULATION,
    MAX_SEQUENCE_LENGTH,
    UPDATES,
    consumed_identity_sha256,
)


def test_training_geometry_matches_the_cross_family_mixtral_point() -> None:
    assert DATA_PRESENTATIONS == 9_655
    assert UPDATES == 256
    assert GRADIENT_ACCUMULATION == 8
    assert CONSUMED_PRESENTATIONS == 2_048
    assert MAX_SEQUENCE_LENGTH == 4_096


def test_consumed_identity_digest_is_order_sensitive() -> None:
    rows = [{"identity_sha256": f"{index:064x}"} for index in range(DATA_PRESENTATIONS)]
    left = consumed_identity_sha256(rows)
    rows[0], rows[1] = rows[1], rows[0]
    assert consumed_identity_sha256(rows) != left


def test_training_uses_harmony_final_targets_and_bounded_logits() -> None:
    path = Path(__file__).with_name("hf_gpt_oss_120b_train_revision.py")
    source = path.read_text()
    tree = ast.parse(source)
    assert "tokenize_training_example" in source
    assert "logits_to_keep=len(response) + 1" in source
    assert 'device_map={"": 0}' in source
    assert 'native_router_expert_trainables": 0' in source
    assert any(isinstance(node, ast.Call) for node in ast.walk(tree))


def _install_fake_loader_modules(
    monkeypatch: pytest.MonkeyPatch, backbone: object
) -> None:
    class _AutoModel:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> object:
            assert kwargs["device_map"] == {"": 0}
            return backbone

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoModelForCausalLM=_AutoModel),
    )
    monkeypatch.setitem(
        sys.modules,
        "kernels",
        types.SimpleNamespace(get_loaded_kernels=lambda: ["torch-cuda"]),
    )


def test_training_loader_reuses_mechanics_cuda_residency_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backbone = object()
    receipt = {"device_map_mode": "absent_single_device_load"}
    _install_fake_loader_modules(monkeypatch, backbone)
    monkeypatch.setattr(training, "_native_mxfp4_load_receipt", lambda value: receipt)
    loaded, observed = training._load_backbone(Path("/model"))
    assert loaded is backbone
    assert observed is receipt


def test_training_loader_converts_residency_failure_to_training_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_loader_modules(monkeypatch, object())

    def reject(_: object) -> dict[str, object]:
        raise GptOssMechanicsError("offloaded")

    monkeypatch.setattr(training, "_native_mxfp4_load_receipt", reject)
    with pytest.raises(training.GptOssTrainingError, match="native MXFP4 training"):
        training._load_backbone(Path("/model"))
