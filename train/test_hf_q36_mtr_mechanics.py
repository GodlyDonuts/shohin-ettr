from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from hf_q36_mtr_mechanics import (
    Q36MTRMechanicsError,
    _state_sha256,
    protected_parameter_receipt,
)


class _TinyMoE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = nn.Linear(4, 2, bias=False)
        self.expert = nn.Linear(4, 4, bias=False)
        self.adapter_a = nn.Linear(4, 1, bias=False)
        self.router.requires_grad_(False)
        self.expert.requires_grad_(False)


def test_protected_receipt_detects_any_router_or_expert_mutation() -> None:
    model = _TinyMoE()
    before = protected_parameter_receipt(model)
    assert before["router_expert_parameter_count"] == 2
    assert before["router_expert_numel"] == 24
    with torch.no_grad():
        model.router.weight.add_(1)
    after = protected_parameter_receipt(model)
    assert after != before
    assert (
        after["router_expert_receipt_sha256"] != before["router_expert_receipt_sha256"]
    )


def test_protected_receipt_hashes_exact_router_and_expert_bytes() -> None:
    model = _TinyMoE()
    before = protected_parameter_receipt(model)
    with torch.no_grad():
        # `.data` mutation does not offer a dependable autograd version proof.
        model.expert.weight.data[0, 0] += 1
    after = protected_parameter_receipt(model)
    assert (
        after["router_expert_receipt_sha256"] != before["router_expert_receipt_sha256"]
    )


def test_protected_receipt_requires_a_real_moe_surface() -> None:
    model = nn.Linear(4, 4)
    model.requires_grad_(False)
    with pytest.raises(Q36MTRMechanicsError):
        protected_parameter_receipt(model)


def test_trainable_state_digest_is_name_shape_dtype_and_byte_bound() -> None:
    first = {"adapter": torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)}
    second = {"adapter": torch.tensor([[1.0, 3.0]], dtype=torch.bfloat16)}
    renamed = {"different": first["adapter"]}
    assert _state_sha256(first) != _state_sha256(second)
    assert _state_sha256(first) != _state_sha256(renamed)
    assert _state_sha256(first) == _state_sha256({"adapter": first["adapter"].clone()})


def test_mechanics_wrapper_is_no_score_single_h100() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "train" / "jobs" / "q36_mtr_mechanics.sbatch").read_text()
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in source
    assert "#SBATCH --no-requeue" in source
    assert "q36_require_authorization" in source
    assert "hf_q36_mtr_mechanics.py" in source
    assert "score_completion" not in source
    assert "ASSESSOR" not in source
    assert "sbatch " not in source


def test_nf4_role_and_mechanics_do_not_move_the_quantized_wrapper() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "train/hf_q36_mtr_train_role.py",
        "train/hf_q36_mtr_mechanics.py",
    ):
        source = (root / relative).read_text()
        assert ').to("cuda:0")' not in source
