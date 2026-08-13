from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from hf_q36_mtr_mechanics import (
    Q36MTRMechanicsError,
    _state_sha256,
    causal_draft_intervention_receipt,
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


class _ToyCausalTextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.embed_tokens = nn.Embedding(64, 8)
        self.config = SimpleNamespace(num_experts_per_tok=2)

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        use_cache: bool,
        output_router_logits: bool,
    ) -> SimpleNamespace:
        assert use_cache is False
        assert output_router_logits is True
        assert position_ids.tolist() == [
            list(range(inputs_embeds.shape[1])) for _ in range(inputs_embeds.shape[0])
        ]
        scores = torch.zeros(
            inputs_embeds.shape[0],
            inputs_embeds.shape[1],
            inputs_embeds.shape[1],
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        causal = torch.ones_like(scores, dtype=torch.bool).tril()
        visible_keys = attention_mask[:, None, :].bool()
        weights = torch.softmax(scores.masked_fill(~(causal & visible_keys), -1e9), -1)
        hidden = inputs_embeds + torch.matmul(weights, inputs_embeds)
        router_logits = torch.stack(
            (
                hidden[..., 0],
                -hidden[..., 0],
                hidden[..., 1],
                -hidden[..., 1],
            ),
            dim=-1,
        ).reshape(-1, 4)
        return SimpleNamespace(
            last_hidden_state=hidden,
            router_logits=(router_logits,),
        )


class _ToyCausalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_model = _ToyCausalTextModel()


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


def test_causal_intervention_hides_only_draft_information() -> None:
    receipt = causal_draft_intervention_receipt(
        _ToyCausalModel(),
        prompt=[3, 5, 7, 11, 13],
        response=[17, 19],
        draft_mask=[1, 0, 0, 1, 1],
        pad_token_id=0,
    )
    assert receipt["token_count_exact"] is True
    assert receipt["position_geometry_exact"] is True
    assert receipt["draft_hidden_response_max_abs_delta"] == 0.0
    assert receipt["draft_hidden_counterfactual_invariant"] is True
    assert receipt["aligned_response_max_abs_delta"] > 0.01
    assert receipt["aligned_counterfactual_sensitive"] is True
    assert receipt["native_router"]["aligned_route_sensitive"] is True
    assert receipt["native_router"]["draft_hidden_route_invariant"] is True
    assert receipt["native_router"]["aligned"]["router_max_abs_delta"] > 0.01
    assert receipt["native_router"]["draft_hidden"]["topk_assignment_changes"] == 0


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
