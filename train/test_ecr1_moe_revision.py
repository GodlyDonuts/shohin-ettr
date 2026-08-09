import sys

import pytest
import torch
import torch.nn as nn

from ecr1_moe_revision import (
    ECR1Config,
    ECR1Error,
    ExpertConditionedResidualMoE,
    expert_code_diagnostics,
)
from train_ecr1_product import parse_args


class FakeGate(nn.Module):
    def __init__(self, hidden=8, experts=4, top_k=2):
        super().__init__()
        self.hidden_dim = hidden
        self.num_experts = experts
        self.top_k = top_k
        self.norm_topk_prob = True
        self.weight = nn.Parameter(torch.randn(experts, hidden))

    def forward(self, hidden):
        logits = torch.nn.functional.linear(hidden, self.weight)
        probability = logits.softmax(dim=-1)
        weights, indices = probability.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return logits, weights, indices


class FakeExperts(nn.Module):
    def __init__(self, hidden=8, experts=4):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(experts, hidden, hidden))

    def forward(self, hidden, indices, weights):
        output = torch.zeros_like(hidden)
        for token in range(hidden.shape[0]):
            for position in range(indices.shape[1]):
                output[token] += weights[token, position] * torch.mv(
                    self.weights[indices[token, position]], hidden[token]
                )
        return output


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = FakeGate()
        self.experts = FakeExperts()

    def forward(self, hidden):
        flat = hidden.reshape(-1, hidden.shape[-1])
        _, weights, indices = self.gate(flat)
        return self.experts(flat, indices, weights).reshape_as(hidden)


def config(mode="expert_conditioned", rank=3):
    return ECR1Config(8, 4, 2, controlled_layers=1, rank=rank, alpha=float(rank), mode=mode)


def test_zero_initialization_is_exact_base_parity():
    torch.manual_seed(4)
    base = FakeBlock()
    wrapped = ExpertConditionedResidualMoE(base, config())
    hidden = torch.randn(2, 5, 8)
    assert torch.equal(wrapped(hidden), base(hidden))
    assert all(not parameter.requires_grad for parameter in base.parameters())


def test_expert_identity_controls_residual_and_permutation_changes_it():
    torch.manual_seed(5)
    wrapped = ExpertConditionedResidualMoE(FakeBlock(), config())
    with torch.no_grad():
        wrapped.adapter_b.weight.fill_(0.1)
        wrapped.expert_codes.copy_(torch.arange(12).reshape(4, 3) / 3)
    hidden = torch.randn(1, 6, 8)
    normal = wrapped(hidden)
    wrapped.set_code_intervention("permutation")
    permuted = wrapped(hidden)
    wrapped.set_code_intervention("zero")
    zeroed = wrapped(hidden)
    assert not torch.equal(normal, permuted)
    assert not torch.equal(normal, zeroed)


def test_shared_control_has_exact_requested_parameter_count():
    wrapped = ExpertConditionedResidualMoE(FakeBlock(), config("shared", rank=4))
    trainable = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
    assert trainable == 2 * 8 * 4
    try:
        wrapped.set_code_intervention("zero")
    except ECR1Error:
        pass
    else:
        raise AssertionError("shared control accepted an expert-code intervention")


def test_olmoe_parameter_receipts_are_exact():
    ecr = 4 * (31 * 2048 + 2048 * 31 + 64 * 31)
    shared = 4 * (32 * 2048 + 2048 * 32)
    assert ecr == 515_840
    assert shared == 524_288


def test_olmoe_depth_followup_parameter_receipts_are_exact():
    ecr = 16 * (8 * 2048 + 2048 * 8 + 64 * 8)
    shared = 16 * (8 * 2048 + 2048 * 8)
    assert ecr == 532_480
    assert shared == 524_288


def test_receipt_separates_load_and_per_token_entropy():
    wrapped = ExpertConditionedResidualMoE(FakeBlock(), config())
    wrapped(torch.randn(2, 4, 8))
    receipt = wrapped.receipt()
    assert receipt["tokens"] == 8
    assert 0 <= receipt["load_entropy_normalized"] <= 1
    assert 0 <= receipt["mean_token_entropy_normalized"] <= 1
    assert receipt["mean_top8_top9_logit_margin"] >= 0


def test_code_diagnostics_report_rank_and_cosine():
    diagnostics = expert_code_diagnostics(torch.eye(4))
    assert diagnostics["effective_rank"] == 4
    assert diagnostics["pairwise_cosine_abs_mean"] == 0


@pytest.mark.parametrize(
    ("mode", "layers", "rank", "alpha"),
    (
        ("expert_conditioned", 4, 31, 31),
        ("shared", 4, 32, 32),
        ("expert_conditioned", 16, 8, 8),
        ("shared", 16, 8, 8),
    ),
)
def test_parser_accepts_only_frozen_geometry(
    monkeypatch, mode, layers, rank, alpha
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ecr1_product.py",
            "--model-root",
            "model",
            "--model-revision",
            "revision",
            "--data",
            "data.jsonl",
            "--output",
            "output",
            "--mode",
            mode,
            "--controlled-layers",
            str(layers),
            "--rank",
            str(rank),
            "--alpha",
            str(alpha),
        ],
    )
    args = parse_args()
    assert (args.controlled_layers, args.rank, args.alpha) == (
        layers,
        rank,
        float(alpha),
    )


def test_parser_rejects_unfrozen_geometry(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ecr1_product.py",
            "--model-root",
            "model",
            "--model-revision",
            "revision",
            "--data",
            "data.jsonl",
            "--output",
            "output",
            "--mode",
            "expert_conditioned",
            "--controlled-layers",
            "16",
            "--rank",
            "9",
            "--alpha",
            "9",
        ],
    )
    with pytest.raises(SystemExit):
        parse_args()
