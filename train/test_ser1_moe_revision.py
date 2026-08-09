import torch
import torch.nn as nn

from ser1_moe_revision import SER1Config, SER1Error, SelectedExpertResidualMoE


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


def config(mode="selected_expert", rank=1):
    return SER1Config(8, 4, 2, controlled_layers=1, rank=rank, alpha=float(rank), mode=mode)


def test_zero_initialization_is_exact_base_parity():
    torch.manual_seed(11)
    base = FakeBlock()
    wrapped = SelectedExpertResidualMoE(base, config())
    hidden = torch.randn(2, 5, 8)
    assert torch.equal(wrapped(hidden), base(hidden))
    assert all(not parameter.requires_grad for parameter in base.parameters())


def test_whole_expert_transform_is_causal_under_permutation():
    torch.manual_seed(12)
    wrapped = SelectedExpertResidualMoE(FakeBlock(), config())
    with torch.no_grad():
        wrapped.adapter_b.copy_(torch.arange(32).reshape(4, 8, 1) / 20)
    hidden = torch.randn(1, 6, 8)
    normal = wrapped(hidden)
    wrapped.set_code_intervention("permutation")
    permuted = wrapped(hidden)
    wrapped.set_code_intervention("zero")
    zeroed = wrapped(hidden)
    assert not torch.equal(normal, permuted)
    assert not torch.equal(normal, zeroed)


def test_shared_control_rejects_expert_intervention():
    wrapped = SelectedExpertResidualMoE(FakeBlock(), config("shared", rank=2))
    assert sum(p.numel() for p in wrapped.parameters() if p.requires_grad) == 32
    try:
        wrapped.set_code_intervention("permutation")
    except SER1Error:
        pass
    else:
        raise AssertionError("shared control accepted expert intervention")


def test_frozen_olmoe_parameter_and_active_compute_receipts():
    selected = 16 * 64 * (1 * 2048 + 2048 * 1)
    shared_flop = 16 * (8 * 2048 + 2048 * 8)
    shared_parameter = 16 * (64 * 2048 + 2048 * 64)
    selected_active_macs = 16 * 8 * (2048 + 2048)
    shared_flop_macs = 16 * (8 * 2048 + 2048 * 8)
    assert selected == 4_194_304
    assert shared_flop == 524_288
    assert shared_parameter == 4_194_304
    assert selected_active_macs == shared_flop_macs


def test_receipt_names_load_entropy_honestly():
    wrapped = SelectedExpertResidualMoE(FakeBlock(), config())
    wrapped(torch.randn(2, 4, 8))
    receipt = wrapped.receipt()
    assert 0 <= receipt["load_entropy"] <= 1
    assert 0 <= receipt["mean_token_entropy_normalized"] <= 1
    assert "load_entropy_normalized" not in receipt
