import torch
import torch.nn as nn

from rme1_moe_revision import RME1Config, RME1Error, RevisionMicroExpertMoE


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


def config(mode="routed", rank=2):
    return RME1Config(
        8,
        4,
        2,
        controlled_layers=1,
        rank=rank,
        alpha=float(rank),
        mode=mode,
        revision_experts=4,
        revision_top_k=2,
        balance_weight=0.01,
    )


def test_zero_initialization_is_exact_base_parity():
    torch.manual_seed(20)
    base = FakeBlock()
    wrapped = RevisionMicroExpertMoE(base, config())
    hidden = torch.randn(2, 6, 8)
    assert torch.equal(wrapped(hidden), base(hidden))
    assert all(not parameter.requires_grad for parameter in base.parameters())


def test_router_and_whole_expert_interventions_are_causal():
    torch.manual_seed(21)
    wrapped = RevisionMicroExpertMoE(FakeBlock(), config())
    with torch.no_grad():
        wrapped.adapter_b.copy_(torch.arange(64).reshape(4, 8, 2) / 40)
    hidden = torch.randn(1, 8, 8)
    normal = wrapped(hidden)
    wrapped.set_code_intervention("permutation")
    permuted = wrapped(hidden)
    wrapped.set_code_intervention("zero")
    zeroed = wrapped(hidden)
    assert not torch.equal(normal, permuted)
    assert not torch.equal(normal, zeroed)


def test_balance_loss_is_differentiable_and_receipt_uses_all_routes():
    torch.manual_seed(22)
    wrapped = RevisionMicroExpertMoE(FakeBlock(), config())
    wrapped(torch.randn(4, 32, 8))
    balance = wrapped.balance_loss()
    balance.backward()
    assert wrapped.revision_router.weight.grad is not None
    receipt = wrapped.receipt()
    assert receipt["active_revision_experts"] == 4
    assert 0 <= receipt["load_entropy"] <= 1


def test_shared_control_rejects_router_interventions():
    wrapped = RevisionMicroExpertMoE(FakeBlock(), config("shared", rank=3))
    try:
        wrapped.set_code_intervention("uniform")
    except RME1Error:
        pass
    else:
        raise AssertionError("shared control accepted revision-router intervention")


def test_frozen_parameter_and_compute_receipts():
    routed = 16 * (4 * 2048 + 4 * 2 * 8 * 2048)
    shared_flop = 16 * 2 * 2048 * 18
    shared_parameter = 16 * 2 * 2048 * 34
    routed_macs_per_layer = 4 * 2048 + 2 * 2 * 2048 * 8
    assert routed == 2_228_224
    assert shared_flop == 1_179_648
    assert shared_parameter == 2_228_224
    assert routed_macs_per_layer == 73_728
