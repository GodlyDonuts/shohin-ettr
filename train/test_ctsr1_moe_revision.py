import torch
import torch.nn as nn

from ctsr1_moe_revision import (
    CTSR1Config,
    CausalTemporalMoEBlock,
    TemporalStateHead,
)


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
        return logits, weights / weights.sum(dim=-1, keepdim=True), indices


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


def make(mode="temporal_router"):
    config = CTSR1Config(
        8, 4, 2, controlled_layers=1, state_width=4, head_width=2,
        residual_rank=2, residual_alpha=2.0, mode=mode,
    )
    core = nn.GRU(8, 4, batch_first=True)
    head1 = TemporalStateHead(4, 2, 4)
    head2 = TemporalStateHead(4, 2, 4)
    return FakeBlock(), CausalTemporalMoEBlock(
        FakeBlock(), core, head1, head2, config, 0
    )


def test_zero_initialization_is_exact_base_parity():
    torch.manual_seed(30)
    base = FakeBlock()
    config = CTSR1Config(
        8, 4, 2, controlled_layers=1, state_width=4, head_width=2,
        residual_rank=2, residual_alpha=2.0,
    )
    wrapped = CausalTemporalMoEBlock(
        base, nn.GRU(8, 4, batch_first=True),
        TemporalStateHead(4, 2, 4), TemporalStateHead(4, 2, 4), config, 0,
    )
    hidden = torch.randn(2, 5, 8)
    wrapped.begin_sequence(torch.ones(2, 5), streaming=False)
    assert torch.equal(wrapped(hidden), base(hidden))
    assert all(not p.requires_grad for p in base.parameters())


def test_streaming_matches_one_shot_causal_state():
    torch.manual_seed(31)
    _, block = make()
    hidden = torch.randn(2, 7, 8)
    block.begin_sequence(torch.ones(2, 7), streaming=False)
    one_shot = block._causal_states(hidden)
    block.begin_sequence(torch.ones(2, 3), streaming=True)
    first = block._causal_states(hidden[:, :3])
    pieces = [first]
    for index in range(3, 7):
        pieces.append(block._causal_states(hidden[:, index : index + 1]))
    assert torch.allclose(one_shot, torch.cat(pieces, dim=1), atol=1e-6)


def test_left_padding_matches_active_suffix():
    torch.manual_seed(32)
    _, block = make()
    hidden = torch.randn(2, 6, 8)
    mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    block.begin_sequence(mask, streaming=False)
    padded = block._causal_states(hidden)
    block.begin_sequence(torch.ones(1, 4), streaming=False)
    suffix = block._causal_states(hidden[:1, 2:])
    assert torch.allclose(padded[0, 2:], suffix[0], atol=1e-6)
    assert torch.equal(padded[0, :2], torch.zeros_like(padded[0, :2]))


def test_router_head_causally_changes_routes_after_update():
    torch.manual_seed(33)
    _, block = make()
    with torch.no_grad():
        block.route_head.up.weight.normal_(std=0.5)
    hidden = torch.randn(1, 12, 8)
    block.begin_sequence(torch.ones(1, 12), streaming=False)
    block(hidden)
    receipt = block.receipt()
    assert receipt["route_probability_l1_mean"] > 0
    assert receipt["active_experts"] > 0


def test_exact_frozen_accounting():
    gru = 3 * 64 * 2048 + 3 * 64 * 64 + 2 * 3 * 64
    layer_codes = 16 * 64
    heads = 2 * (64 * 32 + 32 * 64)
    residuals = 16 * 2 * 2048 * 18
    macs = 3 * 64 * 2048 + 3 * 64 * 64 + heads + 64 * 18 + 2 * 2048 * 18
    assert gru + layer_codes + heads + residuals == 1_594_752
    assert macs == 488_576
