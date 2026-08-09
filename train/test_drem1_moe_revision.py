import torch
import torch.nn as nn

from drem1_moe_revision import (
    DREM1Config,
    DraftConditionedMoEBlock,
    DraftStateController,
    aligned_generation_draft_indicator,
    pool_source_and_draft,
)


class FakeGate(nn.Module):
    def __init__(self, hidden: int, experts: int, top_k: int) -> None:
        super().__init__()
        self.hidden_dim = hidden
        self.num_experts = experts
        self.top_k = top_k
        self.norm_topk_prob = True
        self.weight = nn.Parameter(torch.randn(experts, hidden))

    def forward(self, hidden_states: torch.Tensor):
        logits = torch.nn.functional.linear(hidden_states, self.weight)
        probabilities = logits.softmax(dim=-1)
        weights, indices = probabilities.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return logits, weights, indices


class FakeExperts(nn.Module):
    def __init__(self, hidden: int, experts: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.randn(experts, hidden, hidden))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        output = torch.zeros_like(hidden_states)
        for token in range(hidden_states.shape[0]):
            for position in range(top_k_index.shape[1]):
                expert = top_k_index[token, position]
                output[token] += top_k_weights[token, position] * torch.mv(
                    self.weights[expert], hidden_states[token]
                )
        return output


class FakeBlock(nn.Module):
    def __init__(self, hidden: int, experts: int, top_k: int) -> None:
        super().__init__()
        self.gate = FakeGate(hidden, experts, top_k)
        self.experts = FakeExperts(hidden, experts)

    def forward(self, hidden_states):
        batch, sequence, hidden = hidden_states.shape
        flattened = hidden_states.reshape(-1, hidden)
        _, weights, indices = self.gate(flattened)
        return self.experts(flattened, indices, weights).reshape(batch, sequence, hidden)


class CharacterTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        del add_special_tokens
        output = {"input_ids": [ord(character) for character in text]}
        if return_offsets_mapping:
            output["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return output


def config() -> DREM1Config:
    return DREM1Config(
        hidden_size=8,
        num_experts=4,
        experts_per_token=2,
        controlled_layers=2,
        controller_width=6,
        adapter_rank=3,
        recurrent_steps=2,
    )


def test_source_draft_pooling_is_disjoint() -> None:
    features = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
    attention = torch.ones(1, 4)
    draft = torch.tensor([[0, 0, 1, 1]])
    source_mean, draft_mean = pool_source_and_draft(features, attention, draft)
    assert torch.equal(source_mean, features[:, :2].mean(dim=1))
    assert torch.equal(draft_mean, features[:, 2:].mean(dim=1))


def test_generation_indicator_aligns_under_left_padding() -> None:
    prompts = [
        "Internal draft:\nabc\n\nReturn x\n\nOriginal problem:y",
        "zzInternal draft:\ndefgh\n\nReturn x\n\nOriginal problem:y",
    ]
    width = max(map(len, prompts))
    ids = torch.zeros(2, width, dtype=torch.long)
    attention = torch.zeros(2, width, dtype=torch.long)
    for row, prompt in enumerate(prompts):
        encoded = torch.tensor([ord(character) for character in prompt])
        ids[row, -len(encoded) :] = encoded
        attention[row, -len(encoded) :] = 1
    indicator = aligned_generation_draft_indicator(
        CharacterTokenizer(), prompts, ids, attention
    )
    assert int(indicator[0].sum()) == 3
    assert int(indicator[1].sum()) == 5
    assert torch.equal(indicator.bool() & ~attention.bool(), torch.zeros_like(attention).bool())


def test_controller_emits_tied_recurrent_layer_states() -> None:
    controller = DraftStateController(config())
    features = torch.randn(2, 5, 8)
    attention = torch.ones(2, 5)
    draft = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 0, 0]])
    states = controller(features, attention, draft)
    assert len(states) == 2
    assert all(state.shape == (2, 6) for state in states)
    assert not torch.equal(states[0], states[1])
    masked = controller(features, attention, draft, context_control="draft_masked")
    assert not torch.equal(states[-1], masked[-1])


def test_zero_initialized_intervention_preserves_base_exactly() -> None:
    torch.manual_seed(3)
    base = FakeBlock(8, 4, 2)
    wrapped = DraftConditionedMoEBlock(base, config())
    hidden = torch.randn(2, 3, 8)
    expected = base(hidden)
    wrapped.set_controller_state(torch.randn(2, 6))
    actual = wrapped(hidden)
    assert torch.equal(actual, expected)
    assert all(not parameter.requires_grad for parameter in base.parameters())


def test_ablation_modes_freeze_inactive_intervention_parameters() -> None:
    router = DraftConditionedMoEBlock(FakeBlock(8, 4, 2), config())
    router.configure_trainable_mode("router_only")
    assert router.route_out.weight.requires_grad
    assert not router.expert_up.requires_grad
    expert = DraftConditionedMoEBlock(FakeBlock(8, 4, 2), config())
    expert.configure_trainable_mode("expert_only")
    assert expert.expert_up.requires_grad
    assert not expert.route_out.weight.requires_grad


def test_full_intervention_changes_router_and_selected_expert_output() -> None:
    torch.manual_seed(5)
    wrapped = DraftConditionedMoEBlock(FakeBlock(8, 4, 2), config())
    with torch.no_grad():
        wrapped.route_out.weight.fill_(2.0)
        wrapped.expert_up.fill_(0.1)
    hidden = torch.randn(2, 3, 8)
    state = torch.randn(2, 6)
    wrapped.set_controller_state(state, mode="full")
    full = wrapped(hidden)
    assert float(wrapped.last_metrics["route_probability_l1"].detach()) > 0
    wrapped.set_controller_state(state, mode="router_only")
    router_only = wrapped(hidden)
    wrapped.set_controller_state(state, mode="expert_only")
    expert_only = wrapped(hidden)
    assert not torch.equal(full, router_only)
    assert not torch.equal(full, expert_only)
