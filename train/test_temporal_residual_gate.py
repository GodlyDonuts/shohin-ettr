from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from temporal_residual_gate import (
    MultiTrajectoryResidualGate,
    MultiTrajectoryResidualGateConfig,
    MultiTrajectoryGatedProductModel,
    TemporalResidualGate,
    TemporalResidualGateConfig,
    TemporalResidualGateError,
    TemporalGatedProductModel,
    install_temporal_residual_gates,
    multi_trajectory_branch_layers,
    temporal_branch_layers,
)


class _Native(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.scale


def _block(weight: float = 0.25) -> TemporalResidualGate:
    return TemporalResidualGate(
        _Native(),
        TemporalResidualGateConfig(
            hidden_size=2,
            rank=1,
            alpha=1.0,
            initial_revision_weight=weight,
        ),
        owner_a=torch.tensor([[1.0, 0.0]]),
        owner_b=torch.tensor([[1.0], [0.0]]),
        revision_a=torch.tensor([[0.0, 1.0]]),
        revision_b=torch.tensor([[0.0], [1.0]]),
    )


def test_initial_gate_is_exact_output_space_interpolation() -> None:
    block = _block()
    hidden = torch.tensor([[[4.0, 8.0], [2.0, 6.0]]])
    observed = block(hidden)
    native = hidden * 2.0
    owner = torch.stack((hidden[..., 0], torch.zeros_like(hidden[..., 0])), dim=-1)
    revision = torch.stack((torch.zeros_like(hidden[..., 1]), hidden[..., 1]), dim=-1)
    expected = native + owner * 0.75 + revision * 0.25
    torch.testing.assert_close(observed, expected)
    assert block.receipt()["mean_revision_weight"] == pytest.approx(0.25)


def test_only_scalar_gate_surface_is_trainable() -> None:
    block = _block()
    trainable = {
        name: parameter
        for name, parameter in block.named_parameters()
        if parameter.requires_grad
    }
    assert set(trainable) == {"gate_weight", "gate_bias"}
    assert block.trainable_parameter_count() == 3
    assert not block.base.scale.requires_grad
    assert {name for name, _ in block.named_buffers()} == {
        "owner_a",
        "owner_b",
        "revision_a",
        "revision_b",
    }


def test_gate_receives_gradient_while_frozen_surfaces_do_not() -> None:
    block = _block()
    hidden = torch.tensor([[[4.0, 8.0], [2.0, 6.0]]])
    block(hidden).square().sum().backward()
    assert block.gate_weight.grad is not None
    assert block.gate_bias.grad is not None
    assert torch.isfinite(block.gate_weight.grad).all()
    assert torch.isfinite(block.gate_bias.grad).all()
    assert float(block.gate_weight.grad.norm()) > 0.0
    assert float(block.gate_bias.grad.norm()) > 0.0
    assert block.base.scale.grad is None


def test_gate_learns_conditional_revision_beyond_any_global_blend() -> None:
    block = _block(weight=0.5)
    with torch.no_grad():
        block.base.scale.zero_()
    hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    target = hidden.clone()
    global_blend = block(hidden).detach()
    global_loss = torch.nn.functional.mse_loss(global_blend, target)
    optimizer = torch.optim.AdamW(
        [block.gate_weight, block.gate_bias], lr=0.1, weight_decay=0.0
    )
    for _ in range(500):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(block(hidden), target)
        loss.backward()
        optimizer.step()
    learned = block(hidden).detach()
    learned_loss = torch.nn.functional.mse_loss(learned, target)
    assert float(learned_loss) < 1e-4
    assert float(learned_loss) < float(global_loss) / 1000.0
    gates = block._gate(hidden).detach()
    assert float(gates[0, 0, 0]) < 0.02
    assert float(gates[0, 1, 0]) > 0.98


def test_gate_can_recover_owner_or_revision_endpoints() -> None:
    block = _block()
    hidden = torch.tensor([[[4.0, 8.0]]])
    with torch.no_grad():
        block.gate_bias.fill_(-30.0)
    owner_output = block(hidden)
    torch.testing.assert_close(owner_output, torch.tensor([[[12.0, 16.0]]]))
    with torch.no_grad():
        block.gate_bias.fill_(30.0)
    revision_output = block(hidden)
    torch.testing.assert_close(revision_output, torch.tensor([[[8.0, 24.0]]]))


def test_gate_rejects_invalid_config_or_branch_geometry() -> None:
    with pytest.raises(TemporalResidualGateError):
        TemporalResidualGateConfig(2, 1, 1.0, 1.0).validate()
    with pytest.raises(TemporalResidualGateError):
        TemporalResidualGate(
            _Native(),
            TemporalResidualGateConfig(2, 1, 1.0),
            owner_a=torch.zeros(2, 2),
            owner_b=torch.zeros(2, 1),
            revision_a=torch.zeros(1, 2),
            revision_b=torch.zeros(2, 1),
        )


def _multi_block(
    router_features: str = "hidden_only",
    routing_structure: str = "flat",
) -> MultiTrajectoryResidualGate:
    return MultiTrajectoryResidualGate(
        _Native(),
        MultiTrajectoryResidualGateConfig(
            hidden_size=2,
            rank=1,
            alpha=1.0,
            branch_names=("owner", "revision", "draft_hidden"),
            initial_weights=(0.8, 0.1, 0.1),
            router_features=router_features,
            routing_structure=routing_structure,
        ),
        branches={
            "owner": (
                torch.tensor([[1.0, 0.0]]),
                torch.tensor([[1.0], [0.0]]),
            ),
            "revision": (
                torch.tensor([[0.0, 1.0]]),
                torch.tensor([[0.0], [1.0]]),
            ),
            "draft_hidden": (
                torch.tensor([[1.0, 1.0]]),
                torch.tensor([[0.5], [0.5]]),
            ),
        },
    )


def test_multi_trajectory_gate_starts_at_exact_categorical_mix() -> None:
    block = _multi_block()
    hidden = torch.tensor([[[4.0, 8.0]]])
    native = hidden * 2.0
    owner = torch.tensor([[[4.0, 0.0]]])
    revision = torch.tensor([[[0.0, 8.0]]])
    draft_hidden = torch.tensor([[[6.0, 6.0]]])
    expected = native + owner * 0.8 + revision * 0.1 + draft_hidden * 0.1
    torch.testing.assert_close(block(hidden), expected)
    assert block.receipt()["mean_branch_weights"] == pytest.approx(
        {"owner": 0.8, "revision": 0.1, "draft_hidden": 0.1}
    )
    assert block.trainable_parameter_count() == 9
    assert not block.base.scale.requires_grad


def test_multi_trajectory_supervision_moves_probability_to_selected_branch() -> None:
    block = _multi_block()
    hidden = torch.tensor([[[1.0, -1.0], [0.5, 0.5]]])
    response_mask = torch.tensor([[1, 0]])
    initial = block._gate(hidden).detach()[0, 0, 2]
    optimizer = torch.optim.AdamW(
        [block.gate_weight, block.gate_bias], lr=0.1, weight_decay=0.0
    )
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        block(hidden)
        loss = block.routing_supervision_loss((0.0, 0.0, 1.0), response_mask)
        loss.backward()
        optimizer.step()
    assert block._gate(hidden).detach()[0, 0, 2] > initial + 0.8


def test_correct_set_mass_supervision_preserves_any_correct_branch() -> None:
    block = _multi_block()
    hidden = torch.tensor([[[1.0, -1.0]]])
    response_mask = torch.tensor([[1]])
    block(hidden)
    all_correct = block.routing_supervision_loss(
        (1.0 / 3.0,) * 3,
        response_mask,
        objective="correct_set_mass",
    )
    owner_or_revision = block.routing_supervision_loss(
        (0.5, 0.5, 0.0),
        response_mask,
        objective="correct_set_mass",
    )
    soft_cross_entropy = block.routing_supervision_loss((0.5, 0.5, 0.0), response_mask)
    torch.testing.assert_close(all_correct, torch.tensor(0.0))
    torch.testing.assert_close(owner_or_revision, -torch.log(torch.tensor(0.9)))
    assert soft_cross_entropy > owner_or_revision
    with pytest.raises(TemporalResidualGateError):
        block.routing_supervision_loss(
            (1.0, 0.0, 0.0), response_mask, objective="unknown"
        )


def test_geometry_router_preserves_initial_mix_and_exposes_disagreement() -> None:
    block = _multi_block("trajectory_geometry")
    hidden = torch.tensor([[[4.0, 8.0], [2.0, -1.0]]])
    residuals = block._residuals(hidden)
    features = block._geometry_features(hidden, residuals)
    assert features.shape == (1, 2, 9)
    assert torch.isfinite(features).all()
    torch.testing.assert_close(
        block._gate(hidden, residuals),
        torch.tensor([[[0.8, 0.1, 0.1], [0.8, 0.1, 0.1]]]),
    )
    assert block.geometry_weight is not None
    assert block.trainable_parameter_count() == 36


def test_geometry_router_learns_nonlinear_trajectory_conflict() -> None:
    block = _multi_block("trajectory_geometry")
    hidden = torch.tensor([[[0.5, 0.0], [-0.5, 0.0], [2.0, 0.0], [-2.0, 0.0]]])
    residuals = block._residuals(hidden).detach()
    with torch.no_grad():
        block.gate_weight.zero_()
    block.gate_weight.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [block.geometry_weight, block.gate_bias], lr=0.1, weight_decay=0.0
    )
    target = torch.tensor([0, 0, 2, 2])
    initial = block._gate(hidden, residuals).detach()
    for _ in range(150):
        optimizer.zero_grad(set_to_none=True)
        probabilities = block._gate(hidden, residuals)
        loss = torch.nn.functional.nll_loss(
            probabilities.squeeze(0).clamp_min(1.0e-8).log(), target
        )
        loss.backward()
        optimizer.step()
    learned = block._gate(hidden, residuals).detach()
    assert learned[0, :2, 0].min() > 0.98
    assert learned[0, 2:, 2].min() > initial[0, 2:, 2].max() + 0.8

    hidden_only = _multi_block()
    optimizer = torch.optim.AdamW(
        [hidden_only.gate_weight, hidden_only.gate_bias],
        lr=0.1,
        weight_decay=0.0,
    )
    for _ in range(500):
        optimizer.zero_grad(set_to_none=True)
        probabilities = hidden_only._gate(hidden)
        loss = torch.nn.functional.nll_loss(
            probabilities.squeeze(0).clamp_min(1.0e-8).log(), target
        )
        loss.backward()
        optimizer.step()
    hidden_predictions = hidden_only._gate(hidden).argmax(dim=-1).squeeze(0)
    assert int(hidden_predictions.eq(target).sum()) <= 3


def test_geometry_router_requires_residuals_and_valid_mode() -> None:
    block = _multi_block("trajectory_geometry")
    with pytest.raises(TemporalResidualGateError, match="residuals are absent"):
        block._gate(torch.ones(1, 1, 2))
    with pytest.raises(TemporalResidualGateError):
        MultiTrajectoryResidualGateConfig(
            2,
            1,
            1.0,
            ("owner", "revision"),
            (0.9, 0.1),
            "unknown",
        ).validate()


def test_hierarchical_router_preserves_mix_and_decouples_sibling_choice() -> None:
    block = _multi_block("trajectory_geometry", "hierarchical")
    hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    residuals = block._residuals(hidden)
    initial = block._gate(hidden, residuals)
    torch.testing.assert_close(
        initial,
        torch.tensor([[[0.8, 0.1, 0.1], [0.8, 0.1, 0.1]]]),
    )
    assert block.trainable_parameter_count() == 24
    with torch.no_grad():
        block.gate_weight.zero_()
        block.geometry_weight.zero_()
        block.gate_bias[0] = -30.0
        block.gate_bias[1] = 30.0
    revision = block._gate(hidden, residuals)
    assert revision[..., 0].max() < 1e-6
    assert revision[..., 1].min() > 0.999
    with torch.no_grad():
        block.gate_bias[1] = -30.0
    draft = block._gate(hidden, residuals)
    assert draft[..., 2].min() > 0.999


def test_hierarchical_router_requires_exact_role_semantics() -> None:
    with pytest.raises(TemporalResidualGateError):
        MultiTrajectoryResidualGateConfig(
            2,
            1,
            1.0,
            ("revision", "draft_hidden"),
            (0.9, 0.1),
            "trajectory_geometry",
            "hierarchical",
        ).validate()


def test_multi_trajectory_gate_rejects_branch_order_or_invalid_target() -> None:
    config = MultiTrajectoryResidualGateConfig(
        2, 1, 1.0, ("owner", "revision"), (0.9, 0.1)
    )
    branch = (torch.zeros(1, 2), torch.zeros(2, 1))
    with pytest.raises(TemporalResidualGateError):
        MultiTrajectoryResidualGate(
            _Native(), config, branches={"revision": branch, "owner": branch}
        )
    block = _multi_block()
    block(torch.ones(1, 1, 2))
    with pytest.raises(TemporalResidualGateError):
        block.routing_supervision_loss((1.0, 0.0), torch.ones(1, 1))


class _DecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _Native()


class _TextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 2)
        self.layers = nn.ModuleList([_DecoderLayer() for _ in range(3)])


def _role_state(offset: float) -> dict[str, torch.Tensor]:
    state = {}
    for index in (1, 2):
        state[f"backbone.model.layers.{index}.mlp.adapter_a.weight"] = torch.tensor(
            [[1.0 + offset, 2.0 + offset]], dtype=torch.float32
        )
        state[f"backbone.model.layers.{index}.mlp.adapter_b.weight"] = torch.tensor(
            [[3.0 + offset], [4.0 + offset]], dtype=torch.float32
        )
    return state


def test_real_role_state_mapping_and_final_layer_installation() -> None:
    config = TemporalResidualGateConfig(2, 1, 1.0)
    owner = _role_state(0.0)
    revision = _role_state(0.5)
    branches = temporal_branch_layers(owner, revision, config, (1, 2))
    assert set(branches) == {1, 2}
    assert set(branches[1]) == {"owner_a", "owner_b", "revision_a", "revision_b"}
    model = _TextModel()
    native = model.layers[0].mlp
    blocks = install_temporal_residual_gates(model, owner, revision, config, (1, 2))
    assert model.layers[0].mlp is native
    assert tuple(model.layers[index].mlp for index in (1, 2)) == blocks
    assert all(isinstance(block, TemporalResidualGate) for block in blocks)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    assert sum(parameter.numel() for parameter in trainable) == 6


def test_real_role_state_mapping_rejects_missing_or_nonfinite_tensor() -> None:
    config = TemporalResidualGateConfig(2, 1, 1.0)
    owner = _role_state(0.0)
    revision = _role_state(0.5)
    revision.pop("backbone.model.layers.2.mlp.adapter_b.weight")
    with pytest.raises(TemporalResidualGateError):
        temporal_branch_layers(owner, revision, config, (1, 2))
    revision = _role_state(0.5)
    revision["backbone.model.layers.2.mlp.adapter_b.weight"][0, 0] = float("nan")
    with pytest.raises(TemporalResidualGateError):
        temporal_branch_layers(owner, revision, config, (1, 2))


def test_multi_trajectory_product_installs_three_real_role_states() -> None:
    config = MultiTrajectoryResidualGateConfig(
        2,
        1,
        1.0,
        ("owner", "revision", "draft_hidden"),
        (0.8, 0.1, 0.1),
    )
    states = {
        "owner": _role_state(0.0),
        "revision": _role_state(0.5),
        "draft_hidden": _role_state(1.0),
    }
    layers = multi_trajectory_branch_layers(states, config, (1, 2))
    assert tuple(layers[1]) == config.branch_names
    text_model = _TextModel()
    backbone = nn.Module()
    backbone.model = text_model
    model = MultiTrajectoryGatedProductModel(
        backbone,
        text_model,
        nn.Linear(2, 8, bias=False),
        config,
        role_states=states,
        controlled_layer_indices=(1, 2),
    )
    assert model.trainable_parameter_count() == 18
    assert len(model.trainable_parameter_name_sha256()) == 64
    assert model.routing_receipt() == {
        "branch_names": ["owner", "revision", "draft_hidden"],
        "controlled_layer_indices": [1, 2],
        "layers": [{"tokens": 0}, {"tokens": 0}],
    }
    ids = torch.tensor([[0, 1, 2]])
    attention = torch.ones_like(ids)
    model.prepare_generation_draft_attention(object(), ["prompt"], ids, attention)
    embeddings, observed_attention = model.generation_embeddings(ids, attention)
    assert embeddings.shape == (1, 3, 2)
    assert torch.equal(observed_attention, attention)


def test_product_surface_exposes_exact_trainables_and_generation_state() -> None:
    text_model = _TextModel()
    backbone = nn.Module()
    backbone.model = text_model
    lm_head = nn.Linear(2, 8, bias=False)
    model = TemporalGatedProductModel(
        backbone,
        text_model,
        lm_head,
        TemporalResidualGateConfig(2, 1, 1.0),
        owner_state=_role_state(0.0),
        revision_state=_role_state(0.5),
        controlled_layer_indices=(1, 2),
    )
    assert model.trainable_parameter_count() == 6
    assert len(model.trainable_parameter_name_sha256()) == 64
    assert all(not parameter.requires_grad for parameter in model.lm_head.parameters())
    assert model.routing_receipt() == {
        "controlled_layer_indices": [1, 2],
        "layers": [{"tokens": 0}, {"tokens": 0}],
    }
    ids = torch.tensor([[0, 1, 2], [3, 4, 0]])
    attention = torch.tensor([[1, 1, 1], [1, 1, 0]])
    model.prepare_generation_draft_attention(
        object(), ["prompt one", "prompt two"], ids, attention
    )
    assert model.generation_position_ids().tolist() == [[0, 1, 2], [0, 1, 0]]
    embeddings, observed_attention = model.generation_embeddings(ids, attention)
    assert embeddings.shape == (2, 3, 2)
    assert torch.equal(observed_attention, attention)
    with pytest.raises(TemporalResidualGateError):
        model.generation_embeddings(ids.flip(1), attention)


def test_routing_supervision_drives_owner_and_revision_extremes() -> None:
    block = _block(weight=0.1)
    hidden = torch.tensor([[[1.0, -1.0], [0.5, 0.5]]])
    attention = torch.tensor([[1, 0]])
    block(hidden)
    owner_loss = block.routing_supervision_loss(0.0, attention)
    revision_loss = block.routing_supervision_loss(1.0, attention)
    assert owner_loss < revision_loss
    (owner_loss + revision_loss).backward()
    assert block.gate_weight.grad is not None
    with pytest.raises(TemporalResidualGateError):
        block.routing_supervision_loss(0.0, torch.ones(1, 3))
