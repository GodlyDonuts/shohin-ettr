from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from temporal_residual_gate import (
    TemporalResidualGate,
    TemporalResidualGateConfig,
    TemporalResidualGateError,
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
