from __future__ import annotations

import pytest
import torch

from diverge_vcr1_workspace import (
    TemporalCorrectionConfig,
    TemporalCorrectionReactor,
    VCR1WorkspaceError,
)


def _inputs():
    torch.manual_seed(7)
    memory = torch.randn(2, 7, 12)
    active = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool
    )
    question = torch.tensor(
        [[1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool
    )
    draft = torch.tensor(
        [[0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 0, 0]], dtype=torch.bool
    )
    return memory, active, question, draft


def test_temporal_correction_shapes_and_gradients() -> None:
    config = TemporalCorrectionConfig(
        backbone_width=12,
        workspace_width=16,
        workspace_slots=3,
        recurrent_steps=3,
        attention_heads=4,
        ff_multiplier=2,
    )
    reactor = TemporalCorrectionReactor(config)
    output = reactor(*_inputs())
    assert output.prefix_states.shape == (2, 3, 12)
    assert output.validity_logits.shape == (2,)
    assert output.correction_strength.shape == (2,)
    assert output.step_delta_norms.shape == (2, 3)
    output.prefix_states.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in reactor.parameters()
    )


def test_role_blind_is_exactly_invariant_to_segment_swap() -> None:
    reactor = TemporalCorrectionReactor(
        TemporalCorrectionConfig(
            backbone_width=12,
            workspace_width=16,
            workspace_slots=3,
            recurrent_steps=2,
            attention_heads=4,
            ff_multiplier=2,
        )
    ).eval()
    memory, active, question, draft = _inputs()
    first = reactor(memory, active, question, draft, role_blind=True).prefix_states
    second = reactor(memory, active, draft, question, role_blind=True).prefix_states
    assert torch.equal(first, second)


def test_role_swap_and_reset_are_causal_controls() -> None:
    reactor = TemporalCorrectionReactor(
        TemporalCorrectionConfig(
            backbone_width=12,
            workspace_width=16,
            workspace_slots=3,
            recurrent_steps=2,
            attention_heads=4,
            ff_multiplier=2,
        )
    ).eval()
    memory, active, question, draft = _inputs()
    treatment = reactor(memory, active, question, draft).prefix_states
    swapped = reactor(memory, active, question, draft, swap_roles=True).prefix_states
    reset = reactor(memory, active, question, draft, reset_prefix=True).prefix_states
    assert not torch.equal(treatment, swapped)
    assert torch.count_nonzero(reset) == 0


def test_invalid_masks_fail_closed() -> None:
    reactor = TemporalCorrectionReactor(
        TemporalCorrectionConfig(backbone_width=12, workspace_width=16)
    )
    memory, active, question, draft = _inputs()
    with pytest.raises(VCR1WorkspaceError, match="overlap"):
        reactor(memory, active, question, question)
    with pytest.raises(VCR1WorkspaceError, match="needs question"):
        reactor(memory, active, torch.zeros_like(question), draft)


def test_invalid_width_is_rejected() -> None:
    with pytest.raises(VCR1WorkspaceError, match="divide"):
        TemporalCorrectionConfig(
            backbone_width=12,
            workspace_width=15,
            attention_heads=4,
        )
