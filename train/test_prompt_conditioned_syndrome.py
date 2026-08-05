import torch

from prompt_conditioned_syndrome import (
    MinimumNormSyndromeProjector,
    PromptConditionedCheckCompiler,
    PromptConditionedSyndromeCore,
    SyndromeConfig,
    syndrome,
)


def _fixture() -> tuple[SyndromeConfig, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    config = SyndromeConfig(
        input_width=12,
        state_width=12,
        slots=6,
        checks=3,
        heads=3,
        steps=4,
        min_steps=1,
        ff_multiplier=2,
    )
    source = torch.randn(3, 9, config.input_width)
    mask = torch.ones(3, 9, dtype=torch.bool)
    mask[1, -2:] = False
    state = torch.randn(3, config.slots, config.state_width)
    return config, source, mask, state


def test_compiler_emits_orthonormal_sticky_factors() -> None:
    config, source, mask, state = _fixture()
    checks = PromptConditionedCheckCompiler(config)(source, mask, state)
    slot_gram = torch.einsum(
        "bcs,bds->bcd", checks.slot_factors, checks.slot_factors
    )
    feature_gram = checks.feature_factors @ checks.feature_factors.T
    identity = torch.eye(config.checks)
    assert torch.allclose(slot_gram, identity.expand_as(slot_gram), atol=1e-5)
    assert torch.allclose(feature_gram, identity, atol=1e-5)
    assert torch.allclose(
        checks.reference_syndrome,
        syndrome(state, checks.slot_factors, checks.feature_factors),
    )


def test_projection_removes_prompt_specific_syndrome_error() -> None:
    config, source, mask, state = _fixture()
    checks = PromptConditionedCheckCompiler(config)(source, mask, state)
    proposed = state + 0.5 * torch.randn_like(state)
    corrected, pre, post, correction = MinimumNormSyndromeProjector(config)(
        proposed, checks
    )
    assert pre.square().mean().sqrt() > 1e-3
    assert post.square().mean().sqrt() < pre.square().mean().sqrt() / 1000
    assert correction.square().mean().sqrt() > 0
    assert corrected.shape == state.shape


def test_projection_is_differentiable() -> None:
    config, source, mask, state = _fixture()
    compiler = PromptConditionedCheckCompiler(config)
    projector = MinimumNormSyndromeProjector(config)
    source.requires_grad_()
    state.requires_grad_()
    checks = compiler(source, mask, state)
    corrected, _, _, _ = projector(state + 0.1, checks)
    corrected.square().mean().backward()
    assert source.grad is not None and torch.isfinite(source.grad).all()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert compiler.slot_logits.weight.grad is not None


def test_check_geometry_changes_with_source() -> None:
    config, source, mask, state = _fixture()
    compiler = PromptConditionedCheckCompiler(config)
    first = compiler(source, mask, state)
    second = compiler(source.roll(1, 0), mask.roll(1, 0), state)
    assert not torch.allclose(first.slot_factors, second.slot_factors)


def test_complete_core_is_deterministic_and_keeps_checks() -> None:
    config, source, mask, _ = _fixture()
    torch.manual_seed(19)
    core = PromptConditionedSyndromeCore(config).eval()
    first = core(source, mask)
    second = core(source, mask)
    assert len(first.steps) == config.steps
    assert torch.equal(first.stop_step, second.stop_step)
    assert torch.allclose(first.final_state, second.final_state)
    final_error = syndrome(
        first.final_state,
        first.checks.slot_factors,
        first.checks.feature_factors,
    ) - first.checks.reference_syndrome
    assert final_error.square().mean().sqrt() < 1e-4
