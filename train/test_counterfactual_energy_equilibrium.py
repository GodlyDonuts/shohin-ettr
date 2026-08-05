import inspect

import torch
import torch.nn.functional as F

from counterexample_guided_revision import RevisionConfig
from counterfactual_energy_equilibrium import (
    CounterfactualEnergyEquilibriumCore,
    PositiveEnergyPreconditioner,
)


def _inputs(config: RevisionConfig, batch: int = 4, evidence: int = 7):
    torch.manual_seed(67)
    source = torch.randn(batch, evidence, config.width)
    mask = torch.ones(batch, evidence, dtype=torch.bool)
    outcomes = torch.randint(config.outcome_classes, (batch, evidence))
    query = torch.randn(batch, config.width)
    return source, mask, source.clone(), outcomes, mask.clone(), query


def test_positive_preconditioner_never_reverses_gradient() -> None:
    config = RevisionConfig(width=24, heads=3)
    preconditioner = PositiveEnergyPreconditioner(config)
    state = torch.randn(5, config.slots, config.width)
    gradient = torch.randn_like(state)
    next_state, _ = preconditioner(state, gradient)
    directional_change = ((next_state - state) * gradient).sum((-2, -1))
    assert directional_change.le(0).all()


def test_energy_core_is_finite_with_second_order_gradient() -> None:
    config = RevisionConfig(width=24, heads=3, slots=6, rounds=2)
    model = CounterfactualEnergyEquilibriumCore(config, "energy")
    logits, trajectory = model(*_inputs(config))
    target = torch.randint(config.answer_classes, (logits.shape[0],))
    loss = F.cross_entropy(logits, target) + trajectory.final_evidence_energy.mean()
    loss.backward()
    gradient = model.energy_preconditioner.scale[0].weight.grad
    assert torch.isfinite(loss)
    assert gradient is not None
    assert torch.isfinite(gradient).all()


def test_zero_gradient_freezes_energy_state_and_query_is_late() -> None:
    config = RevisionConfig(width=24, heads=3, slots=6, rounds=2)
    model = CounterfactualEnergyEquilibriumCore(config, "energy")
    source, mask, probes, outcomes, evidence_mask, _ = _inputs(config)
    trajectory = model.deliberate(
        source,
        mask,
        probes,
        outcomes,
        evidence_mask,
        zero_energy_gradient=True,
    )
    assert torch.allclose(trajectory.initial_state, trajectory.final_state)
    assert "query" not in inspect.signature(model.deliberate).parameters


def test_energy_and_recurrent_arms_are_parameter_matched() -> None:
    config = RevisionConfig(width=24, heads=3)
    counts = {
        arm: sum(
            parameter.numel()
            for parameter in CounterfactualEnergyEquilibriumCore(
                config, arm
            ).parameters()
        )
        for arm in ("energy", "recurrent")
    }
    assert len(set(counts.values())) == 1
