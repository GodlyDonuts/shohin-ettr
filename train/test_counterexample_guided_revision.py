import inspect

import torch

from counterexample_guided_revision import (
    CounterexampleGuidedRevisionCore,
    RevisionConfig,
)


def _inputs(config: RevisionConfig, batch: int = 3, evidence: int = 7):
    torch.manual_seed(41)
    source = torch.randn(batch, evidence, config.width)
    mask = torch.ones(batch, evidence, dtype=torch.bool)
    outcomes = torch.randint(config.outcome_classes, (batch, evidence))
    query = torch.randn(batch, config.width)
    return source, mask, source.clone(), outcomes, mask.clone(), query


def test_sparse_revision_changes_only_budgeted_slots() -> None:
    config = RevisionConfig(width=24, heads=3, slots=6, revision_slots=2, rounds=1)
    model = CounterexampleGuidedRevisionCore(config, "guided")
    _, trajectory = model(*_inputs(config))
    changed = trajectory.final_state.ne(trajectory.initial_state).any(-1)
    assert changed.sum(-1).eq(config.revision_slots).all()
    assert trajectory.steps[0].slot_mask.sum(-1).eq(config.revision_slots).all()


def test_guided_and_fixed_have_identical_parameters() -> None:
    config = RevisionConfig(width=24, heads=3)
    guided = CounterexampleGuidedRevisionCore(config, "guided")
    fixed = CounterexampleGuidedRevisionCore(config, "fixed")
    guided_count = sum(parameter.numel() for parameter in guided.parameters())
    fixed_count = sum(parameter.numel() for parameter in fixed.parameters())
    assert guided_count == fixed_count


def test_revision_is_finite_and_differentiable() -> None:
    config = RevisionConfig(width=24, heads=3, rounds=2)
    model = CounterexampleGuidedRevisionCore(config, "guided")
    logits, trajectory = model(*_inputs(config))
    loss = logits.square().mean() + trajectory.final_state.square().mean()
    loss.backward()
    assert torch.isfinite(logits).all()
    assert model.revision.gate.weight.grad is not None
    assert torch.isfinite(model.revision.gate.weight.grad).all()


def test_query_is_late_and_shuffled_outcomes_change_revision() -> None:
    config = RevisionConfig(width=24, heads=3, rounds=2)
    model = CounterexampleGuidedRevisionCore(config, "guided")
    source, mask, probes, outcomes, evidence_mask, query = _inputs(config)
    normal = model.deliberate(source, mask, probes, outcomes, evidence_mask)
    shuffled = model.deliberate(
        source, mask, probes, outcomes, evidence_mask, shuffle_outcomes=True
    )
    assert "query" not in inspect.signature(model.deliberate).parameters
    assert not torch.allclose(normal.final_state, shuffled.final_state)
    first = model.read_answer(normal.final_state, query)
    second = model.read_answer(normal.final_state, -query)
    assert not torch.allclose(first, second)
