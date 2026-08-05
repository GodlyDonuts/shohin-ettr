import torch
import torch.nn.functional as F

from counterexample_guided_revision import RevisionConfig
from query_valued_revision import QueryValuedRevisionCore


def _inputs(config: RevisionConfig, batch: int = 4, evidence: int = 7):
    torch.manual_seed(53)
    source = torch.randn(batch, evidence, config.width)
    mask = torch.ones(batch, evidence, dtype=torch.bool)
    outcomes = torch.randint(config.outcome_classes, (batch, evidence))
    query = torch.randn(batch, config.width)
    return source, mask, source.clone(), outcomes, mask.clone(), query


def test_value_revision_has_hard_forward_budget_and_selector_gradient() -> None:
    config = RevisionConfig(width=24, heads=3, slots=6, rounds=2)
    model = QueryValuedRevisionCore(config, "utility")
    logits, trajectory = model(*_inputs(config))
    loss = F.cross_entropy(
        logits, torch.randint(config.answer_classes, (logits.shape[0],))
    )
    loss.backward()
    assert trajectory.steps[0].selected_probe.shape[-1] == config.probes_per_round
    assert trajectory.steps[0].slot_mask.sum(-1).eq(config.revision_slots).all()
    gradient = model.selector.output.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_value_revision_arms_are_parameter_matched() -> None:
    config = RevisionConfig(width=24, heads=3)
    counts = {
        arm: sum(
            parameter.numel()
            for parameter in QueryValuedRevisionCore(config, arm).parameters()
        )
        for arm in ("utility", "fixed", "residual")
    }
    assert len(set(counts.values())) == 1


def test_selector_query_controls_utility_but_not_fixed_schedule() -> None:
    config = RevisionConfig(width=24, heads=3, rounds=2)
    inputs = _inputs(config)
    utility = QueryValuedRevisionCore(config, "utility")
    normal = utility.deliberate(*inputs)
    shuffled = utility.deliberate(*inputs, shuffle_selector_query=True)
    assert any(
        not torch.equal(left.selected_probe, right.selected_probe)
        for left, right in zip(normal.steps, shuffled.steps, strict=True)
    )

    fixed = QueryValuedRevisionCore(config, "fixed")
    normal_fixed = fixed.deliberate(*inputs)
    shuffled_fixed = fixed.deliberate(*inputs, shuffle_selector_query=True)
    assert all(
        torch.equal(left.selected_probe, right.selected_probe)
        for left, right in zip(normal_fixed.steps, shuffled_fixed.steps, strict=True)
    )
