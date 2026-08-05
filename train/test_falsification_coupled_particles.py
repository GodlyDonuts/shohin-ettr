import inspect

import torch

from falsification_coupled_particles import (
    FalsificationCoupledParticleCore,
    ParticleConfig,
    ParticleDynamicsError,
    behaviorally_equivalent,
    gather_complete_states,
)


def _inputs(config: ParticleConfig, batch: int = 3, evidence: int = 6):
    torch.manual_seed(7)
    source = torch.randn(batch, evidence, config.width)
    mask = torch.ones(batch, evidence, dtype=torch.bool)
    outcomes = torch.randint(config.outcome_classes, (batch, evidence))
    query = torch.randn(batch, config.width)
    return source, mask, source.clone(), outcomes, mask.clone(), query


def test_whole_state_gather_never_mixes_fields() -> None:
    states = torch.arange(2 * 5 * 3 * 4).view(2, 5, 3, 4)
    indices = torch.tensor([[4, 1], [0, 3]])
    selected = gather_complete_states(states, indices)
    for batch in range(2):
        for output in range(2):
            assert torch.equal(selected[batch, output], states[batch, indices[batch, output]])


def test_fcpt_is_finite_and_retains_real_lineages() -> None:
    config = ParticleConfig(width=24, heads=3, rounds=2, outcome_classes=5, answer_classes=7)
    model = FalsificationCoupledParticleCore(config, "fcpt")
    inputs = _inputs(config)
    logits, trajectory = model(*inputs)
    assert logits.shape == (3, config.answer_classes)
    assert torch.isfinite(logits).all()
    assert len(trajectory.rounds) == config.rounds
    assert trajectory.final_lineage.ge(0).all()
    loss = logits.square().mean() + trajectory.final_log_weight.mean()
    loss.backward()
    assert model.proposer.branch_seed.grad is not None
    assert torch.isfinite(model.proposer.branch_seed.grad).all()


def test_fcpt_selection_is_particle_permutation_invariant() -> None:
    config = ParticleConfig(width=16, heads=4, particles=4, branches=2, rounds=1)
    model = FalsificationCoupledParticleCore(config, "fcpt")
    batch, candidates = 2, config.particles * config.branches
    torch.manual_seed(11)
    states = torch.randn(batch, candidates, config.slots, config.width)
    scores = torch.randn(batch, candidates)
    lineage = torch.arange(candidates)[None].expand(batch, -1)
    selected, _ = model._whole_select(states, scores, lineage, config.particles)
    permutation = torch.tensor([6, 2, 1, 7, 4, 0, 5, 3])
    permuted, _ = model._whole_select(
        states[:, permutation], scores[:, permutation], lineage[:, permutation], config.particles
    )
    assert torch.equal(selected.lineage, permuted.lineage)
    assert torch.allclose(selected.state, permuted.state)


def test_soft_control_erases_lineage_while_fcpt_does_not() -> None:
    config = ParticleConfig(width=16, heads=4, rounds=1)
    inputs = _inputs(config, batch=2)
    fcpt = FalsificationCoupledParticleCore(config, "fcpt")
    soft = FalsificationCoupledParticleCore(config, "soft")
    _, full_trajectory = fcpt(*inputs)
    _, soft_trajectory = soft(*inputs)
    assert full_trajectory.final_lineage.ge(0).all()
    assert soft_trajectory.final_lineage.eq(-1).all()


def test_query_is_absent_from_deliberation_api() -> None:
    config = ParticleConfig(width=16, heads=4, rounds=1)
    model = FalsificationCoupledParticleCore(config, "fcpt")
    source, mask, probes, outcomes, evidence_mask, query = _inputs(config, batch=2)
    trajectory = model.deliberate(source, mask, probes, outcomes, evidence_mask)
    assert "query" not in inspect.signature(model.deliberate).parameters
    first = model.read_answer(trajectory.final_state, query)
    second = model.read_answer(trajectory.final_state, -query)
    assert not torch.allclose(first, second)


def test_behavioral_equivalence_uses_consequences_not_coordinates() -> None:
    first = torch.tensor([[[4.0, 0.0], [0.0, 3.0], [2.0, 1.0]]])
    same_behavior = first * 7.0
    different = first.clone()
    different[:, 1] = torch.tensor([5.0, 0.0])
    mask = torch.tensor([[True, True, False]])
    assert behaviorally_equivalent(first, same_behavior, mask).item()
    assert not behaviorally_equivalent(first, different, mask).item()
    try:
        behaviorally_equivalent(first, different, mask.float())
    except ParticleDynamicsError:
        pass
    else:
        raise AssertionError("non-boolean equivalence mask was accepted")
