from dataclasses import replace

import torch
import torch.nn.functional as F

from falsification_coupled_particles import ParticleConfig
from fcpt_plural_reasoning import (
    BoardConfig,
    EpisodeEncoder,
    FAMILIES,
    PluralReasoner,
    batch_sha256,
    behavior_loss,
    behavioral_diversity,
    generate_batch,
)


def test_all_families_are_deterministic_and_in_range() -> None:
    config = BoardConfig(width=24)
    hashes = []
    for family in range(len(FAMILIES)):
        first = generate_batch(16, 4, config, seed=71, family=family)
        second = generate_batch(16, 4, config, seed=71, family=family)
        assert batch_sha256(first) == batch_sha256(second)
        assert first.answer.min() >= 0
        assert first.answer.max() < config.modulus
        assert first.evidence_mask.any(-1).all()
        hashes.append(batch_sha256(first))
    assert len(set(hashes)) == len(FAMILIES)


def test_mixed_batch_cycles_families() -> None:
    config = BoardConfig(width=24)
    batch = generate_batch(12, 3, config, seed=13)
    assert set(batch.family.tolist()) == {0, 1, 2}


def test_episode_encoder_does_not_embed_outcomes() -> None:
    torch.manual_seed(17)
    config = BoardConfig(width=24)
    encoder = EpisodeEncoder(config)
    batch = generate_batch(4, 3, config, seed=19)
    source, probes, query = encoder(batch)
    altered = replace(batch, outcomes=(batch.outcomes + 1) % config.modulus)
    changed_source, changed_probes, changed_query = encoder(altered)
    assert torch.equal(source, changed_source)
    assert torch.equal(probes, changed_probes)
    assert torch.equal(query, changed_query)


def test_plural_reasoner_one_step_is_finite() -> None:
    torch.manual_seed(23)
    board = BoardConfig(width=24)
    particles = ParticleConfig(
        width=24,
        heads=3,
        rounds=2,
        outcome_classes=board.modulus,
        answer_classes=board.modulus,
    )
    model = PluralReasoner(board, particles, "fcpt")
    batch = generate_batch(6, 3, board, seed=29)
    logits, trajectory = model(batch)
    loss = (
        F.cross_entropy(logits, batch.answer)
        + 0.5 * behavior_loss(trajectory, batch)
        - 0.05 * behavioral_diversity(trajectory, batch)
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.core.proposer.branch_seed.grad is not None
    assert torch.isfinite(model.core.proposer.branch_seed.grad).all()
