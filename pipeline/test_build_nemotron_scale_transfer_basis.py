from __future__ import annotations

import torch

from build_nemotron_scale_transfer_basis import build_factor, choose_identities


def test_identity_selection_is_disjoint_deterministic_and_excluded() -> None:
    first = choose_identities(128, 24, 32, {1, 7, 99}, seed=42)
    second = choose_identities(128, 24, 32, {1, 7, 99}, seed=42)
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    anchor, holdout = first
    assert set(anchor.tolist()).isdisjoint(holdout.tolist())
    assert not ({1, 7, 99} & set(anchor.tolist() + holdout.tolist()))


def test_factor_recovers_shared_semantic_subspace() -> None:
    generator = torch.Generator().manual_seed(7)
    vocabulary = 512
    latent = 20
    super_width = 48
    ultra_width = 72
    semantics = torch.randn(vocabulary, latent, generator=generator)
    super_projection = torch.randn(latent, super_width, generator=generator)
    ultra_projection = torch.randn(latent, ultra_width, generator=generator)
    super_embedding = semantics @ super_projection
    ultra_embedding = semantics @ ultra_projection
    anchor, holdout = choose_identities(vocabulary, 192, 192, set(), seed=11)
    factor, metrics = build_factor(
        super_embedding,
        ultra_embedding,
        anchor,
        holdout,
        ridge=1e-4,
        validation_directions=32,
        seed=13,
    )
    assert factor["super_anchor"].shape == (192, super_width)
    assert factor["ultra_anchor"].shape == (192, ultra_width)
    assert factor["ultra_kernel_cholesky"].shape == (192, 192)
    assert metrics["holdout_correlation_mean"] > 0.95
    assert metrics["holdout_correlation_min"] > 0.85
