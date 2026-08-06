"""Mechanics tests for DIVERGE-LTM1 complete latent trajectories."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from diverge_ltm1_workspace import (
    FactorizedLatentTrajectoryWorkspace,
    LatentTrajectoryConfig,
    complete_trajectory_marginal_loss,
    latent_trajectory_architecture_sha256,
    ordered_trace_targets,
    trajectory_alignment_energy,
)


class DivergeLTM1WorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2026080601)
        self.config = LatentTrajectoryConfig(
            backbone_width=48,
            latent_width=24,
            trajectory_slots=3,
            recurrent_steps=4,
            fault_bits=2,
            attention_heads=4,
            ff_multiplier=2,
        )

    def test_all_binary_assignments_exist_once(self) -> None:
        module = FactorizedLatentTrajectoryWorkspace(self.config)
        self.assertEqual(
            module.assignments.tolist(),
            [[0, 0], [1, 0], [0, 1], [1, 1]],
        )
        self.assertEqual(
            len({tuple(row) for row in module.assignments.tolist()}),
            self.config.candidate_count,
        )

    def test_complete_candidate_geometry_is_stable(self) -> None:
        module = FactorizedLatentTrajectoryWorkspace(self.config)
        features = torch.randn(2, 7, self.config.backbone_width)
        mask = torch.tensor([[1] * 7, [1, 1, 1, 1, 0, 0, 0]])
        output = module(features, mask)
        self.assertEqual(
            output.candidate_prefixes.shape,
            (2, 4, 3, self.config.backbone_width),
        )
        self.assertEqual(output.trajectory_probes.shape, (2, 4, 4, 48))
        self.assertEqual(output.prior_logits.shape, (2, 4))
        self.assertEqual(output.stop_logits.shape, (2, 4, 4))
        self.assertTrue(torch.isfinite(output.candidate_prefixes).all())

    def test_gradient_reaches_every_architectural_interface(self) -> None:
        module = FactorizedLatentTrajectoryWorkspace(self.config)
        features = torch.randn(2, 5, 48, requires_grad=True)
        output = module(features, torch.ones(2, 5, dtype=torch.long))
        targets = torch.randn(2, 4, 48)
        active = torch.ones(2, 4, dtype=torch.bool)
        trace = trajectory_alignment_energy(output.trajectory_probes, targets, active)
        nll = output.candidate_prefixes.square().mean(dim=(2, 3))
        result = complete_trajectory_marginal_loss(
            nll,
            trace,
            output.prior_logits,
            trace_weight=0.25,
            balance_weight=0.01,
        )
        (result.loss + 0.01 * module.halting_regularizer(output)).backward()
        self.assertGreater(float(features.grad.abs().sum()), 0.0)
        for parameter in (
            module.initial_slots,
            module.factor_slots,
            module.shared_seed.weight,
            module.factor_seed.weight,
            module.prior_head.weight,
            module.output_projection.weight,
            module.trace_projection.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)
        self.assertTrue(
            all(parameter.grad is not None for parameter in module.cell.parameters())
        )

    def test_selection_returns_one_stored_lineage_not_a_mean(self) -> None:
        module = FactorizedLatentTrajectoryWorkspace(self.config)
        output = module(torch.randn(2, 4, 48), torch.ones(2, 4, dtype=torch.long))
        selected, indices = module.select_prefix(output, strategy="highest_prior")
        batch = torch.arange(2)
        self.assertTrue(
            torch.equal(selected, output.candidate_prefixes[batch, indices])
        )
        reset, reset_indices = module.select_prefix(output, strategy="reset")
        self.assertTrue(torch.equal(indices, reset_indices))
        self.assertEqual(int(torch.count_nonzero(reset)), 0)

    def test_trace_targets_are_contiguous_and_detached(self) -> None:
        embedding = nn.Embedding(20, 6)
        targets, active = ordered_trace_targets(
            embedding,
            [[1, 2, 3, 4], [5, 6]],
            recurrent_steps=4,
        )
        self.assertEqual(targets.shape, (2, 4, 6))
        self.assertEqual(active.tolist(), [[True] * 4, [True, True, False, False]])
        self.assertFalse(targets.requires_grad)
        self.assertTrue(torch.equal(targets[0, 0], embedding.weight[1].float()))
        self.assertTrue(torch.equal(targets[1, 1], embedding.weight[6].float()))

    def test_whole_candidate_permutation_preserves_marginal_loss(self) -> None:
        nll = torch.tensor([[0.7, 0.2, 1.3, 0.9], [0.4, 1.1, 0.8, 0.6]])
        trace = torch.tensor([[0.1, 0.4, 0.7, 0.2], [0.8, 0.3, 0.2, 0.5]])
        logits = torch.tensor([[0.2, -0.3, 0.7, 0.1], [-0.2, 0.5, 0.3, 0.8]])
        original = complete_trajectory_marginal_loss(
            nll,
            trace,
            logits,
            trace_weight=0.5,
            balance_weight=0.01,
        )
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = complete_trajectory_marginal_loss(
            nll[:, permutation],
            trace[:, permutation],
            logits[:, permutation],
            trace_weight=0.5,
            balance_weight=0.01,
        )
        self.assertTrue(torch.allclose(original.loss, permuted.loss))

    def test_architecture_hash_is_deterministic(self) -> None:
        self.assertEqual(
            latent_trajectory_architecture_sha256(self.config),
            latent_trajectory_architecture_sha256(self.config),
        )


if __name__ == "__main__":
    unittest.main()
