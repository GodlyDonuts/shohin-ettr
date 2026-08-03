"""Mechanics tests for the product reasoning workspace."""

from __future__ import annotations

from dataclasses import replace
import unittest

import torch

from integrated_reasoning_workspace import (
    DenseReasoningWorkspace,
    IntegratedReasoningWorkspace,
    IntegratedWorkspaceConfig,
    IntegratedWorkspaceError,
    workspace_architecture_sha256,
)


class IntegratedReasoningWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(31)
        self.config = IntegratedWorkspaceConfig(
            backbone_width=64,
            workspace_width=32,
            workspace_slots=4,
            recurrent_steps=3,
            attention_heads=4,
            ff_multiplier=2,
        )

    def test_shapes_are_stable(self) -> None:
        module = IntegratedReasoningWorkspace(self.config)
        features = torch.randn(2, 7, 64)
        mask = torch.tensor([[1] * 7, [1, 1, 1, 1, 0, 0, 0]])
        output = module(features, mask)
        self.assertEqual(output.prefix_states.shape, (2, 4, 64))
        self.assertEqual(output.workspace_states.shape, (2, 4, 32))
        self.assertEqual(output.stop_logits.shape, (2, 3))
        self.assertEqual(output.step_deltas.shape, (2, 3))
        self.assertTrue(torch.isfinite(output.prefix_states).all())

    def test_gradient_reaches_prompt_and_tied_cell(self) -> None:
        module = IntegratedReasoningWorkspace(self.config)
        features = torch.randn(2, 5, 64, requires_grad=True)
        output = module(features, torch.ones(2, 5, dtype=torch.long))
        loss = output.prefix_states.square().mean() + module.halting_regularizer(output)
        loss.backward()
        self.assertIsNotNone(features.grad)
        self.assertGreater(float(features.grad.abs().sum()), 0.0)
        self.assertTrue(
            all(parameter.grad is not None for parameter in module.cell.parameters())
        )

    def test_step_count_changes_compute_not_parameter_count(self) -> None:
        short = IntegratedReasoningWorkspace(self.config)
        long = IntegratedReasoningWorkspace(
            replace(self.config, recurrent_steps=9)
        )
        self.assertEqual(
            short.trainable_parameter_count(), long.trainable_parameter_count()
        )

    def test_empty_prompt_is_rejected(self) -> None:
        module = IntegratedReasoningWorkspace(self.config)
        with self.assertRaises(IntegratedWorkspaceError):
            module(torch.randn(1, 3, 64), torch.zeros(1, 3))

    def test_architecture_hash_is_deterministic(self) -> None:
        self.assertEqual(
            workspace_architecture_sha256(self.config),
            workspace_architecture_sha256(self.config),
        )

    def test_dense_control_is_capacity_matched_and_finite(self) -> None:
        recurrent = IntegratedReasoningWorkspace(
            IntegratedWorkspaceConfig(backbone_width=1024, workspace_width=512)
        )
        dense = DenseReasoningWorkspace(
            IntegratedWorkspaceConfig(backbone_width=1024, workspace_width=192)
        )
        relative = abs(
            recurrent.trainable_parameter_count() - dense.trainable_parameter_count()
        ) / recurrent.trainable_parameter_count()
        self.assertLess(relative, 0.05)
        output = dense(
            torch.randn(2, 7, 1024),
            torch.ones(2, 7, dtype=torch.long),
        )
        self.assertEqual(output.prefix_states.shape, (2, 16, 1024))
        self.assertTrue(torch.isfinite(output.prefix_states).all())


if __name__ == "__main__":
    unittest.main()
