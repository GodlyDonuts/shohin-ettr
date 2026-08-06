from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from diverge_vmt1_workspace import (
    VerifiedTrajectoryError,
    complete_trace_cost_matrix,
    paired_ordered_trace_targets,
    verified_pair_assignment_objective,
)


class VMTWorkspaceTest(unittest.TestCase):
    def test_pair_targets_and_cost_geometry(self) -> None:
        embedding = nn.Embedding(32, 8)
        responses = [[[1, 2, 3, 4], [5, 6, 7]], [[8, 9], [10, 11, 12]]]
        targets, active = paired_ordered_trace_targets(embedding, responses, 4)
        probes = targets.clone()
        cost = complete_trace_cost_matrix(probes, targets, active)
        self.assertEqual(tuple(cost.shape), (2, 2, 2))
        self.assertTrue(torch.allclose(cost[:, 0, 0], torch.zeros(2), atol=1e-5))
        self.assertTrue(torch.allclose(cost[:, 1, 1], torch.zeros(2), atol=1e-5))

    def test_assignment_selects_correct_content_in_both_orientations(self) -> None:
        trace_cost = torch.tensor(
            [[[0.01, 0.8], [0.9, 0.02]], [[0.03, 0.7], [0.8, 0.01]]],
            requires_grad=True,
        )
        validity = torch.tensor([[2.0, -2.0], [-2.0, 2.0]], requires_grad=True)
        correct = torch.tensor([[True, False], [False, True]])
        nll = torch.tensor([[0.2, 0.8], [0.9, 0.1]], requires_grad=True)
        result = verified_pair_assignment_objective(
            trace_cost,
            validity,
            correct,
            nll,
            assignment_temperature=0.1,
            validity_margin=1.0,
            trace_weight=1.0,
            validity_weight=0.25,
        )
        self.assertEqual(result.selector_correct.tolist(), [True, True])
        self.assertEqual(result.swapped_selector_correct.tolist(), [False, False])
        self.assertTrue(
            (result.matched_trace_cosine > result.crossed_trace_cosine).all()
        )
        result.loss.backward()
        self.assertIsNotNone(trace_cost.grad)
        self.assertIsNotNone(validity.grad)
        self.assertIsNotNone(nll.grad)

    def test_permuting_internal_lineages_preserves_objective(self) -> None:
        trace_cost = torch.tensor([[[0.1, 0.7], [0.8, 0.2]]])
        validity = torch.tensor([[1.5, -0.5]])
        correct = torch.tensor([[True, False]])
        nll = torch.tensor([[0.3, 0.9]])
        first = verified_pair_assignment_objective(
            trace_cost,
            validity,
            correct,
            nll,
            assignment_temperature=0.1,
            validity_margin=1.0,
            trace_weight=1.0,
            validity_weight=0.25,
        )
        second = verified_pair_assignment_objective(
            trace_cost.flip(dims=(1,)),
            validity.flip(dims=(1,)),
            correct,
            nll.flip(dims=(1,)),
            assignment_temperature=0.1,
            validity_margin=1.0,
            trace_weight=1.0,
            validity_weight=0.25,
        )
        self.assertTrue(torch.allclose(first.loss, second.loss, atol=1e-6))

    def test_invalid_correctness_fails_closed(self) -> None:
        with self.assertRaises(VerifiedTrajectoryError):
            verified_pair_assignment_objective(
                torch.zeros(1, 2, 2),
                torch.zeros(1, 2),
                torch.tensor([[True, True]]),
                torch.zeros(1, 2),
                assignment_temperature=0.1,
                validity_margin=1.0,
                trace_weight=1.0,
                validity_weight=0.25,
            )


if __name__ == "__main__":
    unittest.main()
