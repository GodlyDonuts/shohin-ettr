#!/usr/bin/env python3
"""Mechanical tests for the exact EIC1 candidate involution."""

from __future__ import annotations

import unittest

import torch

from diverge_eic1_runtime import EIC1Config, project_candidate_scores


class EIC1RuntimeTest(unittest.TestCase):
    def test_involution_projection_is_equivariant(self) -> None:
        normal = torch.tensor([[3.0, -2.0], [0.25, 7.0]])
        swapped = torch.tensor([[5.0, 1.0], [-4.0, 6.0]])
        projected = project_candidate_scores(
            normal, swapped, mode="involution"
        )
        projected_after_swap = project_candidate_scores(
            swapped, normal, mode="involution"
        ).flip(dims=(-1,))
        self.assertTrue(torch.equal(projected, projected_after_swap))

    def test_duplicate_forward_does_not_force_equivariance(self) -> None:
        normal = torch.tensor([[3.0, -2.0]])
        swapped = torch.tensor([[5.0, 1.0]])
        projected = project_candidate_scores(normal, normal, mode="duplicate")
        swapped_projected = project_candidate_scores(
            swapped, swapped, mode="duplicate"
        ).flip(dims=(-1,))
        self.assertFalse(torch.equal(projected, swapped_projected))

    def test_frozen_geometry_rejects_unknown_mode(self) -> None:
        EIC1Config(projection_mode="involution").validate()
        with self.assertRaises(RuntimeError):
            EIC1Config(projection_mode="invalid").validate()  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
