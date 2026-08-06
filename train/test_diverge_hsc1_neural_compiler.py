#!/usr/bin/env python3
"""Differentiable dynamic-program tests for DIVERGE-HSC1."""

from __future__ import annotations

import unittest

import torch

from diverge_hsc1_neural_compiler import (
    torch_batched_option_log_partition,
    torch_cut_log_partition,
    torch_gold_option_score,
    torch_option_log_partition,
)
from diverge_hsc1_structured_compiler import (
    _logsumexp,
    cut_log_partition,
    gold_option_path,
    path_log_partition,
    semantic_templates,
)
from diverge_sc1_source_compiler import ROLE_COUNT, generate_episode


class DivergeHSC1NeuralTests(unittest.TestCase):
    def test_differentiable_cut_partition_matches_cpu(self) -> None:
        generator = torch.Generator().manual_seed(202608057201)
        values = torch.randn(
            3,
            11,
            generator=generator,
            dtype=torch.float64,
            requires_grad=True,
        )
        expected = cut_log_partition(values.tolist())
        actual = torch_cut_log_partition(values)
        self.assertAlmostEqual(float(actual.detach()), expected, places=10)
        actual.backward()
        self.assertIsNotNone(values.grad)
        self.assertTrue(bool(torch.isfinite(values.grad).all()))

    def test_differentiable_option_partition_matches_cpu(self) -> None:
        generator = torch.Generator().manual_seed(202608057202)
        role = torch.randn(12, ROLE_COUNT, generator=generator, dtype=torch.float64)
        margins = (role - role[:, 0].unsqueeze(-1)).tolist()
        expected = _logsumexp(
            path_log_partition(margins, template.labels)
            for template in semantic_templates()
        )
        actual = torch_option_log_partition(role)
        self.assertAlmostEqual(float(actual), expected, places=10)

    def test_batched_option_partition_matches_individual(self) -> None:
        generator = torch.Generator().manual_seed(202608057204)
        role = torch.randn(5, 13, ROLE_COUNT, generator=generator)
        batched = torch_batched_option_log_partition(role)
        individual = torch.stack([torch_option_log_partition(row) for row in role])
        self.assertTrue(torch.allclose(batched, individual, atol=1e-6, rtol=1e-6))

    def test_gold_path_score_and_gradients_are_finite(self) -> None:
        episode = generate_episode(seed=202608057203, cohort="train")
        record = episode.records[0]
        option = record.options[0]
        positions = [
            index
            for index in range(record.start, record.end)
            if episode.tokens[index] == "option"
        ]
        trailer = next(
            index
            for index in range(record.start, record.end)
            if episode.tokens[index] == "glossary"
        )
        span = (positions[0], positions[1])
        if option.alias_span[0] >= positions[1]:
            span = (positions[1], trailer)
        template, path = gold_option_path(option, span[0])
        role = torch.randn(span[1] - span[0], ROLE_COUNT, requires_grad=True)
        loss = torch_option_log_partition(role) - torch_gold_option_score(
            role, template.labels, path
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(role.grad)
        self.assertTrue(bool(torch.isfinite(role.grad).all()))


if __name__ == "__main__":
    unittest.main()
