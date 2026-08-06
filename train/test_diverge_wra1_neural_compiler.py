#!/usr/bin/env python3
"""Supervision and assignment tests for the WRA1 neural compiler."""

from __future__ import annotations

import unittest

import torch

from diverge_sc1_source_compiler import generate_episode
from diverge_wra1_neural_compiler import _ce, option_targets


class DivergeWRA1NeuralTests(unittest.TestCase):
    def test_complete_option_targets_are_segment_local(self) -> None:
        episode = generate_episode(seed=202608056611, cohort="renderer_shift")
        for record in episode.records:
            width = record.end - record.start
            for option in record.options:
                target = option_targets(record, option, halt_index=width)
                self.assertTrue(0 <= target.alias_start < width)
                self.assertTrue(0 <= target.alias_length < 4)
                self.assertTrue(0 <= target.prior_pointer < width)
                self.assertTrue(0 <= target.action_1_pointer < width)
                self.assertTrue(0 <= target.action_2_pointer_or_halt <= width)
                if len(option.action_positions) == 1:
                    self.assertEqual(target.action_2_pointer_or_halt, width)

    def test_cross_entropy_prefers_exact_pointer(self) -> None:
        logits = torch.tensor([-4.0, 7.0, -2.0])
        self.assertLess(float(_ce(logits, 1)), float(_ce(logits, 0)))


if __name__ == "__main__":
    unittest.main()
