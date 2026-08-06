#!/usr/bin/env python3
"""Exhaustive parity tests for DIVERGE-HSC1 dynamic programs."""

from __future__ import annotations

import random
import unittest

from diverge_hsc1_structured_compiler import (
    calibrated_scores,
    cut_log_partition,
    cut_viterbi,
    decode_hierarchical,
    exact,
    exhaustive_cut_partition,
    exhaustive_path_partition,
    malformed_option_width,
    path_log_partition,
    path_viterbi,
    run_gate,
    semantic_templates,
)
from diverge_sc1_source_compiler import ROLE_COUNT, generate_episode


class DivergeHSC1Tests(unittest.TestCase):
    def test_cut_dynamic_program_matches_exhaustive_reference(self) -> None:
        rng = random.Random(202608056801)
        for width in range(5, 10):
            cuts = tuple(
                tuple(rng.uniform(-2.0, 2.0) for _ in range(width)) for _ in range(3)
            )
            expected_partition, expected_best = exhaustive_cut_partition(cuts)
            self.assertAlmostEqual(
                cut_log_partition(cuts), expected_partition, places=10
            )
            self.assertEqual(cut_viterbi(cuts), expected_best)

    def test_template_dynamic_program_matches_exhaustive_reference(self) -> None:
        rng = random.Random(202608056802)
        for template in semantic_templates():
            length = len(template.labels) + 3
            margins = tuple(
                tuple(rng.uniform(-2.0, 2.0) for _ in range(ROLE_COUNT))
                for _ in range(length)
            )
            expected_partition, expected_best = exhaustive_path_partition(
                margins, template.labels
            )
            self.assertAlmostEqual(
                path_log_partition(margins, template.labels),
                expected_partition,
                places=10,
            )
            self.assertEqual(path_viterbi(margins, template.labels), expected_best)

    def test_calibrated_complete_packet_is_exact(self) -> None:
        for index, cohort in enumerate(
            ("train", "lexical_shift", "renderer_shift", "composition_shift")
        ):
            episode = generate_episode(seed=202608056810 + index, cohort=cohort)
            receipt = decode_hierarchical(episode.tokens, calibrated_scores(episode))
            self.assertTrue(exact(episode, receipt))

    def test_malformed_option_fails_closed(self) -> None:
        episode = generate_episode(seed=202608056820, cohort="train")
        scores = calibrated_scores(episode)
        receipt = decode_hierarchical(episode.tokens, malformed_option_width(scores))
        self.assertTrue(receipt.failed)
        self.assertEqual(receipt.failure_reason, "option-score-width")

    def test_small_gate(self) -> None:
        report = run_gate(count=32, seed=202608056830)
        self.assertTrue(report["passed"])
        self.assertEqual(report["grammar"]["templates"], 128)
        self.assertEqual(report["grammar"]["pair_matrix_entries"], 0)
        self.assertEqual(report["rates"]["linear_accounting"], 1.0)


if __name__ == "__main__":
    unittest.main()
