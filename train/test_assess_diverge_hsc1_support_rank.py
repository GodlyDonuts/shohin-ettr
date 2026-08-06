#!/usr/bin/env python3
"""Unit tests for the frozen HSC1 support-rank assessor."""

from __future__ import annotations

import itertools
import unittest

from assess_diverge_hsc1_support_rank import (
    RankedOption,
    cut_k_best,
    cue_k_best,
    option_k_best,
    path_k_best,
    template_k_best,
)
from diverge_hsc1_structured_compiler import (
    _margins,
    exhaustive_paths,
    semantic_templates,
)
from diverge_sc1_source_compiler import OTHER, ROLE_COUNT


class HSC1SupportRankTests(unittest.TestCase):
    def test_path_k_best_matches_exhaustive(self) -> None:
        margins = tuple(
            tuple(((row + 2) * (role + 3)) % 11 / 7.0 for role in range(ROLE_COUNT))
            for row in range(8)
        )
        labels = semantic_templates()[17].labels
        expected = sorted(
            (
                (
                    sum(
                        margins[position][label]
                        for position, label in zip(path, labels, strict=True)
                    ),
                    path,
                )
                for path in exhaustive_paths(len(margins), labels)
            ),
            key=lambda item: (-item[0], item[1]),
        )[:13]
        actual = path_k_best(margins, labels, 13)
        self.assertEqual([row.path for row in actual], [path for _, path in expected])
        for row, (score, _) in zip(actual, expected, strict=True):
            self.assertAlmostEqual(row.score, score, places=12)

    def test_option_k_best_matches_exhaustive(self) -> None:
        role = tuple(
            tuple(((row + 5) * (value + 7)) % 19 / 9.0 for value in range(ROLE_COUNT))
            for row in range(7)
        )
        margins = _margins(role)
        expected = []
        for template_index, template in enumerate(semantic_templates()):
            for path in exhaustive_paths(len(margins), template.labels):
                expected.append(
                    RankedOption(
                        sum(
                            margins[position][label]
                            for position, label in zip(
                                path, template.labels, strict=True
                            )
                        ),
                        template_index,
                        path,
                    )
                )
        expected.sort(
            key=lambda item: (
                -item.score,
                item.path,
                semantic_templates()[item.template_index].prior_class,
                semantic_templates()[item.template_index].program,
                semantic_templates()[item.template_index].alias_length,
                semantic_templates()[item.template_index].component_order,
            )
        )
        self.assertEqual(option_k_best(role, 31), tuple(expected[:31]))

    def test_cut_k_best_matches_exhaustive(self) -> None:
        cuts = tuple(
            tuple(((channel + 3) * (position + 2)) % 13 / 5.0 for position in range(9))
            for channel in range(3)
        )
        expected = sorted(
            (
                (
                    cuts[0][left] + cuts[1][middle] + cuts[2][trailer],
                    (left, middle, trailer),
                )
                for left, middle, trailer in itertools.combinations(range(1, 9), 3)
            ),
            key=lambda item: (-item[0], item[1]),
        )[:17]
        actual = cut_k_best(cuts, 17)
        self.assertEqual([(row.score, row.path) for row in actual], expected)

    def test_template_k_best_matches_exhaustive_marginals(self) -> None:
        role = tuple(
            tuple(((row + 11) * (value + 5)) % 23 / 8.0 for value in range(ROLE_COUNT))
            for row in range(9)
        )
        actual = template_k_best(role, 128)
        self.assertEqual(len(actual), 128)
        self.assertEqual({template for _, template in actual}, set(range(128)))
        self.assertTrue(
            all(actual[index][0] >= actual[index + 1][0] for index in range(127))
        )

    def test_cue_k_best_uses_margin_and_header_only(self) -> None:
        scores = (
            (10.0, 9.0, 11.0),
            (0.0, 4.0, 3.0),
            (0.0, 100.0, 100.0),
        )
        self.assertEqual(
            cue_k_best(scores, 2, 3),
            ((4.0, 1, 1), (3.0, 1, 2), (1.0, 0, 2)),
        )

    def test_other_baseline_does_not_change_option_order(self) -> None:
        role = [[0.0] * ROLE_COUNT for _ in range(8)]
        for row, values in enumerate(role):
            values[OTHER] = float(row * 100)
            for label in range(1, ROLE_COUNT):
                values[label] = values[OTHER] + float(((row + 5) * (label + 7)) % 19)
        first = option_k_best(role, 16)
        for values in role:
            for label in range(ROLE_COUNT):
                values[label] += 23.0
        self.assertEqual(first, option_k_best(role, 16))


if __name__ == "__main__":
    unittest.main()
