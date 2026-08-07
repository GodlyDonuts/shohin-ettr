#!/usr/bin/env python3
"""Small data-only tests for the frozen NPL2 development evaluator."""

from __future__ import annotations

import unittest

from diverge_npl1_data import natural_public_record
from diverge_npl2_runtime import typed_episode_from_public
from diverge_pl1_data import build_episode
from eval_diverge_npl2_development import _feedback_work, _semantic_floor


class NPL2DevelopmentTest(unittest.TestCase):
    def test_every_legal_feedback_code_is_materialized(self) -> None:
        hidden = build_episode(split="npl2-eval-test", seed=29, serial=0)
        public = natural_public_record(hidden)
        typed = typed_episode_from_public(public)
        work = _feedback_work([public], [typed])
        expected = sum(8 * (1 + len(program.symbols)) for program in typed.acquisition)
        self.assertEqual(len(work), expected)
        self.assertEqual(
            len({(row["attempt"], row["branch"], row["code"]) for row in work}),
            expected,
        )

    def test_semantic_floor_is_conjunctive(self) -> None:
        good = {
            "overall": {"exact": 1000, "total": 1000},
            "by_renderer": {
                str(index): {"exact": 100, "total": 100} for index in range(3)
            },
        }
        self.assertTrue(_semantic_floor(good, 0.995, 3))
        bad = {
            **good,
            "by_renderer": {**good["by_renderer"], "0": {"exact": 98, "total": 100}},
        }
        self.assertFalse(_semantic_floor(bad, 0.995, 3))


if __name__ == "__main__":
    unittest.main()
