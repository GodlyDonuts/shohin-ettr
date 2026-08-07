#!/usr/bin/env python3
"""Unit tests for NPL2 confirmation aggregation mechanics."""

from __future__ import annotations

import unittest

from aggregate_diverge_npl2_confirmation import _bootstrap, _semantic_rate


class NPL2AggregateTest(unittest.TestCase):
    def test_bootstrap_detects_strict_gain(self) -> None:
        result = _bootstrap([32] * 64, [0] * 64, "test")
        self.assertEqual(result["lower_95"], 1.0)

    def test_semantic_rate_uses_exact_counts(self) -> None:
        self.assertEqual(
            _semantic_rate({"overall": {"exact": 995, "total": 1000}}), 0.995
        )


if __name__ == "__main__":
    unittest.main()
