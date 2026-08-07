#!/usr/bin/env python3
"""Tests for fail-closed EIC1 confirmation authorization."""

from __future__ import annotations

import copy
import unittest

from authorize_diverge_eic1_confirmation import authorize


class EIC1AuthorizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assessment = {
            "schema": "shohin-diverge-eic1-assessment-v1",
            "passed": True,
            "selected": "shohin_involution",
            "confirmation_access_authorized": True,
        }
        self.board = {
            "schema": "shohin-diverge-eic1-confirmation-report-v1",
            "board_sha256": "board",
            "model_score_used_for_selection": False,
            "generated_before_eic1_development_result": True,
            "overlap": {"source": 0, "query": 0},
        }

    def test_authorized(self) -> None:
        authorize(self.assessment, self.board, board_sha256="board")

    def test_failed_assessment_rejects(self) -> None:
        assessment = copy.deepcopy(self.assessment)
        assessment["passed"] = False
        with self.assertRaises(RuntimeError):
            authorize(assessment, self.board, board_sha256="board")

    def test_overlap_rejects(self) -> None:
        board = copy.deepcopy(self.board)
        board["overlap"]["query"] = 1
        with self.assertRaises(RuntimeError):
            authorize(self.assessment, board, board_sha256="board")


if __name__ == "__main__":
    unittest.main()
