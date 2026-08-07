#!/usr/bin/env python3
"""Focused tests for the DIVERGE-MZE1 executor and NPL2 integration seam."""

from __future__ import annotations

from pathlib import Path
import hashlib
import unittest

import torch

import diverge_npl2_runtime as npl2_runtime
from diverge_mze1_runtime import PresentedZ97Executor, ROW_CANDIDATES
from diverge_npl1_data import natural_public_record, render_feedback
from diverge_npl2_runtime import (
    DecodedEvidence,
    run_natural_episode,
    typed_episode_from_public,
)
from diverge_pl1_data import apply_operation, build_episode
from diverge_pl1_runtime import run_episode


GOLD_ROWS = (
    ((1, 1), (0, 1)),
    ((1, 0), (1, 1)),
    ((1, -1), (0, 1)),
    ((1, 0), (-1, 1)),
    ((2, 1), (0, 1)),
    ((1, 0), (1, 2)),
    ((0, 1), (1, 0)),
    ((-1, 1), (0, 1)),
)


def exact_model() -> PresentedZ97Executor:
    model = PresentedZ97Executor()
    with torch.no_grad():
        model.row_logits.fill_(-32.0)
        for operation, rows in enumerate(GOLD_ROWS):
            for output, row in enumerate(rows):
                model.row_logits[operation, output, ROW_CANDIDATES.index(row)] = 32.0
    return model


class MZE1Test(unittest.TestCase):
    def test_hard_learned_law_matches_every_exact_transition(self) -> None:
        model = exact_model()
        for operation in range(8):
            for x in range(97):
                for y in range(97):
                    self.assertEqual(
                        model.transition(operation, (x, y)),
                        apply_operation(operation, (x, y)),
                    )

    def test_outcome_loss_prefers_compatible_rows(self) -> None:
        model = PresentedZ97Executor()
        operations = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
        states = torch.tensor(
            [
                [2, 3],
                [5, 7],
                [11, 13],
                [17, 19],
                [23, 29],
                [31, 37],
                [41, 43],
                [47, 53],
            ],
            dtype=torch.long,
        )
        targets = torch.tensor(
            [
                apply_operation(int(operation), tuple(state))
                for operation, state in zip(operations, states, strict=True)
            ],
            dtype=torch.long,
        )
        loss = model.outcome_nll(operations, states, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.row_logits.grad.abs().sum()), 0.0)

    def test_candidate_runtime_does_not_import_exact_operation(self) -> None:
        source = (
            Path(__file__)
            .with_name("diverge_mze1_runtime.py")
            .read_text(encoding="utf-8")
        )
        self.assertNotIn("diverge_pl1_data", source)
        self.assertNotIn("oracle_transition", source)

    def test_learned_transition_injection_preserves_npl2_semantics(self) -> None:
        model = exact_model()
        episode = build_episode(split="mze1-test", seed=41, serial=0)
        public = natural_public_record(episode)
        typed = typed_episode_from_public(public)
        evidence = {}
        for plan in public["feedback_plan"]:
            attempt = int(plan["attempt"])
            branch = int(plan["branch"])
            depth = len(typed.acquisition[attempt].symbols)
            for code in (0, *range(2, depth + 2)):
                text = render_feedback(plan, code)
                evidence[(attempt, branch, code)] = DecodedEvidence(
                    attempt=attempt,
                    target_branch=str(plan["target_branch"]),
                    distractor_branch=str(plan["distractor_branch"]),
                    certificate_code=code,
                    commitment=hashlib.sha256(text.encode("ascii")).hexdigest(),
                )
        selectors = tuple(int(query["register_index"]) for query in public["queries"])
        oracle = run_episode(episode, arm="PL1", seed=2026080799)
        original = npl2_runtime.apply_operation
        try:
            npl2_runtime.apply_operation = model.transition
            learned = run_natural_episode(
                typed,
                episode,
                evidence=evidence,
                query_selectors=selectors,
                arm="PL1",
                proposal_arm="PL1",
                seed=2026080799,
            )
        finally:
            npl2_runtime.apply_operation = original
        self.assertEqual(learned.selected_mapping, oracle.selected_mapping)
        self.assertEqual(learned.transfer_exact, oracle.transfer_exact)
        self.assertEqual(learned.query_exact, 2 * oracle.transfer_exact)


if __name__ == "__main__":
    unittest.main()
