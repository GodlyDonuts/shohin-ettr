#!/usr/bin/env python3
"""Unit tests for the frozen ENI1 admission conditions."""

from __future__ import annotations

import unittest

import torch

from diverge_eic1_runtime import render_claim_prompt
from eval_diverge_eni1_semantics import _rename_query_records, gate_conditions


def query_score(exact: int = 8192) -> dict:
    return {
        "overall": {"exact": exact, "total": 8192},
        "by_renderer": {
            str(index): {"exact": 1365 if index < 4 else 1366, "total": 1365 if index < 4 else 1366}
            for index in range(6)
        },
        "mean_signed_margin": 1.0,
        "_predictions": [0, 1],
        "_scores": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    }


def evidence_score() -> dict:
    return {
        "overall": {"joint_exact": 24576, "total": 24576},
        "by_renderer": {
            str(index): {"joint_exact": 8192, "total": 8192}
            for index in range(3)
        },
    }


class ENI1GateTest(unittest.TestCase):
    def test_episode_local_rename_preserves_canonical_prompt(self) -> None:
        record = {
            "source_text": "Answer from register zibapuxucu; reject decoy register levawoqawu.",
            "symbols": ["unusedalias", "zibapuxucu", "levawoqawu"],
            "symbol_role_ids": [0, 1],
            "renderer": 0,
            "mode": "register_query",
        }
        renamed = _rename_query_records([record])[0]
        self.assertEqual(
            [render_claim_prompt(record, index) for index in (0, 1)],
            [render_claim_prompt(renamed, index) for index in (0, 1)],
        )

    def test_exact_gate_passes(self) -> None:
        normal = query_score()
        swapped = query_score()
        scrubbed = query_score(4096)
        scrubbed["mean_signed_margin"] = 0.0
        renamed = query_score()
        conditions = gate_conditions(
            world_exact=7168,
            evidence=evidence_score(),
            normal=normal,
            swapped=swapped,
            scrubbed=scrubbed,
            renamed=renamed,
            prompts_exact=True,
            projection_error=0.0,
            protected_hashes_exact=True,
        )
        self.assertTrue(all(conditions.values()))

    def test_swap_failure_rejects(self) -> None:
        normal = query_score()
        swapped = query_score(4096)
        scrubbed = query_score(4096)
        scrubbed["mean_signed_margin"] = 0.0
        conditions = gate_conditions(
            world_exact=7168,
            evidence=evidence_score(),
            normal=normal,
            swapped=swapped,
            scrubbed=scrubbed,
            renamed=query_score(),
            prompts_exact=True,
            projection_error=0.0,
            protected_hashes_exact=True,
        )
        self.assertFalse(conditions["mapped_swap_at_least_99_5_percent"])


if __name__ == "__main__":
    unittest.main()
