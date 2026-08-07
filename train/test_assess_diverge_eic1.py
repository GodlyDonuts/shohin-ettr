#!/usr/bin/env python3
"""Unit tests for EIC1 matched-arm promotion logic."""

from __future__ import annotations

import copy
import unittest

from assess_diverge_eic1 import assess


def arm(mode: str, normal: int, swap: int, scrub: int, passed: bool) -> dict:
    training = {
        "base_sha256": "base",
        "tokenizer_sha256": "tok",
        "public_data_sha256": "public",
        "supervisor_data_sha256": "supervisor",
        "seed": 1,
        "updates": 2,
        "pair_batch_size": 3,
        "learning_rate": 0.1,
        "consistency_weight": 0.25,
        "lora_projection_count": 4,
        "trainable_parameters": 5,
        "total_parameters": 6,
        "logical_public_rows": 7,
        "unique_source_rows": 8,
        "backbone_forwards_per_source": 2,
        "training_fit": {"true_exact": 100000},
        "projection_mode": mode,
    }
    development = {
        "normal": {"overall": {"exact": normal}},
        "mention_swap": {"overall": {"exact": swap}},
        "scrub_context": {"overall": {"exact": scrub}},
        "projection_identity_max_absolute_error": 0.0 if mode == "involution" else 1.0,
        "promotion_gate": {"passed": passed},
    }
    return {
        "training": training,
        "development": development,
        "training_report_sha256": mode + "-train",
        "development_report_sha256": mode + "-dev",
    }


class EIC1AssessorTest(unittest.TestCase):
    def test_pass(self) -> None:
        arms = {
            "shohin_involution": arm("involution", 768, 768, 384, True),
            "shohin_duplicate": arm("duplicate", 768, 384, 384, False),
            "smollm2_involution": arm("involution", 768, 768, 384, True),
            "smollm2_duplicate": arm("duplicate", 768, 512, 384, False),
        }
        self.assertTrue(assess(arms)["passed"])

    def test_small_swap_gain_fails(self) -> None:
        arms = {
            "shohin_involution": arm("involution", 768, 768, 384, True),
            "shohin_duplicate": arm("duplicate", 768, 640, 384, False),
            "smollm2_involution": arm("involution", 768, 768, 384, True),
            "smollm2_duplicate": arm("duplicate", 768, 512, 384, False),
        }
        self.assertFalse(assess(arms)["passed"])

    def test_compute_mismatch_fails(self) -> None:
        arms = {
            "shohin_involution": arm("involution", 768, 768, 384, True),
            "shohin_duplicate": arm("duplicate", 768, 384, 384, False),
            "smollm2_involution": arm("involution", 768, 768, 384, True),
            "smollm2_duplicate": arm("duplicate", 768, 512, 384, False),
        }
        broken = copy.deepcopy(arms)
        broken["shohin_duplicate"]["training"]["updates"] = 3
        self.assertFalse(assess(broken)["passed"])


if __name__ == "__main__":
    unittest.main()
