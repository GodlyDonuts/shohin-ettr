#!/usr/bin/env python3
"""Tests for conditional DIVERGE-NPL2 confirmation generation."""

from __future__ import annotations

import unittest

from build_diverge_npl2_confirmation import CONFIRMATION_SEEDS, EPISODES_PER_SEED
from diverge_npl1_data import natural_program_identities, natural_public_record
from diverge_pl1_data import build_split


class NPL2ConfirmationDataTest(unittest.TestCase):
    def test_seed_contract(self) -> None:
        self.assertEqual(len(CONFIRMATION_SEEDS), 5)
        self.assertEqual(len(set(CONFIRMATION_SEEDS)), 5)
        self.assertEqual(EPISODES_PER_SEED, 256)

    def test_first_episodes_are_disjoint_and_valid(self) -> None:
        episodes = [
            build_split(split=f"npl2_confirmation_{seed}", seed=seed, count=1)[0]
            for seed in CONFIRMATION_SEEDS
        ]
        self.assertEqual(len({episode.episode_id for episode in episodes}), 5)
        self.assertEqual(
            len({identity for episode in episodes for identity in natural_program_identities(episode)}),
            5 * 28,
        )
        for episode in episodes:
            record = natural_public_record(episode)
            self.assertEqual(len(record["feedback_plan"]), 96)
            self.assertEqual(len(record["queries"]), 32)


if __name__ == "__main__":
    unittest.main()

