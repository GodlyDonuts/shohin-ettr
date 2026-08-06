#!/usr/bin/env python3
"""Focused tests for DIVERGE-WRA1 whole-record mechanics."""

from __future__ import annotations

from dataclasses import replace
import unittest

from diverge_sc1_source_compiler import generate_episode
from diverge_wra1_whole_record import (
    calibrated_scores,
    decode_reference,
    decode_whole_records,
    detect_segments,
    duplicate_first_slot,
    exact,
    run_gate,
    seal_source_packet,
    shuffle_lineage,
    swap_slots,
)


class DivergeWholeRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.episode = generate_episode(seed=202608056511, cohort="composition_shift")
        self.scores = calibrated_scores(self.episode, seed=202608056512)

    def test_calibrated_scores_reconstruct_without_gold_runtime_objects(self) -> None:
        receipt = decode_whole_records(self.episode.tokens, self.scores)
        reference = decode_reference(self.episode.tokens, self.scores)
        self.assertTrue(exact(self.episode, receipt))
        self.assertEqual(receipt.records, reference.records)
        self.assertEqual(receipt.option_objects, 2 * len(self.episode.records))
        self.assertEqual(receipt.record_objects, len(self.episode.records))

    def test_slot_identity_is_exchangeable_but_field_lineage_is_not(self) -> None:
        self.assertTrue(
            exact(
                self.episode,
                decode_whole_records(self.episode.tokens, swap_slots(self.scores)),
            )
        )
        self.assertFalse(
            exact(
                self.episode,
                decode_whole_records(self.episode.tokens, shuffle_lineage(self.scores)),
            )
        )

    def test_duplicate_complete_slots_fail_closed(self) -> None:
        receipt = decode_whole_records(
            self.episode.tokens, duplicate_first_slot(self.scores)
        )
        self.assertTrue(receipt.failed)
        self.assertIn(
            receipt.failure_reason, {"overlapping-aliases", "shared-option-field"}
        )

    def test_boundary_contract_fails_closed(self) -> None:
        odd = replace(self.scores, boundary=(12.0, *self.scores.boundary[1:]))
        # Flip one true terminal boundary off, leaving an odd number of positives.
        values = list(odd.boundary)
        values[self.episode.records[-1].end] = -12.0
        segments, reason, overflow = detect_segments(values, len(self.episode.tokens))
        self.assertEqual(segments, ())
        self.assertEqual(reason, "odd-boundary-count")
        self.assertFalse(overflow)

    def test_source_is_dead_after_seal(self) -> None:
        receipt = decode_whole_records(self.episode.tokens, self.scores)
        packet = seal_source_packet(self.episode.tokens, receipt)
        poison = seal_source_packet(
            tuple("poison" for _ in self.episode.tokens), receipt
        )
        self.assertEqual(packet.records, poison.records)

    def test_small_cpu_gate(self) -> None:
        report = run_gate(count=64, seed=202608056520)
        self.assertTrue(report["passed"])
        self.assertEqual(report["accounting"]["pair_matrix_entries"], 0)


if __name__ == "__main__":
    unittest.main()
