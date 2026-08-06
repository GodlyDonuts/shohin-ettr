#!/usr/bin/env python3
"""Focused tests for DIVERGE-SC1 source-only structured compilation."""

from __future__ import annotations

import unittest

from diverge_sc1_source_compiler import (
    PROGRAM_ROLES,
    alpha_rename_episode,
    calibrated_scores,
    decode_independent,
    decode_joint,
    decode_reference,
    exact,
    generate_episode,
    seal_source,
)


class DivergeSC1SourceCompilerTests(unittest.TestCase):
    def _episode(self, seed: int = 202608056100, cohort: str = "composition_shift"):
        episode = generate_episode(seed=seed, cohort=cohort)
        scores = calibrated_scores(episode, seed=seed + 99)
        return episode, scores

    def test_joint_decoder_matches_independent_reference(self) -> None:
        episode, scores = self._episode()
        joint = decode_joint(episode.tokens, scores)
        reference = decode_reference(episode.tokens, scores)
        self.assertFalse(joint.overflow)
        self.assertTrue(exact(episode, joint))
        self.assertEqual(joint.score, reference.score)
        self.assertEqual(joint.records, reference.records)

    def test_local_decoding_is_not_the_joint_mechanism(self) -> None:
        episode, scores = self._episode(seed=202608056101, cohort="lexical_shift")
        self.assertTrue(exact(episode, decode_joint(episode.tokens, scores)))
        self.assertFalse(exact(episode, decode_independent(episode.tokens, scores)))

    def test_program_order_survives_source_canonicalization(self) -> None:
        episode, scores = self._episode(seed=202608056102, cohort="renderer_shift")
        receipt = decode_joint(episode.tokens, scores)
        self.assertTrue(exact(episode, receipt))
        for decoded, gold in zip(receipt.records, episode.records, strict=True):
            for option, target in zip(decoded.options, gold.options, strict=True):
                self.assertEqual(option.program, target.program)
                self.assertEqual(
                    tuple(
                        max(
                            range(len(scores.role[position])),
                            key=scores.role[position].__getitem__,
                        )
                        for position in option.action_positions
                    ),
                    PROGRAM_ROLES[target.program],
                )

    def test_occurrence_ids_do_not_collapse_equal_nominals(self) -> None:
        for seed in range(202608056100, 202608056140):
            episode = generate_episode(seed=seed, cohort="composition_shift")
            scores = calibrated_scores(episode, seed=seed + 1)
            receipt = decode_joint(episode.tokens, scores)
            packet = seal_source(episode.tokens, receipt)
            options = [option for record in packet.records for option in record.options]
            by_nominal = {}
            for option in options:
                by_nominal.setdefault(option.nominal_commitment, []).append(option)
            repeats = [rows for rows in by_nominal.values() if len(rows) > 1]
            if repeats:
                self.assertTrue(
                    all(
                        len({row.occurrence_id for row in rows}) == len(rows)
                        for rows in repeats
                    )
                )
                return
        self.fail("test seeds did not generate a repeated nominal alias")

    def test_alpha_rename_and_post_seal_poison(self) -> None:
        episode, scores = self._episode(seed=202608056103, cohort="composition_shift")
        receipt = decode_joint(episode.tokens, scores)
        packet = seal_source(episode.tokens, receipt)
        renamed = alpha_rename_episode(episode)
        self.assertTrue(exact(renamed, decode_joint(renamed.tokens, scores)))
        poisoned = seal_source(tuple("poison" for _ in episode.tokens), receipt)
        self.assertEqual(packet.records, poisoned.records)
        self.assertNotEqual(packet.source_commitment, poisoned.source_commitment)

    def test_records_and_options_never_overlap(self) -> None:
        episode, scores = self._episode(seed=202608056104, cohort="train")
        receipt = decode_joint(episode.tokens, scores)
        self.assertTrue(exact(episode, receipt))
        for left, right in zip(receipt.records, receipt.records[1:], strict=False):
            self.assertLessEqual(left.end, right.start)
        for record in receipt.records:
            self.assertLessEqual(record.options[0].end, record.options[1].start)


if __name__ == "__main__":
    unittest.main()
