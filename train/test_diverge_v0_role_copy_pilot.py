#!/usr/bin/env python3
"""Focused tests for DIVERGE token-role/source-copy mechanics."""

from __future__ import annotations

import unittest

import torch

from diverge_v0_neural_pilot import OptionExample, _render_option
from diverge_v0_role_copy_pilot import (
    ACTION_BASE,
    CANDIDATE_CUE,
    PRIOR_FAVORED,
    PRIOR_RESERVE,
    decode_option_roles,
    option_role_labels,
    record_role_labels,
)


def _character_offsets(text: str):
    return tuple((index, index + 1) for index in range(len(text)))


class DivergeRoleCopyTests(unittest.TestCase):
    def test_record_cues_are_source_local(self) -> None:
        candidate = "candidate alternatives in workshop: alpha; versus beta."
        background = "ignore this background record for workshop: alpha; besides beta."
        candidate_labels = record_role_labels(
            candidate, _character_offsets(candidate), is_fault_line=True
        )
        background_labels = record_role_labels(
            background, _character_offsets(background), is_fault_line=False
        )
        self.assertEqual(sum(label == CANDIDATE_CUE for label in candidate_labels), 22)
        self.assertTrue(any(label != 0 for label in background_labels))

    def test_option_roles_preserve_program_order_and_prior(self) -> None:
        for program in range(4):
            for prior in range(2):
                text = _render_option("alias", program, prior, renderer=3)
                option = OptionExample("alias", text, program, prior)
                labels = option_role_labels(option, text, _character_offsets(text))
                self.assertIn(PRIOR_FAVORED + prior, labels)
                self.assertTrue(any(label >= ACTION_BASE for label in labels))

    def test_hard_role_decoder_recovers_all_finite_programs(self) -> None:
        rows = []
        lengths = []
        expected_prior = []
        for program in range(4):
            prior = program % 2
            text = _render_option("alias", program, prior, renderer=2)
            option = OptionExample("alias", text, program, prior)
            labels = option_role_labels(option, text, _character_offsets(text))
            logits = torch.full((len(labels), 7), -8.0)
            logits[torch.arange(len(labels)), torch.tensor(labels)] = 8.0
            rows.append(logits)
            lengths.append(len(labels))
            expected_prior.append(prior)
        width = max(lengths)
        padded = torch.full((4, width, 7), -8.0)
        for index, row in enumerate(rows):
            padded[index, : row.shape[0]] = row
        programs, priors = decode_option_roles(padded, torch.tensor(lengths))
        self.assertEqual(programs, [0, 1, 2, 3])
        self.assertEqual(priors, expected_prior)
        self.assertEqual(PRIOR_RESERVE, 2)


if __name__ == "__main__":
    unittest.main()
