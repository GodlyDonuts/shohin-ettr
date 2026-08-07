#!/usr/bin/env python3
"""Pre-generation contract tests for the EIC1 confirmation board."""

from __future__ import annotations

import unittest

from diverge_eic1_confirmation_data import NAMES, query_text
from diverge_iem1_data import _symbol_role_ids


class EIC1ConfirmationDataTest(unittest.TestCase):
    def test_entity_bank_is_parser_legal_and_unique(self) -> None:
        self.assertEqual(len(NAMES), 32)
        self.assertEqual(len(set(NAMES)), 32)
        self.assertTrue(all(name.isascii() and name.isalpha() and name.islower() for name in NAMES))

    def test_every_query_renderer_exposes_both_roles(self) -> None:
        symbols = (NAMES[0], NAMES[1])
        for renderer in range(6):
            for order in (0, 1):
                text = query_text(
                    renderer,
                    order,
                    target=symbols[0],
                    distractor=symbols[1],
                )
                roles = _symbol_role_ids(
                    text,
                    symbols,
                    target=symbols[0],
                    distractor=symbols[1],
                )
                self.assertIn(0, roles)
                self.assertIn(1, roles)


if __name__ == "__main__":
    unittest.main()
