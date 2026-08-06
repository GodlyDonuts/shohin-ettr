#!/usr/bin/env python3
"""Data-boundary tests for the DIVERGE-SC1 neural compiler."""

from __future__ import annotations

import unittest

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from diverge_sc1_neural_compiler import (
    encode_source,
    gold_boundaries,
    gold_pairs,
    gold_role_targets,
)
from diverge_sc1_source_compiler import (
    ALIAS_BEGIN,
    OTHER,
    generate_episode,
)


class DivergeSC1NeuralBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.episode = generate_episode(
            seed=202608056200,
            cohort="composition_shift",
        )
        vocabulary = {"[UNK]": 0}
        for token in self.episode.tokens:
            vocabulary.setdefault(token, len(vocabulary))
        self.tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()

    def test_raw_encoding_has_no_record_or_option_input(self) -> None:
        encoded = encode_source(self.tokenizer, self.episode.tokens)
        self.assertEqual(encoded.word_count, len(self.episode.tokens))
        self.assertEqual(set(encoded.token_to_word), set(range(len(self.episode.tokens))))

    def test_supervision_is_complete_and_source_local(self) -> None:
        roles = gold_role_targets(self.episode)
        boundaries = gold_boundaries(self.episode)
        positive, active = gold_pairs(self.episode)
        self.assertEqual(len(roles), len(self.episode.tokens))
        self.assertEqual(len(boundaries), len(self.episode.tokens) + 1)
        self.assertTrue(positive)
        self.assertTrue(active)
        for record in self.episode.records:
            for option in record.options:
                self.assertEqual(roles[option.alias_span[0]], ALIAS_BEGIN)
                self.assertNotEqual(roles[option.alias_span[0]], OTHER)
                self.assertIn(record.start, [i for i, value in enumerate(boundaries) if value])
                self.assertIn(record.end, [i for i, value in enumerate(boundaries) if value])


if __name__ == "__main__":
    unittest.main()
