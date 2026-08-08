"""CPU tests for transferable temporal-revision prompt masking."""

from __future__ import annotations

import unittest

from ttr1_revision import (
    internal_draft_char_span,
    internal_revision_prompt,
    tokenize_with_draft_mask,
)


class CharacterTokenizer:
    """Minimal fast-tokenizer analogue with exact character offsets."""

    def __call__(self, text, **kwargs):
        self.kwargs = kwargs
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


class TTR1RevisionTests(unittest.TestCase):
    def test_mask_preserves_ids_and_hides_only_draft(self) -> None:
        prompt = internal_revision_prompt("Compute 2+2.", "Maybe 5.", "math500")
        rendered = f"System: careful\n\nUser: {prompt}\n\nAssistant:"
        tokenizer = CharacterTokenizer()
        input_ids, attention, span = tokenize_with_draft_mask(tokenizer, rendered)
        start, end = span
        self.assertEqual(len(input_ids), len(attention))
        self.assertEqual(attention[start:end], [0] * (end - start))
        self.assertTrue(all(attention[:start]))
        self.assertTrue(all(attention[end:]))
        self.assertEqual("".join(map(chr, input_ids[start:end])), "Maybe 5.")
        self.assertTrue(tokenizer.kwargs["return_offsets_mapping"])

    def test_code_prompt_has_same_parseable_boundary(self) -> None:
        prompt = internal_revision_prompt("Write add().", "def add(): pass", "mbpp")
        start, end = internal_draft_char_span(prompt)
        self.assertEqual(prompt[start:end], "def add(): pass")

    def test_nested_marker_inside_draft_remains_masked(self) -> None:
        prompt = internal_revision_prompt("q", "Internal draft:\nbad", "math500")
        start, end = internal_draft_char_span(prompt)
        self.assertEqual(prompt[start:end], "Internal draft:\nbad")


if __name__ == "__main__":
    unittest.main()
