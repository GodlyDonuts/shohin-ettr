import unittest

import torch

from pushdown_stack_typed_compiler import load_stack_program
from test_pushdown_stack_typed_compiler import row
from train_pstc1_stack import tokenize_sources


class FakeTokenizer:
    is_fast = True

    def __call__(self, texts, **kwargs):
        width = max(len(text) for text in texts)
        ids = torch.zeros(len(texts), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        offsets = torch.zeros(len(texts), width, 2, dtype=torch.long)
        for index, text in enumerate(texts):
            ids[index, : len(text)] = torch.tensor([ord(character) for character in text])
            mask[index, : len(text)] = 1
            offsets[index, : len(text), 0] = torch.arange(len(text))
            offsets[index, : len(text), 1] = torch.arange(1, len(text) + 1)
        return {"input_ids": ids, "attention_mask": mask, "offset_mapping": offsets}


class PSTC1TrainingTest(unittest.TestCase):
    def test_tokenizer_owns_all_candidates(self):
        program = load_stack_program(row())
        encoded, candidates, receipt = tokenize_sources(
            FakeTokenizer(), [program], torch.device("cpu"), 256
        )
        self.assertEqual(candidates[0, :3].sum().item(), 3)
        self.assertEqual(receipt["maximum_tokens"], len(program.question))
        self.assertEqual(encoded["input_ids"].shape[0], 1)


if __name__ == "__main__":
    unittest.main()
