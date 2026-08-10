import unittest

import torch

from fixed_slot_typed_compiler import compile_typed_program
from train_fstc1_skeleton import tokenize_sources


class FakeTokenizer:
    is_fast = True

    def __call__(self, texts, **kwargs):
        width = max(len(text) for text in texts)
        ids = torch.zeros(len(texts), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        offsets = torch.zeros(len(texts), width, 2, dtype=torch.long)
        for row, text in enumerate(texts):
            ids[row, : len(text)] = torch.tensor([ord(char) for char in text])
            mask[row, : len(text)] = 1
            offsets[row, : len(text), 0] = torch.arange(len(text))
            offsets[row, : len(text), 1] = torch.arange(1, len(text) + 1)
        return {"input_ids": ids, "attention_mask": mask, "offset_mapping": offsets}


class FSTC1TrainingTest(unittest.TestCase):
    def test_candidate_token_masks_cover_every_span(self):
        program = compile_typed_program(
            {
                "identity_sha256": "a" * 64,
                "question": "Calculate 12 + 34.",
                "records": [
                    {
                        "operation": "ADD",
                        "operands": [
                            {"numerator": 12, "denominator": 1},
                            {"numerator": 34, "denominator": 1},
                        ],
                        "result": {"numerator": 46, "denominator": 1},
                        "dependencies": [],
                    }
                ],
            }
        )
        encoded, candidates, receipt = tokenize_sources(
            FakeTokenizer(), [program], torch.device("cpu"), 256
        )
        self.assertEqual(receipt["charged_source_tokens"], len(program.question))
        self.assertEqual(candidates[0, 0].sum().item(), 2)
        self.assertEqual(candidates[0, 1].sum().item(), 2)
        self.assertEqual(encoded["input_ids"].shape[0], 1)


if __name__ == "__main__":
    unittest.main()
