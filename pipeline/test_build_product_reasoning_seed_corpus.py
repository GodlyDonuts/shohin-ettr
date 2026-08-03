"""Tests for multi-domain product-reasoning seed curation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from build_product_reasoning_seed_corpus import (
    extract_openthoughts_row,
    load_eval_contamination,
    parse_caps,
    prompt_sha256,
    select_rows,
    word_ngrams,
)


def sample(question: str, response: str, domain: str) -> dict[str, object]:
    return {
        "domain": domain,
        "source": "fixture",
        "conversations": [
            {"from": "human", "value": question},
            {"from": "assistant", "value": response},
        ],
    }


class ProductReasoningSeedCorpusTests(unittest.TestCase):
    def test_extracts_conversation_schema(self) -> None:
        row = extract_openthoughts_row(
            sample(
                "What is the result of this sufficiently detailed problem?",
                "We compute the result carefully. Therefore the final answer is 7.",
                "math",
            )
        )
        self.assertIsNotNone(row)
        self.assertEqual(row["domain"], "math")  # type: ignore[index]

    def test_selects_one_trace_per_prompt_and_filters_eval(self) -> None:
        question = "A sufficiently detailed unique prompt asks for a result."
        rows = [
            sample(question, "Short but valid reasoning. " * 10, "math"),
            sample(
                question,
                " ".join(f"Step {index} is careful and checks the result." for index in range(8))
                + " Therefore final answer: 7.",
                "math",
            ),
            sample(
                "A second detailed coding prompt asks for a program.",
                "We derive the algorithm, prove its complexity, and test edge cases. " * 4
                + "```python\ndef solve():\n    print(7)\n```",
                "code",
            ),
            sample(
                "A third detailed science prompt asks for an explanation.",
                " ".join(
                    f"Evidence item {index} supports the causal explanation."
                    for index in range(8)
                )
                + " Therefore the answer is A.",
                "science",
            ),
        ]
        selected, report = select_rows(
            rows,
            {"math": 1, "code": 1, "science": 1},
            31,
            set(),
            set(),
        )
        self.assertEqual(len(selected), 3)
        self.assertEqual(report["selected_by_domain"], {"code": 1, "math": 1, "science": 1})
        self.assertTrue(all(row["verification"] == "teacher_trace_unverified" for row in selected))

    def test_caps_require_every_domain(self) -> None:
        self.assertEqual(
            parse_caps(["math=1", "code=2", "science=3"]),
            {"math": 1, "code": 2, "science": 3},
        )

    def test_eval_contamination_loads_mbpp_and_bbh_prompt_fields(self) -> None:
        mbpp = "Write a Python function that returns the sum of two distinct integers."
        bbh = "Determine whether this formal logical argument is valid or invalid."
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "eval.jsonl"
            path.write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for row in ({"text": mbpp}, {"input": bbh})
                ),
                encoding="utf-8",
            )
            exact, ngrams = load_eval_contamination([path])

        self.assertEqual(exact, {prompt_sha256(mbpp), prompt_sha256(bbh)})
        self.assertTrue(word_ngrams(mbpp) <= ngrams)
        self.assertTrue(word_ngrams(bbh) <= ngrams)


if __name__ == "__main__":
    unittest.main()
