"""Tests for product-reasoning corpus re-admission."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from postfilter_product_reasoning_seed_corpus import postfilter


class ProductReasoningPostfilterTests(unittest.TestCase):
    def test_rejects_exact_and_ngram_overlap_across_eval_schemas(self) -> None:
        exact = "Write a function that returns the sum of two distinct integers."
        bbh = (
            "Alice knows Bob and Bob knows Carol and Carol knows David while David knows Eve "
            "and Eve knows Frank and Frank knows Grace."
        )
        overlap = bbh + " Explain the final conclusion carefully."
        clean = "A novel algebra task asks for the product of seven and eleven."
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            eval_path = root / "eval.jsonl"
            output = root / "output.jsonl"
            report_path = root / "report.json"
            source.write_text(
                "".join(
                    json.dumps({"question": question, "response": "reasoning"}) + "\n"
                    for question in (exact, overlap, clean)
                ),
                encoding="utf-8",
            )
            eval_path.write_text(
                json.dumps({"text": exact}) + "\n" + json.dumps({"input": bbh}) + "\n",
                encoding="utf-8",
            )

            report = postfilter(source, [eval_path], output, report_path)
            rows = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertEqual([row["question"] for row in rows], [clean])
        self.assertEqual(
            report["counters"],
            {
                "admitted_rows": 1,
                "eval_13gram_rejected": 1,
                "eval_exact_rejected": 1,
                "source_rows": 3,
            },
        )


if __name__ == "__main__":
    unittest.main()
