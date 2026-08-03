from __future__ import annotations

import json
from pathlib import Path

from pipeline.rescore_product_reasoning_report import rescore_report


def test_rescore_repairs_decimal_and_currency_answer_errors(tmp_path: Path) -> None:
    source = tmp_path / "gsm8k.json"
    source.write_text(
        json.dumps(
            {
                "accuracy": 0.5,
                "correct": 1,
                "results": [
                    {
                        "completion": r"Therefore \boxed{2.00}.",
                        "correct": False,
                        "gold": "2",
                        "prediction": "2.00",
                    },
                    {
                        "completion": "The answer is 1 dollar 40 cents.",
                        "correct": True,
                        "gold": "1",
                        "prediction": "1",
                    },
                ],
                "schema": "shohin-hf-product-reasoning-eval-v2",
                "status": "complete",
                "task": "gsm8k",
                "total": 2,
            }
        ),
        encoding="utf-8",
    )

    report = rescore_report(source)

    assert report["correct"] == 1
    assert report["rescore_changes"] == {"false_to_true": 1, "true_to_false": 1}
    assert report["results"][0]["correct"] is True
    assert report["results"][1]["prediction"] == "1.4"
    assert report["results"][1]["correct"] is False
