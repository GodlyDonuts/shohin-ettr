from __future__ import annotations

import json
from pathlib import Path

from pipeline.rescore_product_reasoning_report import rescore_report
from hf_product_reasoning_eval import has_explicit_final_answer


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
    assert report["rescore_backend"] == "shohin-answer-v4-explicit-cap"


def test_capped_fallback_number_requires_explicit_final_answer(tmp_path: Path) -> None:
    source = tmp_path / "gsm8k_capped.json"
    source.write_text(
        json.dumps(
            {
                "accuracy": 1.0,
                "correct": 3,
                "results": [
                    {
                        "completion": "The algebra is impossible. Given $2 for grape...",
                        "correct": True,
                        "gold": "2",
                        "max_token_exhausted": True,
                        "prediction": "2",
                    },
                    {
                        "completion": "Work continues. The answer is 2.",
                        "correct": True,
                        "gold": "2",
                        "max_token_exhausted": True,
                        "prediction": "2",
                    },
                    {
                        "completion": "A concise uncapped derivation ends with 2",
                        "correct": True,
                        "gold": "2",
                        "max_token_exhausted": False,
                        "prediction": "2",
                    },
                ],
                "schema": "shohin-hf-product-reasoning-eval-v2",
                "status": "complete",
                "task": "gsm8k",
                "total": 3,
            }
        ),
        encoding="utf-8",
    )

    report = rescore_report(source)

    assert report["correct"] == 2
    assert report["cap_exhausted_without_explicit_answer"] == 1
    assert report["results"][0]["prediction"] is None
    assert report["results"][0]["correct"] is False
    assert report["results"][1]["correct"] is True
    assert report["results"][2]["correct"] is True


def test_capped_report_replays_saved_finalization(tmp_path: Path) -> None:
    source = tmp_path / "math_finalized.json"
    source.write_text(
        json.dumps(
            {
                "accuracy": 1.0,
                "correct": 1,
                "results": [
                    {
                        "completion": "An unfinished derivation reaches",
                        "correct": True,
                        "finalization_completion": r"\boxed{7}",
                        "gold": "7",
                        "max_token_exhausted": True,
                        "prediction": "7",
                    }
                ],
                "schema": "shohin-hf-product-reasoning-eval-v3",
                "status": "complete",
                "task": "math500",
                "total": 1,
            }
        )
    )

    report = rescore_report(source)

    assert report["correct"] == 1
    assert report["rescore_change_count"] == 0
    assert report["results"][0]["prediction"] == "7"


def test_explicit_final_answer_markers() -> None:
    assert has_explicit_final_answer(r"Therefore \boxed{4}.")
    assert has_explicit_final_answer("The final answer: B")
    assert has_explicit_final_answer("answer is 12")
    assert not has_explicit_final_answer("We used option B in a partial thought")


def test_aime_rescore_rejects_expression_prefix_false_positive(tmp_path: Path) -> None:
    source = tmp_path / "aime.json"
    source.write_text(
        json.dumps(
            {
                "status": "complete",
                "task": "aime",
                "total": 2,
                "correct": 2,
                "accuracy": 1.0,
                "results": [
                    {
                        "completion": r"The answer is 25^{9/5}.",
                        "prediction": "25",
                        "gold": "025",
                        "correct": True,
                        "max_token_exhausted": False,
                    },
                    {
                        "completion": r"Therefore, the answer is 25.",
                        "prediction": "25",
                        "gold": "025",
                        "correct": True,
                        "max_token_exhausted": False,
                    },
                ],
            }
        )
    )
    report = rescore_report(source)
    assert report["correct"] == 1
    assert report["results"][0]["prediction"] is None
    assert report["results"][1]["prediction"] == "25"
    assert report["rescore_backend"] == "shohin-aime-v1-explicit-final-integer"
