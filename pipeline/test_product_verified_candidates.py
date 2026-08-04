"""Focused semantic-verifier selection tests."""

from __future__ import annotations

from hf_product_candidate_verifier import counterbalanced_score, verifier_prompt
from select_product_verified_candidates import select


def test_counterbalancing_cancels_fixed_a_label_bias() -> None:
    unbiased = counterbalanced_score(3.0, 3.0)
    correct_evidence = counterbalanced_score(5.0, 1.0)
    assert unbiased == 0.0
    assert correct_evidence == 2.0


def test_verifier_prompt_contains_no_gold_or_correctness_field() -> None:
    prompt = verifier_prompt("What is 2+3?", "Answer: 5", reversed_labels=False)
    assert "What is 2+3?" in prompt
    assert "Answer: 5" in prompt
    assert "A means correct" in prompt


def test_selector_uses_score_and_not_correctness(tmp_path) -> None:
    source = tmp_path / "scores.jsonl"
    source.write_text(
        "\n".join(
            [
                '{"identity_sha256":"x","task":"gsm8k","sample_index":0,'
                '"verifier_score":-1.0,"correct":true}',
                '{"identity_sha256":"x","task":"gsm8k","sample_index":1,'
                '"verifier_score":2.0,"correct":false}',
            ]
        )
        + "\n"
    )
    report = select([source])
    assert report["selected_correct"] == 0
    assert report["results"][0]["selected_sample_index"] == 1
