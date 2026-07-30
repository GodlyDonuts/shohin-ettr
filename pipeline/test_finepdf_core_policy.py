from pipeline.finepdf_core_policy import (
    POLICY_SCHEMA,
    classify_finepdf_candidate,
)


def classify(text: str, scores=(2.5,), *, domain="example.org"):
    return classify_finepdf_candidate(
        text=text,
        metadata={"fw_edu_scores": list(scores)},
        domain=domain,
    )


def test_download_aggregation_is_hard_rejected_even_with_high_score():
    decision = classify(
        "Thank you utterly much for downloading this physics answers ebook. "
        + "technical words " * 500,
        scores=(4.0, 4.0),
        domain="repository.example.edu",
    )
    assert decision.tier == "reject"
    assert "download_aggregation_spam" in decision.reason_codes


def test_low_density_newsletter_is_not_promoted_by_authority_domain():
    decision = classify(
        "Weekly newsletter issue 14. Diary dates and parent reminders. "
        + "school community news " * 500,
        scores=(3.5,),
        domain="school.example.edu",
    )
    assert decision.tier == "reject"
    assert "low_density:newsletter" in decision.reason_codes


def test_two_ended_high_score_long_form_is_core():
    decision = classify(
        "A rigorous discussion of computation and systems. " * 500,
        scores=(2.7, 2.8),
    )
    assert decision.tier == "core"
    assert "high_score_core" in decision.reason_codes
    assert "two_ended_score_continuity" in decision.reason_codes


def test_structured_research_can_rescue_moderate_score():
    decision = classify(
        (
            "Abstract. This technical report studies a deterministic method. "
            "Methods. We define the experiment and controls. "
            "Results. The measured result is reported with uncertainty. "
            "References. "
        )
        * 180,
        scores=(1.25, 2.25),
        domain="files.example.edu",
    )
    assert decision.tier == "core"
    assert decision.strong_signal_count >= 1
    assert "structured_core" in decision.reason_codes


def test_unstructured_moderate_score_stays_residual():
    decision = classify(
        "A coherent but specialized discussion of a local subject. " * 250,
        scores=(1.75,),
    )
    assert decision.tier == "residual"
    assert "requires_equal_token_ablation" in decision.reason_codes


def test_missing_score_fails_closed():
    decision = classify_finepdf_candidate(
        text="A substantive document. " * 500,
        metadata={},
        domain="example.edu",
    )
    assert decision.schema == POLICY_SCHEMA
    assert decision.tier == "reject"
    assert "missing_education_score" in decision.reason_codes


def test_low_score_substantive_document_is_residual_not_destroyed():
    decision = classify(
        "A coherent historical account of an experimental system. " * 500,
        scores=(1.25,),
    )
    assert decision.tier == "residual"
    assert "low_education_score" in decision.reason_codes
    assert "hard_reject" not in decision.reason_codes


def test_answer_key_with_formal_exposition_is_not_blanket_rejected():
    decision = classify(
        (
            "Answer key and annotated test. "
            "Theorem 1 establishes the invariant. Proof. "
            "Worked example 1 derives the result step by step. "
        )
        * 220,
        scores=(2.2,),
    )
    assert decision.tier == "core"
    assert "low_density:answer_key_without_exposition" not in decision.reason_codes


def test_decision_is_deterministic_and_serializable():
    left = classify("Chapter 4. Exercise 1. Worked example. " * 400, scores=(2.0,))
    right = classify("Chapter 4. Exercise 1. Worked example. " * 400, scores=(2.0,))
    assert left == right
    assert left.to_dict()["reason_codes"] == left.reason_codes
