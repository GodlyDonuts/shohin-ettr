from score_dense_public_campaign import correctbench_answer, summarize


def test_correctbench_parser_prefers_explicit_last_answer() -> None:
    assert correctbench_answer("A is tempting, but the answer is (C)") == "C"
    assert correctbench_answer("Therefore \\boxed{B}") == "B"


def test_paired_summary_reports_lift_and_retention() -> None:
    metrics = summarize(
        [
            {"unchanged_correct": True, "revision_correct": True},
            {"unchanged_correct": False, "revision_correct": True},
            {"unchanged_correct": True, "revision_correct": False},
            {"unchanged_correct": False, "revision_correct": True},
        ]
    )
    assert metrics["unchanged_score"] == 50.0
    assert metrics["trained_revision_score"] == 75.0
    assert metrics["paired_delta_points"] == 25.0
    assert metrics["baseline_correct_retention"] == 0.5
