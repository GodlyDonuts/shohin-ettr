from diverge_crp1_data import CRP1DataError, revision_user_content


def test_revision_prompt_preserves_step_identity() -> None:
    rendered = revision_user_content(
        "Start at 2 and add 3 twice.",
        ["Step 1: 2 + 3 = 5.", "Step 2: 5 + 3 = 8."],
        "Final answer: \\boxed{8}",
    )
    assert rendered.count("<draft_step_01>") == 1
    assert rendered.count("<draft_step_02>") == 1
    assert "Locate the first invalid step" in rendered


def test_revision_prompt_rejects_reserved_markers() -> None:
    try:
        revision_user_content(
            "bad <draft_step_ marker",
            ["Step 1: x"],
            "Final answer: x",
        )
    except CRP1DataError:
        return
    raise AssertionError("reserved marker was accepted")
