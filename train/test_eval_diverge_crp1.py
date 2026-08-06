from eval_diverge_crp1 import extract_answer, extract_error_step


def test_extracts_last_boxed_answer_and_error_step() -> None:
    completion = (
        "Error step: 4\nCorrection: changed it.\n"
        "Final answer: \\boxed{old}\nFinal answer: \\boxed{17,9}"
    )
    assert extract_error_step(completion) == 4
    assert extract_answer(completion) == "17,9"


def test_extracts_no_error_and_fails_closed() -> None:
    assert extract_error_step("Error step: NONE") == 0
    assert extract_error_step("No structured line") is None
    assert extract_answer("Final answer: 12") is None
