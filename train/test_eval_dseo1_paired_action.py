from eval_dseo1_paired_action import answer_correct, parse_action_completion


def test_parse_action_requires_first_nonempty_exact_line() -> None:
    assert parse_action_completion("\n<KEEP>\nThe answer is 3.") == (
        "<KEEP>",
        "The answer is 3.",
    )
    assert parse_action_completion("prefix <KEEP>\nThe answer is 3.")[0] is None


def test_answer_correct_handles_numeric_and_choice() -> None:
    numeric = {"gold_answer": "3.0", "corruption_family": "numeric_final"}
    choice = {"gold_answer": "B", "corruption_family": "choice_final"}
    assert answer_correct(numeric, "The final answer is 3.")
    assert answer_correct(choice, "Therefore \\boxed{\\text{B}}")
