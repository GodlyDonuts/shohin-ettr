from eval_dset1_span_edit import evaluate_completion


def _row(member="fault"):
    return {
        "script": "<REPLACE_LAST>\nC\nB\n" if member == "fault" else "<KEEP>\n",
        "action": "<REPLACE_LAST>" if member == "fault" else "<KEEP>",
        "draft": "The answer is C." if member == "fault" else "The answer is B.",
        "final_response": "The answer is B.",
        "corruption_family": "choice_final",
        "gold_answer": "B",
    }


def test_correct_replace_executes_verified_trajectory() -> None:
    result = evaluate_completion(_row(), "<REPLACE_LAST>\nC\nB\n")
    assert result["script_exact"]
    assert result["execution_correct"]
    assert result["trajectory_exact"]


def test_wrong_keep_on_fault_fails_answer() -> None:
    result = evaluate_completion(_row(), "<KEEP>\n")
    assert not result["script_exact"]
    assert not result["execution_correct"]


def test_clean_keep_copies_exactly() -> None:
    result = evaluate_completion(_row("clean"), "<KEEP>\n")
    assert result["script_exact"]
    assert result["execution_correct"]
    assert result["trajectory_exact"]
