from build_dseo1_paired_data import (
    boxed_inner_span,
    final_answer_span,
    mutate_surface,
    presentation_fits,
    presentations,
    split_members,
)


def _candidate(identity: str, family: str) -> dict:
    return {
        "source_identity_sha256": identity,
        "source_line": 1,
        "training_group": "science" if family == "choice_final" else "math",
        "task": "broad_reasoning",
        "raw_question": "Question?",
        "clean_response": "Reasoning. \\boxed{10}",
        "fault_response": "Reasoning. \\boxed{11}",
        "fault_action": "<FIX_FINAL>",
        "corruption_family": family,
        "changed_character_span": [23, 24],
        "changed_token_span": [4, 5],
        "gold_answer": "10",
    }


def test_boxed_inner_span_balances_nested_text() -> None:
    text = "first \\boxed{2} then \\boxed{\\text{B}}"
    start, end = boxed_inner_span(text)
    assert text[start:end] == "\\text{B}"


def test_final_answer_span_prefers_expected_inside_box() -> None:
    text = "10 is considered, therefore \\boxed{10}"
    start, end = final_answer_span(text, "10")
    assert text[start:end] == "10"
    assert start > text.index("considered")


def test_final_answer_span_accepts_answer_sentence() -> None:
    text = "Work here. The answer is -3.50e+2."
    start, end = final_answer_span(text, "-3.50e+2")
    assert text[start:end] == "-3.50e+2"


def test_mutate_surface_is_width_preserving_and_unequal() -> None:
    assert mutate_surface("B") == "C"
    assert mutate_surface("-3.50e+2") == "-3.50e+0"
    assert mutate_surface("symbolic") is None


def test_split_members_is_disjoint_and_stratified() -> None:
    candidates = {
        "numeric_final": [_candidate(f"n{index}", "numeric_final") for index in range(25)],
        "choice_final": [_candidate(f"c{index}", "choice_final") for index in range(10)],
    }
    train, diagnostic, quotas = split_members(candidates, 16, 8)
    assert len(train) == 16
    assert len(diagnostic) == 8
    assert quotas["choice_final"] == {"train": 2, "diagnostic": 1}
    assert {row["source_identity_sha256"] for row in train}.isdisjoint(
        row["source_identity_sha256"] for row in diagnostic
    )


def test_presentations_hold_source_and_final_fixed() -> None:
    rows = presentations(_candidate("source", "numeric_final"))
    assert [row["pair_member"] for row in rows] == ["clean", "fault"]
    assert rows[0]["source_identity_sha256"] == rows[1]["source_identity_sha256"]
    assert rows[0]["final_response"] == rows[1]["final_response"]
    assert rows[0]["action"] == "<KEEP>"
    assert rows[1]["action"] == "<FIX_FINAL>"
    assert rows[0]["draft_sha256"] != rows[1]["draft_sha256"]


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return list(range(len(text.split())))


def _render(_tokenizer, messages, *, enable_thinking):
    assert not enable_thinking
    return " ".join(message["content"] for message in messages)


def test_presentation_fits_checks_both_pair_members() -> None:
    item = _candidate("source", "numeric_final")
    assert presentation_fits(_Tokenizer(), _render, "system", item, 200)
    assert not presentation_fits(_Tokenizer(), _render, "system", item, 5)
