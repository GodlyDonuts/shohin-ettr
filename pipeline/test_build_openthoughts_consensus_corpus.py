from __future__ import annotations

from collections import Counter

from build_openthoughts_consensus_corpus import (
    QuestionVotes,
    boxed_answers,
    collect_votes,
    consensus_answer,
    extract_final_answer,
    normalize_answer,
    select_consensus_rows,
)


def raw(question: str, response: str, domain: str = "math") -> dict[str, object]:
    padded_response = (
        "We derive the result carefully from the supplied conditions and check each "
        "intermediate implication before reporting the requested value. " * 2
    ) + response
    return {
        "domain": domain,
        "source": "fixture",
        "difficulty": 3,
        "conversations": [
            {"from": "human", "value": question},
            {"from": "assistant", "value": padded_response},
        ],
    }


def test_answer_extraction() -> None:
    assert boxed_answers(r"x then \boxed{\frac{1}{2}}") == [r"\frac{1}{2}"]
    assert extract_final_answer(r"analysis \boxed{3} later \boxed{7}") == "7"
    assert extract_final_answer("Reasoning. Final answer: (B).") == "(b)"
    assert normalize_answer(r" $\dfrac{ 1 }{ 2 }$. ") == r"\frac{1}{2}"


def test_consensus_gate() -> None:
    votes = QuestionVotes("question", "math", "fixture", 1, annotations=16)
    votes.answers = Counter({"7": 10, "8": 3, "9": 1})
    assert consensus_answer(votes, 8, 8, 0.60, 3) == ("7", 10, 3, 10 / 14)
    votes.answers = Counter({"7": 7, "8": 6})
    assert consensus_answer(votes, 8, 8, 0.60, 3) is None


def test_two_pass_selection() -> None:
    question = "What is the exact integer obtained after adding three and four?"
    rows = [raw(question, f"<think>work {i}</think> Final answer: 7") for i in range(10)]
    rows += [raw(question, f"<think>wrong {i}</think> Final answer: 8") for i in range(3)]
    rows += [raw(question, "No explicit result here.") for _ in range(3)]
    rows += [raw("Write a complete program that returns seven for every input.", "```python\nprint(7)\n```", "code")]
    votes, counters = collect_votes(rows, set(), set())
    identity = next(iter(votes))
    decision = consensus_answer(votes[identity], 8, 8, 0.60, 3)
    assert decision is not None
    selected = select_consensus_rows(rows, {identity: decision}, votes)
    assert len(selected) == 1
    assert selected[0]["consensus_answer"] == "7"
    assert selected[0]["consensus_votes"] == 10
    assert selected[0]["answers_extracted"] == 13
    assert selected[0]["verification"] == "annotation_consensus_v1"
    assert counters["schema_or_domain_rejected"] == 1


if __name__ == "__main__":
    test_answer_extraction()
    test_consensus_gate()
    test_two_pass_selection()
