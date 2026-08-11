import pytest

from kcr1_branch_transducer import (
    CONTINUE,
    KEEP,
    RESTART,
    KCR1TransducerError,
    execute_transaction,
    kcr1_prompt,
    parse_transaction,
    render_transaction,
)


def test_each_branch_executes_distinct_semantics() -> None:
    draft = "work so far"
    assert execute_transaction(draft, render_transaction(KEEP)) == draft
    assert execute_transaction(draft, render_transaction(CONTINUE, " then done")) == (
        "work so far then done"
    )
    assert execute_transaction(draft, render_transaction(RESTART, "new solution")) == (
        "new solution"
    )


def test_parser_fails_closed() -> None:
    for value in ("KEEP", "<CONTINUE>", "<RESTART>", "<UNKNOWN>\nx"):
        with pytest.raises(KCR1TransducerError):
            parse_transaction(value)
    with pytest.raises(KCR1TransducerError):
        parse_transaction("<KEEP>\nnot allowed")
    with pytest.raises(KCR1TransducerError):
        render_transaction(KEEP, "payload")


def test_prompt_exposes_only_controller_termination_state() -> None:
    cutoff = kcr1_prompt("problem", "partial", exhausted=True, task="math")
    stopped = kcr1_prompt("problem", "complete", exhausted=False, task="code")
    assert "DRAFT_STATUS: CUTOFF" in cutoff
    assert "DRAFT_STATUS: STOPPED" in stopped
    assert "correct" not in cutoff.casefold()
    assert "verifier" not in cutoff.casefold()
    assert "executable Python" in stopped
