"""Prompt and exact execution mechanics for KCR1 causal revision branches."""

from __future__ import annotations

from dataclasses import dataclass


KEEP = "<KEEP>"
CONTINUE = "<CONTINUE>"
RESTART = "<RESTART>"
ACTIONS = (KEEP, CONTINUE, RESTART)


class KCR1TransducerError(ValueError):
    """A KCR1 prompt, transaction, or execution contract differs."""


@dataclass(frozen=True)
class KCR1Transaction:
    action: str
    payload: str


def render_transaction(action: str, payload: str = "") -> str:
    if action not in ACTIONS:
        raise KCR1TransducerError("unknown KCR1 action")
    if action == KEEP:
        if payload:
            raise KCR1TransducerError("KEEP cannot carry a payload")
        return KEEP
    if not payload:
        raise KCR1TransducerError("CONTINUE/RESTART requires a payload")
    return f"{action}\n{payload}"


def parse_transaction(value: str) -> KCR1Transaction:
    action, separator, payload = value.partition("\n")
    if action not in ACTIONS:
        raise KCR1TransducerError("unknown or malformed KCR1 action")
    if action == KEEP:
        if separator and payload.strip():
            raise KCR1TransducerError("KEEP carries trailing content")
        return KCR1Transaction(action=KEEP, payload="")
    if not separator or not payload:
        raise KCR1TransducerError("CONTINUE/RESTART payload is absent")
    return KCR1Transaction(action=action, payload=payload)


def execute_transaction(draft: str, value: str) -> str:
    transaction = parse_transaction(value)
    if transaction.action == KEEP:
        return draft
    if transaction.action == CONTINUE:
        return draft + transaction.payload
    if transaction.action == RESTART:
        return transaction.payload
    raise AssertionError("unreachable KCR1 action")


def kcr1_prompt(source: str, draft: str, *, exhausted: bool, task: str) -> str:
    if not source.strip() or not draft:
        raise KCR1TransducerError("source or draft is empty")
    status = "CUTOFF" if exhausted else "STOPPED"
    output_rule = (
        "For code, payload text must be executable Python without Markdown fences."
        if task == "code"
        else "For non-code, payload text must contain the exact final answer in \\boxed{}."
    )
    return (
        "Choose and execute exactly one revision transaction.\n"
        "<KEEP> preserves the draft exactly.\n"
        "<CONTINUE> appends your payload to the draft.\n"
        "<RESTART> replaces the draft with your payload.\n"
        "Return only the action token and any required payload.\n"
        f"{output_rule}\n\n"
        f"SOURCE:\n{source}\n\n"
        f"DRAFT_STATUS: {status}\n"
        f"DRAFT:\n{draft}"
    )
