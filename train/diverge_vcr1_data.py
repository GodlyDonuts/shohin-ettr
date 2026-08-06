"""Exact rendering and token admission for temporal correction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hf_product_reasoning_train import (
    PRODUCT_SYSTEM_PROMPT,
    render_reasoning_messages,
)


QUESTION_OPEN = "<original_problem>"
QUESTION_CLOSE = "</original_problem>"
DRAFT_OPEN = "<first_pass_draft>"
DRAFT_CLOSE = "</first_pass_draft>"


class VCR1DataError(RuntimeError):
    """A correction row cannot satisfy the exact token contract."""


@dataclass(frozen=True)
class CorrectionTokens:
    prompt_ids: list[int]
    response_ids: list[int]
    question_mask: list[bool]
    draft_mask: list[bool]


def correction_user_content(question: str, draft: str) -> str:
    question = question.strip()
    draft = draft.strip()
    if not question or not draft:
        raise VCR1DataError("question and draft must be nonempty")
    forbidden = (QUESTION_OPEN, QUESTION_CLOSE, DRAFT_OPEN, DRAFT_CLOSE)
    if any(marker in question or marker in draft for marker in forbidden):
        raise VCR1DataError("source text contains a reserved correction marker")
    return (
        f"{QUESTION_OPEN}\n{question}\n{QUESTION_CLOSE}\n\n"
        f"{DRAFT_OPEN}\n{draft}\n{DRAFT_CLOSE}\n\n"
        "Verify the first-pass draft against the original problem. If the draft "
        "is correct, preserve its result. If it is wrong, repair the reasoning "
        "and answer. Return one complete response with concise verifiable "
        "reasoning and put only the final answer inside \\boxed{}."
    )


def render_correction_prompt(tokenizer: Any, question: str, draft: str) -> str:
    return render_reasoning_messages(
        tokenizer,
        [
            {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": correction_user_content(question, draft),
            },
        ],
        enable_thinking=False,
    )


def _content_span(rendered: str, opening: str, closing: str) -> tuple[int, int]:
    if rendered.count(opening) != 1 or rendered.count(closing) != 1:
        raise VCR1DataError("rendered correction markers are not unique")
    start = rendered.index(opening) + len(opening)
    if start < len(rendered) and rendered[start] == "\n":
        start += 1
    end = rendered.index(closing, start)
    while end > start and rendered[end - 1] == "\n":
        end -= 1
    if end <= start:
        raise VCR1DataError("rendered correction span is empty")
    return start, end


def _overlaps(offset: tuple[int, int], span: tuple[int, int]) -> bool:
    start, end = offset
    return end > span[0] and start < span[1]


def tokenize_correction_example(
    tokenizer: Any,
    question: str,
    draft: str,
    target: str | None,
    *,
    max_sequence_length: int,
    workspace_slots: int,
) -> CorrectionTokens | None:
    if max_sequence_length <= workspace_slots + 8:
        raise VCR1DataError("correction sequence budget is too small")
    rendered = render_correction_prompt(tokenizer, question, draft)
    question_span = _content_span(rendered, QUESTION_OPEN, QUESTION_CLOSE)
    draft_span = _content_span(rendered, DRAFT_OPEN, DRAFT_CLOSE)
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    prompt_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [tuple(map(int, value)) for value in encoded["offset_mapping"]]
    if len(prompt_ids) != len(offsets) or not prompt_ids:
        raise VCR1DataError("tokenizer offsets differ from prompt tokens")
    question_mask = [_overlaps(offset, question_span) for offset in offsets]
    draft_mask = [_overlaps(offset, draft_span) for offset in offsets]
    if not any(question_mask) or not any(draft_mask):
        raise VCR1DataError("tokenizer lost a correction segment")
    if any(q and d for q, d in zip(question_mask, draft_mask, strict=True)):
        raise VCR1DataError("tokenizer correction segments overlap")

    response_ids: list[int] = []
    if target is not None:
        target = target.strip()
        if not target:
            raise VCR1DataError("correction target is empty")
        response_ids = [
            int(value) for value in tokenizer.encode(target, add_special_tokens=False)
        ]
        if tokenizer.eos_token_id is None:
            raise VCR1DataError("tokenizer exposes no EOS token")
        response_ids.append(int(tokenizer.eos_token_id))
    total = len(prompt_ids) + workspace_slots + len(response_ids)
    if total > max_sequence_length:
        return None
    return CorrectionTokens(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        question_mask=question_mask,
        draft_mask=draft_mask,
    )


__all__ = [
    "CorrectionTokens",
    "DRAFT_CLOSE",
    "DRAFT_OPEN",
    "QUESTION_CLOSE",
    "QUESTION_OPEN",
    "VCR1DataError",
    "correction_user_content",
    "render_correction_prompt",
    "tokenize_correction_example",
]
